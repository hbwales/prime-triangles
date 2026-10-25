// Segmented mod-30 prime sieve, returns per-row counts for CSV output.
// Build (x64 VS developer shell): nvcc -O3 -std=c++20 -arch=native --cudart=static tri.cu -o tri.exe
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <io.h>

using u8 = unsigned char;
using u32 = unsigned int;
using u64 = unsigned long long;
using i32 = int;
using i64 = long long;
using Clock = std::chrono::steady_clock;

constexpr u32 max_rows = 0xffffffffu / 30;      // locate() forms 30*remainder and up to 30*p in u32.
constexpr u32 batch_bytes = 256 * 1024 * 1024;
constexpr u32 tile_bytes = 96 * 1024;
constexpr u32 mark_threads = 1024;
constexpr u32 sparse_prime_limit = 2000000;
constexpr u32 sparse_window_bytes = 16 * 1024 * 1024;
static_assert(batch_bytes + max_rows < (1u << 29)); // Sparse carry packs a 29-bit byte offset and 3-bit phase.
static_assert(batch_bytes <= 1024 * 1024 * 1024);  // Must fit within i32 including one prime step.
static_assert(u64(batch_bytes) + 3ull * max_rows < 0x80000000ull);

constexpr u32 residue_mask = (1u<<1) | (1u<<7) | (1u<<11) | (1u<<13) | (1u<<17) | (1u<<19) | (1u<<23) | (1u<<29);
constexpr u8 host_bits[30] = {0,1,0,0,0,0,0,2,0,0,0,4,0,8,0,0,0,16,0,32,0,0,0,64,0,0,0,0,0,128};
__device__ const u32 bits[30] = {0,1,0,0,0,0,0,2,0,0,0,4,0,8,0,0,0,16,0,32,0,0,0,64,0,0,0,0,0,128};
__device__ const u32 residues[8] = {1,7,11,13,17,19,23,29};
__device__ const u32 through[30] = {0,1,1,1,1,1,1,3,3,3,3,7,7,15,15,15,15,31,31,63,63,63,63,127,127,127,127,127,127,255};

constexpr u32 pattern_period = 7 * 11 * 13 * 17;
static_assert(30ull * (sparse_window_bytes + pattern_period) <= 0xffffffffull);
constexpr u32 small_prime_bits = host_bits[7] | host_bits[11] | host_bits[13] | host_bits[17];

struct BasePrime {
    u32 prime;
    u32 reciprocal32; // floor(2^32 / p)
    u64 reciprocal64; // floor(2^64 / p)
};

// Reduce the hot per-tile arithmetic to a bounded 32-bit problem.
// Preparation does the full-range division once per prime per batch, rather than per tile.
struct alignas(16) PreparedPrime {
    u32 prime;
    i32 anchor;
    u32 reciprocal32;
    u32 wheel;
}; static_assert(sizeof(PreparedPrime)==16);

struct BatchParams {
    u64 first_byte;
    u32 byte_count;
    u32 num_primes;
    u32 first_row;
    u32 row_count;
    u32 pattern_word_offset;
};

struct RowCounts {
    u32 left;
    u32 centre;
    u32 right;
};

// floor(2^32/p) gives an underestimate within one for every u32 n.
// A high multiply and one exact correction replace all floating-point work.
__device__ u32 floor32(u32 n, u32 p, u32 reciprocal32) {
    u32 q = __umulhi(n, reciprocal32);
    return q + u32(n - q*p >= p);
}

// The same reciprocal identity applies to full-range 64-bit integers.
__device__ u64 floor64(u64 n, u32 p, u64 reciprocal64) {
    u64 q = __umul64hi(n, reciprocal64);
    return q + u64(n- q*p >= p);
}

__global__ void prepare_kernel(const BasePrime* base_primes, PreparedPrime* prepared_primes, BatchParams batch) {
    u32 k = blockIdx.x * 256 + threadIdx.x;
    if(k >= batch.num_primes) return;
    u32 p = base_primes[k].prime;
    // Low 24 bits hold eight 3-bit wheel-bit indices, high bits enforce p*p for the first turn;
    u64 turn = max(u64(p/30), floor64(batch.first_byte, p, base_primes[k].reciprocal64));
    u32 wheel = (turn == p/30 ? p % 30 : 0) << 24;
    #pragma unroll
    for (u32 phase = 0; phase < 8; ++phase) {
        u32 r = (p % 30) * residues[phase] % 30;
        wheel |= __popc(residue_mask & ((1u << r) - 1u)) << (phase*3);
    }
    i32 anchor = i32(i64(turn*p) - i64(batch.first_byte)); // byte position relative to this batch.
    prepared_primes[k] = {p, anchor, base_primes[k].reciprocal32, wheel};
}

__device__ void locate(PreparedPrime prime_record, u32 tile_begin, i32& turn_byte, u32& min_multiplier_residue) {
    u32 p = prime_record.prime;
    i32 delta = i32(tile_begin) - prime_record.anchor;
    u32 turn = delta > 0 ? floor32(u32(delta), p, prime_record.reciprocal32) : 0;
    turn_byte = prime_record.anchor + i32(turn * p);
    i32 remainder = i32(tile_begin) - turn_byte;
    u32 numerator = remainder > 0 ? u32(remainder) * 30 : 0;
    min_multiplier_residue = floor32(numerator, p, prime_record.reciprocal32);
    if(min_multiplier_residue * p < numerator) ++min_multiplier_residue;
    if(turn==0) min_multiplier_residue = max(min_multiplier_residue, (prime_record.wheel>>24));
}

__global__ void mark_kernel(u32* sieve, const PreparedPrime* prepared_primes, const u32* pattern, BatchParams batch) {
    extern __shared__ u32 tile[];
    u32 tile_begin = blockIdx.x * tile_bytes;
    u32 tile_size = min(tile_bytes, batch.byte_count - tile_begin);
    // Initialize directly in shared memory: no separate global fill/read pass.
    for (u32 i=threadIdx.x; i<tile_size/4; i+=mark_threads) {
        u32 word = tile_begin/4 + i;
        u32 value = pattern[(batch.pattern_word_offset + word) % pattern_period];
        if(batch.first_byte==0 && word==0) value=(value | small_prime_bits) & ~1u;
        tile[i]=value;
    }
    __syncthreads();

    // A group of 32 invocations shares each dense prime: eight wheel phases, with four interleaved progressions per phase.
    // Shared atomics commute.
    // CUDA warps contain 32 lanes; each handles one dense prime at a time.
    u32 lane = threadIdx.x % 32;
    for(u32 k = threadIdx.x / 32; k < batch.num_primes; k += mark_threads / 32) {
        PreparedPrime prime_record = prepared_primes[k];
        u32 p = prime_record.prime;
        if(p > 1024u) break;
        // A future anchor cannot strike this tile.
        // locate() enforces p*p even when that square falls later than the anchor within its wheel turn.
        if(prime_record.anchor >= i32(tile_begin+tile_size)) continue;
        i32 turn_byte;
        u32 min_multiplier_residue;
        locate(prime_record, tile_begin, turn_byte, min_multiplier_residue);
        u32 phase = lane % 8;
        i32 strike_byte = turn_byte + i32((p*residues[phase]) / 30);
        if(residues[phase] < min_multiplier_residue) strike_byte += p;
        u32 at = u32(strike_byte - i32(tile_begin)) + (lane/8)*p;
        u32 bit = (1u<<((prime_record.wheel>>(phase*3))&7));
        for(;at<tile_size; at+=4*p) atomicAnd(&tile[at>>2], ~(bit<<((at&3)*8)));
    }

    // Load one record ahead so its cache latency overlaps the current prime's arithmetic/strikes.
    // Guard both the initial and final partial iterations.
    PreparedPrime next = {};
    if(threadIdx.x < batch.num_primes) next = prepared_primes[threadIdx.x];
    for (u32 k = threadIdx.x; k < batch.num_primes; k += mark_threads) {
        PreparedPrime prime_record=next;
        if(k+mark_threads < batch.num_primes) next = prepared_primes[k+mark_threads];
        u32 p = prime_record.prime;
        if(p <= 1024u) continue;
        if(prime_record.anchor >= i32(tile_begin + tile_size)) continue;
        i32 turn_byte;
        u32 min_multiplier_residue;
        locate(prime_record, tile_begin, turn_byte, min_multiplier_residue);
        if(p > tile_bytes) {
            // Sparse primes usually have no hit.
            // Start at the next candidate multiplier and stop as soon as its byte lies beyond this tile.
            u32 phase = __popc(residue_mask & ((1u<<min_multiplier_residue)-1u));
            if(phase == 8) {phase=0; turn_byte+=p;}
            for(;;) {
                i32 strike_byte = turn_byte + i32((p*residues[phase])/30);
                if(strike_byte >= i32(tile_begin+tile_size)) break;
                u32 at = u32(strike_byte-i32(tile_begin));
                u32 bit = (1u << ((prime_record.wheel >> (phase*3)) & 7));
                atomicAnd(&tile[at>>2], ~(bit<<((at&3) * 8)));
                if(++phase == 8) {phase=0; turn_byte+=p;}
            }
            continue;
        }
        for (u32 phase = 0; phase < 8; ++phase) {
            // p*(30*q+r)/30 = p*q + floor(p*r/30); the remaining division is only 32-bit.
            // Each phase then has byte stride p.
            i32 strike_byte = turn_byte + i32((p*residues[phase])/30);
            if(residues[phase] < min_multiplier_residue) strike_byte+=p;
            u32 at = u32(strike_byte-i32(tile_begin));
            u32 bit = (1u<<((prime_record.wheel>>(phase*3))&7));
            for(;at < tile_size; at += p) atomicAnd(&tile[at>>2], ~(bit<<((at&3)*8)));
        }
    }
    __syncthreads();
    for(u32 i=threadIdx.x; i<tile_size/4; i+=mark_threads) sieve[tile_begin/4+i] = tile[i];
}


// One thread walks a sparse prime's hits in multiplier order across windows and batches.
// Carry packs the next byte offset in bits 0..28 and its wheel phase in bits 29..31.
__global__ void scatter_kernel(u32* sieve, const PreparedPrime* primes, u32* sparse_state,
    u32 sparse_begin, u32 previous_active, u32 num_primes, u32 byte_count, u32 window_offset, u32 window_mod) {
    u32 k = sparse_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_primes) return;
    PreparedPrime record = primes[k];
    u32 p = record.prime, at, phase;
    if (k >= previous_active) {
        // Newly active primes use locate() throughout their first batch to enforce p*p.
        i32 turn_byte;
        u32 min_residue;
        locate(record, window_offset, turn_byte, min_residue);
        phase = __popc(residue_mask & ((1u << min_residue) - 1u));
        if (phase == 8) { phase = 0; turn_byte += p; }
        at = u32(turn_byte + i32(p * residues[phase] / 30) - i32(window_offset));
    } else {
        u32 state = sparse_state[k - sparse_begin];
        at = state & 0x1fffffffu;
        phase = state >> 29;
    }
    while (at < byte_count) {
        u32 product = p * residues[phase];
        u32 bit = 1u << ((record.wheel >> (phase * 3)) & 7);
        // All multiples of 7, 11, 13 and 17 were already cleared by the pre-sieve.
        u32 value = 30 * (window_mod + at) + product % 30;
        if (value % 7 && value % 11 && value % 13 && value % 17) atomicAnd(&sieve[at >> 2], ~(bit << ((at & 3) * 8)));
        u32 old_offset = product / 30;
        if (++phase == 8) { phase = 0; at += p; }
        at += p * residues[phase] / 30 - old_offset;
    }
    sparse_state[k - sparse_begin] = (at - byte_count) | (phase << 29);
}

__device__ u32 count_range(u64 a, u64 b, const u32* sieve, BatchParams batch) {
    a = max(a, batch.first_byte * 30ull);
    b = min(b, (batch.first_byte + batch.byte_count) * 30ull - 1ull);
    if (a > b) return 0;
    u32 first_word = u32(a / 120ull - batch.first_byte / 4ull);
    u32 last_word  = u32(b / 120ull - batch.first_byte / 4ull);
    u32 a_offset = u32(a % 120ull), b_offset = u32(b % 120ull);
    u32 lowbit  = a_offset / 30 * 8 + (a_offset % 30 == 0 ? 0 : __popc(through[a_offset % 30 - 1]));
    u32 highbit = b_offset / 30 * 8 + __popc(through[b_offset % 30]);
    u32 lowmask = 0xffffffffu << lowbit;
    u32 highmask = highbit == 32 ? 0xffffffffu : (1u << highbit) - 1;
    u32 result = 0;
    for (u32 i = first_word + threadIdx.x; i <= last_word; i += 256) {
        u32 word = sieve[i];
        if (i == first_word) word &= lowmask;
        if (i == last_word) word &= highmask;
        result += __popc(word);
    }
    return result;
}

__global__ void count_kernel(const u32* sieve, RowCounts* counts, BatchParams batch) {
    __shared__ uint2 sums[8];
    u32 row = batch.first_row + blockIdx.x;
    u64 n = u64(row);
    u64 a   = (n-1ull) * (n-1ull) + 1ull;
    u64 mid = n * n - n + 1ull;
    u64 b   = n * n;
    u32 left  = count_range(a, mid-1ull, sieve, batch);
    u32 right = count_range(mid+1ull, b, sieve, batch);

    // Reduce from 256 lanes => 8 => 1; (a warp = 32 lanes)
    left  = __reduce_add_sync(0xffffffffu, left);
    right = __reduce_add_sync(0xffffffffu, right);
    if (threadIdx.x % 32 == 0) sums[threadIdx.x/32] = make_uint2(left, right);
    __syncthreads();
    if (threadIdx.x >= 32) return;
    left  = __reduce_add_sync(0xffffffffu, threadIdx.x < 8 ? sums[threadIdx.x].x : 0);
    right = __reduce_add_sync(0xffffffffu, threadIdx.x < 8 ? sums[threadIdx.x].y : 0);
    // A row may span batches, but only one group updates it in each batch.
    // The default CUDA stream serializes those partial contributions.
    if (threadIdx.x == 0) {
        u32 centre = 0;
        if (mid >= batch.first_byte * 30ull && mid < (batch.first_byte + batch.byte_count) * 30ull) {
            u32 i = u32(mid / 30ull - batch.first_byte);
            centre = u32((sieve[i >> 2] & (bits[u32(mid % 30ull)] << ((i & 3)*8))) != 0);
        }
        counts[row].left += left;
        counts[row].centre += centre;
        counts[row].right += right;
    }
}


static std::atomic<bool> stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free);
static void interrupt(int) {
    std::signal(SIGINT, interrupt); // The CRT resets the handler before invoking it.
    stop_requested.store(true, std::memory_order_relaxed);
}

static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::format("{}: {}", operation, cudaGetErrorString(result)));
}
#define CHECK(call) check((call), #call)

static void mark_batch(u32* sieve, const PreparedPrime* primes, const u32* pattern,
    u32* sparse_state, u32 sparse_begin, u32 previous_active, BatchParams batch) {
    BatchParams tile_batch = batch;
    tile_batch.num_primes = std::min(batch.num_primes, sparse_begin);
    mark_kernel<<<(batch.byte_count + tile_bytes - 1) / tile_bytes, mark_threads, tile_bytes>>>(sieve, primes, pattern, tile_batch);
    CHECK(cudaGetLastError());
    if (batch.num_primes > sparse_begin) {
        for (u32 offset = 0; offset < batch.byte_count; offset += sparse_window_bytes) {
            u32 window_bytes = std::min(sparse_window_bytes, batch.byte_count - offset);
            scatter_kernel<<<(batch.num_primes - sparse_begin + 255) / 256, 256>>>(
                sieve + offset / 4, primes, sparse_state,
                sparse_begin, previous_active, batch.num_primes, window_bytes, offset, u32((batch.first_byte+offset)%pattern_period));
            CHECK(cudaGetLastError());
        }
    }
}

static void write_output(std::FILE* stream, std::string_view text) {
    if (std::fwrite(text.data(), 1, text.size(), stream) != text.size())
        throw std::runtime_error("Output write failed");
}

static void flush_output() {
    if (std::fflush(stdout)) throw std::runtime_error("CSV flush failed");
}

static void write_csv_row(u64 row, u64 left, u64 centre, u64 right) {
    // Four u64 values (up to 20 digits each), three commas and a newline.
    char buffer[84];
    char* end = buffer;
    for (u64 value : {row, left, centre, right}) {
        end = std::to_chars(end, buffer + sizeof(buffer), value).ptr;
        *end++ = ',';
    }
    end[-1] = '\n';
    write_output(stdout, {buffer, std::size_t(end - buffer)});
}

static u64 integer_sqrt(u64 n) {
    if (!n) return 0;
    u64 r = static_cast<u64>(std::sqrt(static_cast<double>(n)));
    while (r > n/r) --r;
    while (r+1 <= n/(r+1)) ++r;
    return r;
}

static std::vector<BasePrime> make_base_primes(u32 max_prime) {
    std::vector<u8> sieve(std::size_t(max_prime)+1, 1);
    for (u32 p=2; p<=max_prime/p; ++p)
        if (sieve[p]) for (u32 n=p*p; n<=max_prime; n+=p) sieve[n]=0;
    std::vector<BasePrime> base_primes;
    auto append = [&](u32 p) {
        // p is odd, so it cannot divide 2^64: floor((2^64-1)/p) == floor(2^64/p).
        base_primes.push_back({p, u32((1ull<<32)/p), ~0ull/p});
    };
    for (u32 p=19; p<=max_prime; p+=2) if (sieve[p]) append(p);
    if (base_primes.empty()) append(19); // Harmless dummy for tiny inputs.
    return base_primes;
}

static std::array<u32, pattern_period> make_presieve_pattern() {
    std::array<u8, pattern_period> pattern;
    pattern.fill(255);
    for (u32 p : {7u, 11u, 13u, 17u})
        for (u32 n=p; n<pattern_period*30; n+=2*p) pattern[n/30] &= ~host_bits[n%30];
    std::array<u32, pattern_period> words{};
    for (u32 i=0; i<pattern_period; ++i)
        for (u32 j=0; j<4; ++j) words[i] |= u32(pattern[(i*4+j) % pattern_period]) << (j*8);
    return words;
}

struct Race {
    u64 left=0, centre=0, right=0, left_rows=0, right_rows=0, ties=0, switches=0, last_switch=0;
    i32 previous_leader=0;
    i64 difference() const { return i64(right) - i64(left); }
    void emit(u64 row, const RowCounts& counts) {
        left += counts.left; centre += counts.centre; right += counts.right;
        i32 leader = (right>left)-(left>right);
        if (leader<0) ++left_rows; else if (leader>0) ++right_rows; else ++ties;
        if (leader && leader!=previous_leader) { ++switches; last_switch=row; previous_leader=leader; }
        write_csv_row(row, left, centre, right);
    }
};

static void run(u32 num_rows) {
    const auto start = Clock::now();
    CHECK(cudaSetDevice(0));
    CHECK(cudaFuncSetAttribute(mark_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, tile_bytes));
    // run() is called once; process exit reclaims these GPU allocations and events.
    u32 *gpu_sieve;
    u32 *gpu_pattern;
    RowCounts* gpu_counts;
    BasePrime* gpu_base_primes;
    PreparedPrime* gpu_prepared_primes;
    u32* gpu_sparse_state = nullptr;
    std::array<cudaEvent_t,4> events{};
    std::array<double,3> stage_seconds{};
    for (auto& event : events) CHECK(cudaEventCreate(&event));
    const auto base_primes = make_base_primes(num_rows);
    const u64 last_number = u64(num_rows) * num_rows;
    const u64 total_bytes = ((last_number/30+1)+3) & ~3ull;
    CHECK(cudaMalloc(&gpu_sieve, std::size_t(std::min(total_bytes, u64(batch_bytes)))));
    CHECK(cudaMalloc(&gpu_base_primes, base_primes.size()*sizeof(BasePrime)));
    CHECK(cudaMemcpy(gpu_base_primes, base_primes.data(), base_primes.size()*sizeof(BasePrime), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&gpu_prepared_primes, base_primes.size()*sizeof(PreparedPrime)));
    // Short runs benefit from using the original tiled path throughout.
    const u32 tiled_prime_limit = num_rows <= 8000000 ? 8000000 : sparse_prime_limit;
    u32 sparse_begin = 0;
    while (sparse_begin < base_primes.size() && base_primes[sparse_begin].prime <= tiled_prime_limit) ++sparse_begin;
    if (sparse_begin < base_primes.size())
        CHECK(cudaMalloc(&gpu_sparse_state, (base_primes.size() - sparse_begin) * sizeof(u32)));
    const auto pattern = make_presieve_pattern();
    CHECK(cudaMalloc(&gpu_pattern, sizeof(pattern)));
    CHECK(cudaMemcpy(gpu_pattern, pattern.data(), sizeof(pattern), cudaMemcpyHostToDevice));
    std::vector<RowCounts> counts; // Reused readback slice; never allocate num_rows CPU entries.
    CHECK(cudaMalloc(&gpu_counts, (std::size_t(num_rows) + 1) * sizeof(RowCounts)));
    CHECK(cudaMemset(gpu_counts, 0, (std::size_t(num_rows) + 1) * sizeof(RowCounts)));
    // 2, 3 and 5 have no wheel bits: seed rows 2 and 3, clipping for tiny inputs.
    constexpr std::array<RowCounts, 4> small_counts{{{}, {}, {1, 1, 0}, {1, 0, 0}}};
    CHECK(cudaMemcpy(gpu_counts, small_counts.data(),
        std::min(std::size_t(num_rows) + 1, small_counts.size()) * sizeof(RowCounts), cudaMemcpyHostToDevice));

    write_output(stdout, "n,l,c,r\n");
    flush_output();
    u64 rows_written=0, batches=0;
    u32 num_active_primes=0;
    Race race;
    auto last_progress = start;
    double gpu_seconds=0, readback_seconds=0, output_seconds=0;
    for (u64 first_byte=0; first_byte<total_bytes && !stop_requested.load(std::memory_order_relaxed); first_byte+=batch_bytes) {
        u32 byte_count = u32(std::min(total_bytes - first_byte, u64(batch_bytes)));
        u64 last_batch_number = std::min((first_byte+byte_count)*30-1, last_number);
        u32 first_row = first_byte ? u32(integer_sqrt(first_byte*30-1)+1) : 1;
        u64 last_complete_row = integer_sqrt(last_batch_number);
        u64 last_batch_row = last_complete_row + (last_complete_row*last_complete_row < last_batch_number);
        const u32 previous_active = num_active_primes;
        while (num_active_primes < base_primes.size() && u64(base_primes[num_active_primes].prime)*base_primes[num_active_primes].prime<=last_batch_number)
            ++num_active_primes;
        const auto gpu_start = Clock::now();
        const BatchParams batch{
            first_byte,
            byte_count,
            std::max(num_active_primes, 1u),
            first_row,
            u32(last_batch_row - first_row + 1),
            u32((first_byte/4) % pattern_period)
        };
        // All grids fit this GPU's 2^31-1 x dimension, even at the row limit.
        CHECK(cudaEventRecord(events[0]));
        prepare_kernel<<<(batch.num_primes+255)/256, 256>>>(gpu_base_primes, gpu_prepared_primes, batch);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(events[1]));
        mark_batch(gpu_sieve, gpu_prepared_primes, gpu_pattern, gpu_sparse_state,
            sparse_begin, previous_active, batch);
        CHECK(cudaEventRecord(events[2]));
        count_kernel<<<batch.row_count, 256>>>(gpu_sieve, gpu_counts, batch);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(events[3]));
        CHECK(cudaEventSynchronize(events[3]));
        for (std::size_t i = 0; i < stage_seconds.size(); ++i) {
            float milliseconds;
            CHECK(cudaEventElapsedTime(&milliseconds, events[i], events[i+1]));
            stage_seconds[i] += milliseconds / 1000.0;
        }
        gpu_seconds += seconds(gpu_start);
        ++batches;

        std::size_t num_complete = std::size_t(last_complete_row - rows_written);
        const auto readback_start = Clock::now();
        counts.resize(num_complete);
        if (num_complete) CHECK(cudaMemcpy(counts.data(), gpu_counts + rows_written + 1,
            num_complete * sizeof(RowCounts), cudaMemcpyDeviceToHost));
        readback_seconds += seconds(readback_start);

        const auto output_start=Clock::now();
        for (const RowCounts& row_counts : counts) race.emit(++rows_written, row_counts);
        output_seconds+=seconds(output_start);
        if (seconds(last_progress) >= 5) {
            flush_output();
            write_output(stderr, std::format("# Progress: row={}; batches={}; switches={}; last_switch={}; R-L={}; seconds={:.3f}\n",
                rows_written, batches, race.switches, race.last_switch, race.difference(), seconds(start)));
            last_progress = Clock::now();
        }
    }

    const auto flush_start = Clock::now();
    flush_output();
    output_seconds += seconds(flush_start);
    if (stop_requested.load(std::memory_order_relaxed)) write_output(stderr, "# Interrupted; complete CSV records flushed\n");
    write_output(stderr, std::format("# Time:{:.6f}s; GPU:{:.6f}; prepare:{:.6f}; mark:{:.6f}; count:{:.6f}; to_cpu:{:.6f}; output_csv:{:.6f}\n",
        seconds(start), gpu_seconds, stage_seconds[0], stage_seconds[1], stage_seconds[2], readback_seconds, output_seconds));
    write_output(stderr, std::format("# row:{}; num:{}; L:{}; R:{}; C:{}; R-L:{}\n",
        rows_written, rows_written*rows_written, race.left, race.right, race.centre, race.difference()));
    write_output(stderr, std::format("# L>R:{}; R>L:{}; R=L:{}\n", race.left_rows, race.right_rows, race.ties));
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected one positive row limit");
        u32 num_rows = 0;
        const std::string_view text = argv[1];
        const auto [end, error] = std::from_chars(text.data(), text.data()+text.size(), num_rows);
        if (error != std::errc{} || end != text.data() + text.size() || !num_rows || num_rows > max_rows)
            throw std::runtime_error(std::format("Row limit must be an integer from 1 to {}", max_rows));
        if (std::signal(SIGINT, interrupt) == SIG_ERR) throw std::runtime_error("Signal handler installation failed");
        if (_setmode(_fileno(stdout), _O_BINARY) == -1) throw std::runtime_error("Cannot set binary stdout");
        if (_setmode(_fileno(stderr), _O_BINARY) == -1) throw std::runtime_error("Cannot set binary stderr");
        static char csv_buffer[65536];
        if (std::setvbuf(stdout, csv_buffer, _IOFBF, sizeof(csv_buffer))) throw std::runtime_error("Cannot buffer stdout");
        run(num_rows);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "# Error: %s\n", e.what());
        return 1;
    }
}
