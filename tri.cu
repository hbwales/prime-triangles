// RTX 5080 Laptop (SM 12.0): segmented mod-30 sieve, 96 KiB tiles, 1024 threads.
// Only per-row counts return to the CPU for CSV. No alternate GPU configurations.
// NVCC compiles host and kernels together; no runtime compiler or DLL lookup.
// Build (x64 VS developer shell): nvcc -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -lineinfo --cudart=static -Xcompiler /O2 tri.cu -o tri.exe
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <io.h>

using u32 = unsigned int;
using u64 = unsigned long long;
using i64 = long long;
using Clock = std::chrono::steady_clock;
constexpr u32 batch_bytes = 256 * 1024 * 1024;
constexpr u32 tile_bytes = 96 * 1024, mark_threads = 1024;
// locate() forms 30*remainder and up to 30*p in u32. See the bounds audit.
constexpr u32 max_supported_rows = 0xffffffffu / 30;
// Relative signed byte arithmetic must stay within i32, including one prime step.
static_assert(batch_bytes <= 1024 * 1024 * 1024);
static_assert(u64(batch_bytes)+3ull*max_supported_rows < 0x80000000ull);
constexpr u32 pattern_period = 7 * 11 * 13 * 17;
constexpr unsigned char host_bits[30] = {0,1,0,0,0,0,0,2,0,0,0,4,0,8,0,0,0,16,0,32,0,0,0,64,0,0,0,0,0,128};
// Exact reciprocals floor(2^32/p) and floor(2^64/p), computed once on the CPU.
struct Job { u32 prime, reciprocal; u64 reciprocal64; };
struct Params { u64 base_block; u32 byte_count, job_count, first_row, row_count, template_offset; };
static_assert(sizeof(void*) == 8 && sizeof(Job) == 16 && offsetof(Job, reciprocal64) == 8 && sizeof(Params) == 32);

using uint = unsigned int;
using uint64_t = unsigned long long;
// Warp lanes select different entries: ordinary read-only loads avoid constant
// memory's serialization of distinct addresses within a warp.
__device__ const uint bits[30] = {0,1,0,0,0,0,0,2,0,0,0,4,0,8,0,0,0,16,0,32,0,0,0,64,0,0,0,0,0,128};
__device__ const uint residues[8] = {1,7,11,13,17,19,23,29};
__device__ const uint through[30] = {0,1,1,1,1,1,1,3,3,3,3,7,7,15,15,15,15,31,31,63,63,63,63,127,127,127,127,127,127,255};
const uint residue_mask = (1u<<1)|(1u<<7)|(1u<<11)|(1u<<13)|(1u<<17)|(1u<<19)|(1u<<23)|(1u<<29);
// The same reciprocal identity applies to full-range 64-bit integers.
__device__ uint64_t floor64(uint64_t n,uint p,uint64_t reciprocal) {
    uint64_t q=__umul64hi(n,reciprocal);
    return q+uint64_t(n-q*p>=p);
}

// Reduce the hot per-tile arithmetic to a bounded 32-bit problem. Preparation
// does the full-range division once per prime per batch, rather than per tile.
// Keep all four fields in one aligned vector load. CUDA allocations satisfy
// this alignment, and every record occupies exactly 16 bytes.
struct __align__(16) Prepared { uint prime; int anchor; uint reciprocal, wheel; };
static_assert(sizeof(Prepared)==16);
extern "C" __global__ void prepare_kernel(const Job* jobs, Prepared* prepared, Params pc) {
    uint k=blockIdx.x*256+threadIdx.x;
    if(k>=pc.job_count) return;
    uint p=jobs[k].prime;
    uint64_t turn=floor64(pc.base_block,p,jobs[k].reciprocal64);
    turn=max(turn,uint64_t(p/30));
    // The low 24 bits hold eight 3-bit wheel-bit indices. High bits enforce p*p
    // for the first turn; anchor is a byte position relative to this batch.
    uint wheel=(turn==p/30 ? p%30 : 0)<<24;
    #pragma unroll
    for(uint phase=0;phase<8;++phase) {
        uint r=(p%30)*residues[phase]%30;
        wheel|=__popc(residue_mask&((1u<<r)-1u))<<(phase*3);
    }
    prepared[k]={p,int((long long)(turn*p)-(long long)pc.base_block),jobs[k].reciprocal,wheel};
}
// floor(2^32/p) gives an underestimate within one for every u32 n.
// A high multiply and one exact correction replace all floating-point work.
__device__ uint floor32(uint n,uint p,uint reciprocal) {
    uint q=__umulhi(n,reciprocal);
    return q+uint(n-q*p>=p);
}
__device__ void locate(Prepared job,uint begin,int& first,uint& residue) {
    uint p=job.prime;
    int delta=int(begin)-job.anchor;
    uint turn=delta>0 ? floor32(uint(delta),p,job.reciprocal) : 0;
    first=job.anchor+int(turn*p);
    int remainder=int(begin)-first;
    uint numerator=remainder>0 ? uint(remainder)*30 : 0;
    residue=floor32(numerator,p,job.reciprocal);
    if(residue*p<numerator) ++residue;
    if(turn==0) residue=max(residue,(job.wheel>>24));
}

extern "C" __global__ void mark_kernel(uint* sieve, const Prepared* jobs, const uint* pattern, Params pc) {
    extern __shared__ uint tile[];
    uint begin = blockIdx.x * tile_bytes;
    uint bytes = min(tile_bytes, pc.byte_count - begin);
    // Initialize directly in shared memory: no separate global fill/read pass.
    for (uint i=threadIdx.x; i<bytes/4; i+=mark_threads) {
        uint word=begin/4+i;
        uint value=pattern[(pc.template_offset+word)%17017];
        if(pc.base_block==0 && word==0) value=(value|30u)&~1u;
        tile[i]=value;
    }
    __syncthreads();


    // A group of 32 invocations shares each dense prime: eight wheel phases,
    // with four interleaved progressions per phase. Shared atomics commute.
    // CUDA warps contain 32 lanes; each handles one dense prime at a time.
    uint lane=threadIdx.x%32;
    for(uint k=threadIdx.x/32;k<pc.job_count;k+=mark_threads/32) {
        Prepared job=jobs[k];
        uint p=job.prime;
        if(p>1024u) break;
        // A future anchor cannot strike this tile. locate() enforces p*p even
        // when that square falls later than the anchor within its wheel turn.
        if(job.anchor>=int(begin+bytes)) continue;
        int first;
        uint residue;
        locate(job,begin,first,residue);
        uint phase=lane%8;
        int initial=first+int((p*residues[phase])/30);
        if(residues[phase]<residue) initial+=p;
        uint at=uint(initial-int(begin))+(lane/8)*p;
        uint bit=(1u<<((job.wheel>>(phase*3))&7));
        for(;at<bytes;at+=4*p)
            atomicAnd(&tile[at>>2],~(bit<<((at&3)*8)));
    }
    // Load one record ahead so its cache latency overlaps the current prime's
    // arithmetic/strikes. Guard both the initial and final partial iterations.
    Prepared next={};
    if(threadIdx.x<pc.job_count) next=jobs[threadIdx.x];
    for (uint k=threadIdx.x; k<pc.job_count; k+=mark_threads) {
        Prepared job=next;
        if(k+mark_threads<pc.job_count) next=jobs[k+mark_threads];
        uint p=job.prime;
        if(p<=1024u) continue;
        if(job.anchor>=int(begin+bytes)) continue;
        int first;
        uint residue;
        locate(job,begin,first,residue);
        if(p>tile_bytes) {
            // Sparse primes usually have no hit. Start at the next candidate
            // multiplier and stop as soon as its byte lies beyond this tile.
            uint phase=__popc(residue_mask & ((1u<<residue)-1u));
            if(phase==8) {phase=0;first+=p;}
            for(;;) {
                int initial=first+int((p*residues[phase])/30);
                if(initial>=int(begin+bytes)) break;
                uint at=uint(initial-int(begin));
                uint bit=(1u<<((job.wheel>>(phase*3))&7));
                atomicAnd(&tile[at>>2],~(bit<<((at&3)*8)));
                if(++phase==8) {phase=0;first+=p;}
            }
            continue;
        }
        for (uint phase=0;phase<8;++phase) {
            // p*(30*q+r)/30 = p*q + floor(p*r/30); the remaining
            // division is only 32-bit. Each phase then has byte stride p.
            int initial=first+int((p*residues[phase])/30);
            if(residues[phase]<residue) initial+=p;
            uint at=uint(initial-int(begin));
            uint bit=(1u<<((job.wheel>>(phase*3))&7));
            for(;at<bytes;at+=p)
                atomicAnd(&tile[at>>2],~(bit<<((at&3)*8)));
        }
    }
    __syncthreads();
    for(uint i=threadIdx.x;i<bytes/4;i+=mark_threads)
        sieve[begin/4+i]=tile[i];
}
__device__ uint rangeCount(uint64_t a, uint64_t b, const uint* sieve, Params pc) {
    a = max(a, pc.base_block * 30ull);
    b = min(b, (pc.base_block + pc.byte_count) * 30ull - 1ull);
    if (a > b) return 0;
    uint first = uint(a / 120ull - pc.base_block / 4ull);
    uint last = uint(b / 120ull - pc.base_block / 4ull);
    uint ar = uint(a % 120ull), br = uint(b % 120ull);
    uint lowbit = ar / 30 * 8 + (ar % 30 == 0 ? 0 : __popc(through[ar % 30 - 1]));
    uint highbit = br / 30 * 8 + __popc(through[br % 30]);
    uint lowmask = 0xffffffffu << lowbit;
    uint highmask = highbit == 32 ? 0xffffffffu : (1u << highbit) - 1;
    uint result = 0;
    for (uint i = first + threadIdx.x; i <= last; i += 256) {
        uint word = sieve[i];
        if (i == first) word &= lowmask;
        if (i == last) word &= highmask;
        result += __popc(word);
    }
    return result;
}

extern "C" __global__ void count_kernel(const uint* sieve, uint* counts, Params pc) {
    __shared__ uint2 sums[8];
    uint row = pc.first_row + blockIdx.x;
    uint64_t k = uint64_t(row);
    uint64_t a = (k-1ull)*(k-1ull)+1ull, mid = k*k-k+1ull, b = k*k;
    uint l = rangeCount(a, mid-1ull, sieve, pc), r = rangeCount(mid+1ull, b, sieve, pc);
    // Blackwell reduces 32 integer lanes in hardware. Only eight warp totals
    // cross shared memory, replacing the old 256-entry tree and eight barriers.
    l=__reduce_add_sync(0xffffffffu,l);
    r=__reduce_add_sync(0xffffffffu,r);
    if (threadIdx.x%32==0) sums[threadIdx.x/32]=make_uint2(l,r);
    __syncthreads();
    if (threadIdx.x>=32) return;
    l=__reduce_add_sync(0xffffffffu,threadIdx.x<8 ? sums[threadIdx.x].x : 0);
    r=__reduce_add_sync(0xffffffffu,threadIdx.x<8 ? sums[threadIdx.x].y : 0);
    // A row may span batches, but only one group updates it in each batch.
    // The default CUDA stream serializes those partial contributions.
    if (threadIdx.x == 0) {
        uint c=0;
        if (mid >= pc.base_block*30ull && mid < (pc.base_block+pc.byte_count)*30ull) {
            uint i = uint(mid / 30ull - pc.base_block);
            c = uint((sieve[i >> 2] & (bits[uint(mid % 30ull)] << ((i & 3)*8))) != 0);
        }
        counts[row*3] += l;
        counts[row*3+1] += c;
        counts[row*3+2] += r;
    }
}


static std::atomic<bool> stop_requested{false};
static BOOL WINAPI interrupt(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    stop_requested.store(true, std::memory_order_relaxed);
    return TRUE;
}
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static void check(cudaError_t result, const char* operation) {
    if (result!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
#define CHECK(call) check((call), #call)

struct Cuda {
    Cuda() = default;
    Cuda(const Cuda&) = delete;
    Cuda& operator=(const Cuda&) = delete;
    std::array<void*,5> buffers{};
    std::array<cudaEvent_t,4> events{};
    std::array<double,3> stage_seconds{};
    bool initialized=false;

    ~Cuda() {
        if (initialized) cudaDeviceSynchronize();
        for (auto buffer : buffers) if (buffer) cudaFree(buffer);
        for (auto event : events) if (event) cudaEventDestroy(event);
    }
    void init() {
        CHECK(cudaSetDevice(0));
        initialized=true;
        CHECK(cudaFuncSetAttribute(mark_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,tile_bytes));
        for (auto& event : events) CHECK(cudaEventCreate(&event));
    }
    void allocate(size_t index,size_t bytes,const void* data=nullptr) {
        CHECK(cudaMalloc(&buffers[index],bytes));
        if (data) CHECK(cudaMemcpy(buffers[index],data,bytes,cudaMemcpyHostToDevice));
    }
    void batch(Params pc) {
        // All grids fit this GPU's 2^31-1 x dimension, even at the row limit.
        CHECK(cudaEventRecord(events[0]));
        prepare_kernel<<<(pc.job_count+255)/256,256>>>(static_cast<const Job*>(buffers[1]),static_cast<Prepared*>(buffers[4]),pc);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(events[1]));
        mark_kernel<<<(pc.byte_count+tile_bytes-1)/tile_bytes,mark_threads,tile_bytes>>>(static_cast<u32*>(buffers[0]),
            static_cast<const Prepared*>(buffers[4]),static_cast<const u32*>(buffers[2]),pc);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(events[2]));
        count_kernel<<<pc.row_count,256>>>(static_cast<const u32*>(buffers[0]),static_cast<u32*>(buffers[3]),pc);
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(events[3]));
        CHECK(cudaEventSynchronize(events[3]));
        for (size_t i=0;i<stage_seconds.size();++i) {
            float milliseconds;
            CHECK(cudaEventElapsedTime(&milliseconds,events[i],events[i+1]));
            stage_seconds[i]+=milliseconds/1000.0;
        }
    }
};

static u64 integer_root(u64 n) {
    if (!n) return 0;
    u64 r = static_cast<u64>(std::sqrt(static_cast<double>(n)));
    while (r > n/r) --r;
    while (r+1 <= n/(r+1)) ++r;
    return r;
}
static std::vector<Job> make_jobs(u32 limit) {
    std::vector<unsigned char> sieve(size_t(limit)+1, 1);
    for (u32 p=2; p<=limit/p; ++p)
        if (sieve[p]) for (u32 n=p*p; n<=limit; n+=p) sieve[n]=0;
    std::vector<Job> jobs;
    auto append = [&](u32 p) {
        // 2^64 is not representable in u64; divide 2^64-1 and account for the
        // remaining one. All calculations here and in the kernels are exact.
        jobs.push_back({p,u32((1ull<<32)/p),(~0ull/p)+u64((~0ull%p)==p-1)});
    };
    for (u32 p=19; p<=limit; p+=2) if (sieve[p]) append(p);
    if (jobs.empty()) append(19); // Harmless dummy for tiny limits.
    return jobs;
}
struct Race {
    u64 left=0, centre=0, right=0, left_rows=0, right_rows=0, ties=0, switches=0, last_switch=0;
    int previous=0;
    i64 difference() const { return i64(right)-i64(left); }
    void emit(u64 row, const u32* counts) {
        left+=counts[0]; centre+=counts[1]; right+=counts[2];
        int leader=(right>left)-(left>right);
        if (leader<0) ++left_rows; else if (leader>0) ++right_rows; else ++ties;
        if (leader && leader!=previous) { ++switches; last_switch=row; previous=leader; }
        if (std::fprintf(stdout,"%llu,%llu,%llu,%llu\n",row,left,centre,right)<0) throw std::runtime_error("CSV write failed");
    }
};
static void run(u32 max_rows) {
    const auto start = Clock::now();
    Cuda gpu;
    gpu.init();
    std::fprintf(stderr,"# RTX 5080 Laptop; batch=%u MiB; shared tile=%u KiB; mark threads=%u\n",batch_bytes/1048576,tile_bytes/1024,mark_threads);
    const auto jobs = make_jobs(max_rows);
    const u64 total_bytes = ((u64(max_rows)*max_rows/30+1)+3)&~3ull;
    gpu.allocate(0, size_t(std::min(total_bytes,u64(batch_bytes))));
    gpu.allocate(1, jobs.size()*sizeof(Job), jobs.data());
    gpu.allocate(4, jobs.size()*sizeof(Prepared));
    std::array<unsigned char,pattern_period> pattern;
    pattern.fill(255);
    for (u32 p : {7u,11u,13u,17u})
        for (u32 n=p; n<pattern_period*30; n+=2*p) pattern[n/30]&=~host_bits[n%30];
    std::array<u32,pattern_period> words{};
    for (u32 i=0; i<pattern_period; ++i)
        for (u32 j=0; j<4; ++j) words[i]|=u32(pattern[(i*4+j)%pattern_period])<<(j*8);
    gpu.allocate(2, sizeof(words), words.data());
    std::vector<u32> counts((size_t(max_rows)+1)*3);
    if (max_rows>=2) { counts[6]=1; counts[7]=1; }
    if (max_rows>=3) counts[9]=1;
    gpu.allocate(3, counts.size()*sizeof(u32), counts.data());
    if (std::fputs("n,l,c,r\n",stdout)<0 || std::fflush(stdout)) throw std::runtime_error("CSV write failed");
    u64 row=0, batches=0;
    u32 active=0;
    Race race;
    auto last_progress=start;
    double gpu_seconds=0, readback_seconds=0, output_seconds=0;
    for (u64 base=0; base<total_bytes && !stop_requested.load(std::memory_order_relaxed); base+=batch_bytes) {
        u32 bytes=u32(std::min(total_bytes-base,u64(batch_bytes)));
        u64 hi=std::min((base+bytes)*30-1,u64(max_rows)*max_rows);
        u32 first=base ? u32(integer_root(base*30-1)+1) : 1;
        u64 complete=integer_root(hi), last=complete+(complete*complete<hi);
        while (active<jobs.size() && u64(jobs[active].prime)*jobs[active].prime<=hi) ++active;
        const auto gpu_start=Clock::now();
        gpu.batch({base,bytes,std::max(active,1u),first,u32(last-first+1),u32((base/4)%pattern_period)});
        gpu_seconds+=seconds(gpu_start);
        ++batches;
        size_t from=size_t(row+1)*3, to=size_t(complete+1)*3;
        const auto readback_start=Clock::now();
        if (to>from) CHECK(cudaMemcpy(counts.data()+from,static_cast<const u32*>(gpu.buffers[3])+from,(to-from)*sizeof(u32),cudaMemcpyDeviceToHost));
        readback_seconds+=seconds(readback_start);
        const auto output_start=Clock::now();
        while (row<complete) { ++row; race.emit(row,counts.data()+size_t(row)*3); }
        output_seconds+=seconds(output_start);
        if (seconds(last_progress)>=5) {
            if (std::fflush(stdout)) throw std::runtime_error("CSV flush failed");
            std::fprintf(stderr,"# Progress: row=%llu; batches=%llu; switches=%llu; last_switch=%llu; R-L=%lld; seconds=%.3f\n",row,batches,race.switches,race.last_switch,race.difference(),seconds(start));
            last_progress=Clock::now();
        }
    }
    const auto flush_start=Clock::now();
    if (std::fflush(stdout)) throw std::runtime_error("CSV flush failed");
    output_seconds+=seconds(flush_start);
    if (stop_requested.load(std::memory_order_relaxed)) std::fputs("# Interrupted; complete CSV records flushed\n",stderr);
    std::fprintf(stderr,"# GPU batch submit/wait seconds %.6f; batches %llu\n",gpu_seconds,batches);
    std::fprintf(stderr,"# GPU timestamp seconds: init=%.6f; mark=%.6f; count=%.6f\n",gpu.stage_seconds[0],gpu.stage_seconds[1],gpu.stage_seconds[2]);
    std::fprintf(stderr,"# Host seconds: readback=%.6f; CSV=%.6f\n",readback_seconds,output_seconds);
    std::fprintf(stderr,"# Completed %llu; last number %llu; seconds %.6f; switches %llu\n",row,row*row,seconds(start),race.switches);
    std::fprintf(stderr,"# Left %llu; right %llu; centre %llu; right minus left %lld\n",race.left,race.right,race.centre,race.difference());
    std::fprintf(stderr,"# left_lead_rows=%llu; right_lead_rows=%llu; tied_rows=%llu\n",race.left_rows,race.right_rows,race.ties);
}
int main(int argc, char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("Expected one positive row limit");
        u32 limit=0;
        const std::string_view text=argv[1];
        const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),limit);
        if (error!=std::errc{} || end!=text.data()+text.size() || !limit || limit>max_supported_rows)
            throw std::runtime_error("Row limit must be an integer from 1 to " + std::to_string(max_supported_rows));
        if (!SetConsoleCtrlHandler(interrupt,TRUE)) throw std::runtime_error("Control handler installation failed");
        if (_setmode(_fileno(stdout),_O_BINARY)==-1) throw std::runtime_error("Cannot set binary stdout");
        if (_setmode(_fileno(stderr),_O_BINARY)==-1) throw std::runtime_error("Cannot set binary stderr");
        static char csv_buffer[65536];
        if (std::setvbuf(stdout,csv_buffer,_IOFBF,sizeof(csv_buffer))) throw std::runtime_error("Cannot buffer stdout");
        run(limit);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"# Error: %s\n",e.what()); return 1; }
}
