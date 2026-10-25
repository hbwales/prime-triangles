# PrimeTri

Count primes on the left, centre and right of a triangle of consecutive integers.

## 1. Setup

Install these development tools, in this order:
1. [Visual Studio 2026 Build Tools](https://visualstudio.microsoft.com/downloads/). Select **Desktop development with C++**, including the x64 MSVC compiler, C++ libraries and Windows SDK.
2. [NVIDIA CUDA Toolkit 13.4](https://developer.nvidia.com/cuda-downloads). Include NVCC, CUDA headers, compiler dependencies and CUDA Runtime development libraries.
3. A compatible NVIDIA GPU driver. The tested driver is **592.01**.

## 2. Build instructions

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd C:\Users\hb_wa\OneDrive\PrimeTri
nvcc -O3 -std=c++20 -arch=native -lineinfo --cudart=static -Xcompiler /O2 tri.cu -o tri.exe
```

The target flag detects the GPUs visible on the build machine and embeds native code for their architectures (SM 12.0 on the tested GPU), without a PTX fallback.
`--cudart=static` links CUDA Runtime into the executable;
`-lineinfo` preserves source mapping for profiling.
NVCC's assembler uses its default optimization level 3;
`/O2` optimizes host code.

Run with exactly one positive row limit:

```bat
tri.exe 5000000 > counts.csv 2> progress.txt
```

From PowerShell, use `.\tri.exe` instead. Redirect large CSV output to a file for practical performance.
CSV columns are cumulative prime totals through each completed row:

```csv
n,l,c,r
1,0,0,0
2,1,1,0
3,2,2,0
```

- `n`: row number.
- `l`, `c`, `r`: cumulative left, centre and right prime counts.

Progress is printed to stderr approximately every five seconds, including the current row, elapsed time, number of lead switches, last switch and `R-L`.
Final stderr totals include rows led by each side and ties. The timing field `prepare` measures the preparation kernel; the reported difference is `R-L`. A first non-tied lead counts as a switch; ties leave the previous leader unchanged.

**Ctrl+C** requests a graceful stop: the current batch finishes, and complete CSV records are flushed.

Accepted limits are **1 through 143,165,576 rows**, inclusive.
Memory use grows with the row limit: at the cap the principal GPU buffers need about **2.12 GiB**, including sparse-strike state, plus CUDA overhead and host memory.
A full run to the cap has not been performed.

## 3. Results & benchmarks

These earlier measurements use the RTX 5080 Laptop GPU (60 SMs, 16 GB).
They predate the latest refactors; current measurements and calibrated estimates
are in section 6. Times include initialization, GPU work, CSV output and teardown.

```
# Time: 1.748256s; row:1,000,000; num: 1,000,000,000,000; L: 18,803,924,009; R: 18,803,899,891; C: 88,118; R-L: -24,118; L>R:  732,963; R>L:  267,022; R=L:15
# Time: 5.029270s; row:2,000,000; num: 4,000,000,000,000; L: 71,483,078,303; R: 71,482,962,655; C:167,168; R-L:-115,648; L>R:1,522,133; R>L:  477,845; R=L:22
# Time:12.964886s; row:3,000,000; num: 9,000,000,000,000; L:156,300,214,489; R:156,299,896,461; C:243,158; R-L:-318,028; L>R:2,522,133; R>L:  477,845; R=L:22
# Time:22.276794s; row:4,000,000; num:16,000,000,000,000; L:272,415,182,694; R:272,415,316,395; C:317,592; R-L: 133,701; L>R:2,684,618; R>L:1,315,359; R=L:23
# Time:37.526960s; row:5,000,000; num:25,000,000,000,000; L:419,268,696,913; R:419,268,951,845; C:390,752; R-L: 254,932; L>R:2,684,618; R>L:2,315,359; R=L:23
```

The complete five-million-row CSV SHA-256 is:

```text
423181dd1aa39eeb9963b9025bafa0acb3dca2b34701cff64d2fffb989fb9bcf
```

Verify a completed output in PowerShell:

```powershell
Get-FileHash .\counts.csv -Algorithm SHA256
```

## 4. Algorithm

### Triangle and cumulative counts

Row `n` contains the integers from `(n-1)^2 + 1` through `n^2`, so it has
`2n-1` entries. Its centre is `n^2 - n + 1`. Primes smaller than the centre
belong to the left, the centre prime belongs to the centre column, and larger
primes belong to the right. The program counts each row, then accumulates its
three counts for CSV output. The integer 1 is not prime.

### Modulo-30 wheel and pre-sieving

Except for 2, 3 and 5, every prime lies in one of eight residues modulo 30:
`1, 7, 11, 13, 17, 19, 23, 29`. One byte represents these eight candidates
among 30 consecutive integers. A set bit means the candidate survives the sieve.

A repeating pattern removes multiples of 7, 11, 13 and 17 before GPU marking.
The CPU generates base primes from 19 through the requested row limit: that
limit is the square root of the largest integer being tested. Special handling
preserves the small primes themselves and clears the bit for 1.

### Batches and prepared prime records

The GPU processes at most 256 MiB of wheel bytes per batch, representing about
8.05 billion consecutive integers. Only base primes whose squares do not
exceed the batch's upper endpoint become active.

A preparation kernel computes each active prime's anchor relative to the batch
and packs wheel information into a 16-byte aligned record. An eight-iteration
unrolled loop computes the packed wheel-bit indices. Exact reciprocal
high-multiply operations replace repeated integer division, with a correction
step that preserves exact results. Preparation performs wide arithmetic once
per prime per batch; tile marking mostly uses bounded 32-bit arithmetic.

### Shared-memory marking

Each 1,024-thread block sieves a 96 KiB tile in shared memory. It initializes
the tile directly from the repeating pre-sieve pattern.

- For dense primes up to 1,024, a warp cooperates on one prime: eight wheel
  phases, with four interleaved progressions per phase.
- Other primes through the tiled cutoff are assigned to individual threads. Threads
  load one prime record ahead to overlap metadata latency with current-prime work.
- Sparse primes use the next eligible wheel multiplier and stop when it lies
  beyond the tile, avoiding eight complete progression loops.

Shared-memory `atomicAnd` clears composite bits safely when threads strike the
same word. After synchronization, the block writes its tile to global memory.
Runs through 8m rows use this tiled path alone. Larger runs tile primes through
2m, then process the remaining primes in 16 MiB global-sieve windows. Each
thread owns a prime and walks its hits in wheel-multiplier order. Global
`atomicAnd` combines overlapping clears. A single u32 retains its next byte
offset (29 bits) and wheel phase (3 bits), across windows and batches.
Newly active primes initialize from their square restriction; stream ordering
protects persistent state. At 50m this carry occupies 10.88 MiB, versus
75.11 MiB for the previous eight-offset implementation.

### Counting and output

One 256-thread block counts a row's left and right intervals using masked word
loads and population counts. Hardware integer warp reductions combine the
partial sums; only eight warp totals pass through shared memory. Thread zero
tests the centre candidate and writes the three row totals.

The GPU count array is zeroed on-device and receives only the four initial
row records containing the small-prime seeds. CPU readback reuses a small
vector, at most about 1.03 MiB, rather than retaining every row's counts.

A row can span batches. Default-stream ordering safely accumulates its partial
counts, and the CPU emits it only when the row is complete. Only row counts
return to the CPU, rather than the whole sieve. The CPU maintains cumulative
totals, lead switches and lead/tie row counts while writing buffered CSV. Each
CSV row is formatted with `std::to_chars` into a fixed stack buffer.

## 5. Short benchmarks and profiling

Build the paired benchmark with the same compiler flags, substituting
`bench.cu -o bench.exe`. It includes the production implementation and compares
the original all-prime tiled path against `mark_batch`, validating every
intersecting row's counts. The benchmark jumps directly to a late range; it
does not compute all earlier rows or produce cumulative CSV.

```powershell
.\bench.exe 50000000 268435456 6
```

Arguments are the row-limit configuration, bytes per batch, and measured paired
batches (defaults: 50m, 256 MiB, 4). One warm-up is excluded. Baseline/candidate
order alternates. On the RTX 5080 Laptop GPU, a six-pair sample measured late-50m
marking at 84.18 ms before and 42.07 ms with windowed sparse strikes. Late-5m
marking was 13.21 versus 13.17 ms. These are kernel samples, not full-run timings;
clocks, power state and range affect results.

Nsight Compute 2026.3.1 CLI was downloaded as NVIDIA's portable archive for this
work; it need not be globally installed. With `ncu` on your path, profile one
sparse window after the warm-up:

```powershell
ncu --set basic --kernel-name regex:scatter_kernel --launch-skip 16 --launch-count 1 --export scatter-window .\bench.exe 50000000 268435456 1
```

Profiling replays kernels and changes elapsed times; use an unprofiled benchmark
for performance comparisons. Nsight Compute identified substantial cache and
compute traffic in the original marking path, rather than saturated DRAM.
Fused counting and a persistent group-bucket prototype were tested and rejected
because their measured total costs were higher. The kept path avoids queue
construction. Nsight Systems would be useful for CPU/transfer overlap analysis,
but was not required to identify this kernel bottleneck.

### Earlier ordered-walk trials

A second sweep tested cutoffs from 250k to 12m, windows from 8 to 32 MiB,
48/64/96 KiB tiles, 512/768/1024 marking threads, pre-sieving with 19,
skipping known multiples of 7, eight-lane sparse groups, compressed phase
origins, and an ordered-hit walk. That version used the ordered walk,
a 4m cutoff for large runs, and 16 MiB windows. Runs through 8m retain the
original tiled path to avoid a small-range regression. Other tile sizes,
thread counts, the larger pre-sieve and the redundant-strike filter did not
show a useful gain. Two million was marginally faster at large ranges but
slowed marking near 3m by about 12%; four million was the better compromise.

Direct comparisons against the previous 8m-cutoff, eight-offset windowed path
(one warm-up and twelve measured paired batches, alternating order):

| Late range | Eight-offset path | Ordered walk, 4m cutoff |
|---|---:|---:|
| 10m | 19.56 ms | 15.78 ms |
| 25m | 25.38 ms | 19.10 ms |
| 50m | 35.81 ms | 22.04 ms |

The ordered walk matched the throughput of the faster phase-origin prototypes
while keeping one thread and one state word per prime, with no subgroup
synchronization. A complete 5m CSV matched the recorded SHA-256; additional
checks covered activation, nonstandard batch/window boundaries, the maximum
row limit, and final partial batches. Large-range timings remain samples,
not complete-run measurements.

A follow-up Nsight Compute sample of the ordered-walk kernel reached about
81% L2 throughput, versus 16% DRAM and 12% SM compute throughput. Guards
using the pre-sieve pattern or a relaxed atomic load before clearing were
also tested; neither improved performance consistently across ranges, so
neither is enabled. Profiling replay times are excluded from benchmarks.

### Arithmetic filter and cutoff retuning (current)

The sparse loop now tests the target's divisibility by 7, 11, 13 and 17 before
issuing its global atomic. The pre-sieve already cleared these bits. For
window base byte `b`, hit byte offset `a`, and multiplier residue `r`, compute
`z = 30 * (b % 17017 + a) + (p * r) % 30`. It has the same remainders as the
actual target modulo every pre-sieve prime. A compile-time bound protects
32-bit arithmetic. This avoids about 32.3% of sparse global atomics while
preserving the hit walk. It uses no additional buffers or includes.

With the filter, a 2m cutoff and 16 MiB windows performed better than the
previous 4m cutoff. Runs through 8m remain on the original tiled path.
Compact 8/12-byte sparse records, structure-of-arrays metadata, and larger
windows did not improve performance enough to keep.

Sixteen measured paired batches, alternating order, comparing the previous
unfiltered 4m-cutoff walk with the filtered 2m-cutoff walk:

| Late range | Before | After |
|---|---:|---:|
| 10m | 15.44 ms | 13.14 ms |
| 25m | 18.97 ms | 15.55 ms |
| 50m | 21.85 ms | 17.72 ms |
| 100m | 24.07 ms | 19.56 ms |

A complete current 10m run took **189.7 s**, writing 405,147,067 CSV bytes.
Every byte matched the prefix of the existing 35m output. Its SHA-256 is
`de001233726b1135233800be1b6305edb56bcd262f98e3c8311b844b39f38c97`.
Activation at the new cutoff, irregular batch lengths, final partial batches,
tiny inputs and the row cap were also checked. The filtered sparse-kernel
Nsight sample measured about 81% L2 throughput, 19% DRAM throughput, 22% SM
compute throughput, 18 registers/thread and 92% achieved occupancy.

## 6. Scaling and runtime model

The detailed derivation and a graphical calculator are in
[the standalone walkthrough](how-it-works.html#scaling).
[Measurement and fit data](scaling-model.json) includes exact prime sums,
logical strike counts, approximate visit counts, timing samples and the
sustained-run calibration. The graph is also available as [SVG](scaling.svg).

![Batch-cost measurements and cumulative runtime model](scaling.svg)

N rows cover **N² integers**. With B wheel bytes per batch, the batch count is
approximately `N² / (30 B)`. Let C be the tiled-prime cutoff, W the sparse
window size, t the tile size, Π(u) the number of primes from 19 through u,
and S₂(u) their squared-prime sum. Ignoring boundary rounding:

- Tile visits: `[N² Π(C) − S₂(C)] / (30 t)`.
- Sparse visits: `[N² (Π(N) − Π(C)) − (S₂(N) − S₂(C))] / (30 W)`.
- Preparation visits: `[N² Π(N) − S₂(N)] / (30 B)`.

For large runs C = 2m is fixed, so tile visits grow quadratically. Sparse and
preparation visits eventually grow as `N³ / log N`, divided by W and B.
The wheel-hit count grows as `N² log log N`. Filtering reduces atomics by a
constant factor; it does not skip loop iterations. Thus the operation-count
model is

```text
Θ(N² log log N + N³/(W log N) + N³/(B log N))
```

With fixed B and W, this is ultimately Θ(N³/log N), rather than a purely
quadratic sieve. At 50m, however, each active sparse prime has a modulo-30 candidate hit in every
full window: its maximum byte gap is ceil(p/5), and 50m < 5W ≈ 83.9m. Queue
scheduling would add traffic to work that mostly already has a hit. Achieving
a better eventual bound would require avoiding both window-wide prime scans
and repeated preparation of established primes.

The empirical combined prepare/mark/count rate, in milliseconds per full
256 MiB batch, is fitted as:

```text
r(n) = 10.580 + 1.136 * (π(n) − π(2m))/1,000,000
                 + 35.482 * (H(n) − H(2m))
H(u) = sum of 1/p over primes p <= u
```

Eleven late-range samples from 8.1m through the row cap give 3.3% RMS fit
error; leave-one-out prediction gives 5.1% RMS and 9.4% maximum error.
These are fit checks, not complete-run validations. The final intervals of
the complete 10m run were **1.219× slower** than the short-sample rate fit.
For N >= 10m, the cumulative model anchors to that full run and uses:

```text
T(N) = 189.7 + 1.219/(30*B*1000) * integral[x=(10m)²..N²] r(sqrt(x)) dx
```

The integral has a direct prime-sum form, recorded in the walkthrough. A
compact interpolation of this calibrated model over **10–50m** is:

```text
T(N) ≈ 190 seconds * (N / 10,000,000)^2.25
```

This exponent describes the fitted operating range, not the asymptotic
complexity. The earlier 1¾-hour 50m estimate used short samples alone; it did
not include the sustained-run correction.

| Rows | Current elapsed time | Basis |
|---:|---:|---|
| 10k | ~0.1 s, warm startup | Measured |
| 100k | 0.12 s | Measured |
| 1m | 0.92 s | Complete run; unchanged tiled path |
| 5m | 37.8 s | Complete run; unchanged tiled path |
| 10m | 3 min 10 s | Complete current run |
| 25m | ~25 min; allow 19–32 min | Calibrated model |
| 50m | ~1 h 58 min; allow 1 h 29 min–2 h 28 min | Calibrated model |

The ±25% planning allowance expresses power, cooling, storage and model
uncertainty; it is not a statistical confidence interval. No full 25m or
50m run was performed. Cold CUDA initialization added up to about 2.5 s in
small runs. Space is O(N + π(N) + B), dominated by the full GPU row-count array
at large N.
