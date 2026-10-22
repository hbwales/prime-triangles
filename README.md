# PrimeTri

Count primes on the left, centre and right of a triangle of consecutive integers.
`tri.cu` is a standalone CUDA C++ program specialized for an NVIDIA GeForce
RTX 5080 Laptop GPU (compute capability 12.0), running Windows x64.

## 1. Setup

Install these development tools, in this order:

1. [Visual Studio 2026 Build Tools](https://visualstudio.microsoft.com/downloads/).
   Select **Desktop development with C++**, including the x64 MSVC compiler,
   C++ libraries and Windows SDK. The full Visual Studio IDE is optional.
2. [NVIDIA CUDA Toolkit 13.4](https://developer.nvidia.com/cuda-downloads).
   Include NVCC, CUDA headers, compiler dependencies and CUDA Runtime development
   libraries. cuBLAS, cuFFT, cuDNN, examples and Visual Studio integration are not
   needed by this program.
3. A compatible NVIDIA GPU driver. The tested driver is **592.01**.

Tested on 22 October 2026 with CUDA Toolkit **13.4.3**, NVCC **13.4.92**, Visual
Studio Build Tools **18.11.1**, and MSVC **14.51.36231**. NVIDIA documents supported
host compilers in the [Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

There are no third-party source dependencies, package manager requirements,
CMake files or runtime kernel compilation. The built executable needs the NVIDIA
driver; the Toolkit, Build Tools and NVRTC DLLs are not needed to run it.

This version embeds native SM 12.0 device code only. It has no PTX JIT or
alternate-GPU fallback. Its 96 KiB shared-memory tiles and 1,024-thread marking
blocks are chosen for the tested GPU.

## 2. Build instructions

Open **x64 Native Tools Command Prompt for VS 2026**, and change directory:

```bat
cd /d C:\Users\hb_wa\OneDrive\PrimeTri
nvcc --version
cl
```

Build the single source file:

```bat
nvcc -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -lineinfo --cudart=static -Xcompiler /O2 tri.cu -o tri.exe
```

If `nvcc` is not on PATH in an already-open terminal, reopen the developer
prompt or invoke `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe`
by its full quoted path. The installed MSVC environment can also be initialized
from Command Prompt with:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

The target flag embeds native SM 12.0 code. `--cudart=static` links CUDA Runtime
into the executable; `-lineinfo` preserves source mapping for profiling. NVCC's
assembler uses its default optimization level 3; `/O2` optimizes host code.

Run with exactly one positive row limit:

```bat
tri.exe 5000000 > counts.csv 2> progress.txt
```

From PowerShell, use `.\tri.exe` instead. Redirect large CSV output to a file
for practical performance. CSV columns are cumulative prime totals through each
completed row:

```csv
n,l,c,r
1,0,0,0
2,1,1,0
3,2,2,0
```

- `n`: row number.
- `l`, `c`, `r`: cumulative left, centre and right prime counts.

Progress is printed to stderr approximately every five seconds, including the
current row, elapsed time, number of lead switches, last switch and `R-L`.
Final stderr totals include rows led by each side and ties. A first non-tied
lead counts as a switch; ties leave the previous leader unchanged.

**Ctrl+C** or **Ctrl+Break** requests a graceful stop: the current batch finishes,
and complete CSV records are flushed. Forced process termination or power loss
does not provide that guarantee. There is no checkpoint/resume mode.

Accepted limits are **1 through 143,165,576 rows**, inclusive. The largest row
ends at **20,496,382,151,411,776**. This cap comes from bounded 32-bit arithmetic
in prime-location calculations; larger runs need code changes. Memory use grows
with the row limit: at the cap the principal GPU buffers need about **2.09 GiB**,
plus CUDA overhead and host memory. A full run to the cap has not been performed.

## 3. Results & benchmarks

Measurements use the RTX 5080 Laptop GPU (60 SMs, 16 GB), native file output,
and the compiler/driver versions above. Laptop clocks were not fixed. Wall times
include initialization, GPU work, CSV output and teardown.

| Five-million-row build | Wall times | Median |
| --- | --- | --- |
| Initial NVCC version with GPU selection | 37.075 / 37.025 s | 37.050 s |
| Specialized version | 37.135 / 37.186 s | 37.160 s |

The 0.30% difference does not establish a meaningful throughput change. The
specialization primarily simplifies code and reduces executable size. A separate
final-build validation run completed in **36.646 s**, with GPU preparation
**0.145 s**, marking **33.716 s**, and counting **1.456 s**. That single run is
not a paired speedup measurement. A smoke benchmark of the packaged `tri.cu`
completed one million rows in **0.980 s**, with the known full CSV hash; this
is also a single-run measurement.

The source is 397 lines, versus 431 in the initial NVCC conversion. With identical
native-only target flags, executable size fell from **382,464 to 289,280 bytes**
(24.4%). Counting reduction storage fell from 3,072 to 64 bytes. Marking uses
40 registers per thread, counting 34; neither kernel spills registers.

At **5,000,000 rows**, covering integers through **25,000,000,000,000**:

| Result | Value |
| --- | ---: |
| Left primes | 419,268,696,913 |
| Centre primes | 390,752 |
| Right primes | 419,268,951,845 |
| Right minus left | +254,932 |
| Rows with a left lead | 2,684,618 |
| Rows with a right lead | 2,315,359 |
| Tied rows | 23 |
| Lead switches, including the first lead | 2,614 |

The complete five-million-row CSV SHA-256 is:

```text
423181dd1aa39eeb9963b9025bafa0acb3dca2b34701cff64d2fffb989fb9bcf
```

Verify a completed output in PowerShell:

```powershell
Get-FileHash .\counts.csv -Algorithm SHA256
```

Validation performed during development:

- Every row through 100,000 agrees with an independent reference.
- Complete five-million-row CSV hashes agree across full benchmark runs.
- The actual NVCC CUBIN agrees with deterministic Miller–Rabin primality tests
  for 163,840 candidate numbers in five windows near rows 10,000,001,
  100,000,000 and the maximum row, including boundary and centre counts.
- Nine real Ctrl+C/Ctrl+Break scenarios preserve complete records, including
  interruption during blocked stdout and startup at the maximum limit.
- Malformed, zero, excessive and extra arguments are rejected.

Compiler screening tested restrict pointers, extra vectorization, register caps
32/48, cache policy changes, AVX2 host code, explicit phase unrolling, launch
bounds and assembler optimization level 2. No convincing benefit was found.
Several options produced byte-identical GPU instructions, despite timing
variation. Sampled full batches near 5m, 10m and 50m rows had identical sieve
and count checksums across candidates; these were not complete runs to 50m.

Fast-math flags do not help the integer sieve. Further gains likely require
reducing marking work rather than adding compiler switches.

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
and packs wheel information into a 16-byte aligned record. Exact reciprocal
high-multiply operations replace repeated integer division, with a correction
step that preserves exact results. Preparation performs wide arithmetic once
per prime per batch; tile marking mostly uses bounded 32-bit arithmetic.

### Shared-memory marking

Each 1,024-thread block sieves a 96 KiB tile in shared memory. It initializes
the tile directly from the repeating pre-sieve pattern.

- For dense primes up to 1,024, a warp cooperates on one prime: eight wheel
  phases, with four interleaved progressions per phase.
- Other primes are assigned to individual threads. Threads load one prime
  record ahead to overlap metadata latency with current-prime work.
- Sparse primes use the next eligible wheel multiplier and stop when it lies
  beyond the tile, avoiding eight complete progression loops.

Shared-memory `atomicAnd` clears composite bits safely when threads strike the
same word. Marking starts at each prime's square, preserving the prime itself.
After synchronization, the block writes its completed tile to global memory.

### Counting and output

One 256-thread block counts a row's left and right intervals using masked word
loads and population counts. Hardware integer warp reductions combine the
partial sums; only eight warp totals pass through shared memory. Thread zero
tests the centre candidate and writes the three row totals.

A row can span batches. Default-stream ordering safely accumulates its partial
counts, and the CPU emits it only when the row is complete. Only row counts
return to the CPU, rather than the whole sieve. The CPU maintains cumulative
totals, lead switches and lead/tie row counts while writing buffered CSV.

Marking accounts for about 92% of measured execution time. Carrying sparse-prime
strike positions between batches is a possible larger redesign; previous
per-batch bucket experiments lost time rebuilding queues, so avoiding that
construction cost would be essential.
