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
nvcc -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -lineinfo --cudart=static -Xcompiler /O2 tri.cu -o tri.exe
```

The target flag embeds native SM 12.0 code. 
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
Final stderr totals include rows led by each side and ties. A first non-tied lead counts as a switch; ties leave the previous leader unchanged.

**Ctrl+C** requests a graceful stop: the current batch finishes, and complete CSV records are flushed. 

Accepted limits are **1 through 143,165,576 rows**, inclusive. 
Memory use grows with the row limit: at the cap the principal GPU buffers need about **2.09 GiB**, plus CUDA overhead and host memory. 
A full run to the cap has not been performed.

## 3. Results & benchmarks

Measurements use the RTX 5080 Laptop GPU (60 SMs, 16 GB). 
Times include initialization, GPU work, CSV output and teardown.

```
# Time: 1.748256s; row:1,000,000; num: 1,000,000,000,000; L: 18,803,924,009; R: 18,803,899,891; C: 88,118; L-R: -24,118; L>R:  732,963; R>L:  267,022; R=L:15
# Time: 5.029270s; row:2,000,000; num: 4,000,000,000,000; L: 71,483,078,303; R: 71,482,962,655; C:167,168; L-R:-115,648; L>R:1,522,133; R>L:  477,845; R=L:22
# Time:12.964886s; row:3,000,000; num: 9,000,000,000,000; L:156,300,214,489; R:156,299,896,461; C:243,158; L-R:-318,028; L>R:2,522,133; R>L:  477,845; R=L:22
# Time:22.276794s; row:4,000,000; num:16,000,000,000,000; L:272,415,182,694; R:272,415,316,395; C:317,592; L-R: 133,701; L>R:2,684,618; R>L:1,315,359; R=L:23
# Time:37.526960s; row:5,000,000; num:25,000,000,000,000; L:419,268,696,913; R:419,268,951,845; C:390,752; L-R: 254,932; L>R:2,684,618; R>L:2,315,359; R=L:23
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
