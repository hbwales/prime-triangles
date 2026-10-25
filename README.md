# PrimeTri

Count primes on the left, centre and right of a triangle of consecutive integers.
See the [illustrated guide](how-it-works.html) for more details.

## Setup

Install [Visual Studio 2026 Build Tools](https://visualstudio.microsoft.com/downloads/)
with **Desktop development with C++** (x64 MSVC, C++ libraries and Windows SDK),
then [CUDA Toolkit 13.4](https://developer.nvidia.com/cuda-downloads) and a compatible NVIDIA driver.

## Build

From the repository directory in an x64 Native Tools Command Prompt:

```bat
nvcc -O3 -std=c++20 -arch=native -lineinfo --cudart=static -Xcompiler /O2 tri.cu -o tri.exe
```

To initialize MSVC from an ordinary Command Prompt first, run:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

## Run

```bat
tri.exe 50000000 > counts.csv 2> progress.txt
```

Use `.\tri.exe` in PowerShell. 
Supply one row count from **1 to 143,165,576**.
CSV columns `n,l,c,r` contain the row number and cumulative left, centre and right prime totals.
Progress and final timings go to stderr. 

**Ctrl+C** finishes the current batch and flushes complete rows.
