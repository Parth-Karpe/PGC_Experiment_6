# CUDA Experiment 6 --- Memory Optimization

This experiment demonstrates how memory access patterns affect CUDA
kernel performance. It compares an **uncoalesced matrix transpose** with
a **coalesced matrix transpose**, and executes a **naive matrix
multiplication** kernel.

## Hardware and software

  Item                 Configuration
  -------------------- ----------------------------
  GPU                  NVIDIA GeForce RTX 5060 Ti
  CUDA Toolkit         13.4
  Compilation target   `sm_120`
  Operating system     Windows
  Working directory    `C:\CUDA_Labs`

## Experiment objectives

-   Compare the execution time of uncoalesced and coalesced matrix
    transpose.
-   Observe the effect of coalesced global-memory access on memory
    bandwidth.
-   Execute naive matrix multiplication on the GPU.
-   Verify the correctness of the output using the program's `PASS`
    checks.

## Implementations

### 1. Uncoalesced matrix transpose

The naive transpose reads adjacent input elements but produces strided
output writes across threads in a warp. This access pattern is less
efficient for global-memory transactions.

### 2. Coalesced matrix transpose

The optimized transpose uses a shared-memory tile to rearrange data
before writing the transposed result. The implementation uses a padded
shared-memory tile of `[32][33]`. The extra column helps avoid
shared-memory bank conflicts during transposed access. A block-level
synchronization ensures that the tile is fully loaded before it is read
in the transposed order.

### 3. Naive matrix multiplication

Each CUDA thread computes one output element `C[i][j]` by multiplying
the corresponding row of matrix `A` with the corresponding column of
matrix `B` and accumulating the products. This naive implementation does
not use shared-memory tiling for input reuse.

## Matrix and launch configuration

  -------------------------------------------------------------------------
  Operation               Matrix size   Block dimensions    Grid dimensions
  ---------------- ------------------ ------------------ ------------------
  Matrix transpose      `4096 × 4096`          `32 × 32`        `128 × 128`

  Naive matrix          `1024 × 1024`          `16 × 16`  Determined by the
  multiplication                                            output size and
                                                           block dimensions
  -------------------------------------------------------------------------

## Prerequisites

1.  An NVIDIA GPU supported by the installed CUDA Toolkit.
2.  NVIDIA GPU driver and CUDA Toolkit installed.
3.  Microsoft Visual Studio C++ Build Tools (or a compatible MSVC
    compiler).
4.  `nvcc` and the MSVC compiler available in the developer command
    environment.

Check the CUDA compiler:

``` bat
nvcc --version
```

Check that the GPU is visible to the driver:

``` bat
nvidia-smi
```

## Build and run

Open **x64 Native Tools Command Prompt for Visual Studio** or the
CUDA-enabled Visual Studio Developer Command Prompt.

Move to the experiment directory:

``` bat
cd /d C:\CUDA_Labs
```

Compile the CUDA source:

``` bat
nvcc -O2 -arch=sm_120 exp6.cu -o exp6.exe
```

Run the experiment:

``` bat
exp6.exe
```

The program runs both transpose implementations and naive matrix
multiplication, then prints execution times, bandwidth or GFLOPS,
speedup, and correctness status.

## Experimental results

The following table contains the three complete runs recorded during the
experiment. All three runs reported `PASS` for both transpose
implementations and naive matrix multiplication.

### Matrix transpose

  --------------------------------------------------------------------------------
  Run             Uncoalesced   Uncoalesced    Coalesced    Coalesced      Speedup
                    time (ms)     bandwidth    time (ms)    bandwidth 
                                     (GB/s)                    (GB/s) 
  ------------- ------------- ------------- ------------ ------------ ------------
  1                     0.791        169.65        0.455       294.91        1.74×

  2                     0.822        163.28        0.447       300.29        1.84×

  3                     0.814        164.85        0.429       312.94        1.90×

  **Average**       **0.809**    **165.93**    **0.444**   **302.71**    **1.83×**
  --------------------------------------------------------------------------------

### Naive matrix multiplication

  Run             Execution time (ms)   Performance (GFLOPS) Correctness
  ------------- --------------------- ---------------------- -------------
  1                             1.546                1388.80 PASS
  2                             1.527                1406.63 PASS
  3                             1.516                1416.18 PASS
  **Average**               **1.530**            **1403.87** **PASS**

*The averages are calculated from the three complete runs. Displayed
values are rounded.*

## Performance formulas

### Transpose bandwidth

The experiment counts both the matrix read and matrix write:

``` text
Bandwidth = (2 × N × N × sizeof(float)) / execution_time
```

Convert the result to GB/s using consistent units.

### Transpose speedup

``` text
Speedup = Uncoalesced execution time / Coalesced execution time
```

### Matrix multiplication performance

For an `N × N` matrix multiplication, the operation count used by the
experiment is:

``` text
FLOPs = 2 × N³
GFLOPS = FLOPs / execution_time_in_seconds / 10⁹
```

## Result and conclusion

Across the three complete runs, the coalesced transpose averaged **0.444
ms**, compared with **0.809 ms** for the uncoalesced transpose. The
average speedup was approximately **1.83×**. Average measured bandwidth
increased from **165.93 GB/s** to **302.71 GB/s**.

Naive matrix multiplication completed successfully in all three runs,
with an average execution time of **1.530 ms** and average measured
performance of **1403.87 GFLOPS**.

The results demonstrate that organizing global-memory accesses so that
adjacent threads access adjacent memory locations can significantly
improve CUDA kernel performance.

## Repository / folder structure

``` text
C:\CUDA_Labs\
│
├── exp6.cu       # CUDA source code
├── exp6.exe      # Compiled executable (generated during build)
└── README.md     # Experiment documentation
```

The executable is generated by the compilation command; it does not need
to be committed if the source code is available.

## Troubleshooting

-   **`nvcc` is not recognized:** Open the CUDA-enabled Visual Studio
    Developer Command Prompt, or ensure the CUDA Toolkit `bin` directory
    is on `PATH`.
-   **MSVC compiler not found:** Install the Visual Studio C++ build
    tools and use the x64 developer command prompt.
-   **GPU architecture error:** Confirm that the installed CUDA Toolkit
    supports the target architecture used in the build command.
-   **Output reports `FAIL`:** Check the kernel indexing, grid and block
    dimensions, memory allocation sizes, and transpose or multiplication
    validation logic.

## Notes

-   The performance numbers are specific to the recorded GPU, driver,
    toolkit, and execution environment. Results can vary between runs
    and systems.
-   An earlier transpose-only validation screenshot reported a separate
    `1.94×` speedup. It is supplementary and is not included in the
    three-run averages above.
-   This experiment implements naive matrix multiplication;
    shared-memory tiled matrix multiplication is not part of the
    measured implementation documented here.
