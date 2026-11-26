
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16

// ============================================
// CUDA ERROR CHECKING
// ============================================

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (%s:%d)\n", \
                cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)


// ============================================
// KERNEL 1: UNCOALESCED MATRIX TRANSPOSE
// ============================================

__global__ void transposeNaive(
    const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];
}


// ============================================
// KERNEL 2: COALESCED MATRIX TRANSPOSE
// ============================================

__global__ void transposeCoalesced(
    const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Read input matrix into shared memory
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];

    __syncthreads();

    // Swap block coordinates for transposed output
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Write transposed data
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}


// ============================================
// KERNEL 3: NAIVE MATRIX MULTIPLICATION
// ============================================

__global__ void matMulNaive(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ============================================
// GPU KERNEL TIMING FOR TRANSPOSE
// ============================================

typedef void (*TKernel)(const float*, float*, int);

float timeTranspose(
    TKernel kernel,
    dim3 grid,
    dim3 block,
    const float *in,
    float *out,
    int N,
    int reps)
{
    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up
    kernel<<<grid, block>>>(in, out, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Start timing
    CHECK(cudaEventRecord(start));

    for (int i = 0; i < reps; i++)
        kernel<<<grid, block>>>(in, out, N);

    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;

    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return ms / reps;
}


// ============================================
// RESULT VERIFICATION: TRANSPOSE
// ============================================

bool checkTranspose(
    const float *in,
    const float *out,
    int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (out[j * N + i] != in[i * N + j])
                return false;
        }
    }

    return true;
}


// ============================================
// RESULT VERIFICATION: MATRIX MULTIPLICATION
// ============================================

bool checkMatMul(
    const float *A,
    const float *B,
    const float *C,
    int N)
{
    for (int t = 0; t < 20; t++)
    {
        int i = rand() % N;
        int j = rand() % N;

        double ref = 0.0;

        for (int k = 0; k < N; k++)
        {
            ref += (double)A[i * N + k] * B[k * N + j];
        }

        if (fabs(ref - C[i * N + j]) >
            1e-3 * fabs(ref) + 1e-3)
        {
            return false;
        }
    }

    return true;
}


// ============================================
// MAIN FUNCTION
// ============================================

int main(int argc, char **argv)
{
    int N = 4096;

    if (argc > 1)
        N = atoi(argv[1]);

    if (N <= 0)
    {
        printf("Invalid matrix size.\n");
        return 1;
    }

    printf("====================================\n");
    printf(" CUDA EXPERIMENT 6: MEMORY OPTIMIZATION\n");
    printf("====================================\n");

    cudaDeviceProp prop;

    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("\nPART A: MATRIX TRANSPOSE\n");
    printf("Matrix size: %d x %d\n", N, N);

    size_t bytes = (size_t)N * N * sizeof(float);

    // ----------------------------------------
    // HOST MEMORY ALLOCATION
    // ----------------------------------------

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    if (!h_in || !h_out)
    {
        printf("Host memory allocation failed.\n");
        free(h_in);
        free(h_out);
        return 1;
    }

    // Initialize input matrix
    for (size_t i = 0; i < (size_t)N * N; i++)
        h_in[i] = (float)(i % 1000);

    // ----------------------------------------
    // DEVICE MEMORY ALLOCATION
    // ----------------------------------------

    float *d_in, *d_out;

    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    CHECK(cudaMemcpy(
        d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ----------------------------------------
    // TRANSPOSE GRID AND BLOCK CONFIGURATION
    // ----------------------------------------

    dim3 blockTranspose(TILE, TILE);

    dim3 gridTranspose(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    printf("Block: %d x %d\n", TILE, TILE);
    printf("Grid: %d x %d\n",
           gridTranspose.x, gridTranspose.y);

    // ----------------------------------------
    // UNCOALESCED TRANSPOSE
    // ----------------------------------------

    printf("\nRunning uncoalesced transpose...\n");

    float tN = timeTranspose(
        transposeNaive,
        gridTranspose,
        blockTranspose,
        d_in,
        d_out,
        N,
        20
    );

    CHECK(cudaMemcpy(
        h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool okN = checkTranspose(h_in, h_out, N);

    // ----------------------------------------
    // COALESCED TRANSPOSE
    // ----------------------------------------

    printf("Running coalesced transpose...\n");

    float tC = timeTranspose(
        transposeCoalesced,
        gridTranspose,
        blockTranspose,
        d_in,
        d_out,
        N,
        20
    );

    CHECK(cudaMemcpy(
        h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool okC = checkTranspose(h_in, h_out, N);

    // ----------------------------------------
    // TRANSPOSE PERFORMANCE CALCULATION
    // ----------------------------------------

    double gbN = (2.0 * bytes) / (tN * 1e6);
    double gbC = (2.0 * bytes) / (tC * 1e6);

    printf("\n====================================\n");
    printf("       PART A: RESULTS\n");
    printf("====================================\n");

    printf("Uncoalesced : %.3f ms | %.2f GB/s | %s\n",
           tN, gbN, okN ? "PASS" : "FAIL");

    printf("Coalesced   : %.3f ms | %.2f GB/s | %s\n",
           tC, gbC, okC ? "PASS" : "FAIL");

    printf("\nTranspose Speedup: %.2fx\n", tN / tC);


    // ========================================
    // PART B: NAIVE MATRIX MULTIPLICATION
    // ========================================

    printf("\n====================================\n");
    printf(" PART B: NAIVE MATRIX MULTIPLICATION\n");
    printf("====================================\n");

    int NM = 1024;

    size_t bytesM = (size_t)NM * NM * sizeof(float);

    printf("Matrix size: %d x %d\n", NM, NM);

    // ----------------------------------------
    // HOST MEMORY ALLOCATION
    // ----------------------------------------

    float *hA = (float*)malloc(bytesM);
    float *hB = (float*)malloc(bytesM);
    float *hC = (float*)malloc(bytesM);

    if (!hA || !hB || !hC)
    {
        printf("Matrix multiplication host allocation failed.\n");
        return 1;
    }

    // Initialize matrices with random values
    for (size_t i = 0; i < (size_t)NM * NM; i++)
    {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    // ----------------------------------------
    // GPU MEMORY ALLOCATION
    // ----------------------------------------

    float *dA, *dB, *dC;

    CHECK(cudaMalloc(&dA, bytesM));
    CHECK(cudaMalloc(&dB, bytesM));
    CHECK(cudaMalloc(&dC, bytesM));

    // Copy matrices from CPU to GPU
    CHECK(cudaMemcpy(
        dA, hA, bytesM, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(
        dB, hB, bytesM, cudaMemcpyHostToDevice));

    // ----------------------------------------
    // MATRIX MULTIPLICATION GRID AND BLOCK
    // ----------------------------------------

    dim3 blockMatMul(BLK, BLK);

    dim3 gridMatMul(
        (NM + BLK - 1) / BLK,
        (NM + BLK - 1) / BLK
    );

    // ----------------------------------------
    // WARM-UP EXECUTION
    // ----------------------------------------

    matMulNaive<<<gridMatMul, blockMatMul>>>(
        dA, dB, dC, NM);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // ----------------------------------------
    // EXECUTION TIME MEASUREMENT
    // ----------------------------------------

    cudaEvent_t startM, stopM;

    CHECK(cudaEventCreate(&startM));
    CHECK(cudaEventCreate(&stopM));

    int reps = 5;

    CHECK(cudaEventRecord(startM));

    for (int r = 0; r < reps; r++)
    {
        matMulNaive<<<gridMatMul, blockMatMul>>>(
            dA, dB, dC, NM);
    }

    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(stopM));
    CHECK(cudaEventSynchronize(stopM));

    float totalMs;

    CHECK(cudaEventElapsedTime(
        &totalMs, startM, stopM));

    float tM = totalMs / reps;

    CHECK(cudaEventDestroy(startM));
    CHECK(cudaEventDestroy(stopM));

    // ----------------------------------------
    // COPY RESULT BACK TO CPU
    // ----------------------------------------

    CHECK(cudaMemcpy(
        hC, dC, bytesM, cudaMemcpyDeviceToHost));

    // ----------------------------------------
    // VERIFY MATRIX MULTIPLICATION
    // ----------------------------------------

    bool okM = checkMatMul(hA, hB, hC, NM);

    // ----------------------------------------
    // PERFORMANCE CALCULATION
    // ----------------------------------------

    double gflops =
        2.0 * NM * NM * (double)NM / (tM * 1e6);

    printf("\nMatrix multiplication results:\n");

    printf("Naive MatMul : %.3f ms | %.2f GFLOPS | %s\n",
           tM, gflops, okM ? "PASS" : "FAIL");


    // ========================================
    // MEMORY CLEANUP
    // ========================================

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    free(h_in);
    free(h_out);

    printf("\n====================================\n");
    printf(" EXPERIMENT COMPLETED\n");
    printf("====================================\n");

    return 0;
}
