
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/*
Naive Sgemv kernel

- Each thread calculates one element of the output vector
- The row index is calculated using block index and thread index
- Uses linearized indexing
- Memory accesses are not coalesced
*/
__global__ void naive_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += matd[row * N + col] * vecd[col];
        }
        resd[row] = sum;
    }
}

/*
Runs the naive Sgemv kernel.
*/
void run_kernel_1(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    naive_sgemv_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}
