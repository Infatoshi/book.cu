
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

namespace {
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
} 

/*
Coalesced Warp Sgemv kernel

- Each block is assigned to a row of the matrix A
- Each block calculates one output element of y
- The columns are accessed in coalesced manner by threads
- Performs warp level sum reduction only
- Block size must be equal to number of threads
*/
__global__ void coalesced_warp_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    assert(blockDim.x == warpSize);

    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += matd[bid * N + col] * vecd[col];
    }

    
    
    float sum = warpReduceSum(partial_sum);
    if (tid == 0) {
        resd[bid] = sum;
    }
}

/*
Runs the coalesced warp sgemv kernel.
*/
void run_kernel_2(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int NUM_THREADS = 32;  

    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);

    coalesced_warp_sgmev_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}
