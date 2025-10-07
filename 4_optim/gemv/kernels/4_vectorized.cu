
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

template<typename T>
__device__ __forceinline__ void blockReduceSum(T val, T* smem, int tid, int block_size) {
    int warp_size = 32;
    val = warpReduceSum(val);
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    if (tid < CEIL_DIV(block_size, warp_size)) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }
    if (tid / warp_size == 0) {
        val = warpReduceSum(val);
    }
    if (tid == 0) smem[0] = val;
    __syncthreads();
}
} 

/*
Vectorized Sgemv kernel

- Each block is assigned to a row of the matrix A
- Each block calculates one output element of y
- The columns are accessed in coalesced manner by threads
- Vectorized loads are done for efficient memory bandwidth
- Performs warp level + block level sum reduction
*/
__global__ void vectorized_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];

    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    int n_float4s = N / 4;

    
    
    float4* mat_row = reinterpret_cast<float4*>(matd + bid * N);
    float4* vec = reinterpret_cast<float4*>(vecd);

    
    float partial_sum = 0.f;

    for (int col = tid; col < n_float4s; col += blockDim.x) {
        float4 matval = mat_row[col];
        float4 vecval = vec[col];

        partial_sum += (matval.x * vecval.x +
                        matval.y * vecval.y +
                        matval.z * vecval.z +
                        matval.w * vecval.w);
    }

    
    
    
    blockReduceSum(partial_sum, smem, tid, blockDim.x);
    if (tid == 0) {
        float sum = smem[0];
        resd[bid] = sum;
    }
}

/*
Runs the vectorized sgemv kernel.
*/
void run_kernel_4(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int NUM_THREADS = 64;
    int warp_size = 32;

    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);
    size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    vectorized_sgemv_kernel<<<grid_size, block_size, shared_mem_size>>>(matd, vecd, resd, M, N);
}
