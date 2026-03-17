
/*
Warp-optimized LayerNorm kernel (kernel3 from llm.c)
Uses warp-level reductions for better performance
One warp processes one row

Based on llm.c/llmc/layernorm.cuh kernel3
*/

    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layernorm_kernel_2(
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int idx = blockIdx.x * num_warps + warp_id;
    if(idx >= N) { return; } 

    
    const float* x = inp + idx * C;

    
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        sum += x[i];
    }
    sum = warpReduceSum(sum);
    
    float m = __shfl_sync(0xffffffff, sum, 0) / C;
    
    if(lane_id == 0 && mean != nullptr) {
        mean[idx] = m;
    }

    
    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = warpReduceSum(sum);
    
    float s = __shfl_sync(0xffffffff, sum, 0);
    s = rsqrtf(s / C + 1e-5f);
    
    if(lane_id == 0 && rstd != nullptr) {
        rstd[idx] = s;
    }

    
    float* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        float n = s * (x[c] - m);
        o[c] = n * weight[c] + bias[c];
    }
}

void run_kernel_2(float* out, float* mean, float* rstd, const float* inp,
                  const float* weight, const float* bias, int N, int C) {
    
    int warps_per_block = 4;
    dim3 block_size(warps_per_block * WARP_SIZE);
    dim3 grid_size((N + warps_per_block - 1) / warps_per_block);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    layernorm_kernel_2<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}