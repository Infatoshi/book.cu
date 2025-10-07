
/*
Parallel LayerNorm kernel (kernel2 from llm.c)
Uses thread coarsening and shared memory reductions
Multiple threads cooperate to process each row

Based on llm.c/dev/cuda/layernorm_forward.cu kernel2
*/

    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; 
    int tid = threadIdx.x; 
    const float* x = inp + idx * C;
    
    
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    
    
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; 
    int tid = threadIdx.x; 
    const float* x = inp + idx * C;
    float m = mean[idx];
    
    
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    
    
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    
    
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void normalization_kernel(float* out, const float* inp, const float* mean, const float* rstd,
                                     const float* weight, const float* bias, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int row = idx / C;
        int col = idx % C;
        float m = mean[row];
        float s = rstd[row];
        float n = s * (inp[idx] - m);
        out[idx] = n * weight[col] + bias[col];
    }
}

void run_kernel_1(float* out, float* mean, float* rstd, const float* inp,
                  const float* weight, const float* bias, int N, int C) {
    int block_size = 256;
    int smem_size = block_size * sizeof(float);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    
    mean_kernel<<<N, block_size, smem_size>>>(mean, inp, N, C, block_size);
    rstd_kernel<<<N, block_size, smem_size>>>(rstd, inp, mean, N, C, block_size);
    normalization_kernel<<<(N*C + 255)/256, 256>>>(out, inp, mean, rstd, weight, bias, N, C);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
