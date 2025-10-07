
/*
Naive LayerNorm kernel (kernel1 from llm.c)
Each thread processes one entire row sequentially
Simple two-pass algorithm: mean, then variance

Based on llm.c/dev/cuda/layernorm_forward.cu kernel1
*/

    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
__global__ void layernorm_kernel_0(
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        
        const float* x = inp + idx * C;
        
        
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        
        
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        
        
        float s = 1.0f / sqrtf(v + eps);
        
        
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); 
            float o = n * weight[i] + bias[i]; 
            out_idx[i] = o; 
        }
        
        
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void run_kernel_0(float* out, float* mean, float* rstd, const float* inp, 
                  const float* weight, const float* bias, int N, int C) {
    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    layernorm_kernel_0<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}