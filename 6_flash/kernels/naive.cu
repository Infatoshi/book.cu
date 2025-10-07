
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

/*
Naive Attention Kernel
This is a straightforward implementation that:
1. Computes Q @ K^T
2. Applies scaling
3. Computes softmax
4. Multiplies by V

All steps materialize intermediate results in global memory (HBM).
No tiling, no shared memory optimizations.
*/

template <int BLOCK_SIZE>
__global__ void naive_qk_matmul_kernel(
    float* Q, float* K, float* S,
    int N, int d, float scale
) {
    
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        S[row * N + col] = sum * scale;
    }
}

__global__ void naive_softmax_kernel(float* S, int N) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        float* row_ptr = S + row * N;
        
        
        float max_val = -INFINITY;
        for (int i = 0; i < N; i++) {
            max_val = fmaxf(max_val, row_ptr[i]);
        }
        
        
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            row_ptr[i] = expf(row_ptr[i] - max_val);
            sum += row_ptr[i];
        }
        
        
        for (int i = 0; i < N; i++) {
            row_ptr[i] /= sum;
        }
    }
}

template <int BLOCK_SIZE>
__global__ void naive_sv_matmul_kernel(
    float* S, float* V, float* O,
    int N, int d
) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < d) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += S[row * N + j] * V[j * d + col];
        }
        O[row * d + col] = sum;
    }
}

torch::Tensor naive_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);
    
    float scale = 1.0f / sqrtf((float)d);
    
    auto O = torch::zeros_like(Q);
    
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < nh; h++) {
            int offset = (b * nh + h) * N * d;
            int s_offset = (b * nh + h) * N * N;
            
            float* Q_ptr = Q.data_ptr<float>() + offset;
            float* K_ptr = K.data_ptr<float>() + offset;
            float* V_ptr = V.data_ptr<float>() + offset;
            float* O_ptr = O.data_ptr<float>() + offset;
            
            
            float* S;
            CUDA_CHECK(cudaMalloc(&S, N * N * sizeof(float)));
            
            
            const int BLOCK_SIZE = 16;
            dim3 block_qk(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_qk((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                         (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            naive_qk_matmul_kernel<BLOCK_SIZE><<<grid_qk, block_qk>>>(
                Q_ptr, K_ptr, S, N, d, scale);
            
            
            dim3 block_softmax(256);
            dim3 grid_softmax((N + 255) / 256);
            naive_softmax_kernel<<<grid_softmax, block_softmax>>>(S, N);
            
            
            dim3 block_sv(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_sv((d + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            naive_sv_matmul_kernel<BLOCK_SIZE><<<grid_sv, block_sv>>>(
                S, V_ptr, O_ptr, N, d);
            
            CUDA_CHECK(cudaFree(S));
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    return O;
}
