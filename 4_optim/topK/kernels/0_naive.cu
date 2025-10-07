
/*
Naive TopK kernel: Selection sort approach
For each of K iterations, find the maximum value and its index
Very inefficient but simple to understand

Used in MoE for selecting top K experts based on routing scores
*/

    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
__global__ void topk_kernel_0(
    float* __restrict__ input,
    int* __restrict__ indices,
    float* __restrict__ values,
    int N, int K,
    bool* __restrict__ selected  
) {
    int row = blockIdx.x;
    
    if (row < 1) {  
        
        for (int i = 0; i < N; i++) {
            selected[i] = false;
        }
        
        
        for (int k = 0; k < K; k++) {
            float max_val = -INFINITY;
            int max_idx = -1;
            
            for (int i = 0; i < N; i++) {
                if (!selected[i] && input[i] > max_val) {
                    max_val = input[i];
                    max_idx = i;
                }
            }
            
            if (max_idx >= 0) {
                selected[max_idx] = true;
                values[k] = max_val;
                indices[k] = max_idx;
            }
        }
    }
}

void run_kernel_0(float* input, int* indices, float* values, int N, int K) {
    dim3 block_size(1);
    dim3 grid_size(1);
    
    
    bool* selected;
    CUDA_CHECK(cudaMalloc(&selected, N * sizeof(bool)));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    topk_kernel_0<<<grid_size, block_size>>>(input, indices, values, N, K, selected);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaFree(selected));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
