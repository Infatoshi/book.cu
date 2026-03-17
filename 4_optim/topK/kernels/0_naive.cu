/**
 * CUDA error checking macro
 * Checks CUDA function calls for errors and exits on failure
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * Naive TopK kernel using selection sort approach
 * For each of K iterations, finds the maximum value and its index
 * Very inefficient but simple to understand
 * 
 * Used in MoE (Mixture of Experts) for selecting top K experts based on routing scores
 * 
 * Algorithm:
 * 1. Initialize a boolean array to track selected elements
 * 2. For K iterations:
 *    - Find the maximum unselected element
 *    - Mark it as selected
 *    - Store its value and index
 * 
 * Performance characteristics:
 * - O(N*K) time complexity
 * - Single thread processes entire array
 * - Not optimized for GPU parallelism
 * 
 * @param input Input array (N, device memory)
 * @param indices Output array for top-K indices (K, device memory)
 * @param values Output array for top-K values (K, device memory)
 * @param N Size of input array
 * @param K Number of top elements to select
 * @param selected Boolean array to track selected elements (N, device memory)
 */
__global__ void topk_kernel_0(
    float* __restrict__ input,
    int* __restrict__ indices,
    float* __restrict__ values,
    int N, int K,
    bool* __restrict__ selected  
) {
    int row = blockIdx.x;
    
    // Process only first row (single-threaded implementation)
    if (row < 1) {
        // Initialize selection array: no elements selected yet
        for (int i = 0; i < N; i++) {
            selected[i] = false;
        }
        
        // Find top K elements using selection sort
        for (int k = 0; k < K; k++) {
            float max_val = -INFINITY;
            int max_idx = -1;
            
            // Find maximum unselected element
            for (int i = 0; i < N; i++) {
                if (!selected[i] && input[i] > max_val) {
                    max_val = input[i];
                    max_idx = i;
                }
            }
            
            // Mark element as selected and store result
            if (max_idx >= 0) {
                selected[max_idx] = true;
                values[k] = max_val;
                indices[k] = max_idx;
            }
        }
    }
}

/**
 * Launcher function for naive TopK kernel
 * Configures and launches the kernel with timing instrumentation
 * 
 * @param input Input array (N, device memory)
 * @param indices Output array for top-K indices (K, device memory)
 * @param values Output array for top-K values (K, device memory)
 * @param N Size of input array
 * @param K Number of top elements to select
 */
void run_kernel_0(float* input, int* indices, float* values, int N, int K) {
    // Configure kernel launch parameters (single block, single thread)
    dim3 block_size(1);
    dim3 grid_size(1);
    
    // Allocate device memory for selection tracking array
    bool* selected;
    CUDA_CHECK(cudaMalloc(&selected, N * sizeof(bool)));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start time and launch kernel
    CUDA_CHECK(cudaEventRecord(start));
    topk_kernel_0<<<grid_size, block_size>>>(input, indices, values, N, K, selected);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Compute elapsed time (for benchmarking)
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Clean up device memory and CUDA events
    CUDA_CHECK(cudaFree(selected));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
