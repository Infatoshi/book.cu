/**
 * @file 1_parallel.cu
 * @brief Parallel LayerNorm kernel using thread coarsening and shared memory reductions
 * 
 * This kernel improves upon the naive implementation by using multiple threads
 * to process each row in parallel, leveraging shared memory for efficient reduction.
 * 
 * Optimization Journey:
 * Kernel 0: Naive - one thread processes entire row sequentially
 * Kernel 1: Parallel (this file) - multiple threads cooperate per row with shared memory
 * Kernel 2: Warp-optimized - warp-level primitives for better performance
 * 
 * Key Optimization:
 * - Multiple threads cooperate to process each row
 * - Uses shared memory for efficient block-level reductions
 * - Three-phase approach: mean computation, variance computation, normalization
 * - Parallel reduction tree in shared memory
 * 
 * Performance Impact:
 * - Better GPU utilization through parallel processing
 * - Faster reductions using shared memory (faster than global memory)
 * - Separates computation phases for better instruction scheduling
 * 
 * Based on llm.c/dev/cuda/layernorm_forward.cu kernel2
 */

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
 * @brief Kernel to compute mean for each row
 * 
 * Computes the mean of each row using parallel reduction in shared memory.
 * Each block processes one row, with multiple threads computing partial sums.
 * 
 * Algorithm:
 * 1. Each thread computes partial sum over assigned elements (stride = block_size)
 * 2. Store partial sums in shared memory
 * 3. Perform reduction tree in shared memory (log2(block_size) steps)
 * 4. Thread 0 computes final mean and writes to global memory
 * 
 * @param mean Output array for mean values (N, device memory)
 * @param inp Input tensor (N×C, device memory)
 * @param N Number of rows (batch size)
 * @param C Number of columns (feature dimension)
 * @param block_size Number of threads per block
 */
__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    // Shared memory for block-level reduction
    extern __shared__ float shared[];
    
    // Row index: which row this block processes
    int idx = blockIdx.x; 
    // Thread index within block
    int tid = threadIdx.x; 
    // Pointer to current row
    const float* x = inp + idx * C;
    
    // Phase 1: Each thread computes partial sum
    // Threads access elements with stride = block_size for coalescing
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    // Store partial sum in shared memory
    shared[tid] = sum;
    __syncthreads();
    
    // Phase 2: Parallel reduction tree in shared memory
    // Reduces partial sums to single value using binary tree
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            // Combine two partial sums
            shared[tid] += shared[tid + stride];
        }
    }
    
    // Phase 3: Thread 0 computes final mean
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

/**
 * @brief Kernel to compute reciprocal standard deviation (rstd) for each row
 * 
 * Computes the variance of each row and converts to reciprocal standard deviation.
 * Uses the same parallel reduction approach as mean computation.
 * 
 * Algorithm:
 * 1. Load mean value for this row
 * 2. Each thread computes partial sum of squared differences
 * 3. Store partial sums in shared memory
 * 4. Perform reduction tree in shared memory
 * 5. Thread 0 computes final rstd = 1/sqrt(variance + eps)
 * 
 * @param rstd Output array for reciprocal standard deviation (N, device memory)
 * @param inp Input tensor (N×C, device memory)
 * @param mean Mean values for each row (N, device memory)
 * @param N Number of rows (batch size)
 * @param C Number of columns (feature dimension)
 * @param block_size Number of threads per block
 */
__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    // Shared memory for block-level reduction
    extern __shared__ float shared[];
    
    // Row index: which row this block processes
    int idx = blockIdx.x; 
    // Thread index within block
    int tid = threadIdx.x; 
    // Pointer to current row
    const float* x = inp + idx * C;
    // Load mean for this row
    float m = mean[idx];
    
    // Phase 1: Each thread computes partial sum of squared differences
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;  // Squared difference
    }
    // Store partial sum in shared memory
    shared[tid] = sum;
    __syncthreads();
    
    // Phase 2: Parallel reduction tree in shared memory
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    
    // Phase 3: Thread 0 computes final reciprocal standard deviation
    if (tid == 0) {
        // rstd = 1 / sqrt(variance + eps)
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

/**
 * @brief Kernel to apply normalization and affine transformation
 * 
 * Applies layer normalization formula: out = (inp - mean) / std * weight + bias
 * Each thread processes one element of the output tensor.
 * 
 * @param out Output tensor (N×C, device memory)
 * @param inp Input tensor (N×C, device memory)
 * @param mean Mean values for each row (N, device memory)
 * @param rstd Reciprocal standard deviation for each row (N, device memory)
 * @param weight Scale parameters (C, device memory)
 * @param bias Shift parameters (C, device memory)
 * @param N Number of rows (batch size)
 * @param C Number of columns (feature dimension)
 */
__global__ void normalization_kernel(float* out, const float* inp, const float* mean, const float* rstd,
                                     const float* weight, const float* bias, int N, int C) {
    // Linearized thread index across all blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        // Compute row and column indices
        int row = idx / C;
        int col = idx % C;
        
        // Load mean and rstd for this row
        float m = mean[row];
        float s = rstd[row];
        
        // Apply normalization: (inp - mean) * rstd
        float n = s * (inp[idx] - m);
        
        // Apply affine transformation: normalized * weight + bias
        out[idx] = n * weight[col] + bias[col];
    }
}

/**
 * @brief Launcher function for parallel LayerNorm kernel
 * 
 * Configures and launches the three-phase LayerNorm computation:
 * 1. Mean computation kernel
 * 2. Reciprocal standard deviation computation kernel
 * 3. Normalization and affine transformation kernel
 * 
 * @param out Output tensor (N×C, device memory)
 * @param mean Mean values for each row (N, device memory)
 * @param rstd Reciprocal standard deviation for each row (N, device memory)
 * @param inp Input tensor (N×C, device memory)
 * @param weight Scale parameters (C, device memory)
 * @param bias Shift parameters (C, device memory)
 * @param N Number of rows (batch size)
 * @param C Number of columns (feature dimension)
 */
void run_kernel_1(float* out, float* mean, float* rstd, const float* inp,
                  const float* weight, const float* bias, int N, int C) {
    int block_size = 256;  // Threads per block for reduction kernels
    int smem_size = block_size * sizeof(float);  // Shared memory per block
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Phase 1: Compute mean for each row
    // One block per row, each block has block_size threads
    mean_kernel<<<N, block_size, smem_size>>>(mean, inp, N, C, block_size);
    
    // Phase 2: Compute reciprocal standard deviation for each row
    rstd_kernel<<<N, block_size, smem_size>>>(rstd, inp, mean, N, C, block_size);
    
    // Phase 3: Apply normalization and affine transformation
    // One thread per output element
    normalization_kernel<<<(N*C + 255)/256, 256>>>(out, inp, mean, rstd, weight, bias, N, C);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Compute elapsed time (for benchmarking)
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Clean up CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
