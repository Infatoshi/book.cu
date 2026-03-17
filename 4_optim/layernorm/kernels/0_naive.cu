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
 * Naive LayerNorm kernel implementation
 * Each thread processes one entire row sequentially
 * Simple two-pass algorithm: compute mean, then variance
 * 
 * Layer normalization formula:
 *   out = (inp - mean) / sqrt(variance + eps) * weight + bias
 * 
 * Performance characteristics:
 * - One thread processes one entire row
 * - Sequential computation within each thread
 * - Not optimized for GPU parallelism
 * 
 * Based on llm.c/dev/cuda/layernorm_forward.cu kernel1
 * 
 * @param out Output tensor (N×C, device memory)
 * @param mean Mean values for each row (N, device memory)
 * @param rstd Reciprocal standard deviation (1/sqrt(variance+eps)) for each row (N, device memory)
 * @param inp Input tensor (N×C, device memory)
 * @param weight Scale parameters (C, device memory)
 * @param bias Shift parameters (C, device memory)
 * @param N Number of rows (batch size)
 * @param C Number of columns (feature dimension)
 */
__global__ void layernorm_kernel_0(
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C
) {
    // Calculate row index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;  // Small epsilon to prevent division by zero

    if (idx < N) {
        // Get pointer to current row
        const float* x = inp + idx * C;
        
        // Step 1: Compute mean of the row
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        
        // Step 2: Compute variance of the row
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        
        // Step 3: Compute reciprocal standard deviation
        float s = 1.0f / sqrtf(v + eps);
        
        // Step 4: Normalize and apply affine transformation
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m));  // Normalized value
            float o = n * weight[i] + bias[i];  // Apply scale and shift
            out_idx[i] = o;
        }
        
        // Store mean and reciprocal standard deviation for potential use in backward pass
        mean[idx] = m;
        rstd[idx] = s;
    }
}

/**
 * Launcher function for naive LayerNorm kernel
 * Configures and launches the kernel with timing instrumentation
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
void run_kernel_0(float* out, float* mean, float* rstd, const float* inp, 
                  const float* weight, const float* bias, int N, int C) {
    // Configure kernel launch parameters
    dim3 block_size(256);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start time and launch kernel
    CUDA_CHECK(cudaEventRecord(start));
    layernorm_kernel_0<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Compute elapsed time (for benchmarking)
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    // Clean up CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}