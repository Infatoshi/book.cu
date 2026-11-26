#include "common.h"

/**
 * @file conv1d.cu
 * @brief Naive CUDA implementation of 1D convolution
 * 
 * This file demonstrates sliding window operations and convolution patterns.
 * 1D convolution is fundamental to many operations including:
 * - Signal processing: filtering, smoothing, edge detection
 * - Time series analysis: temporal pattern recognition
 * - Text processing: character/word-level convolutions
 * - Sequence modeling: causal convolutions in WaveNet
 * 
 * Convolution Operation: out[i] = sum(j=0 to kernel_size-1) in[i+j] * kernel[j]
 * 
 * Padding: Valid padding (no padding) - output size = input_size - kernel_size + 1
 * 
 * CPU Implementation: Sequential sliding window over input signal
 * GPU Implementation: Each thread computes one output element independently
 * 
 * Algorithm:
 * - For each output position i:
 *   - Compute dot product of kernel with input window starting at position i
 *   - Window spans from in[i] to in[i + kernel_size - 1]
 * 
 * Memory Access Pattern:
 * - CPU: Sequential access to input, each output reads kernel_size consecutive elements
 * - GPU: Each thread reads kernel_size consecutive elements (coalesced access)
 * 
 * Performance Characteristics:
 * - CPU: O(N*K) time complexity where N=input_size, K=kernel_size
 * - GPU: Highly parallelizable, each output element independent
 * - Memory bandwidth limited (many reads per output element)
 */

/**
 * CPU implementation of 1D convolution
 * 
 * Applies a convolution kernel to a 1D input signal using valid padding.
 * Each output element is computed as the dot product of the kernel with
 * a sliding window of the input signal.
 * 
 * Algorithm:
 * - For each output position i from 0 to output_size-1:
 *   1. Initialize accumulator sum = 0
 *   2. For j from 0 to kernel_size-1:
 *      - sum += in[i+j] * kernel[j]
 *   3. out[i] = sum
 * 
 * Padding: Valid padding (no padding)
 * - Output size = input_size - kernel_size + 1
 * - Ensures kernel always fits within input boundaries
 * 
 * Memory Access Pattern:
 * - Input: Sequential access (cache-friendly)
 * - Kernel: Sequential access (cache-friendly, small size)
 * - Output: Sequential write (cache-friendly)
 * 
 * Time Complexity: O(N * K) where N=input_size, K=kernel_size
 * Space Complexity: O(1) excluding input/output arrays
 * 
 * @param in Input signal (host memory, size input_size)
 * @param out Output signal (host memory, size output_size = input_size - kernel_size + 1)
 *            Stores convolution result: out[i] = sum(j) in[i+j] * kernel[j]
 * @param kernel Convolution kernel (host memory, size kernel_size)
 *               Contains filter weights to be applied
 * @param input_size Size of input signal (must be >= kernel_size)
 * @param kernel_size Size of convolution kernel (must be >= 1)
 */
void conv1d_cpu(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    // Output size is reduced by kernel_size - 1 (valid padding)
    // Formula: output_size = input_size - kernel_size + 1
    // This ensures the kernel always fits within input boundaries
    int output_size = input_size - kernel_size + 1;
    
    // Iterate through each output position
    // Each output element corresponds to one position of the sliding window
    for (int i = 0; i < output_size; ++i) {
        float sum = 0.0f;  // Accumulator for dot product
        
        // Compute dot product of kernel with input window
        // Window starts at position i and spans kernel_size elements
        // Kernel is applied element-wise to the input window
        for (int j = 0; j < kernel_size; ++j) {
            // Access in[i+j]: sequential access (cache-friendly)
            // Access kernel[j]: sequential access (cache-friendly, small kernel)
            // Multiply and accumulate: sum += in[i+j] * kernel[j]
            sum += in[i + j] * kernel[j];
        }
        // Store computed convolution result
        out[i] = sum;
    }
}

/**
 * CUDA kernel for 1D convolution
 * 
 * Parallel GPU implementation where each thread computes one output element.
 * This demonstrates the sliding window pattern in parallel.
 * 
 * Algorithm (same as CPU):
 * - Each thread computes one output element
 * - Thread with index i computes: out[i] = sum(j) in[i+j] * kernel[j]
 * 
 * Thread Indexing:
 * - Global thread ID: output_idx = blockIdx.x * blockDim.x + threadIdx.x
 * - blockIdx.x: block index within the grid (x-dimension)
 * - blockDim.x: number of threads per block (x-dimension)
 * - threadIdx.x: thread index within its block (x-dimension)
 * 
 * Memory Access Pattern:
 * - Input: Each thread reads kernel_size consecutive elements (coalesced)
 * - Kernel: All threads read same kernel (cached in constant memory or texture)
 * - Output: Each thread writes one element (coalesced)
 * 
 * Performance:
 * - Highly parallelizable: each output element independent
 * - Memory bandwidth limited: kernel_size reads per output element
 * - No shared memory: direct global memory access (acceptable for small kernels)
 * 
 * Optimization Opportunities:
 * - Shared memory: Cache input window to reduce global memory accesses
 * - Constant memory: Store kernel in constant memory for fast access
 * - Texture memory: Use texture cache for input signal
 * 
 * @param in Input signal (device memory, size input_size)
 * @param out Output signal (device memory, size output_size = input_size - kernel_size + 1)
 *            Stores convolution result: out[i] = sum(j) in[i+j] * kernel[j]
 * @param kernel Convolution kernel (device memory, size kernel_size)
 *               Contains filter weights to be applied
 * @param input_size Size of input signal (must be >= kernel_size)
 * @param kernel_size Size of convolution kernel (must be >= 1)
 */
__global__ void conv1d_kernel(const float* in, float* out, const float* kernel, int input_size, int kernel_size) {
    // Calculate output index from thread index
    // Formula: global_thread_id = block_id * threads_per_block + thread_id_in_block
    // This maps each thread to a unique output element index
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Output size calculation (valid padding)
    // Formula: output_size = input_size - kernel_size + 1
    int output_size = input_size - kernel_size + 1;

    // Bounds check: critical for correctness when output_size is not divisible by block size
    // Without this check, threads beyond output_size would access invalid memory
    // GPU memory access violations cause undefined behavior or crashes
    if (output_idx < output_size) {
        float sum = 0.0f;  // Accumulator for dot product
        
        // Compute dot product of kernel with input window
        // Window starts at position output_idx and spans kernel_size elements
        // Kernel is applied element-wise to the input window
        for (int k_idx = 0; k_idx < kernel_size; ++k_idx) {
            // Access in[output_idx + k_idx]: sequential access within thread
            // Memory access pattern: coalesced (consecutive threads access consecutive elements)
            // Access kernel[k_idx]: sequential access (small kernel, may be cached)
            // Multiply and accumulate: sum += in[output_idx + k_idx] * kernel[k_idx]
            sum += in[output_idx + k_idx] * kernel[k_idx];
        }
        // Store computed convolution result
        // Memory access: coalesced (threads write consecutive elements)
        out[output_idx] = sum;
    }
}

int main() {
    // Signal and kernel dimensions
    const int input_size = 100000;  
    const int kernel_size = 32;     
    const int output_size = input_size - kernel_size + 1;

    std::cout << "1D Convolution: Input[" << input_size << "] * Kernel[" << kernel_size
              << "] -> Output[" << output_size << "]" << std::endl;

    // Allocate host memory for input, kernel, and output
    float *h_in, *h_kernel, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, input_size);
    allocate_host(&h_kernel, kernel_size);
    allocate_host(&h_out_cpu, output_size);
    allocate_host(&h_out_gpu, output_size);

    // Initialize input signal with test data (pattern 0-9 repeating)
    for (int i = 0; i < input_size; ++i) {
        h_in[i] = static_cast<float>(i % 10);  
    }

    // Initialize kernel as averaging filter (normalized)
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = 1.0f / kernel_size;  
    }

    // Run CPU version with timing
    {
        Timer cpu_timer("CPU 1D Convolution");
        conv1d_cpu(h_in, h_out_cpu, h_kernel, input_size, kernel_size);
    }

    // Allocate device memory
    float *d_in, *d_kernel, *d_out;
    allocate_device(&d_in, input_size);
    allocate_device(&d_kernel, kernel_size);
    allocate_device(&d_out, output_size);

    // Copy data from host to device
    copy_to_device(d_in, h_in, input_size);
    copy_to_device(d_kernel, h_kernel, kernel_size);

    // Configure kernel launch parameters
    int threadsPerBlock = 256;  // Standard block size
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "GPU: Launching " << blocksPerGrid << " blocks with "
              << threadsPerBlock << " threads per block" << std::endl;

    // Run GPU version with timing
    {
        Timer gpu_timer("GPU 1D Convolution");
        conv1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_kernel, input_size, kernel_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result back from device to host
    copy_to_host(h_out_gpu, d_out, output_size);

    // Verify GPU results against CPU results
    if (verify_results(h_out_gpu, h_out_cpu, output_size)) {
        std::cout << "✓ 1D convolution results match!" << std::endl;

        // Display sample input, kernel, and output
        std::cout << "\nInput signal (first 10): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_in[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Kernel (first 10): ";
        for (int i = 0; i < std::min(10, kernel_size); ++i) {
            std::cout << h_kernel[i] << " ";
        }
        if (kernel_size > 10) std::cout << "...";
        std::cout << std::endl;

        std::cout << "Output (first 10): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_out_gpu[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "\nVerification: For averaging kernel, output[" << kernel_size/2 << "] should be ~"
                  << (kernel_size * 4.5f) / kernel_size << " (average of 0-9 pattern)" << std::endl;
        std::cout << "Actual output[" << kernel_size/2 << "] = " << h_out_gpu[kernel_size/2] << std::endl;
    }

    // Clean up memory
    free_host(h_in);
    free_host(h_kernel);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_kernel);
    free_device(d_out);

    return 0;
}
