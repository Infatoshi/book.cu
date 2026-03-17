#include "common.h"

/**
 * @file vector_add.cu
 * @brief Naive CUDA implementation of vector addition (element-wise addition)
 * 
 * This file demonstrates the simplest parallel operation: element-wise vector addition.
 * This operation is fundamental to many neural network operations including:
 * - Bias addition in linear layers
 * - Residual connections in ResNet architectures
 * - Element-wise operations in attention mechanisms
 * 
 * Algorithm: For vectors a, b, and c of length n: c[i] = a[i] + b[i] for all i in [0, n)
 * 
 * CPU Implementation: Sequential loop processing one element at a time
 * GPU Implementation: Each thread processes one element independently in parallel
 * 
 * Memory Access Pattern:
 * - CPU: Sequential memory access (cache-friendly)
 * - GPU: Coalesced global memory access (all threads in a warp access consecutive elements)
 * 
 * Performance Characteristics:
 * - CPU: O(n) time complexity, single-threaded
 * - GPU: O(n/p) where p is number of threads, highly parallelizable
 * - Memory bandwidth limited operation (no computation bottleneck)
 */

/**
 * CPU implementation of vector addition
 * 
 * Sequential element-wise addition of two vectors performed on the CPU.
 * This serves as a reference implementation for correctness verification.
 * 
 * Algorithm:
 * - Iterate through each element index i from 0 to n-1
 * - Compute c[i] = a[i] + b[i]
 * - Memory access is sequential and cache-friendly
 * 
 * @param a Input vector A (host memory, size n)
 * @param b Input vector B (host memory, size n)
 * @param c Output vector C (host memory, size n), stores a[i] + b[i]
 * @param n Number of elements in each vector (must be >= 0)
 */
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    // Sequential loop: process one element at a time
    // Time complexity: O(n)
    // Memory access: Sequential, cache-friendly
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * CUDA kernel for vector addition
 * 
 * Parallel GPU implementation where each thread processes one element.
 * This is the simplest CUDA kernel pattern, demonstrating basic thread indexing.
 * 
 * Algorithm (same as CPU):
 * - Each thread computes one output element
 * - Thread with index i computes c[i] = a[i] + b[i]
 * 
 * Thread Indexing:
 * - Global thread ID: i = blockIdx.x * blockDim.x + threadIdx.x
 * - blockIdx.x: block index within the grid (x-dimension)
 * - blockDim.x: number of threads per block (x-dimension)
 * - threadIdx.x: thread index within its block (x-dimension)
 * 
 * Memory Access Pattern:
 * - All threads in a warp access consecutive memory locations
 * - This enables memory coalescing: multiple memory accesses are combined
 * - Optimal memory bandwidth utilization
 * 
 * Performance:
 * - Highly parallelizable: each element independent
 * - Memory bandwidth limited (very simple computation)
 * - No shared memory needed (direct global memory access is efficient)
 * 
 * @param a Input vector A (device memory, size n)
 * @param b Input vector B (device memory, size n)
 * @param c Output vector C (device memory, size n), stores a[i] + b[i]
 * @param n Number of elements in each vector (must be >= 0)
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // Calculate global thread index across all blocks
    // Formula: global_thread_id = block_id * threads_per_block + thread_id_in_block
    // This maps each thread to a unique element index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check: critical for correctness when n is not divisible by block size
    // Without this check, threads beyond n would access invalid memory
    // GPU memory access violations cause undefined behavior or crashes
    if (i < n) {
        // Perform element-wise addition: c[i] = a[i] + b[i]
        // Each thread performs independent computation
        // Memory accesses are coalesced (consecutive threads access consecutive memory)
        c[i] = a[i] + b[i];
    }
}

/**
 * Main function demonstrating vector addition on both CPU and GPU
 * 
 * This function:
 * 1. Allocates memory for input and output vectors
 * 2. Initializes test data
 * 3. Runs CPU reference implementation with timing
 * 4. Runs GPU kernel implementation with timing
 * 5. Verifies correctness by comparing CPU and GPU results
 * 6. Cleans up allocated memory
 */
int main() {
    // Large vector size for performance demonstration
    // 1 million elements = 4 MB per vector (assuming float = 4 bytes)
    // This size is large enough to demonstrate GPU parallelism benefits
    const int n = 1000000; 

    std::cout << "Vector Addition: " << n << " elements" << std::endl;

    // Allocate host (CPU) memory for input and output vectors
    // h_ prefix denotes host memory pointers
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    allocate_host(&h_a, n);      // Input vector A
    allocate_host(&h_b, n);      // Input vector B
    allocate_host(&h_c_cpu, n);  // CPU output (for verification)
    allocate_host(&h_c_gpu, n);  // GPU output (for comparison)

    // Initialize input vectors with test data
    // Pattern chosen for easy verification: c[i] = a[i] + b[i] = i + 2*i = 3*i
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);        // Vector A: [0, 1, 2, ..., 999999]
        h_b[i] = static_cast<float>(i * 2);  // Vector B: [0, 2, 4, ..., 1999998]
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        Timer cpu_timer("CPU Vector Addition");
        vector_add_cpu(h_a, h_b, h_c_cpu, n);
    } // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_a, *d_b, *d_c;
    allocate_device(&d_a, n);  // Input vector A on GPU
    allocate_device(&d_b, n);  // Input vector B on GPU
    allocate_device(&d_c, n);  // Output vector C on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_a, h_a, n);
    copy_to_device(d_b, h_b, n);

    // Configure kernel launch parameters
    // threadsPerBlock: Number of threads per block (typically 128, 256, or 512)
    // 256 is a good default: balances occupancy and register usage
    // blocksPerGrid: Number of blocks needed to cover all elements
    // Formula: ceil(n / threadsPerBlock) = (n + threadsPerBlock - 1) / threadsPerBlock
    int threadsPerBlock = 256;  // Standard block size (must be multiple of 32 for warp efficiency)
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division

    std::cout << "GPU: Launching " << blocksPerGrid << " blocks with "
              << threadsPerBlock << " threads per block" << std::endl;
    std::cout << "Total threads: " << blocksPerGrid * threadsPerBlock << std::endl;

    // Run GPU version with timing
    // Kernel launch syntax: kernel_name<<<grid_size, block_size>>>(parameters)
    // Kernel launch is asynchronous (returns immediately)
    // cudaDeviceSynchronize() waits for kernel completion
    {
        Timer gpu_timer("GPU Vector Addition");
        vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel completion
    } // Timer prints elapsed time here

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_c_gpu, d_c, n);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_c_gpu, h_c_cpu, n)) {
        std::cout << "✓ Vector addition results match!" << std::endl;
    } else {
        std::cerr << "✗ Error: GPU and CPU results do not match!" << std::endl;
    }

    // Clean up memory
    // Free all allocated memory to prevent leaks
    // Important: free device memory before host memory (good practice)
    free_host(h_a);
    free_host(h_b);
    free_host(h_c_cpu);
    free_host(h_c_gpu);
    free_device(d_a);
    free_device(d_b);
    free_device(d_c);

    return 0;
}
