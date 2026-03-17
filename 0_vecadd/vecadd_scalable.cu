/**
 * @file vecadd_scalable.cu
 * @brief Scalable CUDA vector addition example demonstrating multi-block kernel launches.
 * 
 * This file extends the basic vector addition example to handle vectors of arbitrary size
 * using multiple thread blocks. It demonstrates key concepts for scalable CUDA programming:
 * - Global thread indexing across multiple blocks
 * - Boundary checking to prevent out-of-bounds memory access
 * - Optimal grid/block dimension calculation
 * - Handling vectors larger than a single thread block
 * 
 * Unlike the basic vecadd.cu example, this version can handle vectors of any size,
 * making it suitable for production-level CUDA code.
 * 
 * @author CUDA Book Examples
 * @date Chapter 02: GPU Memory Management and Kernel Launch
 */

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>

/**
 * @brief Scalable CUDA kernel for element-wise vector addition.
 * 
 * This kernel uses global thread indexing to handle vectors of any size, not just
 * those that fit in a single thread block. It demonstrates the standard pattern
 * for writing scalable CUDA kernels:
 * 1. Calculate global thread index from block and thread indices
 * 2. Check bounds before accessing memory
 * 3. Process one element per thread
 * 
 * @note This kernel can handle vectors larger than the maximum threads per block
 * (typically 1024 threads) by using multiple blocks. The grid configuration
 * determines how many blocks are launched.
 * 
 * @param[in] a Input vector A stored in GPU device memory
 * @param[in] b Input vector B stored in GPU device memory
 * @param[out] c Output vector C stored in GPU device memory, where c[i] = a[i] + b[i]
 * @param[in] n Total number of elements in each vector (size of vectors)
 * 
 * @details Thread indexing formula:
 *   - blockIdx.x: Index of the current thread block within the grid
 *   - blockDim.x: Number of threads per block (constant for all blocks)
 *   - threadIdx.x: Index of the current thread within its block
 *   - Global index: i = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * @warning The bounds check (i < n) is critical. Without it, threads that exceed
 * the vector size would access invalid memory, causing undefined behavior.
 */
__global__ void vectorAddScalable(float *a, float *b, float *c, int n) {
    // Calculate global thread index across all blocks
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to ensure we don't access out-of-range elements
    if (i < n) {
        // Perform element-wise addition: c[i] = a[i] + b[i]
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief Main function demonstrating scalable CUDA vector addition workflow.
 * 
 * This function demonstrates how to handle large vectors using multiple thread blocks.
 * The key difference from the basic example is the calculation of grid dimensions
 * to ensure all vector elements are processed, even when the vector size exceeds
 * the maximum threads per block.
 * 
 * @return int Returns 0 on success, non-zero on failure
 */
int main() {
    // ============================================================
    // STEP 1: Define problem size and memory requirements
    // ============================================================
    // Large vector size (1 million elements) to demonstrate scalability
    // This size cannot fit in a single thread block (max ~1024 threads)
    int n = 1000000;
    // Calculate total memory size in bytes for all vectors
    size_t size = n * sizeof(float);

    // ============================================================
    // STEP 2: Allocate host (CPU) memory
    // ============================================================
    // Host memory pointers (h_ prefix convention)
    float *h_a = (float*)malloc(size);  // Input vector A on CPU
    float *h_b = (float*)malloc(size);  // Input vector B on CPU
    float *h_c = (float*)malloc(size);  // Output vector C on CPU

    // Check for host memory allocation failure
    if (h_a == nullptr || h_b == nullptr || h_c == nullptr) {
        std::cerr << "Error: Failed to allocate host memory" << std::endl;
        return 1;
    }

    // ============================================================
    // STEP 3: Initialize input vectors with test data
    // ============================================================
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;        // Vector A: [0, 1, 2, ..., 999999]
        h_b[i] = (float)(i * 2);  // Vector B: [0, 2, 4, ..., 1999998]
        // Expected result: C = [0, 3, 6, 9, ..., 2999997]
    }

    // ============================================================
    // STEP 4: Allocate device (GPU) memory using cudaMalloc
    // ============================================================
    // Device memory pointers (d_ prefix convention)
    float *d_a, *d_b, *d_c;
    
    // cudaMalloc allocates memory on the GPU device
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_a: " << cudaGetErrorString(err) << std::endl;
        free(h_a); free(h_b); free(h_c);
        return 1;
    }
    
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_b: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); free(h_a); free(h_b); free(h_c);
        return 1;
    }
    
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_c: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); free(h_a); free(h_b); free(h_c);
        return 1;
    }

    // ============================================================
    // STEP 5: Copy data from host (CPU) to device (GPU)
    // ============================================================
    // cudaMemcpy performs synchronous memory transfer
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (HostToDevice): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 1;
    }
    
    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (HostToDevice): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 1;
    }

    // ============================================================
    // STEP 6: Configure kernel launch parameters for scalability
    // ============================================================
    // Choose thread block size (typically 256 or 512 for good performance)
    // Common choices: 128, 256, 512, 1024 (max threads per block)
    // 256 is often optimal for memory-bound kernels like vector addition
    int threadsPerBlock = 256;
    
    // Calculate number of blocks needed to cover all elements
    // Formula: ceil(n / threadsPerBlock) = (n + threadsPerBlock - 1) / threadsPerBlock
    // This ensures we have enough threads to cover all elements, even if n is not
    // evenly divisible by threadsPerBlock. Extra threads will be handled by bounds
    // checking in the kernel.
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Example calculation for n=1,000,000, threadsPerBlock=256:
    // blocksPerGrid = (1000000 + 256 - 1) / 256 = 1000255 / 256 = 3908 blocks
    // Total threads = 3908 * 256 = 1,000,448 threads (more than needed, but safe)
    
    std::cout << "Launching " << blocksPerGrid << " blocks with " << threadsPerBlock
              << " threads each (total: " << blocksPerGrid * threadsPerBlock << " threads)" << std::endl;
    std::cout << "Processing " << n << " elements" << std::endl;

    // ============================================================
    // STEP 7: Launch scalable CUDA kernel on the GPU
    // ============================================================
    // Kernel launch syntax: kernelName<<<blocksPerGrid, threadsPerBlock>>>(args)
    // - blocksPerGrid: Number of thread blocks (3908 in this example)
    // - threadsPerBlock: Number of threads per block (256 in this example)
    // - Total threads: 3908 * 256 = 1,000,448 threads
    //
    // This launch is asynchronous - CPU continues immediately after launch
    vectorAddScalable<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error: Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 1;
    }
    
    // Wait for kernel to complete before copying results back
    // cudaDeviceSynchronize ensures all GPU operations are finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error: Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 1;
    }

    // ============================================================
    // STEP 8: Copy results back from device (GPU) to host (CPU)
    // ============================================================
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (DeviceToHost): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 1;
    }

    // ============================================================
    // STEP 9: Verify results for correctness
    // ============================================================
    // For large vectors, checking all elements would be too slow
    // Instead, verify a sample: first 10 and last 10 elements
    bool success = true;
    
    // Check first 10 elements
    for (int i = 0; i < 10 && success; ++i) {
        float expected = h_a[i] + h_b[i];
        float diff = std::abs(expected - h_c[i]);
        if (diff > std::numeric_limits<float>::epsilon()) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] 
                      << ", expected " << expected << std::endl;
            success = false;
        }
    }
    
    // Check last 10 elements
    for (int i = n - 10; i < n && success; ++i) {
        float expected = h_a[i] + h_b[i];
        float diff = std::abs(expected - h_c[i]);
        if (diff > std::numeric_limits<float>::epsilon()) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] 
                      << ", expected " << expected << std::endl;
            success = false;
        }
    }
    
    if (success) {
        std::cout << "Success! All checked elements are correct." << std::endl;
    }

    // ============================================================
    // STEP 10: Clean up allocated memory
    // ============================================================
    // Free host memory (CPU)
    free(h_a);
    free(h_b);
    free(h_c);
    
    // Free device memory (GPU)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return success ? 0 : 1;
}
