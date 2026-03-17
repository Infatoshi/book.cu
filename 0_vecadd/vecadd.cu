/**
 * @file vecadd.cu
 * @brief Basic CUDA vector addition example demonstrating fundamental GPU programming concepts.
 * 
 * This file implements the simplest CUDA program: element-wise addition of two vectors.
 * It demonstrates:
 * - Basic CUDA kernel definition using __global__ qualifier
 * - Simple thread indexing with threadIdx.x
 * - Host-to-device and device-to-host memory transfers
 * - Kernel launch configuration with single thread block
 * 
 * This example uses a fixed-size vector (8 elements) and a single thread block,
 * making it ideal for understanding the basic CUDA programming model.
 * 
 * @author CUDA Book Examples
 * @date Chapter 02: GPU Memory Management and Kernel Launch
 */

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>

/**
 * @brief CUDA kernel for element-wise vector addition.
 * 
 * This kernel runs on the GPU and performs parallel addition of two input vectors.
 * Each thread processes exactly one element, identified by its thread index within
 * the block. This is the simplest possible CUDA kernel pattern.
 * 
 * @note This kernel assumes exactly 8 threads (one per element) and uses only
 * threadIdx.x, making it suitable only for small, fixed-size vectors.
 * 
 * @param[in] a Input vector A stored in GPU device memory
 * @param[in] b Input vector B stored in GPU device memory
 * @param[out] c Output vector C stored in GPU device memory, where c[i] = a[i] + b[i]
 * 
 * @warning This kernel does not perform bounds checking. It assumes the number of
 * threads launched exactly matches the vector size.
 */
__global__ void vectorAdd(float *a, float *b, float *c) {
    // Get the thread index within the block
    int i = threadIdx.x;
    
    // Perform element-wise addition: c[i] = a[i] + b[i]
    c[i] = a[i] + b[i];
}

/**
 * @brief Main function demonstrating basic CUDA vector addition workflow.
 * 
 * This function demonstrates the complete CUDA program execution flow:
 * 1. Allocate host (CPU) memory
 * 2. Initialize input data
 * 3. Allocate device (GPU) memory using cudaMalloc
 * 4. Transfer data from host to device using cudaMemcpy
 * 5. Launch kernel with specified grid/block configuration
 * 6. Transfer results back from device to host
 * 7. Verify correctness
 * 8. Clean up allocated memory
 * 
 * @return int Returns 0 on success, non-zero on failure
 */
int main() {
    // Vector size: fixed at 8 elements for this basic example
    int n = 8;
    // Calculate total memory size in bytes for all vectors
    size_t size = n * sizeof(float);

    // ============================================================
    // STEP 1: Allocate host (CPU) memory using standard malloc
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
    // STEP 2: Initialize input vectors with test data
    // ============================================================
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;        // Vector A: [0, 1, 2, 3, 4, 5, 6, 7]
        h_b[i] = (float)(i * 2);  // Vector B: [0, 2, 4, 6, 8, 10, 12, 14]
        // Expected result: C = [0, 3, 6, 9, 12, 15, 18, 21]
    }

    // ============================================================
    // STEP 3: Allocate device (GPU) memory using cudaMalloc
    // ============================================================
    // Device memory pointers (d_ prefix convention)
    float *d_a, *d_b, *d_c;
    
    // cudaMalloc allocates memory on the GPU device
    // Returns cudaError_t status; should check for errors in production code
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
    // STEP 4: Copy data from host (CPU) to device (GPU)
    // ============================================================
    // cudaMemcpy performs synchronous memory transfer
    // Syntax: cudaMemcpy(destination, source, size, direction)
    // Direction: cudaMemcpyHostToDevice for CPU -> GPU
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
    // STEP 5: Launch CUDA kernel on the GPU
    // ============================================================
    // Kernel launch syntax: kernelName<<<blocksPerGrid, threadsPerBlock>>>(args)
    // - blocksPerGrid: Number of thread blocks (1 in this case)
    // - threadsPerBlock: Number of threads per block (8 in this case)
    // - Total threads: 1 * 8 = 8 threads (one per vector element)
    //
    // The <<< >>> syntax is CUDA-specific and is processed by nvcc compiler
    // This launch is asynchronous - CPU continues immediately after launch
    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);
    
    // Check for kernel launch errors
    // Note: Kernel launches are asynchronous, so we check after the launch
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
    // STEP 6: Copy results back from device (GPU) to host (CPU)
    // ============================================================
    // Direction: cudaMemcpyDeviceToHost for GPU -> CPU
    err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (DeviceToHost): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        free(h_a); free(h_b); free(h_c);
        return 1;
    }

    // ============================================================
    // STEP 7: Verify the results for correctness
    // ============================================================
    bool success = true;
    for (int i = 0; i < n; ++i) {
        // Use floating-point comparison with epsilon tolerance
        // Direct equality (==) is unreliable for floating-point numbers
        float expected = h_a[i] + h_b[i];
        float diff = std::abs(expected - h_c[i]);
        if (diff > std::numeric_limits<float>::epsilon()) {
            std::cout << "Error at index " << i << ": Got " << h_c[i] 
                      << ", expected " << expected << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "Success! All elements are correct." << std::endl;
    }

    // ============================================================
    // STEP 8: Clean up allocated memory
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
