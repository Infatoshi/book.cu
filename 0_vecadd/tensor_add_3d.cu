/**
 * @file tensor_add_3d.cu
 * @brief 3D tensor addition example demonstrating multi-dimensional CUDA indexing.
 * 
 * This file demonstrates how to work with 3D tensors in CUDA, which is common in
 * deep learning applications (e.g., batches of images, feature maps). It shows:
 * - 3D thread block and grid configurations using dim3
 * - Multi-dimensional thread indexing (x, y, z coordinates)
 * - Converting 3D coordinates to 1D memory indices (row-major order)
 * - CPU vs GPU comparison with timing
 * 
 * The example processes a 3D tensor with dimensions (depth, height, width),
 * where each dimension can be independently sized and processed.
 * 
 * @author CUDA Book Examples
 * @date Chapter 02: GPU Memory Management and Kernel Launch
 */

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <chrono>

/**
 * @brief CPU implementation of 3D tensor addition (reference implementation).
 * 
 * This function performs element-wise addition of two 3D tensors on the CPU.
 * It serves as a reference implementation for verifying GPU correctness.
 * The CPU version uses nested loops to iterate through all 3D coordinates.
 * 
 * Memory layout: Tensors are stored in row-major order (C-style), meaning:
 * - Elements are contiguous in the width dimension
 * - Height dimension changes next
 * - Depth dimension changes last
 * 
 * Index calculation: index = d * (height * width) + h * width + w
 * 
 * @param[in] A Input tensor A stored in host (CPU) memory
 * @param[in] B Input tensor B stored in host (CPU) memory
 * @param[out] C Output tensor C stored in host (CPU) memory, where C[d][h][w] = A[d][h][w] + B[d][h][w]
 * @param[in] depth Number of depth layers in the 3D tensor
 * @param[in] height Number of height rows in each depth layer
 * @param[in] width Number of width columns in each row
 * 
 * @note This is a sequential CPU implementation. For large tensors, the GPU
 * version will be significantly faster due to parallel processing.
 */
void tensorAdd3D_cpu(const float* A, const float* B, float* C, int depth, int height, int width) {
    // Iterate through all 3D coordinates
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // Convert 3D coordinates to 1D index (row-major order)
                int index = d * (height * width) + h * width + w;
                // Perform element-wise addition: C[index] = A[index] + B[index]
                C[index] = A[index] + B[index];
            }
        }
    }
}

/**
 * @brief CUDA kernel for 3D tensor addition using multi-dimensional indexing.
 * 
 * This kernel uses 3D thread indexing to map threads directly to 3D tensor coordinates,
 * making the code more intuitive and aligned with the problem structure. Each thread
 * processes exactly one tensor element identified by its 3D coordinates (d, h, w).
 * 
 * Thread indexing:
 * - blockIdx.x/y/z: Block index in each dimension of the grid
 * - blockDim.x/y/z: Number of threads per block in each dimension
 * - threadIdx.x/y/z: Thread index within the block in each dimension
 * - Global coordinates: computed independently for each dimension
 * 
 * The kernel demonstrates how to use dim3 for both grid and block dimensions,
 * allowing natural mapping of threads to multi-dimensional data structures.
 * 
 * @param[in] A Input tensor A stored in GPU device memory
 * @param[in] B Input tensor B stored in GPU device memory
 * @param[out] C Output tensor C stored in GPU device memory, where C[d][h][w] = A[d][h][w] + B[d][h][w]
 * @param[in] depth Number of depth layers in the 3D tensor
 * @param[in] height Number of height rows in each depth layer
 * @param[in] width Number of width columns in each row
 * 
 * @details Memory layout and indexing:
 *   - Tensors are stored in row-major order (C-style)
 *   - 3D coordinate (d, h, w) maps to 1D index: d * (height * width) + h * width + w
 *   - This formula ensures contiguous access patterns for better memory coalescing
 * 
 * @warning The bounds check (d < depth && h < height && w < width) is essential.
 * When grid dimensions are calculated using ceiling division, some threads may
 * be launched beyond the tensor boundaries and must be safely handled.
 */
__global__ void tensorAdd3D_kernel(const float* A, const float* B, float* C, int depth, int height, int width) {
    // Calculate 3D coordinates from thread indices
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // Width (x-dimension)
    int h = blockIdx.y * blockDim.y + threadIdx.y;   // Height (y-dimension)
    int d = blockIdx.z * blockDim.z + threadIdx.z;   // Depth (z-dimension)

    // Bounds check to ensure we don't access out-of-range elements
    if (d < depth && h < height && w < width) {
        // Convert 3D coordinates to 1D index (row-major order)
        int index = d * (height * width) + h * width + w;
        // Perform element-wise addition: C[index] = A[index] + B[index]
        C[index] = A[index] + B[index];
    }
}

/**
 * @brief Main function demonstrating 3D tensor addition with CPU/GPU comparison.
 * 
 * This function demonstrates the complete workflow for 3D tensor operations:
 * 1. Define 3D tensor dimensions
 * 2. Allocate and initialize host memory
 * 3. Allocate device memory
 * 4. Transfer data to device
 * 5. Configure 3D kernel launch parameters
 * 6. Launch kernel with 3D grid/block configuration
 * 7. Transfer results back to host
 * 8. Compare GPU results with CPU reference implementation
 * 9. Measure and report CPU execution time
 * 
 * @return int Returns 0 on success, non-zero on failure
 */
int main() {
    // ============================================================
    // STEP 1: Define 3D tensor dimensions
    // ============================================================
    // Tensor dimensions: depth x height x width
    // Example: 32 depth layers, each with 128x128 height/width
    // This represents a common structure in deep learning (e.g., batch of images)
    int depth = 32, height = 128, width = 128;
    int total_elements = depth * height * width;
    size_t size = total_elements * sizeof(float);

    std::cout << "3D Tensor Addition: " << depth << "x" << height << "x" << width
              << " = " << total_elements << " elements" << std::endl;
    std::cout << "Memory per tensor: " << size / (1024 * 1024) << " MB" << std::endl;

    // ============================================================
    // STEP 2: Allocate host (CPU) memory for input and output tensors
    // ============================================================
    // Host memory pointers (h_ prefix convention)
    float *h_A = (float*)malloc(size);        // Input tensor A on CPU
    float *h_B = (float*)malloc(size);        // Input tensor B on CPU
    float *h_C_cpu = (float*)malloc(size);     // CPU output tensor (reference)
    float *h_C_gpu = (float*)malloc(size);     // GPU output tensor (result)

    // Check for host memory allocation failure
    if (h_A == nullptr || h_B == nullptr || h_C_cpu == nullptr || h_C_gpu == nullptr) {
        std::cerr << "Error: Failed to allocate host memory" << std::endl;
        return 1;
    }

    // ============================================================
    // STEP 3: Initialize input tensors with test data
    // ============================================================
    // Initialize with sequential values for easy verification
    for (int i = 0; i < total_elements; ++i) {
        h_A[i] = (float)i;        // Tensor A: sequential values [0, 1, 2, ...]
        h_B[i] = (float)(i * 2);  // Tensor B: doubled values [0, 2, 4, ...]
        // Expected result: C = [0, 3, 6, 9, ...]
    }

    // ============================================================
    // STEP 4: Allocate device (GPU) memory using cudaMalloc
    // ============================================================
    // Device memory pointers (d_ prefix convention)
    float *d_A, *d_B, *d_C;
    
    // cudaMalloc allocates memory on the GPU device
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_A: " << cudaGetErrorString(err) << std::endl;
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }
    
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }
    
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }

    // ============================================================
    // STEP 5: Copy data from host (CPU) to device (GPU)
    // ============================================================
    // cudaMemcpy performs synchronous memory transfer
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (HostToDevice): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }
    
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (HostToDevice): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }

    // ============================================================
    // STEP 6: Configure 3D kernel launch parameters
    // ============================================================
    // Define thread block dimensions using dim3
    // 8x8x8 = 512 threads per block (within the 1024 thread limit)
    // This block size provides good balance between occupancy and flexibility
    dim3 threadsPerBlock(8, 8, 8);
    
    // Calculate number of blocks needed in each dimension
    // Formula: ceil(dimension / threadsPerDimension)
    // This ensures all tensor elements are covered, even if dimensions are not
    // evenly divisible by the thread block size
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,   // Blocks in x-dimension (width)
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,   // Blocks in y-dimension (height)
        (depth + threadsPerBlock.z - 1) / threadsPerBlock.z     // Blocks in z-dimension (depth)
    );
    
    // Example calculation for dimensions 128x128x32, threadsPerBlock 8x8x8:
    // - blocksPerGrid.x = (128 + 8 - 1) / 8 = 135 / 8 = 16 blocks
    // - blocksPerGrid.y = (128 + 8 - 1) / 8 = 135 / 8 = 16 blocks
    // - blocksPerGrid.z = (32 + 8 - 1) / 8 = 39 / 8 = 4 blocks
    // - Total blocks: 16 * 16 * 4 = 1,024 blocks
    // - Total threads: 1,024 * 512 = 524,288 threads
    
    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y << "x" << blocksPerGrid.z
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y << "x" << threadsPerBlock.z
              << " threads per block" << std::endl;
    std::cout << "Total blocks: " << blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z << std::endl;
    std::cout << "Total threads: " << blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z 
              * threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z << std::endl;

    // ============================================================
    // STEP 7: Launch 3D CUDA kernel on the GPU
    // ============================================================
    // Kernel launch syntax with 3D configuration:
    // kernelName<<<dim3(blocks), dim3(threads)>>>(args)
    //
    // The <<< >>> syntax specifies:
    // - Grid dimensions: blocksPerGrid (3D: x, y, z)
    // - Block dimensions: threadsPerBlock (3D: x, y, z)
    //
    // This launch is asynchronous - CPU continues immediately after launch
    tensorAdd3D_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, depth, height, width);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error: Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }
    
    // Wait for kernel to complete before copying results back
    // cudaDeviceSynchronize ensures all GPU operations are finished
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error: Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }

    // ============================================================
    // STEP 8: Copy results back from device (GPU) to host (CPU)
    // ============================================================
    err = cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed (DeviceToHost): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
        return 1;
    }

    // ============================================================
    // STEP 9: Run CPU version for comparison and timing
    // ============================================================
    // Measure CPU execution time for performance comparison
    auto start = std::chrono::high_resolution_clock::now();
    tensorAdd3D_cpu(h_A, h_B, h_C_cpu, depth, height, width);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ============================================================
    // STEP 10: Verify GPU results against CPU reference
    // ============================================================
    // Compare GPU results with CPU reference implementation
    // This ensures correctness of the GPU kernel
    bool success = true;
    int error_count = 0;
    const int max_errors_to_report = 10;  // Limit error reporting
    
    for (int i = 0; i < total_elements && success; ++i) {
        float diff = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > std::numeric_limits<float>::epsilon()) {
            if (error_count < max_errors_to_report) {
                std::cout << "Mismatch at index " << i << ": GPU=" << h_C_gpu[i] 
                          << ", CPU=" << h_C_cpu[i] << std::endl;
            }
            error_count++;
            success = false;
        }
    }

    if (success) {
        std::cout << "Success! GPU and CPU results match." << std::endl;
        std::cout << "CPU computation took " << cpu_time.count() << " ms" << std::endl;
        std::cout << "GPU computation completed (asynchronous, timing not measured)" << std::endl;
    } else {
        std::cerr << "Error: Found " << error_count << " mismatches between GPU and CPU results" << std::endl;
    }

    // ============================================================
    // STEP 11: Clean up allocated memory
    // ============================================================
    // Free host memory (CPU)
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    // Free device memory (GPU)
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success ? 0 : 1;
}
