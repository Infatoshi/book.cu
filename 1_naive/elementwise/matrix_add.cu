#include "common.h"

/**
 * @file matrix_add.cu
 * @brief Naive CUDA implementation of 2D matrix addition (element-wise addition)
 * 
 * This file demonstrates 2D thread indexing for matrix operations.
 * Matrix addition is fundamental to many neural network operations including:
 * - Batch normalization: adding learned bias terms
 * - Residual connections: adding skip connections in ResNet/Transformer architectures
 * - Layer aggregation: combining outputs from multiple branches
 * 
 * Algorithm: For matrices A, B, and C of size M×N: C[i,j] = A[i,j] + B[i,j] for all i,j
 * 
 * CPU Implementation: Nested loops iterating through rows and columns sequentially
 * GPU Implementation: 2D thread grid where each thread processes one matrix element
 * 
 * Memory Layout:
 * - Matrices stored in row-major order (C-style)
 * - Element at position (row, col) is at index: row * num_cols + col
 * 
 * Memory Access Pattern:
 * - CPU: Row-major access (cache-friendly, sequential within rows)
 * - GPU: 2D grid ensures coalesced access along columns within each warp
 * 
 * Performance Characteristics:
 * - CPU: O(M*N) time complexity, sequential processing
 * - GPU: O(M*N/p) where p is number of threads, highly parallelizable
 * - Memory bandwidth limited operation (computation is trivial)
 */

/**
 * CPU implementation of matrix addition
 * 
 * Sequential element-wise addition of two matrices performed on the CPU.
 * This serves as a reference implementation for correctness verification.
 * 
 * Algorithm:
 * - Iterate through each row from 0 to num_rows-1
 * - For each row, iterate through each column from 0 to num_cols-1
 * - Compute C[row][col] = A[row][col] + B[row][col]
 * - Memory access is row-major (cache-friendly sequential access)
 * 
 * Time Complexity: O(M * N) where M = num_rows, N = num_cols
 * Space Complexity: O(1) excluding input/output matrices
 * 
 * @param A Input matrix A (host memory, row-major order, size num_rows × num_cols)
 * @param B Input matrix B (host memory, row-major order, size num_rows × num_cols)
 * @param C Output matrix C (host memory, row-major order, size num_rows × num_cols)
 *          Stores element-wise sum: C[i,j] = A[i,j] + B[i,j]
 * @param num_rows Number of rows in all matrices (M dimension, must be >= 0)
 * @param num_cols Number of columns in all matrices (N dimension, must be >= 0)
 */
void matrix_add_cpu(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    // Outer loop: iterate through rows
    // Row-major order ensures sequential memory access within each row
    for (int row = 0; row < num_rows; ++row) {
        // Inner loop: iterate through columns
        // Sequential access within row maximizes cache hit rate
        for (int col = 0; col < num_cols; ++col) {
            // Convert 2D coordinates (row, col) to 1D index (row-major order)
            // Formula: index = row * num_cols + col
            // This maps 2D matrix coordinates to linear array index
            int index = row * num_cols + col;
            // Perform element-wise addition: C[index] = A[index] + B[index]
            C[index] = A[index] + B[index];
        }
    }
}

/**
 * CUDA kernel for matrix addition
 * 
 * Parallel GPU implementation using 2D thread indexing to map threads to matrix elements.
 * This demonstrates the standard pattern for 2D array operations in CUDA.
 * 
 * Algorithm (same as CPU):
 * - Each thread computes one output element
 * - Thread at position (row, col) computes C[row][col] = A[row][col] + B[row][col]
 * 
 * Thread Indexing (2D Grid):
 * - Grid dimensions: (blocksPerGrid.x, blocksPerGrid.y)
 * - Block dimensions: (blockDim.x, blockDim.y)
 * - Column index: column = blockIdx.x * blockDim.x + threadIdx.x
 * - Row index: row = blockIdx.y * blockDim.y + threadIdx.y
 * 
 * Memory Access Pattern:
 * - Threads in the same warp (32 consecutive threads) access consecutive columns
 * - This enables memory coalescing: multiple memory accesses are combined
 * - Optimal bandwidth utilization when threadsPerBlock.x is multiple of 32
 * 
 * Performance:
 * - Highly parallelizable: each element independent
 * - Memory bandwidth limited (very simple computation)
 * - No shared memory needed (direct global memory access is efficient)
 * - 2D grid better matches 2D matrix structure than 1D grid
 * 
 * @param A Input matrix A (device memory, row-major order, size num_rows × num_cols)
 * @param B Input matrix B (device memory, row-major order, size num_rows × num_cols)
 * @param C Output matrix C (device memory, row-major order, size num_rows × num_cols)
 *          Stores element-wise sum: C[i,j] = A[i,j] + B[i,j]
 * @param num_rows Number of rows in all matrices (M dimension, must be >= 0)
 * @param num_cols Number of columns in all matrices (N dimension, must be >= 0)
 */
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int num_rows, int num_cols) {
    // Calculate 2D coordinates from thread indices
    // X-dimension (columns): maps to blockIdx.x and threadIdx.x
    // Formula: column = block_index_x * threads_per_block_x + thread_index_x
    int column = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in matrix
    
    // Y-dimension (rows): maps to blockIdx.y and threadIdx.y
    // Formula: row = block_index_y * threads_per_block_y + thread_index_y
    int row = blockIdx.y * blockDim.y + threadIdx.y;      // Row index in matrix

    // Bounds check: critical for correctness when dimensions are not divisible by block size
    // Checks both row and column bounds independently
    // Without this check, threads beyond matrix boundaries would access invalid memory
    if (row < num_rows && column < num_cols) {
        // Convert 2D coordinates (row, column) to 1D index (row-major order)
        // Formula: index = row * num_cols + column
        // This maps 2D matrix coordinates to linear array index
        int index = row * num_cols + column;
        // Perform element-wise addition: C[index] = A[index] + B[index]
        // Each thread performs independent computation
        // Memory accesses are coalesced (consecutive threads in warp access consecutive columns)
        C[index] = A[index] + B[index];
    }
}

/**
 * Main function demonstrating matrix addition on both CPU and GPU
 * 
 * This function:
 * 1. Allocates memory for input and output matrices
 * 2. Initializes test data
 * 3. Runs CPU reference implementation with timing
 * 4. Runs GPU kernel implementation with timing
 * 5. Verifies correctness by comparing CPU and GPU results
 * 6. Cleans up allocated memory
 */
int main() {
    // Matrix dimensions: 1024x1024 = 1,048,576 elements
    // Each matrix is ~4 MB (assuming float = 4 bytes)
    // This size is large enough to demonstrate GPU parallelism benefits
    const int num_rows = 1024;
    const int num_cols = 1024;
    const int n = num_rows * num_cols;  // Total number of elements

    std::cout << "Matrix Addition: " << num_rows << "x" << num_cols << " = " << n << " elements" << std::endl;

    // Allocate host (CPU) memory for input and output matrices
    // h_ prefix denotes host memory pointers
    // All matrices have same dimensions: num_rows × num_cols
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, n);      // Input matrix A
    allocate_host(&h_B, n);      // Input matrix B
    allocate_host(&h_C_cpu, n);  // CPU output (for verification)
    allocate_host(&h_C_gpu, n);  // GPU output (for comparison)

    // Initialize input matrices with test data
    // Pattern chosen for easy verification: C[i] = A[i] + B[i] = i + 2*i = 3*i
    // Matrices stored in row-major order: element at (row, col) is at index row*num_cols + col
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);        // Matrix A: sequential values [0, 1, 2, ..., n-1]
        h_B[i] = static_cast<float>(i * 2);    // Matrix B: doubled values [0, 2, 4, ..., 2*(n-1)]
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        Timer cpu_timer("CPU Matrix Addition");
        matrix_add_cpu(h_A, h_B, h_C_cpu, num_rows, num_cols);
    } // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, n);  // Input matrix A on GPU
    allocate_device(&d_B, n);  // Input matrix B on GPU
    allocate_device(&d_C, n);  // Output matrix C on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_A, h_A, n);
    copy_to_device(d_B, h_B, n);

    // Configure 2D kernel launch parameters
    // threadsPerBlock: 2D block dimensions (16x16 = 256 threads per block)
    // 16x16 is a good default: balances occupancy and warp efficiency
    // Each warp has 32 threads, so 16x16 = 256 = 8 warps per block
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    
    // blocksPerGrid: 2D grid dimensions
    // Formula: ceil(num_cols / threadsPerBlock.x) for x-dimension
    //          ceil(num_rows / threadsPerBlock.y) for y-dimension
    // Ceiling division ensures all matrix elements are covered
    dim3 blocksPerGrid(
        (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Blocks in x-dimension (columns)
        (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y   // Blocks in y-dimension (rows)
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;
    std::cout << "Total threads: " << blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y << std::endl;

    // Run GPU version with timing
    // Kernel launch syntax: kernel_name<<<grid_size, block_size>>>(parameters)
    // dim3 types allow 2D/3D grid and block configurations
    // Kernel launch is asynchronous (returns immediately)
    // cudaDeviceSynchronize() waits for kernel completion
    {
        Timer gpu_timer("GPU Matrix Addition");
        matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_rows, num_cols);
        CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel completion
    } // Timer prints elapsed time here

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_C_gpu, d_C, n);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_C_gpu, h_C_cpu, n)) {
        std::cout << "✓ Matrix addition results match!" << std::endl;
    } else {
        std::cerr << "✗ Error: GPU and CPU results do not match!" << std::endl;
    }

    // Clean up memory
    // Free all allocated memory to prevent leaks
    // Important: free device memory before host memory (good practice)
    free_host(h_A);
    free_host(h_B);
    free_host(h_C_cpu);
    free_host(h_C_gpu);
    free_device(d_A);
    free_device(d_B);
    free_device(d_C);

    return 0;
}
