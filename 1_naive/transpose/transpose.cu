#include "common.h"

/**
 * @file transpose.cu
 * @brief Naive CUDA implementation of matrix transpose
 * 
 * This file demonstrates data movement patterns and memory access optimization challenges.
 * Matrix transpose is fundamental to many operations including:
 * - Attention mechanisms: QK^T requires transposing K
 * - Preparing data for GEMM operations
 * - Data layout transformations
 * 
 * Matrix Transpose: B[j,i] = A[i,j] for all i,j
 * Input: A[num_rows × num_cols], Output: B[num_cols × num_rows]
 * 
 * CPU Implementation: Sequential nested loops, swaps row/column indices
 * GPU Implementation: Each thread moves one element, parallelizes over input matrix
 * 
 * Memory Access Pattern:
 * - CPU: Sequential read from input, strided write to output (poor cache utilization)
 * - GPU: Coalesced read from input, strided write to output (major performance bottleneck)
 * 
 * Performance Characteristics:
 * - CPU: O(M*N) time complexity, sequential processing
 * - GPU: Memory bandwidth limited (one read + one write per element)
 * - Naive implementation suffers from uncoalesced writes
 * - Optimized version uses shared memory to enable coalesced writes
 */

/**
 * CPU implementation of matrix transpose
 * 
 * Sequential transpose operation performed on the CPU.
 * This serves as a reference implementation for correctness verification.
 * 
 * Algorithm:
 * - Iterate through each element of the input matrix
 * - For element at position (row, column):
 *   - Read from input at index (row, column)
 *   - Write to output at index (column, row)
 * - Effectively swaps rows and columns
 * 
 * Memory Access Pattern:
 * - Input: Sequential access (row-major, cache-friendly)
 * - Output: Strided access (column-major pattern, cache-unfriendly)
 * 
 * Time Complexity: O(M * N) where M = num_rows, N = num_cols
 * Space Complexity: O(1) excluding input/output matrices
 * 
 * @param in Input matrix (host memory, row-major order, size num_rows × num_cols)
 * @param out Output transposed matrix (host memory, row-major order, size num_cols × num_rows)
 *            Stores transposed result: out[j][i] = in[i][j]
 * @param num_rows Number of rows in input matrix (M dimension, must be >= 0)
 * @param num_cols Number of columns in input matrix (N dimension, must be >= 0)
 */
void transpose_cpu(const float* in, float* out, int num_rows, int num_cols) {
    // Outer loop: iterate through each row of input matrix
    // Sequential access to input matrix (cache-friendly)
    for (int row = 0; row < num_rows; ++row) {
        // Inner loop: iterate through each column of input matrix
        // Sequential access within row maximizes cache hit rate
        for (int column = 0; column < num_cols; ++column) {
            // Transpose operation: swap row and column indices
            // Input index: (row, column) -> row * num_cols + column (row-major)
            // Output index: (column, row) -> column * num_rows + row (row-major)
            // This writes in column-major pattern (strided access, cache-unfriendly)
            out[column * num_rows + row] = in[row * num_cols + column];
        }
    }
}

/**
 * CUDA kernel for matrix transpose
 * 
 * Parallel GPU implementation where each thread moves one element.
 * This naive implementation demonstrates the memory access challenge.
 * 
 * Algorithm (same as CPU):
 * - Each thread processes one input element
 * - Thread at position (row, column) copies in[row][column] to out[column][row]
 * 
 * Thread Indexing (2D Grid):
 * - Grid dimensions: (blocksPerGrid.x, blocksPerGrid.y) covering input matrix
 * - Block dimensions: (blockDim.x, blockDim.y)
 * - Column index: column = blockIdx.x * blockDim.x + threadIdx.x
 * - Row index: row = blockIdx.y * blockDim.y + threadIdx.y
 * 
 * Memory Access Pattern:
 * - Input: Coalesced read (threads in same row read consecutive elements)
 * - Output: Strided write (threads write to distant locations, uncoalesced)
 * 
 * Performance Issues in Naive Implementation:
 * - Uncoalesced writes: threads write with stride num_rows
 * - Poor memory bandwidth utilization for output
 * - No shared memory: cannot batch writes for coalescing
 * 
 * Optimization Opportunities:
 * - Use shared memory: load input tile to shared memory, transpose in shared memory, write coalesced
 * - Block transpose: transpose tiles of the matrix rather than individual elements
 * - Padding: add padding to reduce bank conflicts
 * 
 * @param in Input matrix (device memory, row-major order, size num_rows × num_cols)
 * @param out Output transposed matrix (device memory, row-major order, size num_cols × num_rows)
 *            Stores transposed result: out[j][i] = in[i][j]
 * @param num_rows Number of rows in input matrix (M dimension, must be >= 0)
 * @param num_cols Number of columns in input matrix (N dimension, must be >= 0)
 */
__global__ void transpose_kernel(const float* in, float* out, int num_rows, int num_cols) {
    // Calculate 2D coordinates from thread indices
    // X-dimension (columns): maps to blockIdx.x and threadIdx.x
    // Formula: column = block_index_x * threads_per_block_x + thread_index_x
    int column = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in input matrix
    
    // Y-dimension (rows): maps to blockIdx.y and threadIdx.y
    // Formula: row = block_index_y * threads_per_block_y + thread_index_y
    int row = blockIdx.y * blockDim.y + threadIdx.y;      // Row index in input matrix

    // Bounds check: critical for correctness when dimensions are not divisible by block size
    // Checks both row and column bounds independently
    // Without this check, threads beyond matrix boundaries would access invalid memory
    if (row < num_rows && column < num_cols) {
        // Convert 2D coordinates to 1D index for input matrix (row-major order)
        // Formula: index = row * num_cols + column
        // Memory access: coalesced (threads in same row access consecutive elements)
        int in_index = row * num_cols + column;
        
        // Convert 2D coordinates to 1D index for output matrix (transposed, row-major order)
        // Formula: index = column * num_rows + row (swapped indices)
        // Memory access: uncoalesced (threads write with stride num_rows)
        // This is the main performance bottleneck: strided writes prevent memory coalescing
        int out_index = column * num_rows + row;
        
        // Transpose: swap row and column indices
        // Read from input (coalesced) and write to output (uncoalesced)
        out[out_index] = in[in_index];
    }
}

/**
 * Main function demonstrating matrix transpose on both CPU and GPU
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

    std::cout << "Matrix Transpose: " << num_rows << "x" << num_cols << " -> "
              << num_cols << "x" << num_rows << std::endl;

    // Allocate host (CPU) memory for input and output matrices
    // h_ prefix denotes host memory pointers
    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, n);      // Input matrix (num_rows × num_cols)
    allocate_host(&h_out_cpu, n); // CPU output (for verification)
    allocate_host(&h_out_gpu, n); // GPU output (for comparison)

    // Initialize input matrix with test data
    // Pattern: element at (row, col) = row * num_cols + col
    // This makes verification easy: transpose should swap row and column indices
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            h_in[i * num_cols + j] = static_cast<float>(i * num_cols + j);
        }
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        Timer cpu_timer("CPU Matrix Transpose");
        transpose_cpu(h_in, h_out_cpu, num_rows, num_cols);
    } // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_in, *d_out;
    allocate_device(&d_in, n);  // Input matrix on GPU
    allocate_device(&d_out, n); // Output matrix on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_in, h_in, n);

    // Configure 2D kernel launch parameters
    // threadsPerBlock: 2D block dimensions (16x16 = 256 threads per block)
    // 16x16 is a good default: balances occupancy and warp efficiency
    // Each warp has 32 threads, so 16x16 = 256 = 8 warps per block
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    
    // blocksPerGrid: 2D grid dimensions covering input matrix
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
        Timer gpu_timer("GPU Matrix Transpose");
        transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
        CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel completion
    } // Timer prints elapsed time here

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_out_gpu, d_out, n);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_out_gpu, h_out_cpu, n)) {
        std::cout << "✓ Matrix transpose results match!" << std::endl;

        // Display sample results for verification
        // Shows original and transposed matrices side-by-side
        std::cout << "\nOriginal matrix (first 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_in[i * num_cols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nTransposed matrix (first 3x3):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_out_gpu[i * num_rows + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "✗ Error: GPU and CPU results do not match!" << std::endl;
    }

    // Clean up memory
    // Free all allocated memory to prevent leaks
    // Important: free device memory before host memory (good practice)
    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
