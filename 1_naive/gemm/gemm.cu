#include "common.h"

/**
 * @file gemm.cu
 * @brief Naive CUDA implementation of General Matrix Multiply (GEMM)
 * 
 * This file demonstrates matrix multiplication, the most computationally intensive operation
 * in deep learning. GEMM is the core operation in:
 * - Linear/fully-connected layers: Y = XW + b
 * - Attention mechanisms: QK^T, attention scores computation
 * - Convolution layers: can be implemented as GEMM via im2col
 * 
 * Matrix Multiplication: C[M×N] = A[M×K] * B[K×N]
 * Formula: C[i,j] = sum(k=0 to K-1) A[i,k] * B[k,j]
 * 
 * CPU Implementation: Triple nested loops, O(M*N*K) time complexity
 * GPU Implementation: Each thread computes one output element, parallelizes over M×N
 * 
 * Memory Access Pattern:
 * - CPU: Sequential access to A rows, strided access to B columns (poor cache utilization)
 * - GPU: Each thread reads one row of A (coalesced) and one column of B (strided, inefficient)
 * 
 * Performance Characteristics:
 * - CPU: O(M*N*K) time, poor cache locality for matrix B
 * - GPU: Computation-bound operation (many multiply-adds per memory access)
 * - Naive implementation: ~10-50 GFLOPS
 * - Optimized (with tiling, shared memory): ~1000-5000 GFLOPS
 * 
 * FLOPs Count: 2 * M * N * K (one multiply + one add per element)
 */

/**
 * CPU implementation of General Matrix Multiply (GEMM)
 * 
 * Computes C = A * B where A is M×K, B is K×N, C is M×N.
 * This is the standard triple-nested loop matrix multiplication algorithm.
 * 
 * Algorithm:
 * - For each output element C[row][column]:
 *   1. Initialize sum = 0
 *   2. For k from 0 to K-1:
 *      - sum += A[row][k] * B[k][column]
 *   3. C[row][column] = sum
 * 
 * Memory Access Pattern:
 * - A: Row-major access (cache-friendly, sequential)
 * - B: Column-major access (cache-unfriendly, strided)
 * - C: Row-major access (cache-friendly, sequential)
 * 
 * Time Complexity: O(M * N * K)
 * Space Complexity: O(1) excluding input/output matrices
 * 
 * @param A Input matrix A (host memory, row-major order, size M × K)
 * @param B Input matrix B (host memory, row-major order, size K × N)
 * @param C Output matrix C (host memory, row-major order, size M × N)
 *          Stores matrix product: C[i,j] = sum(k) A[i,k] * B[k,j]
 * @param M_rows Number of rows in A and C (M dimension, must be >= 0)
 * @param N_cols Number of columns in B and C (N dimension, must be >= 0)
 * @param K_shared_dim Number of columns in A and rows in B (K dimension, must be >= 0)
 *                     This is the "shared" dimension that gets reduced/summed over
 */
void gemm_cpu(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    // Outer loop: iterate through each row of output matrix C
    // Each row corresponds to one row of matrix A
    for (int row = 0; row < M_rows; ++row) {
        // Middle loop: iterate through each column of output matrix C
        // Each column corresponds to one column of matrix B
        for (int column = 0; column < N_cols; ++column) {
            float sum = 0.0f;  // Accumulator for dot product
            
            // Inner loop: compute dot product of row A[row,:] and column B[:,column]
            // This is the reduction operation: sum over the shared dimension K
            for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
                // Access A[row][k_idx]: row-major order, sequential access
                // Access B[k_idx][column]: row-major order, strided access (K elements apart)
                // Strided access to B is cache-unfriendly on CPU
                sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
            }
            // Store computed dot product in output matrix
            C[row * N_cols + column] = sum;
        }
    }
}

/**
 * CUDA kernel for General Matrix Multiply (GEMM)
 * 
 * Parallel GPU implementation where each thread computes one output element.
 * This naive implementation demonstrates the basic parallelization strategy.
 * 
 * Algorithm (same as CPU):
 * - Each thread computes C[row][column] = dot product of A[row,:] and B[:,column]
 * - Thread with 2D index (row, column) computes one output element
 * 
 * Thread Indexing (2D Grid):
 * - Grid dimensions: (blocksPerGrid.x, blocksPerGrid.y) covering M×N output elements
 * - Block dimensions: (blockDim.x, blockDim.y)
 * - Row index: row = blockIdx.y * blockDim.y + threadIdx.y
 * - Column index: column = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * Memory Access Pattern:
 * - A: Each thread reads one row sequentially (coalesced access)
 * - B: Each thread reads one column with stride K (strided access, inefficient)
 * - C: Each thread writes one element (coalesced access)
 * 
 * Performance Issues in Naive Implementation:
 * - No shared memory: each thread loads entire row/column from global memory
 * - Strided access to B: poor memory coalescing
 * - Redundant memory loads: threads in same block load same elements multiple times
 * - No tiling: cannot reuse loaded data
 * 
 * Optimization Opportunities:
 * - Use shared memory to cache tiles of A and B
 * - Tiling: process matrices in blocks to improve data reuse
 * - Loop unrolling: reduce loop overhead
 * - Vectorization: process multiple elements per thread
 * 
 * @param A Input matrix A (device memory, row-major order, size M × K)
 * @param B Input matrix B (device memory, row-major order, size K × N)
 * @param C Output matrix C (device memory, row-major order, size M × N)
 *          Stores matrix product: C[i,j] = sum(k) A[i,k] * B[k,j]
 * @param M_rows Number of rows in A and C (M dimension, must be >= 0)
 * @param N_cols Number of columns in B and C (N dimension, must be >= 0)
 * @param K_shared_dim Number of columns in A and rows in B (K dimension, must be >= 0)
 *                     This is the "shared" dimension that gets reduced/summed over
 */
__global__ void gemm_kernel(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
    // Calculate 2D coordinates from thread indices
    // Y-dimension (rows): maps to blockIdx.y and threadIdx.y
    // Formula: row = block_index_y * threads_per_block_y + thread_index_y
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix C
    
    // X-dimension (columns): maps to blockIdx.x and threadIdx.x
    // Formula: column = block_index_x * threads_per_block_x + thread_index_x
    int column = blockIdx.x * blockDim.x + threadIdx.x; // Column index in output matrix C

    // Bounds check: critical for correctness when dimensions are not divisible by block size
    // Checks both row and column bounds independently
    // Without this check, threads beyond matrix boundaries would access invalid memory
    if (row < M_rows && column < N_cols) {
        float sum = 0.0f;  // Accumulator for dot product
        
        // Compute dot product: sum over the shared dimension K
        // This is the reduction operation that computes one output element
        for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
            // Access A[row][k_idx]: row-major order, sequential access within row
            // Memory access pattern: coalesced (threads in same row access consecutive elements)
            // Access B[k_idx][column]: row-major order, strided access (K elements apart)
            // Memory access pattern: uncoalesced (threads access distant elements)
            // This strided access is the main performance bottleneck in naive GEMM
            sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + column];
        }
        // Store computed dot product in output matrix
        // Memory access: coalesced (threads write consecutive elements)
        C[row * N_cols + column] = sum;
    }
}

/**
 * Main function demonstrating GEMM on both CPU and GPU
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
    // Matrix dimensions: C[M×N] = A[M×K] * B[K×N]
    // M=512, N=512, K=256 results in:
    // - A: 512×256 = 131,072 elements (~512 KB)
    // - B: 256×512 = 131,072 elements (~512 KB)
    // - C: 512×512 = 262,144 elements (~1 MB)
    // - Total operations: 2 * M * N * K = 134,217,728 FLOPs (~134 MFLOPS)
    const int M_rows = 512, N_cols = 512, K_shared_dim = 256;
    const int size_A = M_rows * K_shared_dim;  // Size of matrix A
    const int size_B = K_shared_dim * N_cols;    // Size of matrix B
    const int size_C = M_rows * N_cols;         // Size of matrix C

    std::cout << "GEMM: C[" << M_rows << "x" << N_cols << "] = A[" << M_rows << "x" << K_shared_dim
              << "] * B[" << K_shared_dim << "x" << N_cols << "]" << std::endl;
    std::cout << "Total operations: " << (long long)M_rows * N_cols * K_shared_dim * 2 << " FLOPs" << std::endl;

    // Allocate host (CPU) memory for matrices
    // h_ prefix denotes host memory pointers
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocate_host(&h_A, size_A);      // Input matrix A (M×K)
    allocate_host(&h_B, size_B);      // Input matrix B (K×N)
    allocate_host(&h_C_cpu, size_C);   // CPU output (for verification)
    allocate_host(&h_C_gpu, size_C);   // GPU output (for comparison)

    // Initialize input matrices with test data
    // Pattern chosen for easy verification and debugging
    for (int i = 0; i < size_A; ++i) {
        h_A[i] = static_cast<float>(i % 10);  // Matrix A: values 0-9 repeating
    }
    for (int i = 0; i < size_B; ++i) {
        h_B[i] = static_cast<float>((i * 2) % 10);  // Matrix B: values 0,2,4,6,8 repeating
    }

    // Run CPU version with timing
    // Timer uses RAII pattern: timing starts at construction, ends at destruction
    {
        Timer cpu_timer("CPU GEMM");
        gemm_cpu(h_A, h_B, h_C_cpu, M_rows, N_cols, K_shared_dim);
    } // Timer prints elapsed time here

    // Allocate device (GPU) memory
    // d_ prefix denotes device memory pointers
    // GPU memory allocation is separate from host memory
    float *d_A, *d_B, *d_C;
    allocate_device(&d_A, size_A);  // Input matrix A on GPU
    allocate_device(&d_B, size_B);  // Input matrix B on GPU
    allocate_device(&d_C, size_C);  // Output matrix C on GPU

    // Copy data from host to device
    // This is a synchronous operation (blocks until copy completes)
    // GPU kernels require data to be in device memory
    copy_to_device(d_A, h_A, size_A);
    copy_to_device(d_B, h_B, size_B);

    // Configure 2D kernel launch parameters
    // threadsPerBlock: 2D block dimensions (16x16 = 256 threads per block)
    // 16x16 is a good default: balances occupancy and warp efficiency
    // Each warp has 32 threads, so 16x16 = 256 = 8 warps per block
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    
    // blocksPerGrid: 2D grid dimensions covering output matrix C
    // Formula: ceil(N_cols / threadsPerBlock.x) for x-dimension
    //          ceil(M_rows / threadsPerBlock.y) for y-dimension
    // Ceiling division ensures all output elements are covered
    dim3 blocksPerGrid(
        (N_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Blocks in x-dimension (columns)
        (M_rows + threadsPerBlock.y - 1) / threadsPerBlock.y   // Blocks in y-dimension (rows)
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
        Timer gpu_timer("GPU GEMM");
        gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);
        CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel completion
    } // Timer prints elapsed time here

    // Copy result back from device to host
    // Synchronous operation: blocks until copy completes
    copy_to_host(h_C_gpu, d_C, size_C);

    // Verify GPU results against CPU results
    // Uses floating-point tolerance comparison (default tolerance: 1e-5)
    // This ensures numerical differences don't cause false failures
    if (verify_results(h_C_gpu, h_C_cpu, size_C)) {
        std::cout << "✓ GEMM results match!" << std::endl;

        // Display sample results for verification
        std::cout << "\nSample results (first 3x3 of output matrix):" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << h_C_gpu[i * N_cols + j] << " ";
            }
            std::cout << std::endl;
        }
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
