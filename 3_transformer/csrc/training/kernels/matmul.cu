
#include <c10/cuda/CUDAGuard.h>
/**
 * Forward pass matrix multiplication kernel for transformer training
 * Computes C = A * B where A is M×K, B is K×N, C is M×N
 * Used in attention mechanisms and feed-forward networks
 * 
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
__global__ void matmul_fwd_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in output matrix

    // Bounds check to ensure we don't access out-of-range elements
    if (row < M && col < N) {
        float sum = 0.0f;
        // Compute dot product of row A[row,:] and column B[:,col]
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        // Store result in output matrix
        C[row * N + col] = sum;
    }
}

/**
 * Backward pass kernel for gradient computation with respect to matrix A
 * Computes grad_A = grad_C * B^T where grad_C is M×N, B is K×N, grad_A is M×K
 * Used in backpropagation during transformer training
 * 
 * @param grad_C Gradient with respect to output C (M×N, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param grad_A Gradient with respect to input A (M×K, device memory)
 * @param M Number of rows in grad_C and grad_A
 * @param N Number of columns in grad_C and B
 * @param K Number of columns in grad_A and rows in B
 */
__global__ void matmul_bwd_A_kernel(const float* grad_C, const float* B, float* grad_A,
                                   int M, int N, int K) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in grad_A
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in grad_A

    // Bounds check
    if (row < M && col < K) {
        float sum = 0.0f;
        // Compute dot product of row grad_C[row,:] and row B[col,:] (transpose)
        for (int n = 0; n < N; n++) {
            sum += grad_C[row * N + n] * B[col * N + n];
        }
        // Store gradient in grad_A
        grad_A[row * K + col] = sum;
    }
}

/**
 * Backward pass kernel for gradient computation with respect to matrix B
 * Computes grad_B = A^T * grad_C where A is M×K, grad_C is M×N, grad_B is K×N
 * Used in backpropagation during transformer training
 * 
 * @param A Input matrix A (M×K, device memory)
 * @param grad_C Gradient with respect to output C (M×N, device memory)
 * @param grad_B Gradient with respect to input B (K×N, device memory)
 * @param M Number of rows in A and grad_C
 * @param N Number of columns in grad_C and grad_B
 * @param K Number of columns in A and rows in grad_B
 */
__global__ void matmul_bwd_B_kernel(const float* A, const float* grad_C, float* grad_B,
                                   int M, int N, int K) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in grad_B
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in grad_B

    // Bounds check
    if (row < K && col < N) {
        float sum = 0.0f;
        // Compute dot product of column A[:,row] and column grad_C[:,col]
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_C[m * N + col];
        }
        // Store gradient in grad_B
        grad_B[row * N + col] = sum;
    }
}

/**
 * Batched forward pass matrix multiplication kernel for transformer training
 * Computes C[batch] = A[batch] * B[batch] for multiple batches simultaneously
 * Used for processing multiple sequences in parallel during training
 * 
 * @param A Input matrix A (batch_size × M×K, device memory)
 * @param B Input matrix B (batch_size × K×N, device memory)
 * @param C Output matrix C (batch_size × M×N, device memory)
 * @param batch_size Number of batches to process
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
__global__ void batched_matmul_fwd_kernel(const float* A, const float* B, float* C,
                                         int batch_size, int M, int N, int K) {
    // Calculate 3D coordinates from thread indices
    int batch = blockIdx.z;                              // Batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // Row index within batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index within batch

    // Bounds check for all dimensions
    if (batch < batch_size && row < M && col < N) {
        float sum = 0.0f;
        // Compute dot product for this batch
        for (int k = 0; k < K; k++) {
            int a_idx = batch * M * K + row * K + k;     // Index in A[batch]
            int b_idx = batch * K * N + k * N + col;    // Index in B[batch]
            sum += A[a_idx] * B[b_idx];
        }
        int c_idx = batch * M * N + row * N + col;      // Index in C[batch]
        C[c_idx] = sum;
    }
}

/**
 * Batched backward pass kernel for gradient computation with respect to matrix A
 * Computes grad_A[batch] = grad_C[batch] * B[batch]^T for multiple batches simultaneously
 * Used in backpropagation during transformer training with batched inputs
 * 
 * @param grad_C Gradient with respect to output C (batch_size × M×N, device memory)
 * @param B Input matrix B (batch_size × K×N, device memory)
 * @param grad_A Gradient with respect to input A (batch_size × M×K, device memory)
 * @param batch_size Number of batches to process
 * @param M Number of rows in grad_C and grad_A
 * @param N Number of columns in grad_C and B
 * @param K Number of columns in grad_A and rows in B
 */
__global__ void batched_matmul_bwd_A_kernel(const float* grad_C, const float* B, float* grad_A,
                                           int batch_size, int M, int N, int K) {
    // Calculate 3D coordinates from thread indices
    int batch = blockIdx.z;                              // Batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index within batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index within batch

    // Bounds check for all dimensions
    if (batch < batch_size && row < M && col < K) {
        float sum = 0.0f;
        // Compute dot product of row grad_C[batch][row,:] and row B[batch][col,:] (transpose)
        for (int n = 0; n < N; n++) {
            int grad_c_idx = batch * M * N + row * N + n;  // Index in grad_C[batch]
            int b_idx = batch * K * N + col * N + n;        // Index in B[batch]
            sum += grad_C[grad_c_idx] * B[b_idx];
        }
        int grad_a_idx = batch * M * K + row * K + col;    // Index in grad_A[batch]
        grad_A[grad_a_idx] = sum;
    }
}

/**
 * Batched backward pass kernel for gradient computation with respect to matrix B
 * Computes grad_B[batch] = A[batch]^T * grad_C[batch] for multiple batches simultaneously
 * Used in backpropagation during transformer training with batched inputs
 * 
 * @param A Input matrix A (batch_size × M×K, device memory)
 * @param grad_C Gradient with respect to output C (batch_size × M×N, device memory)
 * @param grad_B Gradient with respect to input B (batch_size × K×N, device memory)
 * @param batch_size Number of batches to process
 * @param M Number of rows in A and grad_C
 * @param N Number of columns in grad_C and grad_B
 * @param K Number of columns in A and rows in grad_B
 */
__global__ void batched_matmul_bwd_B_kernel(const float* A, const float* grad_C, float* grad_B,
                                           int batch_size, int M, int N, int K) {
    // Calculate 3D coordinates from thread indices
    int batch = blockIdx.z;                              // Batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index within batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index within batch

    // Bounds check for all dimensions
    if (batch < batch_size && row < K && col < N) {
        float sum = 0.0f;
        // Compute dot product of column A[batch][:,row] and column grad_C[batch][:,col]
        for (int m = 0; m < M; m++) {
            int a_idx = batch * M * K + m * K + row;         // Index in A[batch]
            int grad_c_idx = batch * M * N + m * N + col;   // Index in grad_C[batch]
            sum += A[a_idx] * grad_C[grad_c_idx];
        }
        int grad_b_idx = batch * K * N + row * N + col;     // Index in grad_B[batch]
        grad_B[grad_b_idx] = sum;
    }
}

/**
 * CUDA wrapper function for forward matrix multiplication
 * Launches the forward pass kernel with appropriate grid and block dimensions
 * 
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void matmul_fwd_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);  // 16x16 = 256 threads per block
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_fwd_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

/**
 * CUDA wrapper function for backward matrix multiplication
 * Launches both gradient computation kernels for A and B
 * 
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param grad_C Gradient with respect to output C (M×N, device memory)
 * @param grad_A Gradient with respect to input A (M×K, device memory)
 * @param grad_B Gradient with respect to input B (K×N, device memory)
 * @param M Number of rows in A and grad_C
 * @param N Number of columns in B and grad_C
 * @param K Number of columns in A and rows in B
 */
void matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                    float* grad_A, float* grad_B, int M, int N, int K) {
    // Launch gradient computation for A
    dim3 threads_A(16, 16);
    dim3 blocks_A((K + threads_A.x - 1) / threads_A.x, (M + threads_A.y - 1) / threads_A.y);
    matmul_bwd_A_kernel<<<blocks_A, threads_A>>>(grad_C, B, grad_A, M, N, K);

    // Launch gradient computation for B
    dim3 threads_B(16, 16);
    dim3 blocks_B((N + threads_B.x - 1) / threads_B.x, (K + threads_B.y - 1) / threads_B.y);
    matmul_bwd_B_kernel<<<blocks_B, threads_B>>>(A, grad_C, grad_B, M, N, K);
}

/**
 * CUDA wrapper function for batched forward matrix multiplication
 * Launches the batched forward pass kernel with 3D grid dimensions
 * 
 * @param A Input matrix A (batch_size × M×K, device memory)
 * @param B Input matrix B (batch_size × K×N, device memory)
 * @param C Output matrix C (batch_size × M×N, device memory)
 * @param batch_size Number of batches to process
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void batched_matmul_fwd_cuda(const float* A, const float* B, float* C,
                            int batch_size, int M, int N, int K) {
    dim3 threads(8, 8);  // Smaller block size for batched operations
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                batch_size);  // 3D grid for batch dimension
    batched_matmul_fwd_kernel<<<blocks, threads>>>(A, B, C, batch_size, M, N, K);
}

/**
 * CUDA wrapper function for batched backward matrix multiplication
 * Launches both batched gradient computation kernels for A and B
 * 
 * @param A Input matrix A (batch_size × M×K, device memory)
 * @param B Input matrix B (batch_size × K×N, device memory)
 * @param grad_C Gradient with respect to output C (batch_size × M×N, device memory)
 * @param grad_A Gradient with respect to input A (batch_size × M×K, device memory)
 * @param grad_B Gradient with respect to input B (batch_size × K×N, device memory)
 * @param batch_size Number of batches to process
 * @param M Number of rows in A and grad_C
 * @param N Number of columns in B and grad_C
 * @param K Number of columns in A and rows in B
 */
void batched_matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                            float* grad_A, float* grad_B,
                            int batch_size, int M, int N, int K) {
    // Launch batched gradient computation for A
    dim3 threads_A(8, 8);
    dim3 blocks_A((K + threads_A.x - 1) / threads_A.x,
                  (M + threads_A.y - 1) / threads_A.y,
                  batch_size);
    batched_matmul_bwd_A_kernel<<<blocks_A, threads_A>>>(grad_C, B, grad_A, batch_size, M, N, K);

    // Launch batched gradient computation for B
    dim3 threads_B(8, 8);
    dim3 blocks_B((N + threads_B.x - 1) / threads_B.x,
                  (K + threads_B.y - 1) / threads_B.y,
                  batch_size);
    batched_matmul_bwd_B_kernel<<<blocks_B, threads_B>>>(A, grad_C, grad_B, batch_size, M, N, K);
}
