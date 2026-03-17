#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * CUDA kernel for matrix multiplication: C = A * B
 * Each thread computes one element of the output matrix
 * 
 * @param A Input matrix A (M×K, device memory)
 * @param B Input matrix B (K×N, device memory)
 * @param C Output matrix C (M×N, device memory)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;     // Column index in output matrix

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
 * PyTorch wrapper for matrix multiplication forward pass
 * Performs matrix multiplication: C = A * B
 * 
 * @param A Input tensor A (M×K, CUDA tensor)
 * @param B Input tensor B (K×N, CUDA tensor)
 * @return Output tensor C (M×N, CUDA tensor)
 */
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    const c10::cuda::CUDAGuard device_guard(A.device());

    // Validate input tensors
    TORCH_CHECK(A.device().type() == torch::kCUDA, "A must be a CUDA tensor");
    TORCH_CHECK(B.device().type() == torch::kCUDA, "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Both tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions don't match");

    // Extract matrix dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    // Configure kernel launch parameters
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}
