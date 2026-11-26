#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * CUDA kernel for General Matrix-Vector multiplication (GEMV)
 * Computes: y = A * x (where A is a matrix and x is a vector)
 * Supports both single and batched operations
 * 
 * @param A Input matrix (batch×M×N or M×N, device memory)
 * @param x Input vector (batch×N or N, device memory)
 * @param y Output vector (batch×M or M, device memory)
 * @param batch Batch size (1 for non-batched, >1 for batched)
 * @param M Number of rows in matrix A
 * @param N Number of columns in matrix A and size of input vector
 */
__global__ void gemv_kernel(const float* A, const float* x, float* y, int batch, int M, int N) {
    // Calculate batch and row indices from 3D thread/block indices
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (batch_idx < batch && row < M) {
        float sum = 0.0f;
        // Get pointers to current batch's data
        const float* A_batch = A + batch_idx * M * N;
        const float* x_batch = x + batch_idx * N;

        // Compute dot product of matrix row and input vector
        for (int col = 0; col < N; col++) {
            sum += A_batch[row * N + col] * x_batch[col];
        }
        // Store result in output vector
        y[batch_idx * M + row] = sum;
    }
}

/**
 * PyTorch wrapper for GEMV forward pass
 * Supports both single and batched matrix-vector multiplication
 * 
 * @param A Input tensor A (M×N for single, batch×M×N for batched, CUDA tensor)
 * @param x Input tensor x (N for single, batch×N for batched, CUDA tensor)
 * @return Output tensor y (M for single, batch×M for batched)
 */
torch::Tensor gemv_forward(torch::Tensor A, torch::Tensor x) {
    const c10::cuda::CUDAGuard device_guard(A.device());

    // Validate input tensors
    TORCH_CHECK(A.device().type() == torch::kCUDA, "A must be a CUDA tensor");
    TORCH_CHECK(x.device().type() == torch::kCUDA, "x must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    // Handle single matrix-vector multiplication (2D matrix, 1D vector)
    if (A.dim() == 2 && x.dim() == 1) {
        TORCH_CHECK(A.size(1) == x.size(0), "Matrix-vector dimensions don't match");

        int M = A.size(0);
        int N = A.size(1);

        // Allocate output tensor
        auto y = torch::zeros({M}, torch::dtype(torch::kFloat32).device(A.device()));

        // Configure kernel launch parameters for single operation
        dim3 threadsPerBlock(1, 256, 1);
        dim3 numBlocks(1, (M + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

        // Launch CUDA kernel
        gemv_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            1, M, N
        );

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        return y;
    }
    // Handle batched matrix-vector multiplication (3D tensor batch, 2D tensor)
    else if (A.dim() == 3 && x.dim() == 2) {
        TORCH_CHECK(A.size(0) == x.size(0) && A.size(2) == x.size(1),
                   "Batch matrix-vector dimensions don't match");

        int batch = A.size(0);
        int M = A.size(1);
        int N = A.size(2);

        // Allocate output tensor
        auto y = torch::zeros({batch, M}, torch::dtype(torch::kFloat32).device(A.device()));

        // Configure kernel launch parameters for batched operation
        dim3 threadsPerBlock(1, 16, 16);
        dim3 numBlocks(1,
                      (M + threadsPerBlock.y - 1) / threadsPerBlock.y,
                      (batch + threadsPerBlock.z - 1) / threadsPerBlock.z);

        // Launch CUDA kernel
        gemv_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            batch, M, N
        );

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

        return y;
    } else {
        TORCH_CHECK(false, "Unsupported tensor dimensions for GEMV");
    }
}
