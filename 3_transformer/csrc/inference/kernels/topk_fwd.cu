#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * CUDA kernel for Top-K selection
 * Finds the K largest values and their indices in each row
 * Uses naive insertion sort algorithm (not optimized for performance)
 * 
 * @param input Input tensor (batch_size × n, device memory)
 * @param values Output tensor for top-K values (batch_size × k, device memory)
 * @param indices Output tensor for top-K indices (batch_size × k, device memory)
 * @param batch_size Number of samples in the batch
 * @param n Size of input vector for each sample
 * @param k Number of top elements to select
 */
__global__ void topk_kernel(const float* input, float* values, int* indices,
                           int batch_size, int n, int k) {
    // Each block processes one sample in the batch
    int batch_idx = blockIdx.x;

    if (batch_idx < batch_size) {
        // Get pointers to current batch's data
        const float* input_row = input + batch_idx * n;
        float* values_row = values + batch_idx * k;
        int* indices_row = indices + batch_idx * k;

        // Initialize top-K arrays with minimum values
        for (int i = 0; i < k; i++) {
            values_row[i] = -INFINITY;
            indices_row[i] = -1;
        }

        // Process each element in the input vector
        for (int i = 0; i < n; i++) {
            float val = input_row[i];

            // Find insertion position in sorted top-K array
            for (int j = 0; j < k; j++) {
                if (val > values_row[j]) {
                    // Shift elements to make room for new value
                    for (int m = k - 1; m > j; m--) {
                        values_row[m] = values_row[m - 1];
                        indices_row[m] = indices_row[m - 1];
                    }
                    // Insert new value at position j
                    values_row[j] = val;
                    indices_row[j] = i;
                    break;
                }
            }
        }
    }
}

/**
 * PyTorch wrapper for Top-K forward pass
 * Finds the K largest values and their indices in each row
 * 
 * @param input Input tensor (batch_size × n, CUDA tensor)
 * @param k Number of top elements to select
 * @return Tuple of (values, indices) tensors, both of shape (batch_size × k)
 */
std::tuple<torch::Tensor, torch::Tensor> topk_forward(torch::Tensor input, int k) {
    const c10::cuda::CUDAGuard device_guard(input.device());

    // Validate input tensor
    TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D tensor (batch, n)");
    TORCH_CHECK(k > 0 && k <= input.size(1), "k must be > 0 and <= input.size(1)");

    // Extract tensor dimensions
    int batch_size = input.size(0);
    int n = input.size(1);

    // Allocate output tensors
    auto values = torch::zeros({batch_size, k}, torch::dtype(torch::kFloat32).device(input.device()));
    auto indices = torch::zeros({batch_size, k}, torch::dtype(torch::kInt32).device(input.device()));

    // Configure kernel launch parameters
    // Each block processes one sample in the batch
    int threads_per_block = 1;
    int num_blocks = batch_size;

    // Launch CUDA kernel
    topk_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size, n, k
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}
