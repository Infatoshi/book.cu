#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <device_launch_parameters.h>

/**
 * CUDA kernel for Layer Normalization forward pass (training)
 * Uses shared memory for efficient parallel reduction across hidden dimension
 * 
 * Algorithm:
 * 1. Each thread computes partial sums over hidden dimension (stride = blockDim.x)
 * 2. Use shared memory reduction to compute mean and variance
 * 3. Normalize: normalized = (x - mean) / sqrt(var + eps)
 * 4. Scale and shift: y = normalized * gamma + beta
 * 
 * This implementation uses block-level parallelism with shared memory reductions
 * for better performance compared to sequential processing.
 * 
 * @param x Input tensor (batch_size × seq_len × n_embd, device memory)
 * @param gamma Scale parameter (n_embd, device memory)
 * @param beta Shift parameter (n_embd, device memory)
 * @param out Output tensor (batch_size × seq_len × n_embd, device memory)
 * @param mean_out Output mean values (batch_size × seq_len, device memory)
 * @param var_out Output variance values (batch_size × seq_len, device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension (embedding size)
 * @param eps Small epsilon to prevent division by zero
 */
__global__ void layernorm_fwd_kernel(const float* x, const float* gamma, const float* beta,
                                   float* out, float* mean_out, float* var_out,
                                   int batch_size, int seq_len, int n_embd, float eps) {
    // Each block processes one (batch, sequence) position
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        // Allocate shared memory for reduction
        extern __shared__ float shared_mem[];
        float* sum_vals = shared_mem;           // For sum(x)
        float* sum_sq_vals = &shared_mem[blockDim.x];  // For sum(x²)

        // Step 1: Each thread computes partial sums over hidden dimension
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float val = x[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
        sum_vals[tid] = local_sum;
        sum_sq_vals[tid] = local_sum_sq;

        // Step 2: Parallel reduction in shared memory to compute total sums
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                sum_vals[tid] += sum_vals[tid + stride];
                sum_sq_vals[tid] += sum_sq_vals[tid + stride];
            }
        }
        __syncthreads();

        // Step 3: Compute mean and variance (only thread 0 needed, but all threads wait)
        float total_sum = sum_vals[0];
        float total_sum_sq = sum_sq_vals[0];
        float mean = total_sum / n_embd;
        float var = (total_sum_sq / n_embd) - (mean * mean);  // Var = E[x²] - E[x]²

        // Save mean and variance for backward pass
        if (tid == 0) {
            int mean_var_idx = batch * seq_len + seq;
            mean_out[mean_var_idx] = mean;
            var_out[mean_var_idx] = var;
        }

        // Step 4: Normalize and apply affine transformation
        float inv_std = rsqrtf(var + eps);  // 1 / sqrt(var + eps)
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float normalized = (x[idx] - mean) * inv_std;
            out[idx] = normalized * gamma[i] + beta[i];
        }
    }
}

/**
 * CUDA kernel for Layer Normalization backward pass (training)
 * Computes gradients with respect to input x, gamma, and beta
 * 
 * Uses shared memory for efficient reduction of intermediate values needed for
 * gradient computation. The gradient formulas are:
 * - grad_gamma = sum(grad_out * normalized)
 * - grad_beta = sum(grad_out)
 * - grad_x = inv_std * (grad_out * gamma - mean(grad_out * gamma) - normalized * mean(grad_out * gamma * normalized))
 * 
 * @param grad_out Gradient with respect to output (batch_size × seq_len × n_embd, device memory)
 * @param x Input tensor from forward pass (batch_size × seq_len × n_embd, device memory)
 * @param gamma Scale parameter (n_embd, device memory)
 * @param mean Mean values from forward pass (batch_size × seq_len, device memory)
 * @param var Variance values from forward pass (batch_size × seq_len, device memory)
 * @param grad_x Gradient with respect to input (batch_size × seq_len × n_embd, device memory)
 * @param grad_gamma Gradient with respect to gamma (n_embd, device memory)
 * @param grad_beta Gradient with respect to beta (n_embd, device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension (embedding size)
 * @param eps Small epsilon (must match forward pass)
 */
__global__ void layernorm_bwd_kernel(const float* grad_out, const float* x,
                                   const float* gamma, const float* mean, const float* var,
                                   float* grad_x, float* grad_gamma, float* grad_beta,
                                   int batch_size, int seq_len, int n_embd, float eps) {
    // Each block processes one (batch, sequence) position
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        // Allocate shared memory for reduction
        extern __shared__ float shared_mem[];
        float* local_sums = shared_mem;  // [sum(grad_out * gamma), sum(grad_out * gamma * normalized)]

        // Get mean and variance from forward pass
        int mean_var_idx = batch * seq_len + seq;
        float mean_val = mean[mean_var_idx];
        float var_val = var[mean_var_idx];
        float inv_std = rsqrtf(var_val + eps);

        // Initialize shared memory
        if (tid < 2) {
            local_sums[tid] = 0.0f;
        }
        __syncthreads();

        // Step 1: Compute partial sums for gradient computation
        float local_sum_grad_gamma = 0.0f;           // sum(grad_out * gamma)
        float local_sum_grad_gamma_x_norm = 0.0f;   // sum(grad_out * gamma * normalized)

        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float normalized = (x[idx] - mean_val) * inv_std;

            local_sum_grad_gamma += grad_out[idx] * gamma[i];
            local_sum_grad_gamma_x_norm += grad_out[idx] * gamma[i] * normalized;
        }

        // Accumulate partial sums using atomic operations
        atomicAdd(&local_sums[0], local_sum_grad_gamma);
        atomicAdd(&local_sums[1], local_sum_grad_gamma_x_norm);
        __syncthreads();

        // Compute mean values needed for gradient computation
        float mean_grad_gamma = local_sums[0] / n_embd;
        float mean_grad_gamma_x_norm = local_sums[1] / n_embd;

        // Step 2: Compute gradient with respect to input x
        // Formula: grad_x = inv_std * (grad_out * gamma - mean(grad_out * gamma) - normalized * mean(grad_out * gamma * normalized))
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float normalized = (x[idx] - mean_val) * inv_std;

            grad_x[idx] = inv_std * (
                grad_out[idx] * gamma[i] -
                mean_grad_gamma -
                normalized * mean_grad_gamma_x_norm
            );
        }
    }
}

/**
 * CUDA kernel for Layer Normalization backward pass - parameter gradients
 * Computes gradients with respect to gamma and beta using atomic operations
 * 
 * This kernel is called separately to accumulate gradients over all (batch, sequence) positions
 * 
 * @param grad_out Gradient with respect to output (batch_size × seq_len × n_embd, device memory)
 * @param x Input tensor from forward pass (batch_size × seq_len × n_embd, device memory)
 * @param mean Mean values from forward pass (batch_size × seq_len, device memory)
 * @param var Variance values from forward pass (batch_size × seq_len, device memory)
 * @param grad_gamma Gradient with respect to gamma (n_embd, device memory) - accumulates
 * @param grad_beta Gradient with respect to beta (n_embd, device memory) - accumulates
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension (embedding size)
 * @param eps Small epsilon (must match forward pass)
 */
__global__ void layernorm_bwd_params_kernel(const float* grad_out, const float* x,
                                          const float* mean, const float* var,
                                          float* grad_gamma, float* grad_beta,
                                          int batch_size, int seq_len, int n_embd, float eps) {
    // Each thread processes one embedding dimension across all (batch, sequence) positions
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int emb = threadIdx.x;

    if (batch < batch_size && seq < seq_len && emb < n_embd) {
        int mean_var_idx = batch * seq_len + seq;
        int tensor_idx = batch * seq_len * n_embd + seq * n_embd + emb;

        // Get mean and variance from forward pass
        float mean_val = mean[mean_var_idx];
        float var_val = var[mean_var_idx];
        float inv_std = rsqrtf(var_val + eps);

        // Compute normalized value
        float normalized = (x[tensor_idx] - mean_val) * inv_std;

        // Accumulate gradients using atomic operations
        // grad_gamma[emb] += grad_out * normalized (accumulated over all batch/seq positions)
        // grad_beta[emb] += grad_out (accumulated over all batch/seq positions)
        atomicAdd(&grad_gamma[emb], grad_out[tensor_idx] * normalized);
        atomicAdd(&grad_beta[emb], grad_out[tensor_idx]);
    }
}

/**
 * CUDA wrapper function for Layer Normalization forward pass (training)
 * Launches the forward kernel with shared memory allocation
 * 
 * @param x Input tensor (device memory)
 * @param gamma Scale parameter (device memory)
 * @param beta Shift parameter (device memory)
 * @param out Output tensor (device memory)
 * @param mean_out Output mean values (device memory)
 * @param var_out Output variance values (device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension
 * @param eps Small epsilon
 */
void layernorm_fwd_cuda(const float* x, const float* gamma, const float* beta,
                       float* out, float* mean_out, float* var_out,
                       int batch_size, int seq_len, int n_embd, float eps) {
    dim3 blocks(batch_size, seq_len);
    int threads = min(256, n_embd);
    size_t shared_mem = 2 * threads * sizeof(float);  // For sum_vals and sum_sq_vals
    layernorm_fwd_kernel<<<blocks, threads, shared_mem>>>(x, gamma, beta, out, mean_out, var_out,
                                                         batch_size, seq_len, n_embd, eps);
}

/**
 * CUDA wrapper function for Layer Normalization backward pass (training)
 * Launches two kernels: one for input gradients, one for parameter gradients
 * 
 * @param grad_out Gradient with respect to output (device memory)
 * @param x Input tensor from forward pass (device memory)
 * @param gamma Scale parameter (device memory)
 * @param mean Mean values from forward pass (device memory)
 * @param var Variance values from forward pass (device memory)
 * @param grad_x Gradient with respect to input (device memory)
 * @param grad_gamma Gradient with respect to gamma (device memory) - must be zero-initialized
 * @param grad_beta Gradient with respect to beta (device memory) - must be zero-initialized
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension
 * @param eps Small epsilon
 */
void layernorm_bwd_cuda(const float* grad_out, const float* x, const float* gamma,
                       const float* mean, const float* var, float* grad_x,
                       float* grad_gamma, float* grad_beta,
                       int batch_size, int seq_len, int n_embd, float eps) {
    // Zero-initialize parameter gradients
    cudaMemset(grad_gamma, 0, n_embd * sizeof(float));
    cudaMemset(grad_beta, 0, n_embd * sizeof(float));

    // Launch kernel for input gradient computation
    dim3 blocks_main(batch_size, seq_len);
    int threads_main = min(256, n_embd);
    size_t shared_mem_main = 2 * sizeof(float);  // For local_sums[0] and local_sums[1]
    layernorm_bwd_kernel<<<blocks_main, threads_main, shared_mem_main>>>(grad_out, x, gamma, mean, var,
                                                                         grad_x, grad_gamma, grad_beta,
                                                                         batch_size, seq_len, n_embd, eps);

    // Launch kernel for parameter gradient accumulation
    dim3 blocks_params(batch_size, seq_len);
    int threads_params = min(256, n_embd);
    layernorm_bwd_params_kernel<<<blocks_params, threads_params>>>(grad_out, x, mean, var,
                                                                   grad_gamma, grad_beta,
                                                                   batch_size, seq_len, n_embd, eps);
}
