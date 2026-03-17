#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <device_launch_parameters.h>
#include <math.h>

/**
 * CUDA kernel for Softmax forward pass (training)
 * Uses shared memory for efficient parallel reduction across hidden dimension
 * 
 * Algorithm (numerically stable):
 * 1. Find maximum value across hidden dimension (shared memory reduction)
 * 2. Compute exp(x - max) for each element
 * 3. Compute sum of exponentials (shared memory reduction)
 * 4. Normalize: softmax = exp(x - max) / sum(exp(x - max))
 * 
 * This implementation uses block-level parallelism with shared memory reductions
 * for better performance compared to sequential processing.
 * 
 * @param x Input logits (batch_size × seq_len × n_embd, device memory)
 * @param out Output probabilities (batch_size × seq_len × n_embd, device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension (embedding size)
 */
__global__ void softmax_fwd_kernel(const float* x, float* out, int batch_size, int seq_len, int n_embd) {
    // Each block processes one (batch, sequence) position
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        // Allocate shared memory for reduction
        extern __shared__ float shared_mem[];
        float* max_val = shared_mem;        // For max reduction
        float* sum_val = &shared_mem[blockDim.x];  // For sum reduction

        // Step 1: Find maximum value across hidden dimension
        float local_max = -INFINITY;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            local_max = fmaxf(local_max, x[idx]);
        }
        max_val[tid] = local_max;

        // Parallel reduction to find global maximum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                max_val[tid] = fmaxf(max_val[tid], max_val[tid + stride]);
            }
        }
        __syncthreads();
        float global_max = max_val[0];

        // Step 2 & 3: Compute exp(x - max) and sum
        float local_sum = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float exp_val = expf(x[idx] - global_max);  // Numerically stable
            out[idx] = exp_val;  // Store exp values temporarily
            local_sum += exp_val;
        }
        sum_val[tid] = local_sum;

        // Parallel reduction to compute total sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                sum_val[tid] += sum_val[tid + stride];
            }
        }
        __syncthreads();
        float global_sum = sum_val[0];

        // Step 4: Normalize to get final softmax probabilities
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            out[idx] /= global_sum;
        }
    }
}

/**
 * CUDA kernel for Softmax backward pass (training)
 * Computes gradient with respect to input logits
 * 
 * Gradient formula for softmax:
 * - Let s = softmax(x), then grad_x = s * (grad_out - sum(grad_out * s))
 * 
 * The key insight is that the gradient of softmax depends on:
 * 1. The output probabilities (softmax values)
 * 2. The dot product of grad_out and softmax output
 * 
 * This implementation uses shared memory for efficient reduction of the dot product.
 * 
 * @param grad_out Gradient with respect to output probabilities (batch_size × seq_len × n_embd, device memory)
 * @param out Softmax output probabilities from forward pass (batch_size × seq_len × n_embd, device memory)
 * @param grad_x Gradient with respect to input logits (batch_size × seq_len × n_embd, device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension (embedding size)
 */
__global__ void softmax_bwd_kernel(const float* grad_out, const float* out,
                                 float* grad_x, int batch_size, int seq_len, int n_embd) {
    // Each block processes one (batch, sequence) position
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        // Allocate shared memory for reduction
        extern __shared__ float shared_dot[];

        // Step 1: Compute dot product sum(grad_out * out) across hidden dimension
        float local_dot = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            local_dot += grad_out[idx] * out[idx];
        }
        shared_dot[tid] = local_dot;

        // Parallel reduction to compute total dot product
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                shared_dot[tid] += shared_dot[tid + stride];
            }
        }
        __syncthreads();
        float sum_grad_out_softmax = shared_dot[0];

        // Step 2: Compute gradient: grad_x = out * (grad_out - sum(grad_out * out))
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            grad_x[idx] = out[idx] * (grad_out[idx] - sum_grad_out_softmax);
        }
    }
}

/**
 * CUDA wrapper function for Softmax forward pass (training)
 * Launches the forward kernel with shared memory allocation
 * 
 * @param x Input logits (device memory)
 * @param out Output probabilities (device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension
 */
void softmax_fwd_cuda(const float* x, float* out, int batch_size, int seq_len, int n_embd) {
    dim3 blocks(batch_size, seq_len);
    int threads = min(256, n_embd);
    size_t shared_mem = 2 * threads * sizeof(float);  // For max_val and sum_val
    softmax_fwd_kernel<<<blocks, threads, shared_mem>>>(x, out, batch_size, seq_len, n_embd);
}

/**
 * CUDA wrapper function for Softmax backward pass (training)
 * Launches the backward kernel with shared memory allocation
 * 
 * @param grad_out Gradient with respect to output (device memory)
 * @param out Softmax output probabilities from forward pass (device memory)
 * @param grad_x Gradient with respect to input logits (device memory)
 * @param batch_size Batch dimension
 * @param seq_len Sequence length dimension
 * @param n_embd Hidden dimension
 */
void softmax_bwd_cuda(const float* grad_out, const float* out, float* grad_x,
                     int batch_size, int seq_len, int n_embd) {
    dim3 blocks(batch_size, seq_len);
    int threads = min(256, n_embd);
    size_t shared_mem = threads * sizeof(float);  // For shared_dot
    softmax_bwd_kernel<<<blocks, threads, shared_mem>>>(grad_out, out, grad_x, batch_size, seq_len, n_embd);
}
