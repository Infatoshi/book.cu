
__global__ void layernorm_fwd_kernel(const float* x, const float* gamma, const float* beta,
                                   float* out, float* mean_out, float* var_out,
                                   int batch_size, int seq_len, int n_embd, float eps) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        extern __shared__ float shared_mem[];
        float* sum_vals = shared_mem;
        float* sum_sq_vals = &shared_mem[blockDim.x];

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

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            if (tid < stride) {
                sum_vals[tid] += sum_vals[tid + stride];
                sum_sq_vals[tid] += sum_sq_vals[tid + stride];
            }
        }
        __syncthreads();

        float total_sum = sum_vals[0];
        float total_sum_sq = sum_sq_vals[0];
        float mean = total_sum / n_embd;
        float var = (total_sum_sq / n_embd) - (mean * mean);

        if (tid == 0) {
            int mean_var_idx = batch * seq_len + seq;
            mean_out[mean_var_idx] = mean;
            var_out[mean_var_idx] = var;
        }

        float inv_std = rsqrtf(var + eps);
        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float normalized = (x[idx] - mean) * inv_std;
            out[idx] = normalized * gamma[i] + beta[i];
        }
    }
}

__global__ void layernorm_bwd_kernel(const float* grad_out, const float* x,
                                   const float* gamma, const float* mean, const float* var,
                                   float* grad_x, float* grad_gamma, float* grad_beta,
                                   int batch_size, int seq_len, int n_embd, float eps) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;

    if (batch < batch_size && seq < seq_len) {
        extern __shared__ float shared_mem[];
        float* local_sums = shared_mem;  

        int mean_var_idx = batch * seq_len + seq;
        float mean_val = mean[mean_var_idx];
        float var_val = var[mean_var_idx];
        float inv_std = rsqrtf(var_val + eps);

        if (tid < 2) {
            local_sums[tid] = 0.0f;
        }
        __syncthreads();

        float local_sum_grad_gamma = 0.0f;
        float local_sum_grad_gamma_x_norm = 0.0f;

        for (int i = tid; i < n_embd; i += blockDim.x) {
            int idx = batch * seq_len * n_embd + seq * n_embd + i;
            float normalized = (x[idx] - mean_val) * inv_std;

            local_sum_grad_gamma += grad_out[idx] * gamma[i];
            local_sum_grad_gamma_x_norm += grad_out[idx] * gamma[i] * normalized;
        }

        atomicAdd(&local_sums[0], local_sum_grad_gamma);
        atomicAdd(&local_sums[1], local_sum_grad_gamma_x_norm);
        __syncthreads();

        float mean_grad_gamma = local_sums[0] / n_embd;
        float mean_grad_gamma_x_norm = local_sums[1] / n_embd;


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

__global__ void layernorm_bwd_params_kernel(const float* grad_out, const float* x,
                                          const float* mean, const float* var,
                                          float* grad_gamma, float* grad_beta,
                                          int batch_size, int seq_len, int n_embd, float eps) {
    int batch = blockIdx.x;
    int seq = blockIdx.y;
    int emb = threadIdx.x;

    if (batch < batch_size && seq < seq_len && emb < n_embd) {
        int mean_var_idx = batch * seq_len + seq;
        int tensor_idx = batch * seq_len * n_embd + seq * n_embd + emb;

        float mean_val = mean[mean_var_idx];
        float var_val = var[mean_var_idx];
        float inv_std = rsqrtf(var_val + eps);

        float normalized = (x[tensor_idx] - mean_val) * inv_std;
        atomicAdd(&grad_gamma[emb], grad_out[tensor_idx] * normalized);
        atomicAdd(&grad_beta[emb], grad_out[tensor_idx]);
    }
}

void layernorm_fwd_cuda(const float* x, const float* gamma, const float* beta,
                       float* out, float* mean_out, float* var_out,
                       int batch_size, int seq_len, int n_embd, float eps) {
    dim3 blocks(batch_size, seq_len);
    int threads = min(256, n_embd);
    size_t shared_mem = 2 * threads * sizeof(float);
    layernorm_fwd_kernel<<<blocks, threads, shared_mem>>>(x, gamma, beta, out, mean_out, var_out,
                                                         batch_size, seq_len, n_embd, eps);
}

void layernorm_bwd_cuda(const float* grad_out, const float* x, const float* gamma,
                       const float* mean, const float* var, float* grad_x,
                       float* grad_gamma, float* grad_beta,
                       int batch_size, int seq_len, int n_embd, float eps) {
    cudaMemset(grad_gamma, 0, n_embd * sizeof(float));
    cudaMemset(grad_beta, 0, n_embd * sizeof(float));

    dim3 blocks_main(batch_size, seq_len);
    int threads_main = min(256, n_embd);
    size_t shared_mem_main = 2 * sizeof(float);  
    layernorm_bwd_kernel<<<blocks_main, threads_main, shared_mem_main>>>(grad_out, x, gamma, mean, var,
                                                                         grad_x, grad_gamma, grad_beta,
                                                                         batch_size, seq_len, n_embd, eps);

    dim3 blocks_params(batch_size, seq_len);
    int threads_params = min(256, n_embd);
    layernorm_bwd_params_kernel<<<blocks_params, threads_params>>>(grad_out, x, mean, var,
                                                                   grad_gamma, grad_beta,
                                                                   batch_size, seq_len, n_embd, eps);
}
