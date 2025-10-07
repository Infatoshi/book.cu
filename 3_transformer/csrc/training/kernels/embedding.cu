
__global__ void embedding_fwd_kernel(const float* weight, const int* indices,
                                   float* out, int num_indices, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * n_embd;

    if (idx < total_elements) {
        int token_idx = idx / n_embd;
        int emb_idx = idx % n_embd;
        int weight_idx = indices[token_idx] * n_embd + emb_idx;
        out[idx] = weight[weight_idx];
    }
}

__global__ void embedding_bwd_kernel(const float* grad_out, const int* indices,
                                   float* grad_weight, int num_indices, int n_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_indices * n_embd;

    if (idx < total_elements) {
        int token_idx = idx / n_embd;
        int emb_idx = idx % n_embd;
        int weight_idx = indices[token_idx] * n_embd + emb_idx;
        atomicAdd(&grad_weight[weight_idx], grad_out[idx]);
    }
}

void embedding_fwd_cuda(const float* weight, const int* indices, float* out,
                       int num_indices, int n_embd) {
    int total_elements = num_indices * n_embd;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    embedding_fwd_kernel<<<blocks, threads>>>(weight, indices, out, num_indices, n_embd);
}

void embedding_bwd_cuda(const float* grad_out, const int* indices, float* grad_weight,
                       int num_indices, int n_embd) {
    int total_elements = num_indices * n_embd;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    embedding_bwd_kernel<<<blocks, threads>>>(grad_out, indices, grad_weight, num_indices, n_embd);
}
