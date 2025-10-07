
__global__ void gelu_fwd_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float arg = val * 0.7071067811865476f;  
        float erf_val = 1.0f - erfc(arg);
        out[idx] = 0.5f * val * (1.0f + erf_val);
    }
}

__global__ void gelu_bwd_kernel(const float* grad_out, const float* x, float* grad_x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float arg = val * 0.7071067811865476f;  
        float erf_val = 1.0f - erfc(arg);

        float d_erf = (2.0f / sqrtf(M_PI)) * expf(-arg * arg) * (1.0f / sqrtf(2.0f));

        grad_x[idx] = grad_out[idx] * (0.5f * (1.0f + erf_val) + 0.5f * val * d_erf);
    }
}

void gelu_fwd_cuda(const float* x, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_fwd_kernel<<<blocks, threads>>>(x, out, size);
}

void gelu_bwd_cuda(const float* grad_out, const float* x, float* grad_x, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_bwd_kernel<<<blocks, threads>>>(grad_out, x, grad_x, size);
}
