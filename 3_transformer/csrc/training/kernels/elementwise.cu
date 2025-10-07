
__global__ void add_fwd_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void add_bwd_kernel(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] = grad_out[idx];
        grad_b[idx] = grad_out[idx];
    }
}

__global__ void mul_fwd_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_bwd_kernel(const float* grad_out, const float* a, const float* b,
                              float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] = grad_out[idx] * b[idx];
        grad_b[idx] = grad_out[idx] * a[idx];
    }
}

void add_fwd_cuda(const float* a, const float* b, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_fwd_kernel<<<blocks, threads>>>(a, b, out, size);
}

void add_bwd_cuda(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bwd_kernel<<<blocks, threads>>>(grad_out, grad_a, grad_b, size);
}

void mul_fwd_cuda(const float* a, const float* b, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_fwd_kernel<<<blocks, threads>>>(a, b, out, size);
}

void mul_bwd_cuda(const float* grad_out, const float* a, const float* b,
                  float* grad_a, float* grad_b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_bwd_kernel<<<blocks, threads>>>(grad_out, a, b, grad_a, grad_b, size);
}
