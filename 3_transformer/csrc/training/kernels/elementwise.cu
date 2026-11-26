/**
#include <c10/cuda/CUDAGuard.h>
 * CUDA kernels for element-wise operations with forward and backward passes
 * Used for training neural networks with automatic differentiation
 */

/**
 * Forward pass kernel for element-wise addition
 * Computes: out = a + b
 * 
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param out Output tensor (device memory)
 * @param size Number of elements
 */
__global__ void add_fwd_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

/**
 * Backward pass kernel for element-wise addition
 * Computes gradients: grad_a = grad_out, grad_b = grad_out
 * (both inputs receive the same gradient)
 * 
 * @param grad_out Gradient from output (device memory)
 * @param grad_a Gradient for input a (device memory)
 * @param grad_b Gradient for input b (device memory)
 * @param size Number of elements
 */
__global__ void add_bwd_kernel(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Both inputs receive the same gradient
        grad_a[idx] = grad_out[idx];
        grad_b[idx] = grad_out[idx];
    }
}

/**
 * Forward pass kernel for element-wise multiplication
 * Computes: out = a * b
 * 
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param out Output tensor (device memory)
 * @param size Number of elements
 */
__global__ void mul_fwd_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

/**
 * Backward pass kernel for element-wise multiplication
 * Computes gradients: grad_a = grad_out * b, grad_b = grad_out * a
 * 
 * @param grad_out Gradient from output (device memory)
 * @param a Input tensor a (needed for gradient computation, device memory)
 * @param b Input tensor b (needed for gradient computation, device memory)
 * @param grad_a Gradient for input a (device memory)
 * @param grad_b Gradient for input b (device memory)
 * @param size Number of elements
 */
__global__ void mul_bwd_kernel(const float* grad_out, const float* a, const float* b,
                              float* grad_a, float* grad_b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Product rule: d/dx (a*b) = b, d/dy (a*b) = a
        grad_a[idx] = grad_out[idx] * b[idx];
        grad_b[idx] = grad_out[idx] * a[idx];
    }
}

/**
 * CUDA wrapper for forward addition
 * Launches the forward addition kernel
 * 
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param out Output tensor (device memory)
 * @param size Number of elements
 */
void add_fwd_cuda(const float* a, const float* b, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_fwd_kernel<<<blocks, threads>>>(a, b, out, size);
}

/**
 * CUDA wrapper for backward addition
 * Launches the backward addition kernel
 * 
 * @param grad_out Gradient from output (device memory)
 * @param grad_a Gradient for input a (device memory)
 * @param grad_b Gradient for input b (device memory)
 * @param size Number of elements
 */
void add_bwd_cuda(const float* grad_out, float* grad_a, float* grad_b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_bwd_kernel<<<blocks, threads>>>(grad_out, grad_a, grad_b, size);
}

/**
 * CUDA wrapper for forward multiplication
 * Launches the forward multiplication kernel
 * 
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param out Output tensor (device memory)
 * @param size Number of elements
 */
void mul_fwd_cuda(const float* a, const float* b, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_fwd_kernel<<<blocks, threads>>>(a, b, out, size);
}

/**
 * CUDA wrapper for backward multiplication
 * Launches the backward multiplication kernel
 * 
 * @param grad_out Gradient from output (device memory)
 * @param a Input tensor a (device memory)
 * @param b Input tensor b (device memory)
 * @param grad_a Gradient for input a (device memory)
 * @param grad_b Gradient for input b (device memory)
 * @param size Number of elements
 */
void mul_bwd_cuda(const float* grad_out, const float* a, const float* b,
                  float* grad_a, float* grad_b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_bwd_kernel<<<blocks, threads>>>(grad_out, a, b, grad_a, grad_b, size);
}
