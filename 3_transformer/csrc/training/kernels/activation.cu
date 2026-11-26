#include <math.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * CUDA kernel for GELU (Gaussian Error Linear Unit) forward pass
 * GELU is commonly used in transformer architectures (e.g., GPT, BERT)
 * 
 * GELU formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 
 * This implementation uses erfc (complementary error function) for numerical stability:
 * - Computes erf(x / sqrt(2)) = 1 - erfc(x / sqrt(2))
 * - Uses sqrt(2) ≈ 0.7071067811865476
 * 
 * @param x Input tensor (device memory)
 * @param out Output tensor (device memory)
 * @param size Number of elements
 */
__global__ void gelu_fwd_kernel(const float* x, float* out, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Compute erf(x / sqrt(2)) using erfc for numerical stability
        float arg = val * 0.7071067811865476f;  // x / sqrt(2)
        float erf_val = 1.0f - erfc(arg);
        // Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        out[idx] = 0.5f * val * (1.0f + erf_val);
    }
}

/**
 * CUDA kernel for GELU backward pass
 * Computes gradient of GELU with respect to input
 * 
 * Gradient formula: dGELU/dx = 0.5 * (1 + erf(x/√2)) + 0.5 * x * d(erf(x/√2))/dx
 * where d(erf(x/√2))/dx = (2/√π) * exp(-(x/√2)^2) * (1/√2)
 * 
 * @param grad_out Gradient with respect to output (device memory)
 * @param x Input tensor from forward pass (device memory)
 * @param grad_x Gradient with respect to input (device memory)
 * @param size Number of elements
 */
__global__ void gelu_bwd_kernel(const float* grad_out, const float* x, float* grad_x, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Compute erf(x / sqrt(2))
        float arg = val * 0.7071067811865476f;  // x / sqrt(2)
        float erf_val = 1.0f - erfc(arg);

        // Compute derivative of erf(x/√2) with respect to x
        // d(erf(x/√2))/dx = (2/√π) * exp(-(x/√2)^2) * (1/√2)
        float d_erf = (2.0f / sqrtf(M_PI)) * expf(-arg * arg) * (1.0f / sqrtf(2.0f));

        // Compute gradient: grad_x = grad_out * dGELU/dx
        grad_x[idx] = grad_out[idx] * (0.5f * (1.0f + erf_val) + 0.5f * val * d_erf);
    }
}

/**
 * CUDA wrapper function for GELU forward pass
 * Launches the forward kernel with appropriate grid and block dimensions
 * 
 * @param x Input tensor (device memory)
 * @param out Output tensor (device memory)
 * @param size Number of elements
 */
void gelu_fwd_cuda(const float* x, float* out, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_fwd_kernel<<<blocks, threads>>>(x, out, size);
}

/**
 * CUDA wrapper function for GELU backward pass
 * Launches the backward kernel with appropriate grid and block dimensions
 * 
 * @param grad_out Gradient with respect to output (device memory)
 * @param x Input tensor from forward pass (device memory)
 * @param grad_x Gradient with respect to input (device memory)
 * @param size Number of elements
 */
void gelu_bwd_cuda(const float* grad_out, const float* x, float* grad_x, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_bwd_kernel<<<blocks, threads>>>(grad_out, x, grad_x, size);
}
