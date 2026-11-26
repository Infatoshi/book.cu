#include <torch/extension.h>

/**
 * PyTorch C++ Extension Bindings for Transformer Training Kernels
 *
 * This file provides Python bindings for CUDA kernels used in transformer training.
 * It includes forward and backward pass implementations for:
 * - Element-wise operations (add, multiply)
 * - Activation functions (GELU)
 * - Matrix operations (matmul, batched matmul)
 * - Normalization (softmax, layer normalization)
 * - Embedding layers
 * 
 * All functions perform input validation and convert PyTorch tensors to CUDA pointers
 * before calling the underlying CUDA kernels.
 */

#include <torch/extension.h>

// Forward declarations of CUDA kernel wrapper functions
// These are implemented in separate .cu files

void add_fwd_cuda(const float* a, const float* b, float* out, int size);
void add_bwd_cuda(const float* grad_out, float* grad_a, float* grad_b, int size);
void mul_fwd_cuda(const float* a, const float* b, float* out, int size);
void mul_bwd_cuda(const float* grad_out, const float* a, const float* b,
                  float* grad_a, float* grad_b, int size);
void gelu_fwd_cuda(const float* x, float* out, int size);
void gelu_bwd_cuda(const float* grad_out, const float* x, float* grad_x, int size);
void matmul_fwd_cuda(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                    float* grad_A, float* grad_B, int M, int N, int K);
void batched_matmul_fwd_cuda(const float* A, const float* B, float* C,
                            int batch_size, int M, int N, int K);
void batched_matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                            float* grad_A, float* grad_B,
                            int batch_size, int M, int N, int K);
void softmax_fwd_cuda(const float* x, float* out, int batch_size, int seq_len, int n_embd);
void softmax_bwd_cuda(const float* grad_out, const float* out, float* grad_x,
                     int batch_size, int seq_len, int n_embd);
void layernorm_fwd_cuda(const float* x, const float* gamma, const float* beta,
                       float* out, float* mean_out, float* var_out,
                       int batch_size, int seq_len, int n_embd, float eps);
void layernorm_bwd_cuda(const float* grad_out, const float* x, const float* gamma,
                       const float* mean, const float* var, float* grad_x,
                       float* grad_gamma, float* grad_beta,
                       int batch_size, int seq_len, int n_embd, float eps);
void embedding_fwd_cuda(const float* weight, const int* indices, float* out,
                       int num_indices, int n_embd);
void embedding_bwd_cuda(const float* grad_out, const int* indices, float* grad_weight,
                       int num_indices, int n_embd);

/**
 * PyTorch binding for element-wise addition forward pass
 * Computes: out = a + b
 * 
 * @param a First input tensor (CUDA tensor)
 * @param b Second input tensor (CUDA tensor)
 * @param out Output tensor (CUDA tensor)
 */
void add_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    // Validate input tensors
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    // Call CUDA kernel
    add_fwd_cuda(a.data_ptr<float>(), b.data_ptr<float>(),
                out.data_ptr<float>(), a.numel());
}

/**
 * PyTorch binding for element-wise addition backward pass
 * Computes gradients: grad_a = grad_out, grad_b = grad_out
 * 
 * @param grad_out Gradient with respect to output (CUDA tensor)
 * @param grad_a Gradient with respect to first input (CUDA tensor)
 * @param grad_b Gradient with respect to second input (CUDA tensor)
 */
void add_bwd(torch::Tensor grad_out, torch::Tensor grad_a, torch::Tensor grad_b) {
    // Validate input tensors
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(grad_out.numel() == grad_a.numel() && grad_out.numel() == grad_b.numel(),
                "tensor sizes must match");

    // Call CUDA kernel
    add_bwd_cuda(grad_out.data_ptr<float>(), grad_a.data_ptr<float>(),
                grad_b.data_ptr<float>(), grad_out.numel());
}

/**
 * PyTorch binding for element-wise multiplication forward pass
 * Computes: out = a * b
 * 
 * @param a First input tensor (CUDA tensor)
 * @param b Second input tensor (CUDA tensor)
 * @param out Output tensor (CUDA tensor)
 */
void mul_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    // Validate input tensors
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    // Call CUDA kernel
    mul_fwd_cuda(a.data_ptr<float>(), b.data_ptr<float>(),
                out.data_ptr<float>(), a.numel());
}

/**
 * PyTorch binding for element-wise multiplication backward pass
 * Computes gradients: grad_a = grad_out * b, grad_b = grad_out * a
 * 
 * @param grad_out Gradient with respect to output (CUDA tensor)
 * @param a First input tensor from forward pass (CUDA tensor)
 * @param b Second input tensor from forward pass (CUDA tensor)
 * @param grad_a Gradient with respect to first input (CUDA tensor)
 * @param grad_b Gradient with respect to second input (CUDA tensor)
 */
void mul_bwd(torch::Tensor grad_out, torch::Tensor a, torch::Tensor b,
            torch::Tensor grad_a, torch::Tensor grad_b) {
    // Validate input tensors
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == grad_out.numel() &&
                a.numel() == grad_a.numel() && a.numel() == grad_b.numel(),
                "tensor sizes must match");

    // Call CUDA kernel
    mul_bwd_cuda(grad_out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
                grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), a.numel());
}

/**
 * PyTorch binding for GELU (Gaussian Error Linear Unit) forward pass
 * Computes: out = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 
 * @param x Input tensor (CUDA tensor)
 * @param out Output tensor (CUDA tensor)
 */
void gelu_fwd(torch::Tensor x, torch::Tensor out) {
    // Validate input tensors
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.numel() == out.numel(), "tensor sizes must match");

    // Call CUDA kernel
    gelu_fwd_cuda(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
}

/**
 * PyTorch binding for GELU backward pass
 * Computes gradient with respect to input
 * 
 * @param grad_out Gradient with respect to output (CUDA tensor)
 * @param x Input tensor from forward pass (CUDA tensor)
 * @param grad_x Gradient with respect to input (CUDA tensor)
 */
void gelu_bwd(torch::Tensor grad_out, torch::Tensor x, torch::Tensor grad_x) {
    // Validate input tensors
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_out.numel() == x.numel() && x.numel() == grad_x.numel(),
                "tensor sizes must match");

    // Call CUDA kernel
    gelu_bwd_cuda(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                 grad_x.data_ptr<float>(), x.numel());
}

/**
 * PyTorch binding for matrix multiplication forward pass
 * Computes: C = A @ B where A is M×K, B is K×N, C is M×N
 * 
 * @param A Input matrix A (M×K, CUDA tensor)
 * @param B Input matrix B (K×N, CUDA tensor)
 * @param C Output matrix C (M×N, CUDA tensor)
 */
void matmul_fwd(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // Validate input tensors
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0) && A.size(0) == C.size(0) && B.size(1) == C.size(1),
                "matrix dimensions must be compatible");

    // Extract matrix dimensions
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    // Call CUDA kernel
    matmul_fwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                   M, N, K);
}

/**
 * PyTorch binding for matrix multiplication backward pass
 * Computes gradients: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
 * 
 * @param A Input matrix A from forward pass (M×K, CUDA tensor)
 * @param B Input matrix B from forward pass (K×N, CUDA tensor)
 * @param grad_C Gradient with respect to output C (M×N, CUDA tensor)
 * @param grad_A Gradient with respect to input A (M×K, CUDA tensor)
 * @param grad_B Gradient with respect to input B (K×N, CUDA tensor)
 */
void matmul_bwd(torch::Tensor A, torch::Tensor B, torch::Tensor grad_C,
               torch::Tensor grad_A, torch::Tensor grad_B) {
    // Validate input tensors
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(grad_C.device().is_cuda(), "grad_C must be a CUDA tensor");
    TORCH_CHECK(grad_A.device().is_cuda(), "grad_A must be a CUDA tensor");
    TORCH_CHECK(grad_B.device().is_cuda(), "grad_B must be a CUDA tensor");

    // Extract matrix dimensions
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    // Call CUDA kernel
    matmul_bwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), grad_C.data_ptr<float>(),
                   grad_A.data_ptr<float>(), grad_B.data_ptr<float>(), M, N, K);
}

/**
 * PyTorch binding for batched matrix multiplication forward pass
 * Computes: C[batch] = A[batch] @ B[batch] for all batches
 * 
 * @param A Input matrix A (batch_size × M×K, CUDA tensor)
 * @param B Input matrix B (batch_size × K×N, CUDA tensor)
 * @param C Output matrix C (batch_size × M×N, CUDA tensor)
 */
void batched_matmul_fwd(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // Validate input tensors
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3, "tensors must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == C.size(0), "batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1) && A.size(1) == C.size(1) && B.size(2) == C.size(2),
                "matrix dimensions must be compatible");

    // Extract dimensions
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    // Call CUDA kernel
    batched_matmul_fwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                           batch_size, M, N, K);
}

/**
 * PyTorch binding for batched matrix multiplication backward pass
 * Computes gradients for each batch separately
 * 
 * @param A Input matrix A from forward pass (batch_size × M×K, CUDA tensor)
 * @param B Input matrix B from forward pass (batch_size × K×N, CUDA tensor)
 * @param grad_C Gradient with respect to output C (batch_size × M×N, CUDA tensor)
 * @param grad_A Gradient with respect to input A (batch_size × M×K, CUDA tensor)
 * @param grad_B Gradient with respect to input B (batch_size × K×N, CUDA tensor)
 */
void batched_matmul_bwd(torch::Tensor A, torch::Tensor B, torch::Tensor grad_C,
                       torch::Tensor grad_A, torch::Tensor grad_B) {
    // Validate input tensors
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(grad_C.device().is_cuda(), "grad_C must be a CUDA tensor");
    TORCH_CHECK(grad_A.device().is_cuda(), "grad_A must be a CUDA tensor");
    TORCH_CHECK(grad_B.device().is_cuda(), "grad_B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && grad_C.dim() == 3 &&
                grad_A.dim() == 3 && grad_B.dim() == 3, "tensors must be 3D");

    // Extract dimensions
    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    // Call CUDA kernel
    batched_matmul_bwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), grad_C.data_ptr<float>(),
                           grad_A.data_ptr<float>(), grad_B.data_ptr<float>(),
                           batch_size, M, N, K);
}

/**
 * PyTorch binding for Softmax forward pass
 * Applies softmax activation across the last dimension (n_embd)
 * 
 * @param x Input logits (batch_size × seq_len × n_embd, CUDA tensor)
 * @param out Output probabilities (batch_size × seq_len × n_embd, CUDA tensor)
 */
void softmax_fwd(torch::Tensor x, torch::Tensor out) {
    // Validate input tensors
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3 && out.dim() == 3, "tensors must be 3D");
    TORCH_CHECK(x.sizes() == out.sizes(), "tensor sizes must match");

    // Extract dimensions
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embd = x.size(2);

    // Call CUDA kernel
    softmax_fwd_cuda(x.data_ptr<float>(), out.data_ptr<float>(),
                    batch_size, seq_len, n_embd);
}

/**
 * PyTorch binding for Softmax backward pass
 * Computes gradient with respect to input logits
 * 
 * @param grad_out Gradient with respect to output (batch_size × seq_len × n_embd, CUDA tensor)
 * @param out Softmax output probabilities from forward pass (batch_size × seq_len × n_embd, CUDA tensor)
 * @param grad_x Gradient with respect to input logits (batch_size × seq_len × n_embd, CUDA tensor)
 */
void softmax_bwd(torch::Tensor grad_out, torch::Tensor out, torch::Tensor grad_x) {
    // Validate input tensors
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_out.dim() == 3 && out.dim() == 3 && grad_x.dim() == 3,
                "tensors must be 3D");
    TORCH_CHECK(grad_out.sizes() == out.sizes() && out.sizes() == grad_x.sizes(),
                "tensor sizes must match");

    // Extract dimensions
    int batch_size = grad_out.size(0);
    int seq_len = grad_out.size(1);
    int n_embd = grad_out.size(2);

    // Call CUDA kernel
    softmax_bwd_cuda(grad_out.data_ptr<float>(), out.data_ptr<float>(),
                    grad_x.data_ptr<float>(), batch_size, seq_len, n_embd);
}

/**
 * PyTorch binding for Layer Normalization forward pass
 * Normalizes input across the last dimension (n_embd) and applies affine transformation
 * 
 * @param x Input tensor (batch_size × seq_len × n_embd, CUDA tensor)
 * @param gamma Scale parameter (n_embd, CUDA tensor)
 * @param beta Shift parameter (n_embd, CUDA tensor)
 * @param out Output tensor (batch_size × seq_len × n_embd, CUDA tensor)
 * @param mean_out Output mean values (batch_size × seq_len, CUDA tensor) - saved for backward pass
 * @param var_out Output variance values (batch_size × seq_len, CUDA tensor) - saved for backward pass
 * @param eps Small epsilon to prevent division by zero
 */
void layernorm_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                  torch::Tensor out, torch::Tensor mean_out, torch::Tensor var_out, float eps) {
    // Validate input tensors
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.device().is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(mean_out.device().is_cuda(), "mean_out must be a CUDA tensor");
    TORCH_CHECK(var_out.device().is_cuda(), "var_out must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3 && out.dim() == 3, "x and out must be 3D");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma and beta must be 1D");
    TORCH_CHECK(gamma.size(0) == x.size(2) && beta.size(0) == x.size(2),
                "gamma and beta size must match embedding dimension");

    // Extract dimensions
    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embd = x.size(2);

    // Call CUDA kernel
    layernorm_fwd_cuda(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                      out.data_ptr<float>(), mean_out.data_ptr<float>(), var_out.data_ptr<float>(),
                      batch_size, seq_len, n_embd, eps);
}

/**
 * PyTorch binding for Layer Normalization backward pass
 * Computes gradients with respect to input x, gamma, and beta
 * 
 * @param grad_out Gradient with respect to output (batch_size × seq_len × n_embd, CUDA tensor)
 * @param x Input tensor from forward pass (batch_size × seq_len × n_embd, CUDA tensor)
 * @param gamma Scale parameter (n_embd, CUDA tensor)
 * @param mean Mean values from forward pass (batch_size × seq_len, CUDA tensor)
 * @param var Variance values from forward pass (batch_size × seq_len, CUDA tensor)
 * @param grad_x Gradient with respect to input (batch_size × seq_len × n_embd, CUDA tensor)
 * @param grad_gamma Gradient with respect to gamma (n_embd, CUDA tensor)
 * @param grad_beta Gradient with respect to beta (n_embd, CUDA tensor)
 * @param eps Small epsilon (must match forward pass)
 */
void layernorm_bwd(torch::Tensor grad_out, torch::Tensor x, torch::Tensor gamma,
                  torch::Tensor mean, torch::Tensor var, torch::Tensor grad_x,
                  torch::Tensor grad_gamma, torch::Tensor grad_beta, float eps) {
    // Validate input tensors
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(mean.device().is_cuda(), "mean must be a CUDA tensor");
    TORCH_CHECK(var.device().is_cuda(), "var must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_gamma.device().is_cuda(), "grad_gamma must be a CUDA tensor");
    TORCH_CHECK(grad_beta.device().is_cuda(), "grad_beta must be a CUDA tensor");

    // Extract dimensions
    int batch_size = grad_out.size(0);
    int seq_len = grad_out.size(1);
    int n_embd = grad_out.size(2);

    // Call CUDA kernel
    layernorm_bwd_cuda(grad_out.data_ptr<float>(), x.data_ptr<float>(), gamma.data_ptr<float>(),
                      mean.data_ptr<float>(), var.data_ptr<float>(), grad_x.data_ptr<float>(),
                      grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
                      batch_size, seq_len, n_embd, eps);
}

/**
 * PyTorch binding for Embedding forward pass
 * Performs embedding lookup: out[i] = weight[indices[i]]
 * 
 * @param weight Embedding weight matrix (vocab_size × n_embd, CUDA tensor)
 * @param indices Token indices (batch_size × seq_len, CUDA tensor, dtype=int)
 * @param out Output embeddings (batch_size × seq_len × n_embd, CUDA tensor)
 */
void embedding_fwd(torch::Tensor weight, torch::Tensor indices, torch::Tensor out) {
    // Validate input tensors
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.device().is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    TORCH_CHECK(out.dim() == 3, "out must be 3D");
    TORCH_CHECK(indices.size(1) == out.size(1) && weight.size(1) == out.size(2),
                "tensor dimensions must be compatible");

    // Extract dimensions
    int num_indices = indices.numel();
    int n_embd = weight.size(1);

    // Call CUDA kernel
    embedding_fwd_cuda(weight.data_ptr<float>(), indices.data_ptr<int>(),
                      out.data_ptr<float>(), num_indices, n_embd);
}

/**
 * PyTorch binding for Embedding backward pass
 * Accumulates gradients into embedding weight matrix using atomic operations
 * 
 * @param grad_out Gradient with respect to output (batch_size × seq_len × n_embd, CUDA tensor)
 * @param indices Token indices from forward pass (batch_size × seq_len, CUDA tensor, dtype=int)
 * @param grad_weight Gradient with respect to embedding weights (vocab_size × n_embd, CUDA tensor)
 */
void embedding_bwd(torch::Tensor grad_out, torch::Tensor indices, torch::Tensor grad_weight) {
    // Validate input tensors
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(indices.device().is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(grad_weight.device().is_cuda(), "grad_weight must be a CUDA tensor");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    TORCH_CHECK(grad_weight.dim() == 2, "grad_weight must be 2D");

    // Extract dimensions
    int num_indices = indices.numel();
    int n_embd = grad_weight.size(1);

    // Call CUDA kernel
    embedding_bwd_cuda(grad_out.data_ptr<float>(), indices.data_ptr<int>(),
                      grad_weight.data_ptr<float>(), num_indices, n_embd);
}

/**
 * Pybind11 module definition
 * Exposes all forward and backward pass functions to Python
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_fwd", &add_fwd, "Add forward");
    m.def("add_bwd", &add_bwd, "Add backward");
    m.def("mul_fwd", &mul_fwd, "Mul forward");
    m.def("mul_bwd", &mul_bwd, "Mul backward");
    m.def("gelu_fwd", &gelu_fwd, "GELU forward");
    m.def("gelu_bwd", &gelu_bwd, "GELU backward");
    m.def("matmul_fwd", &matmul_fwd, "MatMul forward");
    m.def("matmul_bwd", &matmul_bwd, "MatMul backward");
    m.def("batched_matmul_fwd", &batched_matmul_fwd, "Batched MatMul forward");
    m.def("batched_matmul_bwd", &batched_matmul_bwd, "Batched MatMul backward");
    m.def("softmax_fwd", &softmax_fwd, "Softmax forward");
    m.def("softmax_bwd", &softmax_bwd, "Softmax backward");
    m.def("layernorm_fwd", &layernorm_fwd, "LayerNorm forward");
    m.def("layernorm_bwd", &layernorm_bwd, "LayerNorm backward");
    m.def("embedding_fwd", &embedding_fwd, "Embedding forward");
    m.def("embedding_bwd", &embedding_bwd, "Embedding backward");
}
