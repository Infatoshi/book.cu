#include <torch/extension.h>

/**
 * PyTorch C++ Extension Bindings for Transformer Inference Kernels
 * 
 * This file provides Python bindings for CUDA kernels used in transformer inference.
 * Unlike the training bindings, these only include forward pass implementations since
 * inference doesn't require gradient computation.
 * 
 * The bindings connect Python code to CUDA kernels through PyTorch's extension system,
 * allowing seamless integration with PyTorch's tensor operations and autograd system.
 * 
 * Architecture:
 * - Forward-only operations optimized for inference speed
 * - Specialized kernels for single-token generation (GEMV instead of MatMul)
 * - KV cache support for efficient autoregressive generation
 * - MoE (Mixture of Experts) routing support via TopK
 * 
 * Key Operations:
 * - MatMul/GEMV: Matrix operations for attention and feed-forward layers
 * - Element-wise ops: Add/Mul for residual connections
 * - Normalization: LayerNorm for pre-attention/post-attention normalization
 * - Activation: GELU for feed-forward network activation
 * - Attention: Softmax for attention weights computation
 * - Routing: TopK for expert selection in MoE architectures
 */

// Forward declarations of CUDA kernel wrapper functions
// These are implemented in separate .cu files in the kernels/ directory

torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);
torch::Tensor gemv_forward(torch::Tensor A, torch::Tensor x);
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor mul_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor gelu_forward(torch::Tensor x);
torch::Tensor softmax_forward(torch::Tensor x);
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
std::tuple<torch::Tensor, torch::Tensor> topk_forward(torch::Tensor input, int k);

/**
 * Pybind11 module definition for inference extension
 * 
 * This module exposes all forward pass functions to Python, allowing them to be
 * imported as `custom_inference_extension.matmul_forward()`, etc.
 * 
 * The functions are called from Python wrapper modules in wrapper/inference/
 * which provide a cleaner interface and handle tensor shape validation.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward, "Matrix multiplication forward pass");
    m.def("gemv_forward", &gemv_forward, "GEMV (Matrix-Vector) multiplication forward pass");
    m.def("add_forward", &add_forward, "Elementwise addition forward pass");
    m.def("mul_forward", &mul_forward, "Elementwise multiplication forward pass");
    m.def("gelu_forward", &gelu_forward, "GELU activation forward pass");
    m.def("softmax_forward", &softmax_forward, "Softmax forward pass");
    m.def("layernorm_forward", &layernorm_forward, "Layer normalization forward pass");
    m.def("topk_forward", &topk_forward, "Top-K selection forward pass");
}
