/**
 * @file build_naive.cpp
 * @brief PyTorch C++ extension for binding the naive attention kernel.
 *
 * @details
 * This file uses Pybind11 to create a Python module that exposes the
 * `naive_attn_forward` C++/CUDA function to PyTorch. This allows the non-optimized,
 * baseline attention implementation to be called from Python for comparison
 * and testing purposes.
 *
 * The `PYBIND11_MODULE` macro defines the module's entry point, with the module
 * name being provided by the `TORCH_EXTENSION_NAME` macro from the PyTorch
 * build system.
 *
 * `m.def("forward", ...)` binds the `naive_attn_forward` C++ function to a
 * Python function named `forward` in the compiled module.
 */

#include <torch/extension.h>

torch::Tensor naive_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Binds the C++ function to the Python name "forward"
    m.def("forward", torch::wrap_pybind_function(naive_attn_forward), "naive_attn_forward");
}
