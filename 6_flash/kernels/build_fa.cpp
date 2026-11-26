/**
 * @file build_fa.cpp
 * @brief PyTorch C++ extension for binding the FlashAttention kernel.
 *
 * @details
 * This file uses the Pybind11 library to create a Python module that exposes
 * the `fa_forward` C++/CUDA function to PyTorch. This allows the custom
 * FlashAttention kernel, written in CUDA, to be called directly from Python
 * as if it were a standard PyTorch function.
 *
 * The `PYBIND11_MODULE` macro defines the module entry point. The name of the
 * module is specified by `TORCH_EXTENSION_NAME`, which is a macro defined by
 * the PyTorch build system (setuptools).
 *
 * `m.def("forward", ...)` binds the C++ function `fa_forward` to a Python
 * function named `forward` within the compiled module. A docstring is also
 * provided, which will be accessible via `help()` in Python.
 */

#include <torch/extension.h>

torch::Tensor fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Binds the fa_forward C++ function to the Python name "forward"
    m.def("forward", &fa_forward, "Flash Attention forward with WMMA tensor cores");
}

