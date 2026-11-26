// Softmax PyTorch Extension Wrapper
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <torch/extension.h>

void run_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
void run_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
void run_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
void run_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
float run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

torch::Tensor softmax_kernel_0(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    run_kernel_0(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );

    return output;
}

torch::Tensor softmax_kernel_1(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    run_kernel_1(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );

    return output;
}

torch::Tensor softmax_kernel_2(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    run_kernel_2(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );

    return output;
}

torch::Tensor softmax_kernel_3(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    run_kernel_3(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );

    return output;
}

torch::Tensor softmax_kernel_4(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int N = input.size(1);

    auto output = torch::empty_like(input);

    run_kernel_4(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_0", &softmax_kernel_0, "Softmax Kernel 0 (Naive)");
    m.def("kernel_1", &softmax_kernel_1, "Softmax Kernel 1 (Online)");
    m.def("kernel_2", &softmax_kernel_2, "Softmax Kernel 2 (Shared Memory)");
    m.def("kernel_3", &softmax_kernel_3, "Softmax Kernel 3 (Warp Shuffle)");
    m.def("kernel_4", &softmax_kernel_4, "Softmax Kernel 4 (Vectorized)");
}
