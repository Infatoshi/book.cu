// GEMV PyTorch Extension Wrapper
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <torch/extension.h>

void run_kernel_0(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);
void run_kernel_1(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);
void run_kernel_2(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);
void run_kernel_3(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);
void run_kernel_4(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

torch::Tensor gemv_kernel_0(torch::Tensor mat, torch::Tensor vec) {
    TORCH_CHECK(mat.is_cuda(), "Matrix must be a CUDA tensor");
    TORCH_CHECK(vec.is_cuda(), "Vector must be a CUDA tensor");
    TORCH_CHECK(mat.is_contiguous(), "Matrix must be contiguous");
    TORCH_CHECK(vec.is_contiguous(), "Vector must be contiguous");
    TORCH_CHECK(mat.dim() == 2, "Matrix must be 2D");
    TORCH_CHECK(vec.dim() == 1, "Vector must be 1D");

    int M = mat.size(0);
    int N = mat.size(1);
    TORCH_CHECK(vec.size(0) == N, "Vector size must match matrix columns");

    auto result = torch::empty({M}, mat.options());

    run_kernel_0(
        mat.data_ptr<float>(),
        vec.data_ptr<float>(),
        result.data_ptr<float>(),
        M, N
    );

    return result;
}

torch::Tensor gemv_kernel_1(torch::Tensor mat, torch::Tensor vec) {
    TORCH_CHECK(mat.is_cuda(), "Matrix must be a CUDA tensor");
    TORCH_CHECK(vec.is_cuda(), "Vector must be a CUDA tensor");
    TORCH_CHECK(mat.is_contiguous(), "Matrix must be contiguous");
    TORCH_CHECK(vec.is_contiguous(), "Vector must be contiguous");
    TORCH_CHECK(mat.dim() == 2, "Matrix must be 2D");
    TORCH_CHECK(vec.dim() == 1, "Vector must be 1D");

    int M = mat.size(0);
    int N = mat.size(1);
    TORCH_CHECK(vec.size(0) == N, "Vector size must match matrix columns");

    auto result = torch::empty({M}, mat.options());

    run_kernel_1(
        mat.data_ptr<float>(),
        vec.data_ptr<float>(),
        result.data_ptr<float>(),
        M, N
    );

    return result;
}

torch::Tensor gemv_kernel_2(torch::Tensor mat, torch::Tensor vec) {
    TORCH_CHECK(mat.is_cuda(), "Matrix must be a CUDA tensor");
    TORCH_CHECK(vec.is_cuda(), "Vector must be a CUDA tensor");
    TORCH_CHECK(mat.is_contiguous(), "Matrix must be contiguous");
    TORCH_CHECK(vec.is_contiguous(), "Vector must be contiguous");
    TORCH_CHECK(mat.dim() == 2, "Matrix must be 2D");
    TORCH_CHECK(vec.dim() == 1, "Vector must be 1D");

    int M = mat.size(0);
    int N = mat.size(1);
    TORCH_CHECK(vec.size(0) == N, "Vector size must match matrix columns");

    auto result = torch::empty({M}, mat.options());

    run_kernel_2(
        mat.data_ptr<float>(),
        vec.data_ptr<float>(),
        result.data_ptr<float>(),
        M, N
    );

    return result;
}

torch::Tensor gemv_kernel_3(torch::Tensor mat, torch::Tensor vec) {
    TORCH_CHECK(mat.is_cuda(), "Matrix must be a CUDA tensor");
    TORCH_CHECK(vec.is_cuda(), "Vector must be a CUDA tensor");
    TORCH_CHECK(mat.is_contiguous(), "Matrix must be contiguous");
    TORCH_CHECK(vec.is_contiguous(), "Vector must be contiguous");
    TORCH_CHECK(mat.dim() == 2, "Matrix must be 2D");
    TORCH_CHECK(vec.dim() == 1, "Vector must be 1D");

    int M = mat.size(0);
    int N = mat.size(1);
    TORCH_CHECK(vec.size(0) == N, "Vector size must match matrix columns");

    auto result = torch::empty({M}, mat.options());

    run_kernel_3(
        mat.data_ptr<float>(),
        vec.data_ptr<float>(),
        result.data_ptr<float>(),
        M, N
    );

    return result;
}

torch::Tensor gemv_kernel_4(torch::Tensor mat, torch::Tensor vec) {
    TORCH_CHECK(mat.is_cuda(), "Matrix must be a CUDA tensor");
    TORCH_CHECK(vec.is_cuda(), "Vector must be a CUDA tensor");
    TORCH_CHECK(mat.is_contiguous(), "Matrix must be contiguous");
    TORCH_CHECK(vec.is_contiguous(), "Vector must be contiguous");
    TORCH_CHECK(mat.dim() == 2, "Matrix must be 2D");
    TORCH_CHECK(vec.dim() == 1, "Vector must be 1D");

    int M = mat.size(0);
    int N = mat.size(1);
    TORCH_CHECK(vec.size(0) == N, "Vector size must match matrix columns");

    auto result = torch::empty({M}, mat.options());

    run_kernel_4(
        mat.data_ptr<float>(),
        vec.data_ptr<float>(),
        result.data_ptr<float>(),
        M, N
    );

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_0", &gemv_kernel_0, "GEMV Kernel 0 (cuBLAS)");
    m.def("kernel_1", &gemv_kernel_1, "GEMV Kernel 1 (Naive)");
    m.def("kernel_2", &gemv_kernel_2, "GEMV Kernel 2 (Coalesced Warp)");
    m.def("kernel_3", &gemv_kernel_3, "GEMV Kernel 3 (Coalesced Warp+Block)");
    m.def("kernel_4", &gemv_kernel_4, "GEMV Kernel 4 (Vectorized)");
}
