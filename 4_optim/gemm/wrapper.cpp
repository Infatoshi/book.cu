// GEMM PyTorch Extension Wrapper
// Based in part on Simon Boehm's SGEMM tutorial (MIT License)
// https://github.com/siboehm/SGEMM_CUDA

#include <torch/extension.h>

void runCublasGemmFP16(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C);
void runKernel1(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C);
void runKernel2(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C);
void runKernel3(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel4(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel5(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel6(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);

void validate_inputs(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be FP16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be FP16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
}

torch::Tensor kernel_0(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Incompatible matrix dimensions");
    
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    runCublasGemmFP16(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>());
    return C;
}

void kernel_0_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runCublasGemmFP16(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr));
}

torch::Tensor kernel_1(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    runKernel1(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>());
    return C;
}

void kernel_1_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runKernel1(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr));
}

torch::Tensor kernel_2(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    runKernel2(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>());
    return C;
}

void kernel_2_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runKernel2(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr));
}

torch::Tensor kernel_3(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel3(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

void kernel_3_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runKernel3(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr),
        nullptr);
}

torch::Tensor kernel_4(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel4(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

void kernel_4_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runKernel4(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr),
        nullptr);
}

torch::Tensor kernel_5(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel5(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

void kernel_5_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runKernel5(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr),
        nullptr);
}

torch::Tensor kernel_6(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel6(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

void kernel_6_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr) {
    runKernel6(
        M, N, K,
        reinterpret_cast<at::Half *>(A_ptr),
        reinterpret_cast<at::Half *>(B_ptr),
        reinterpret_cast<at::Half *>(C_ptr),
        nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_0", &kernel_0, "cuBLAS GEMM (BF16)");
    m.def("kernel_1", &kernel_1, "Naive GEMM");
    m.def("kernel_2", &kernel_2, "GMEM Coalescing");
    m.def("kernel_3", &kernel_3, "Shared Memory Blocking");
    m.def("kernel_4", &kernel_4, "1D Blocktiling");
    m.def("kernel_5", &kernel_5, "2D Blocktiling");
    m.def("kernel_6", &kernel_6, "Vectorized Memory Access");
    m.def("kernel_0_raw", &kernel_0_raw, "cuBLAS raw pointer entry point");
    m.def("kernel_1_raw", &kernel_1_raw, "Naive raw pointer entry point");
    m.def("kernel_2_raw", &kernel_2_raw, "GMEM Coalescing raw pointer entry point");
    m.def("kernel_3_raw", &kernel_3_raw, "Shared Memory Blocking raw pointer entry point");
    m.def("kernel_4_raw", &kernel_4_raw, "1D Blocktiling raw pointer entry point");
    m.def("kernel_5_raw", &kernel_5_raw, "2D Blocktiling raw pointer entry point");
    m.def("kernel_6_raw", &kernel_6_raw, "Vectorized Memory Access raw pointer entry point");
}
