/**
 * @file wrapper.cpp
 * @brief PyTorch bindings for Tensor Core GEMM kernels
 * 
 * Provides Python interface for all GEMM kernel implementations using Pybind11.
 * Handles input validation, tensor type conversion, and result tensor creation.
 * 
 * Exposed Functions:
 * - kernel_7: cuBLAS with Tensor Cores
 * - kernel_8: WMMA Tensor Cores
 * - kernel_9: WGMMA Basic
 * - kernel_10: WGMMA Larger Tiles
 * - kernel_11: WGMMA Async Loads
 * - kernel_12: WGMMA Max Tiles
 * 
 * Each function also has a "_raw" variant that accepts raw pointers for advanced use cases.
 */
void runKernel8(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel9(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel10(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel11(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);
void runKernel12(int M, int N, int K, at::Half *A, at::Half *B, at::Half *C, int *DB);

void kernel_7_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr = 0);
void kernel_8_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr = 0);
void kernel_9_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr = 0);
void kernel_10_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr = 0);
void kernel_11_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr = 0);
void kernel_12_raw(int M, int N, int K, uint64_t A_ptr, uint64_t B_ptr, uint64_t C_ptr, uint64_t DB_ptr = 0);

/**
 * @brief Validates input tensors for GEMM operations
 * @param A Input matrix A
 * @param B Input matrix B
 * 
 * Checks that:
 * - Tensors are on CUDA device
 * - Data type is FP16 (Float16)
 * - Tensors are contiguous in memory
 * 
 * Throws exceptions if validation fails.
 */
void validate_inputs(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be FP16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be FP16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
}

torch::Tensor kernel_7(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel7(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

torch::Tensor kernel_8(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel8(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

torch::Tensor kernel_9(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel9(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

torch::Tensor kernel_10(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel10(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

torch::Tensor kernel_11(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel11(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

torch::Tensor kernel_12(torch::Tensor A, torch::Tensor B) {
    validate_inputs(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat16).device(A.device()));
    auto DB = torch::full({M * 128}, -1, torch::dtype(torch::kInt32).device(A.device()));
    runKernel12(M, N, K, A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), DB.data_ptr<int>());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_7", &kernel_7, "cuBLAS with Tensor Cores");
    m.def("kernel_8", &kernel_8, "WMMA Tensor Cores");
    m.def("kernel_9", &kernel_9, "WGMMA Basic (FP16 in/out, FP32 accumulate)");
    m.def("kernel_10", &kernel_10, "WGMMA Larger Tiles");
    m.def("kernel_11", &kernel_11, "WGMMA Async Loads");
    m.def("kernel_12", &kernel_12, "WGMMA Max Tiles");
    m.def("kernel_7_raw", &kernel_7_raw, "cuBLAS TC raw pointer entry point");
    m.def("kernel_8_raw", &kernel_8_raw, "WMMA raw pointer entry point");
    m.def("kernel_9_raw", &kernel_9_raw, "WGMMA raw pointer entry point");
    m.def("kernel_10_raw", &kernel_10_raw, "WGMMA Larger Tiles raw pointer entry point");
    m.def("kernel_11_raw", &kernel_11_raw, "WGMMA Async Loads raw pointer entry point");
    m.def("kernel_12_raw", &kernel_12_raw, "WGMMA Max Tiles raw pointer entry point");
}

