
void run_kernel_0(float* input, int* indices, float* values, int N, int K);
void run_kernel_1(float* input, int* indices, float* values, int N, int K);
void run_kernel_2(float* input, int* indices, float* values, int N, int K);

std::tuple<torch::Tensor, torch::Tensor> topk_kernel_0(torch::Tensor input, int K) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D");
    
    int N = input.size(0);
    
    auto values = torch::empty({K}, input.options());
    auto indices = torch::empty({K}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    
    run_kernel_0(
        input.data_ptr<float>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        N, K
    );
    
    return std::make_tuple(values, indices);
}

std::tuple<torch::Tensor, torch::Tensor> topk_kernel_1(torch::Tensor input, int K) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D");
    
    int N = input.size(0);
    
    auto values = torch::empty({K}, input.options());
    auto indices = torch::empty({K}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    
    run_kernel_1(
        input.data_ptr<float>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        N, K
    );
    
    return std::make_tuple(values, indices);
}

std::tuple<torch::Tensor, torch::Tensor> topk_kernel_2(torch::Tensor input, int K) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D");
    
    int N = input.size(0);
    
    auto values = torch::empty({K}, input.options());
    auto indices = torch::empty({K}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));
    
    run_kernel_2(
        input.data_ptr<float>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        N, K
    );
    
    return std::make_tuple(values, indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_0", &topk_kernel_0, "TopK Kernel 0 (Naive Selection Sort)");
    m.def("kernel_1", &topk_kernel_1, "TopK Kernel 1 (Min-Heap)");
    m.def("kernel_2", &topk_kernel_2, "TopK Kernel 2 (Warp-Parallel)");
}
