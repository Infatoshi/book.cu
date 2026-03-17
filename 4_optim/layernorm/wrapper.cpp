
void run_kernel_0(float* out, float* mean, float* rstd, const float* inp, 
                  const float* weight, const float* bias, int N, int C);
void run_kernel_1(float* out, float* mean, float* rstd, const float* inp,
                  const float* weight, const float* bias, int N, int C);
void run_kernel_2(float* out, float* mean, float* rstd, const float* inp,
                  const float* weight, const float* bias, int N, int C);

torch::Tensor layernorm_kernel_0(torch::Tensor inp, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(), "Input must be contiguous");
    
    int N = inp.size(0);
    int C = inp.size(1);
    
    auto out = torch::empty_like(inp);
    auto mean = torch::empty({N}, inp.options());
    auto rstd = torch::empty({N}, inp.options());
    
    run_kernel_0(
        out.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        inp.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C
    );
    
    return out;
}

torch::Tensor layernorm_kernel_1(torch::Tensor inp, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(), "Input must be contiguous");
    
    int N = inp.size(0);
    int C = inp.size(1);
    
    auto out = torch::empty_like(inp);
    auto mean = torch::empty({N}, inp.options());
    auto rstd = torch::empty({N}, inp.options());
    
    run_kernel_1(
        out.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        inp.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C
    );
    
    return out;
}

torch::Tensor layernorm_kernel_2(torch::Tensor inp, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(inp.is_contiguous(), "Input must be contiguous");
    
    int N = inp.size(0);
    int C = inp.size(1);
    
    auto out = torch::empty_like(inp);
    auto mean = torch::empty({N}, inp.options());
    auto rstd = torch::empty({N}, inp.options());
    
    run_kernel_2(
        out.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        inp.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_0", &layernorm_kernel_0, "LayerNorm Kernel 0 (Naive)");
    m.def("kernel_1", &layernorm_kernel_1, "LayerNorm Kernel 1 (Parallel)");
    m.def("kernel_2", &layernorm_kernel_2, "LayerNorm Kernel 2 (Warp)");
}
