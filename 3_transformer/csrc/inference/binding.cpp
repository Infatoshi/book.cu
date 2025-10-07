
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);
torch::Tensor gemv_forward(torch::Tensor A, torch::Tensor x);
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor mul_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor gelu_forward(torch::Tensor x);
torch::Tensor softmax_forward(torch::Tensor x);
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
std::tuple<torch::Tensor, torch::Tensor> topk_forward(torch::Tensor input, int k);

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
