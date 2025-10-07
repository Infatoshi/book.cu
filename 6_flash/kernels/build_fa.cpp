
torch::Tensor fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fa_forward, "Flash Attention forward with WMMA tensor cores");
}

