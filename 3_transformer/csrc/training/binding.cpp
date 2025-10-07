
void add_fwd_cuda(const float* a, const float* b, float* out, int size);
void add_bwd_cuda(const float* grad_out, float* grad_a, float* grad_b, int size);
void mul_fwd_cuda(const float* a, const float* b, float* out, int size);
void mul_bwd_cuda(const float* grad_out, const float* a, const float* b,
                  float* grad_a, float* grad_b, int size);
void gelu_fwd_cuda(const float* x, float* out, int size);
void gelu_bwd_cuda(const float* grad_out, const float* x, float* grad_x, int size);
void matmul_fwd_cuda(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                    float* grad_A, float* grad_B, int M, int N, int K);
void batched_matmul_fwd_cuda(const float* A, const float* B, float* C,
                            int batch_size, int M, int N, int K);
void batched_matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                            float* grad_A, float* grad_B,
                            int batch_size, int M, int N, int K);
void softmax_fwd_cuda(const float* x, float* out, int batch_size, int seq_len, int n_embd);
void softmax_bwd_cuda(const float* grad_out, const float* out, float* grad_x,
                     int batch_size, int seq_len, int n_embd);
void layernorm_fwd_cuda(const float* x, const float* gamma, const float* beta,
                       float* out, float* mean_out, float* var_out,
                       int batch_size, int seq_len, int n_embd, float eps);
void layernorm_bwd_cuda(const float* grad_out, const float* x, const float* gamma,
                       const float* mean, const float* var, float* grad_x,
                       float* grad_gamma, float* grad_beta,
                       int batch_size, int seq_len, int n_embd, float eps);
void embedding_fwd_cuda(const float* weight, const int* indices, float* out,
                       int num_indices, int n_embd);
void embedding_bwd_cuda(const float* grad_out, const int* indices, float* grad_weight,
                       int num_indices, int n_embd);

void add_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    add_fwd_cuda(a.data_ptr<float>(), b.data_ptr<float>(),
                out.data_ptr<float>(), a.numel());
}

void add_bwd(torch::Tensor grad_out, torch::Tensor grad_a, torch::Tensor grad_b) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(grad_out.numel() == grad_a.numel() && grad_out.numel() == grad_b.numel(),
                "tensor sizes must match");

    add_bwd_cuda(grad_out.data_ptr<float>(), grad_a.data_ptr<float>(),
                grad_b.data_ptr<float>(), grad_out.numel());
}

void mul_fwd(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "tensor sizes must match");

    mul_fwd_cuda(a.data_ptr<float>(), b.data_ptr<float>(),
                out.data_ptr<float>(), a.numel());
}

void mul_bwd(torch::Tensor grad_out, torch::Tensor a, torch::Tensor b,
            torch::Tensor grad_a, torch::Tensor grad_b) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(grad_a.device().is_cuda(), "grad_a must be a CUDA tensor");
    TORCH_CHECK(grad_b.device().is_cuda(), "grad_b must be a CUDA tensor");
    TORCH_CHECK(a.numel() == b.numel() && a.numel() == grad_out.numel() &&
                a.numel() == grad_a.numel() && a.numel() == grad_b.numel(),
                "tensor sizes must match");

    mul_bwd_cuda(grad_out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
                grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), a.numel());
}

void gelu_fwd(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.numel() == out.numel(), "tensor sizes must match");

    gelu_fwd_cuda(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
}

void gelu_bwd(torch::Tensor grad_out, torch::Tensor x, torch::Tensor grad_x) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_out.numel() == x.numel() && x.numel() == grad_x.numel(),
                "tensor sizes must match");

    gelu_bwd_cuda(grad_out.data_ptr<float>(), x.data_ptr<float>(),
                 grad_x.data_ptr<float>(), x.numel());
}

void matmul_fwd(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "tensors must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0) && A.size(0) == C.size(0) && B.size(1) == C.size(1),
                "matrix dimensions must be compatible");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    matmul_fwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                   M, N, K);
}

void matmul_bwd(torch::Tensor A, torch::Tensor B, torch::Tensor grad_C,
               torch::Tensor grad_A, torch::Tensor grad_B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(grad_C.device().is_cuda(), "grad_C must be a CUDA tensor");
    TORCH_CHECK(grad_A.device().is_cuda(), "grad_A must be a CUDA tensor");
    TORCH_CHECK(grad_B.device().is_cuda(), "grad_B must be a CUDA tensor");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    matmul_bwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), grad_C.data_ptr<float>(),
                   grad_A.data_ptr<float>(), grad_B.data_ptr<float>(), M, N, K);
}

void batched_matmul_fwd(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3, "tensors must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == C.size(0), "batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1) && A.size(1) == C.size(1) && B.size(2) == C.size(2),
                "matrix dimensions must be compatible");

    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    batched_matmul_fwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                           batch_size, M, N, K);
}

void batched_matmul_bwd(torch::Tensor A, torch::Tensor B, torch::Tensor grad_C,
                       torch::Tensor grad_A, torch::Tensor grad_B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(grad_C.device().is_cuda(), "grad_C must be a CUDA tensor");
    TORCH_CHECK(grad_A.device().is_cuda(), "grad_A must be a CUDA tensor");
    TORCH_CHECK(grad_B.device().is_cuda(), "grad_B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && grad_C.dim() == 3 &&
                grad_A.dim() == 3 && grad_B.dim() == 3, "tensors must be 3D");

    int batch_size = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    batched_matmul_bwd_cuda(A.data_ptr<float>(), B.data_ptr<float>(), grad_C.data_ptr<float>(),
                           grad_A.data_ptr<float>(), grad_B.data_ptr<float>(),
                           batch_size, M, N, K);
}

void softmax_fwd(torch::Tensor x, torch::Tensor out) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3 && out.dim() == 3, "tensors must be 3D");
    TORCH_CHECK(x.sizes() == out.sizes(), "tensor sizes must match");

    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embd = x.size(2);

    softmax_fwd_cuda(x.data_ptr<float>(), out.data_ptr<float>(),
                    batch_size, seq_len, n_embd);
}

void softmax_bwd(torch::Tensor grad_out, torch::Tensor out, torch::Tensor grad_x) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_out.dim() == 3 && out.dim() == 3 && grad_x.dim() == 3,
                "tensors must be 3D");
    TORCH_CHECK(grad_out.sizes() == out.sizes() && out.sizes() == grad_x.sizes(),
                "tensor sizes must match");

    int batch_size = grad_out.size(0);
    int seq_len = grad_out.size(1);
    int n_embd = grad_out.size(2);

    softmax_bwd_cuda(grad_out.data_ptr<float>(), out.data_ptr<float>(),
                    grad_x.data_ptr<float>(), batch_size, seq_len, n_embd);
}

void layernorm_fwd(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                  torch::Tensor out, torch::Tensor mean_out, torch::Tensor var_out, float eps) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.device().is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(mean_out.device().is_cuda(), "mean_out must be a CUDA tensor");
    TORCH_CHECK(var_out.device().is_cuda(), "var_out must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3 && out.dim() == 3, "x and out must be 3D");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma and beta must be 1D");
    TORCH_CHECK(gamma.size(0) == x.size(2) && beta.size(0) == x.size(2),
                "gamma and beta size must match embedding dimension");

    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int n_embd = x.size(2);

    layernorm_fwd_cuda(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                      out.data_ptr<float>(), mean_out.data_ptr<float>(), var_out.data_ptr<float>(),
                      batch_size, seq_len, n_embd, eps);
}

void layernorm_bwd(torch::Tensor grad_out, torch::Tensor x, torch::Tensor gamma,
                  torch::Tensor mean, torch::Tensor var, torch::Tensor grad_x,
                  torch::Tensor grad_gamma, torch::Tensor grad_beta, float eps) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.device().is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(mean.device().is_cuda(), "mean must be a CUDA tensor");
    TORCH_CHECK(var.device().is_cuda(), "var must be a CUDA tensor");
    TORCH_CHECK(grad_x.device().is_cuda(), "grad_x must be a CUDA tensor");
    TORCH_CHECK(grad_gamma.device().is_cuda(), "grad_gamma must be a CUDA tensor");
    TORCH_CHECK(grad_beta.device().is_cuda(), "grad_beta must be a CUDA tensor");

    int batch_size = grad_out.size(0);
    int seq_len = grad_out.size(1);
    int n_embd = grad_out.size(2);

    layernorm_bwd_cuda(grad_out.data_ptr<float>(), x.data_ptr<float>(), gamma.data_ptr<float>(),
                      mean.data_ptr<float>(), var.data_ptr<float>(), grad_x.data_ptr<float>(),
                      grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
                      batch_size, seq_len, n_embd, eps);
}

void embedding_fwd(torch::Tensor weight, torch::Tensor indices, torch::Tensor out) {
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.device().is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    TORCH_CHECK(out.dim() == 3, "out must be 3D");
    TORCH_CHECK(indices.size(1) == out.size(1) && weight.size(1) == out.size(2),
                "tensor dimensions must be compatible");

    int num_indices = indices.numel();
    int n_embd = weight.size(1);

    embedding_fwd_cuda(weight.data_ptr<float>(), indices.data_ptr<int>(),
                      out.data_ptr<float>(), num_indices, n_embd);
}

void embedding_bwd(torch::Tensor grad_out, torch::Tensor indices, torch::Tensor grad_weight) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(indices.device().is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(grad_weight.device().is_cuda(), "grad_weight must be a CUDA tensor");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be 3D");
    TORCH_CHECK(indices.dim() == 2, "indices must be 2D");
    TORCH_CHECK(grad_weight.dim() == 2, "grad_weight must be 2D");

    int num_indices = indices.numel();
    int n_embd = grad_weight.size(1);

    embedding_bwd_cuda(grad_out.data_ptr<float>(), indices.data_ptr<int>(),
                      grad_weight.data_ptr<float>(), num_indices, n_embd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_fwd", &add_fwd, "Add forward");
    m.def("add_bwd", &add_bwd, "Add backward");
    m.def("mul_fwd", &mul_fwd, "Mul forward");
    m.def("mul_bwd", &mul_bwd, "Mul backward");
    m.def("gelu_fwd", &gelu_fwd, "GELU forward");
    m.def("gelu_bwd", &gelu_bwd, "GELU backward");
    m.def("matmul_fwd", &matmul_fwd, "MatMul forward");
    m.def("matmul_bwd", &matmul_bwd, "MatMul backward");
    m.def("batched_matmul_fwd", &batched_matmul_fwd, "Batched MatMul forward");
    m.def("batched_matmul_bwd", &batched_matmul_bwd, "Batched MatMul backward");
    m.def("softmax_fwd", &softmax_fwd, "Softmax forward");
    m.def("softmax_bwd", &softmax_bwd, "Softmax backward");
    m.def("layernorm_fwd", &layernorm_fwd, "LayerNorm forward");
    m.def("layernorm_bwd", &layernorm_bwd, "LayerNorm backward");
    m.def("embedding_fwd", &embedding_fwd, "Embedding forward");
    m.def("embedding_bwd", &embedding_bwd, "Embedding backward");
}
