
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error = %u at %s:%d '%s' \n",
            (unsigned int)result, file, line, func);
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void matmul_naive_quant_kernel(
    const float* x,
    const float* W,
    float* out,
    int K, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        float activation = x[k];
        float weight = W[k * N + col];

        float quant_val = roundf(weight * 7.0f);
        float weight_to_use = quant_val / 7.0f;

        sum += activation * weight_to_use;
    }
    out[col] = sum;
}

__global__ void matmul_awq_quant_kernel(
    const float* x,
    const float* W,
    float* out,
    const float* scales,
    int K, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        float activation = x[k];
        float weight = W[k * N + col];

        float activation_to_use = activation * scales[k];
        float scaled_weight = weight / scales[k];
        float quant_val = roundf(scaled_weight * 7.0f);
        float weight_to_use = quant_val / 7.0f;

        sum += activation_to_use * weight_to_use;
    }
    out[col] = sum;
}

int main() {
    const int M = 1, K = 6, N = 4;

    float h_x[] = { 0.1f, 25.0f, -0.2f, 0.05f, -18.0f, 0.3f };
    float h_W[] = {
        1.1f, -0.4f,  2.3f,  0.9f,
        0.8f,  1.5f, -0.2f,  0.7f,
       -2.5f,  0.1f,  0.3f,  1.4f,
        0.6f, -1.9f,  0.5f,  2.1f,
        1.2f,  0.3f, -0.6f, -1.7f,
       -0.7f,  1.3f, -1.1f,  0.2f
    };
    float h_scales[K];
    float sum_scales = 0.0f;
    for (int k = 0; k < K; ++k) {
        h_scales[k] = 1.0f / (fabsf(h_x[k]) + 0.1f);  
        sum_scales += h_scales[k];
    }
    float avg_scale = sum_scales / K;
    for (int k = 0; k < K; ++k) {
        h_scales[k] /= avg_scale;
    }

    printf("AWQ Scales (computed from activations): [ ");
    for (int i = 0; i < K; ++i) printf("%.4f ", h_scales[i]);
    printf("]\n");

    float h_out_fp32[4], h_out_naive_q[4], h_out_awq_q[4];

    float *d_x, *d_W, *d_scales, *d_out;
    checkCudaErrors(cudaMalloc(&d_x, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_W, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_scales, K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out, M * N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_x, h_x, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_W, h_W, K * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_scales, h_scales, K * sizeof(float), cudaMemcpyHostToDevice));

    int threads_per_block = N;
    int blocks_per_grid = 1;

    printf("--- Running Calculations ---\n\n");

    float *d_W_fp32, *d_out_fp32;
    checkCudaErrors(cudaMalloc(&d_W_fp32, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out_fp32, M * N * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_W_fp32, h_W, K * N * sizeof(float), cudaMemcpyHostToDevice));

    for (int col = 0; col < N; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += h_x[k] * h_W[k * N + col];
        }
        h_out_fp32[col] = sum;
    }

    matmul_naive_quant_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_W, d_out, K, N);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_out_naive_q, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    matmul_awq_quant_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_W, d_out, d_scales, K, N);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_out_awq_q, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("--- Results ---\n\n");
    printf("Input Activation: [ ");
    for (int i = 0; i < K; ++i) printf("%.4f ", h_x[i]);
    printf("]\n\n");

    printf("FP32 Ground Truth: [ ");
    for (int i = 0; i < N; ++i) printf("%.4f ", h_out_fp32[i]);
    printf("]\n");
    printf("-------------------------------------------------------------\n");

    printf("Naive INT4 Quant:  [ ");
    for (int i = 0; i < N; ++i) printf("%.4f ", h_out_naive_q[i]);
    printf("]\n");

    float naive_mse = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = h_out_fp32[i] - h_out_naive_q[i];
        naive_mse += diff * diff;
    }
    naive_mse /= N;
    printf("  -> Naive Quant MSE vs Ground Truth: %.6f\n\n", naive_mse);

    printf("AWQ INT4 Quant:    [ ");
    for (int i = 0; i < N; ++i) printf("%.4f ", h_out_awq_q[i]);
    printf("]\n");

    float awq_mse = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = h_out_fp32[i] - h_out_awq_q[i];
        awq_mse += diff * diff;
    }
    awq_mse /= N;
    printf("  -> AWQ Quant MSE vs Ground Truth:   %.6f\n\n", awq_mse);

    printf("--- Conclusion ---\n");
    if (awq_mse < naive_mse) {
        printf("Success! The Mean Squared Error for AWQ is significantly lower.\n");
        printf("AWQ protected the important weights, preserving the output accuracy.\n");
    } else {
        printf("Error: Something is still wrong in the logic.\n");
    }

    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_W));
    checkCudaErrors(cudaFree(d_scales));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_W_fp32));
    checkCudaErrors(cudaFree(d_out_fp32));
    return 0;
}