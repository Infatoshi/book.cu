

__global__ void quantize_tensorwise(float* input, signed char* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (signed char)roundf(scaled);
}

__global__ void dequantize_tensorwise(signed char* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = (float)input[idx] * scale;
}

float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

int main() {
    const int TENSOR_SIZE = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (TENSOR_SIZE + THREADS - 1) / THREADS;

    srand(time(NULL));

    float* h_tensor = (float*)malloc(TENSOR_SIZE * sizeof(float));
    float* h_output = (float*)malloc(TENSOR_SIZE * sizeof(float));

    float tensor_max_abs = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        h_tensor[i] = rand_normal(0.0f, 2.0f);
        tensor_max_abs = fmaxf(tensor_max_abs, fabsf(h_tensor[i]));
    }

    float scale = tensor_max_abs / 127.0f;

    printf("Tensor-wise Quantization Test\n");
    printf("Tensor size: %d elements\n", TENSOR_SIZE);
    printf("Single scale: %f (applied to entire tensor)\n", scale);
    printf("Memory reduction: 4x (FP32 -> INT8)\n\n");

    float *d_tensor, *d_output;
    signed char *d_quantized;

    cudaMalloc(&d_tensor, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_quantized, TENSOR_SIZE * sizeof(signed char));

    cudaMemcpy(d_tensor, h_tensor, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    quantize_tensorwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, scale, TENSOR_SIZE);
    cudaDeviceSynchronize();

    dequantize_tensorwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, scale, TENSOR_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        float error = h_tensor[i] - h_output[i];
        mse += error * error;
        mae += fabsf(error);
        max_error = fmaxf(max_error, fabsf(error));
    }
    mse /= TENSOR_SIZE;
    mae /= TENSOR_SIZE;

    printf("Quantization Accuracy:\n");
    printf("  MSE: %f\n", mse);
    printf("  MAE: %f\n", mae);
    printf("  Max Error: %f\n", max_error);

    printf("\nSample tensor values:\n");
    printf("  Original -> Quantized -> Dequantized -> Error\n");
    for (int i = 0; i < 5; ++i) {
        float error = h_tensor[i] - h_output[i];
        printf("  %f -> [quantized] -> %f (error: %f)\n",
               h_tensor[i], h_output[i], error);
    }

    printf("\nKey Properties of Tensor-wise Quantization:\n");
    printf("- Same scale applied to ALL %d elements\n", TENSOR_SIZE);
    printf("- Simplest quantization scheme\n");
    printf("- Works best when tensor values have similar ranges\n");
    printf("- Fast and memory efficient\n");
    printf("- May underperform if tensor has varying value ranges\n");

    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_quantized);
    free(h_tensor);
    free(h_output);

    return 0;
}
