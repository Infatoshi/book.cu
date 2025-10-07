






float rand_normal(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * stddev + mean;
}


__global__ void quantize_groupwise(float* input, int8_t* output,
                                   float* group_scales, int group_size, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    
    int group_idx = idx / group_size;
    float scale = group_scales[group_idx];

    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

__global__ void dequantize_groupwise(int8_t* input, float* output,
                                     float* group_scales, int group_size, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    
    int group_idx = idx / group_size;
    float scale = group_scales[group_idx];

    output[idx] = (float)input[idx] * scale;
}

int main() {
    const int TENSOR_SIZE = 512 * 1024;  
    const int GROUP_SIZE = 4096;         
    const int NUM_GROUPS = (TENSOR_SIZE + GROUP_SIZE - 1) / GROUP_SIZE;
    const int THREADS = 256;
    const int BLOCKS = (TENSOR_SIZE + THREADS - 1) / THREADS;

    printf("Group-wise Quantization Test\n");
    printf("Tensor size: %d elements\n", TENSOR_SIZE);
    printf("Group size: %d elements per group\n", GROUP_SIZE);
    printf("Number of groups: %d\n\n", NUM_GROUPS);


    float* h_tensor = (float*)malloc(TENSOR_SIZE * sizeof(float));
    srand(time(NULL));

    for (int g = 0; g < NUM_GROUPS; ++g) {

        float group_std = 1.0f + 0.5f * g;

        int start_idx = g * GROUP_SIZE;
        int end_idx = fminf((g + 1) * GROUP_SIZE, TENSOR_SIZE);

        for (int i = start_idx; i < end_idx; ++i) {
            h_tensor[i] = rand_normal(0.0f, group_std);
        }
    }


    float* h_group_scales = (float*)malloc(NUM_GROUPS * sizeof(float));
    printf("Group scales:\n");
    for (int g = 0; g < NUM_GROUPS; ++g) {
        float group_max_abs = 0.0f;
        int start_idx = g * GROUP_SIZE;
        int end_idx = fminf((g + 1) * GROUP_SIZE, TENSOR_SIZE);

        for (int i = start_idx; i < end_idx; ++i) {
            group_max_abs = fmaxf(group_max_abs, fabsf(h_tensor[i]));
        }

        h_group_scales[g] = group_max_abs / 127.0f;

        if (g < 3 || g >= NUM_GROUPS - 3) {
            printf("  Group %d: %f\n", g, h_group_scales[g]);
        } else if (g == 3) {
            printf("  ... (%d more groups) ...\n", (NUM_GROUPS - 6));
        }
    }

    
    float *d_tensor, *d_output, *d_group_scales;
    int8_t *d_quantized;

    cudaMalloc(&d_tensor, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_group_scales, NUM_GROUPS * sizeof(float));
    cudaMalloc(&d_quantized, TENSOR_SIZE * sizeof(int8_t));

    
    cudaMemcpy(d_tensor, h_tensor, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_scales, h_group_scales, NUM_GROUPS * sizeof(float), cudaMemcpyHostToDevice);

    
    quantize_groupwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, d_group_scales, GROUP_SIZE, TENSOR_SIZE);
    cudaDeviceSynchronize();

    
    dequantize_groupwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, d_group_scales, GROUP_SIZE, TENSOR_SIZE);
    cudaDeviceSynchronize();


    float* h_output = (float*)malloc(TENSOR_SIZE * sizeof(float));
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

    printf("\nQuantization Accuracy:\n");
    printf("  MSE: %f\n", mse);
    printf("  MAE: %f\n", mae);
    printf("  Max Error: %f\n", max_error);
    printf("  Memory reduction: 4x (FP32 -> INT8)\n");


    float tensor_max_abs = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        tensor_max_abs = fmaxf(tensor_max_abs, fabsf(h_tensor[i]));
    }
    float tensor_scale = tensor_max_abs / 127.0f;

    
    float tensor_mse = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        float scaled = h_tensor[i] / tensor_scale;
        scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
        int8_t quantized = (int8_t)roundf(scaled);
        float dequantized = (float)quantized * tensor_scale;
        float error = h_tensor[i] - dequantized;
        tensor_mse += error * error;
    }
    tensor_mse /= TENSOR_SIZE;

    printf("\nComparison with Tensor-wise:\n");
    printf("  Tensor-wise MSE: %f\n", tensor_mse);
    printf("  Group-wise MSE: %f\n", mse);
    printf("  Group-wise is %fx more accurate\n\n", (tensor_mse / mse));

    printf("Key Properties of Group-wise Quantization:\n");
    printf("- Different scale for each group of %d elements\n", GROUP_SIZE);
    printf("- Better accuracy than tensor-wise when groups have different ranges\n");
    printf("- Trade-off: %dx more scale parameters to store\n", NUM_GROUPS);
    printf("- Useful for tensors with locally varying value distributions\n");


    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_group_scales);
    cudaFree(d_quantized);

    free(h_tensor);
    free(h_group_scales);
    free(h_output);

    return 0;
}
