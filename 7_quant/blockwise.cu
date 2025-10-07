

float rand_normal(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * stddev + mean;
}


__global__ void quantize_blockwise(float* input, int8_t* output,
                                   float* block_scales, int block_height, int block_width,
                                   int tensor_height, int tensor_width, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    
    int row = idx / tensor_width;
    int col = idx % tensor_width;

    
    int block_row = row / block_height;
    int block_col = col / block_width;
    int blocks_per_row = (tensor_width + block_width - 1) / block_width;
    int block_idx = block_row * blocks_per_row + block_col;

    float scale = block_scales[block_idx];

    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

__global__ void dequantize_blockwise(int8_t* input, float* output,
                                     float* block_scales, int block_height, int block_width,
                                     int tensor_height, int tensor_width, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    
    int row = idx / tensor_width;
    int col = idx % tensor_width;

    
    int block_row = row / block_height;
    int block_col = col / block_width;
    int blocks_per_row = (tensor_width + block_width - 1) / block_width;
    int block_idx = block_row * blocks_per_row + block_col;

    float scale = block_scales[block_idx];
    output[idx] = (float)input[idx] * scale;
}

int main() {
    const int TENSOR_HEIGHT = 256;
    const int TENSOR_WIDTH = 256;
    const int TENSOR_SIZE = TENSOR_HEIGHT * TENSOR_WIDTH;

    const int BLOCK_HEIGHT = 32;
    const int BLOCK_WIDTH = 32;
    const int BLOCKS_PER_ROW = (TENSOR_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    const int BLOCKS_PER_COL = (TENSOR_HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    const int NUM_BLOCKS = BLOCKS_PER_ROW * BLOCKS_PER_COL;

    const int THREADS = 256;
    const int BLOCKS = (TENSOR_SIZE + THREADS - 1) / THREADS;

    printf("Block-wise Quantization Test\n");
    printf("Tensor size: %dx%d = %d elements\n", TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_SIZE);
    printf("Block size: %dx%d\n", BLOCK_HEIGHT, BLOCK_WIDTH);
    printf("Number of blocks: %d (%dx%d)\n\n", NUM_BLOCKS, BLOCKS_PER_COL, BLOCKS_PER_ROW);

    
    float* h_tensor = (float*)malloc(TENSOR_SIZE * sizeof(float));
    srand(time(NULL));

    for (int row = 0; row < TENSOR_HEIGHT; ++row) {
        for (int col = 0; col < TENSOR_WIDTH; ++col) {
            int idx = row * TENSOR_WIDTH + col;

            
            float center_row = row - TENSOR_HEIGHT / 2.0f;
            float center_col = col - TENSOR_WIDTH / 2.0f;
            float distance_from_center = sqrtf(center_row * center_row + center_col * center_col);

            
            float local_std = 1.0f + 2.0f * expf(-distance_from_center / 100.0f);

            h_tensor[idx] = rand_normal(0.0f, local_std);
        }
    }

    
    float* h_block_scales = (float*)malloc(NUM_BLOCKS * sizeof(float));
    printf("Block scales (showing corner blocks):\n");
    for (int block_row = 0; block_row < BLOCKS_PER_COL; ++block_row) {
        for (int block_col = 0; block_col < BLOCKS_PER_ROW; ++block_col) {
            int block_idx = block_row * BLOCKS_PER_ROW + block_col;

            float block_max_abs = 0.0f;
            int start_row = block_row * BLOCK_HEIGHT;
            int end_row = fminf(start_row + BLOCK_HEIGHT, TENSOR_HEIGHT);
            int start_col = block_col * BLOCK_WIDTH;
            int end_col = fminf(start_col + BLOCK_WIDTH, TENSOR_WIDTH);

            for (int r = start_row; r < end_row; ++r) {
                for (int c = start_col; c < end_col; ++c) {
                    int idx = r * TENSOR_WIDTH + c;
                    block_max_abs = fmaxf(block_max_abs, fabsf(h_tensor[idx]));
                }
            }

            h_block_scales[block_idx] = block_max_abs / 127.0f;

            
            if ((block_row == 0 || block_row == BLOCKS_PER_COL - 1) &&
                (block_col == 0 || block_col == BLOCKS_PER_ROW - 1)) {
                printf("  Block [%d,%d]: %f\n", block_row, block_col, h_block_scales[block_idx]);
            }
        }
    }

    
    float *d_tensor, *d_output, *d_block_scales;
    int8_t *d_quantized;

    cudaMalloc(&d_tensor, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_block_scales, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_quantized, TENSOR_SIZE * sizeof(int8_t));

    
    cudaMemcpy(d_tensor, h_tensor, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_scales, h_block_scales, NUM_BLOCKS * sizeof(float), cudaMemcpyHostToDevice);

    
    quantize_blockwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, d_block_scales,
        BLOCK_HEIGHT, BLOCK_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_SIZE);
    cudaDeviceSynchronize();

    
    dequantize_blockwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, d_block_scales,
        BLOCK_HEIGHT, BLOCK_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_SIZE);
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

    printf("\nKey Properties of Block-wise Quantization:\n");
    printf("- Different scale for each %dx%d block\n", BLOCK_HEIGHT, BLOCK_WIDTH);
    printf("- Ideal for spatially organized data (images, feature maps)\n");
    printf("- %d different scales for adaptive quantization\n", NUM_BLOCKS);
    printf("- Balances local accuracy with parameter overhead\n");
    printf("- Useful when different regions have different value ranges\n");

    
    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_block_scales);
    cudaFree(d_quantized);

    free(h_tensor);
    free(h_block_scales);
    free(h_output);

    return 0;
}
