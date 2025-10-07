

float rand_normal(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * stddev + mean;
}


__global__ void quantize_channelwise(float* input, int8_t* output,
                                     float* channel_scales, int num_channels, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    
    int channel_idx = idx % num_channels;
    float scale = channel_scales[channel_idx];

    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

__global__ void dequantize_channelwise(int8_t* input, float* output,
                                       float* channel_scales, int num_channels, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int channel_idx = idx % num_channels;
    float scale = channel_scales[channel_idx];

    output[idx] = (float)input[idx] * scale;
}

int main() {
    
    const int BATCH_SIZE = 4;
    const int SEQ_LEN = 64;
    const int CHANNELS = 128;
    const int TOTAL_ELEMENTS = BATCH_SIZE * SEQ_LEN * CHANNELS;

    const int THREADS = 256;
    const int BLOCKS = (TOTAL_ELEMENTS + THREADS - 1) / THREADS;

    printf("Channel-wise Quantization Test\n");
    printf("Tensor shape: [%d, %d, %d] (B x T x C)\n", BATCH_SIZE, SEQ_LEN, CHANNELS);
    printf("Total elements: %d\n", TOTAL_ELEMENTS);
    printf("Different scale for each of %d channels\n\n", CHANNELS);

    
    float* h_tensor = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));
    srand(time(NULL));

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int t = 0; t < SEQ_LEN; ++t) {
            for (int c = 0; c < CHANNELS; ++c) {
                int idx = (b * SEQ_LEN * CHANNELS) + (t * CHANNELS) + c;

                
                float channel_scale = 0.5f + 1.5f * (float)c / CHANNELS;

                h_tensor[idx] = rand_normal(0.0f, channel_scale);
            }
        }
    }

    
    float* h_channel_scales = (float*)malloc(CHANNELS * sizeof(float));
    printf("Channel scales (showing range):\n");

    float min_scale = INFINITY, max_scale = 0.0f;
    for (int c = 0; c < CHANNELS; ++c) {
        float channel_max_abs = 0.0f;

        
        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int t = 0; t < SEQ_LEN; ++t) {
                int idx = (b * SEQ_LEN * CHANNELS) + (t * CHANNELS) + c;
                channel_max_abs = fmaxf(channel_max_abs, fabsf(h_tensor[idx]));
            }
        }

        h_channel_scales[c] = channel_max_abs / 127.0f;
        min_scale = fminf(min_scale, h_channel_scales[c]);
        max_scale = fmaxf(max_scale, h_channel_scales[c]);
    }

    printf("  Scale range: %f to %f\n", min_scale, max_scale);
    printf("  Average scale: %f\n\n", (min_scale + max_scale) / 2.0f);

    
    float *d_tensor, *d_output, *d_channel_scales;
    int8_t *d_quantized;

    cudaMalloc(&d_tensor, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_channel_scales, CHANNELS * sizeof(float));
    cudaMalloc(&d_quantized, TOTAL_ELEMENTS * sizeof(int8_t));

    
    cudaMemcpy(d_tensor, h_tensor, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channel_scales, h_channel_scales, CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    
    quantize_channelwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, d_channel_scales, CHANNELS, TOTAL_ELEMENTS);
    cudaDeviceSynchronize();

    
    dequantize_channelwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, d_channel_scales, CHANNELS, TOTAL_ELEMENTS);
    cudaDeviceSynchronize();

    
    float* h_output = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));
    cudaMemcpy(h_output, d_output, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    
    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
        float error = h_tensor[i] - h_output[i];
        mse += error * error;
        mae += fabsf(error);
        max_error = fmaxf(max_error, fabsf(error));
    }
    mse /= TOTAL_ELEMENTS;
    mae /= TOTAL_ELEMENTS;

    printf("Quantization Accuracy:\n");
    printf("  MSE: %f\n", mse);
    printf("  MAE: %f\n", mae);
    printf("  Max Error: %f\n", max_error);
    printf("  Memory reduction: 4x (FP32 -> INT8)\n");

    
    printf("\nPer-channel accuracy (first 5 channels):\n");
    for (int c = 0; c < 5; ++c) {
        float channel_mse = 0.0f;
        int channel_elements = BATCH_SIZE * SEQ_LEN;

        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int t = 0; t < SEQ_LEN; ++t) {
                int idx = (b * SEQ_LEN * CHANNELS) + (t * CHANNELS) + c;
                float error = h_tensor[idx] - h_output[idx];
                channel_mse += error * error;
            }
        }
        channel_mse /= channel_elements;
        printf("  Channel %d MSE: %f\n", c, channel_mse);
    }

    printf("\nKey Properties of Channel-wise Quantization:\n");
    printf("- Different scale for each of %d channels\n", CHANNELS);
    printf("- Channels aggregated across batch and time dimensions\n");
    printf("- Essential when different features have different value ranges\n");
    printf("- Works for any tensor where last dimension represents 'channels'\n\n");

    printf("How to quantize across different dimensions:\n");
    printf("- Per-batch: Different scales for each batch item\n");
    printf("- Per-time: Different scales for each time step (sequence position)\n");
    printf("- Per-channel: Different scales for each feature/channel\n");
    printf("- Per-head: Different scales for each attention head\n");
    printf("- Per-layer: Different scales for each transformer layer\n");
    printf("\nChoose based on which dimension has the most variation in value ranges!\n");

    
    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_channel_scales);
    cudaFree(d_quantized);

    free(h_tensor);
    free(h_channel_scales);
    free(h_output);

    return 0;
}
