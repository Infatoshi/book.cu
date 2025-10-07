

__global__ void quantize_static(float* input, int8_t* output, float static_scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float scaled = input[idx] / static_scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

__global__ void quantize_dynamic(float* input, int8_t* output, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < size) ? fabsf(input[idx]) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    float dynamic_scale = sdata[0] / 127.0f;

    if (idx < size) {
        float scaled = input[idx] / dynamic_scale;
        scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
        output[idx] = (int8_t)roundf(scaled);
    }
}

__global__ void dequantize(int8_t* input, float* output, float scale, int size) {
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
    const int BATCH_SIZE = 256 * 1024;
    const int NUM_BATCHES = 4;
    const int THREADS = 256;
    const int BLOCKS = (BATCH_SIZE + THREADS - 1) / THREADS;

    float** batches = (float**)malloc(NUM_BATCHES * sizeof(float*));
    for (int b = 0; b < NUM_BATCHES; ++b) {
        batches[b] = (float*)malloc(BATCH_SIZE * sizeof(float));
    }
    srand(time(NULL));

    for (int b = 0; b < NUM_BATCHES; ++b) {
        float std = 1.0f + 0.5f * b;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            batches[b][i] = rand_normal(0.0f, std);
        }
    }

    float static_max_abs = 0.0f;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        static_max_abs = fmaxf(static_max_abs, fabsf(batches[0][i]));
    }
    float static_scale = static_max_abs / 127.0f;

    printf("Dynamic vs Static Quantization Test\n");
    printf("Static scale (from batch 0): %f\n\n", static_scale);

    for (int b = 0; b < NUM_BATCHES; ++b) {
        printf("Batch %d:\n", b);

        float* h_input = batches[b];

        float *d_input, *d_static_output, *d_dynamic_output;
        int8_t *d_static_quantized, *d_dynamic_quantized;

        cudaMalloc(&d_input, BATCH_SIZE * sizeof(float));
        cudaMalloc(&d_static_output, BATCH_SIZE * sizeof(float));
        cudaMalloc(&d_dynamic_output, BATCH_SIZE * sizeof(float));
        cudaMalloc(&d_static_quantized, BATCH_SIZE * sizeof(int8_t));
        cudaMalloc(&d_dynamic_quantized, BATCH_SIZE * sizeof(int8_t));

        cudaMemcpy(d_input, h_input, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        quantize_static<<<BLOCKS, THREADS>>>(d_input, d_static_quantized, static_scale, BATCH_SIZE);
        dequantize<<<BLOCKS, THREADS>>>(d_static_quantized, d_static_output, static_scale, BATCH_SIZE);

        quantize_dynamic<<<BLOCKS, THREADS, THREADS * sizeof(float)>>>(d_input, d_dynamic_quantized, BATCH_SIZE);

        float dynamic_max_abs = 0.0f;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            dynamic_max_abs = fmaxf(dynamic_max_abs, fabsf(h_input[i]));
        }
        float dynamic_scale = dynamic_max_abs / 127.0f;
        dequantize<<<BLOCKS, THREADS>>>(d_dynamic_quantized, d_dynamic_output, dynamic_scale, BATCH_SIZE);

        cudaDeviceSynchronize();

        float* h_static_output = (float*)malloc(BATCH_SIZE * sizeof(float));
        float* h_dynamic_output = (float*)malloc(BATCH_SIZE * sizeof(float));

        cudaMemcpy(h_static_output, d_static_output, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dynamic_output, d_dynamic_output, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        float static_mse = 0.0f, dynamic_mse = 0.0f;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            float static_error = h_input[i] - h_static_output[i];
            float dynamic_error = h_input[i] - h_dynamic_output[i];
            static_mse += static_error * static_error;
            dynamic_mse += dynamic_error * dynamic_error;
        }
        static_mse /= BATCH_SIZE;
        dynamic_mse /= BATCH_SIZE;

        printf("  Static MSE: %f\n", static_mse);
        printf("  Dynamic MSE: %f\n", dynamic_mse);
        printf("  Dynamic is %fx more accurate\n\n", static_mse / dynamic_mse);

        cudaFree(d_input);
        cudaFree(d_static_output);
        cudaFree(d_dynamic_output);
        cudaFree(d_static_quantized);
        cudaFree(d_dynamic_quantized);
        free(h_static_output);
        free(h_dynamic_output);
    }

    for (int b = 0; b < NUM_BATCHES; ++b) {
        free(batches[b]);
    }
    free(batches);

    printf("Key Insight:\n");
    printf("- Static: Fast but uses fixed scale from calibration data\n");
    printf("- Dynamic: More accurate but slower (computes scale per batch)\n");

    return 0;
}
