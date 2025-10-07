

__global__ void calibrate_min_max(float* calibration_data, float* scale_output,
                                  float* zero_point_output, int size) {
    extern __shared__ float sdata_min_max[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < size) ? calibration_data[idx] : 0.0f;

    if (tid == 0) {
        sdata_min_max[0] = val;
        sdata_min_max[1] = val;
    }
    __syncthreads();

    atomicMin((int*)&sdata_min_max[0], __float_as_int(val));
    atomicMax((int*)&sdata_min_max[1], __float_as_int(val));
    __syncthreads();

    if (tid == 0) {
        float min_val = sdata_min_max[0];
        float max_val = sdata_min_max[1];
        float range = max_val - min_val;

        *scale_output = range / 255.0f;
        *zero_point_output = -min_val / *scale_output;
    }
}

__global__ void calibrate_percentile(float* calibration_data, float* scale_output, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < size) ? fabsf(calibration_data[idx]) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *scale_output = (sdata[0] * 0.95f) / 127.0f;
    }
}

__global__ void quantize_and_test(float* test_data, int8_t* quantized, float* dequantized,
                                  float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float scaled = (test_data[idx] - zero_point) / scale;
    scaled = fmaxf(fminf(scaled, 255.0f), 0.0f);
    quantized[idx] = (int8_t)roundf(scaled);

    dequantized[idx] = (float)quantized[idx] * scale + zero_point;
}

float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

int main() {
    const int CALIBRATION_SIZE = 100 * 1024;
    const int TEST_SIZE = 50 * 1024;
    const int THREADS = 256;
    const int CALIBRATION_BLOCKS = (CALIBRATION_SIZE + THREADS - 1) / THREADS;
    const int TEST_BLOCKS = (TEST_SIZE + THREADS - 1) / THREADS;

    float* h_calibration = (float*)malloc(CALIBRATION_SIZE * sizeof(float));
    srand(time(NULL));

    float calib_min = INFINITY, calib_max = -INFINITY;
    for (int i = 0; i < CALIBRATION_SIZE; ++i) {
        h_calibration[i] = rand_normal(0.0f, 1.5f);
        calib_min = fminf(calib_min, h_calibration[i]);
        calib_max = fmaxf(calib_max, h_calibration[i]);
    }

    float* h_test = (float*)malloc(TEST_SIZE * sizeof(float));

    for (int i = 0; i < TEST_SIZE; ++i) {
        h_test[i] = rand_normal(0.2f, 2.0f);
        if (i % 1000 == 0) h_test[i] *= 5.0f;
    }

    printf("Calibration Techniques Test\n");
    printf("Calibration data: %d samples\n", CALIBRATION_SIZE);
    printf("Test data: %d samples (with outliers)\n\n", TEST_SIZE);

    float *d_calibration, *d_test, *d_minmax_scale, *d_minmax_zero, *d_percentile_scale;
    float *d_test_dequantized_minmax, *d_test_dequantized_percentile;
    int8_t *d_test_quantized_minmax, *d_test_quantized_percentile;

    cudaMalloc(&d_calibration, CALIBRATION_SIZE * sizeof(float));
    cudaMalloc(&d_test, TEST_SIZE * sizeof(float));
    cudaMalloc(&d_minmax_scale, sizeof(float));
    cudaMalloc(&d_minmax_zero, sizeof(float));
    cudaMalloc(&d_percentile_scale, sizeof(float));
    cudaMalloc(&d_test_dequantized_minmax, TEST_SIZE * sizeof(float));
    cudaMalloc(&d_test_dequantized_percentile, TEST_SIZE * sizeof(float));
    cudaMalloc(&d_test_quantized_minmax, TEST_SIZE * sizeof(int8_t));
    cudaMalloc(&d_test_quantized_percentile, TEST_SIZE * sizeof(int8_t));

    cudaMemcpy(d_calibration, h_calibration, CALIBRATION_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test, TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    calibrate_min_max<<<CALIBRATION_BLOCKS, THREADS, 2 * sizeof(float)>>>(
        d_calibration, d_minmax_scale, d_minmax_zero, CALIBRATION_SIZE);

    float h_minmax_scale, h_minmax_zero;
    cudaMemcpy(&h_minmax_scale, d_minmax_scale, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minmax_zero, d_minmax_zero, sizeof(float), cudaMemcpyDeviceToHost);

    quantize_and_test<<<TEST_BLOCKS, THREADS>>>(d_test, d_test_quantized_minmax,
        d_test_dequantized_minmax, h_minmax_scale, h_minmax_zero, TEST_SIZE);

    calibrate_percentile<<<CALIBRATION_BLOCKS, THREADS, THREADS * sizeof(float)>>>(
        d_calibration, d_percentile_scale, CALIBRATION_SIZE);

    float h_percentile_scale;
    cudaMemcpy(&h_percentile_scale, d_percentile_scale, sizeof(float), cudaMemcpyDeviceToHost);

    quantize_and_test<<<TEST_BLOCKS, THREADS>>>(d_test, d_test_quantized_percentile,
        d_test_dequantized_percentile, h_percentile_scale, 0.0f, TEST_SIZE);

    cudaDeviceSynchronize();

    float* h_dequantized_minmax = (float*)malloc(TEST_SIZE * sizeof(float));
    float* h_dequantized_percentile = (float*)malloc(TEST_SIZE * sizeof(float));

    cudaMemcpy(h_dequantized_minmax, d_test_dequantized_minmax,
               TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dequantized_percentile, d_test_dequantized_percentile,
               TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float minmax_mse = 0.0f, percentile_mse = 0.0f;
    for (int i = 0; i < TEST_SIZE; ++i) {
        float minmax_error = h_test[i] - h_dequantized_minmax[i];
        float percentile_error = h_test[i] - h_dequantized_percentile[i];
        minmax_mse += minmax_error * minmax_error;
        percentile_mse += percentile_error * percentile_error;
    }
    minmax_mse /= TEST_SIZE;
    percentile_mse /= TEST_SIZE;

    printf("Calibration Results:\n");
    printf("Min-Max Calibration:\n");
    printf("  Scale: %f, Zero Point: %f\n", h_minmax_scale, h_minmax_zero);
    printf("  MSE on test data: %f\n\n", minmax_mse);

    printf("95th Percentile Calibration:\n");
    printf("  Scale: %f, Zero Point: 0 (symmetric)\n", h_percentile_scale);
    printf("  MSE on test data: %f\n\n", percentile_mse);

    if (percentile_mse < minmax_mse) {
        printf("Percentile calibration is more robust to outliers!\n");
    } else {
        printf("Min-max calibration works well for this data.\n");
    }

    printf("\nKey Insight:\n");
    printf("- Min-Max: Sensitive to outliers in calibration data\n");
    printf("- Percentile: More robust, ignores extreme values\n");

    cudaFree(d_calibration);
    cudaFree(d_test);
    cudaFree(d_minmax_scale);
    cudaFree(d_minmax_zero);
    cudaFree(d_percentile_scale);
    cudaFree(d_test_dequantized_minmax);
    cudaFree(d_test_dequantized_percentile);
    cudaFree(d_test_quantized_minmax);
    cudaFree(d_test_quantized_percentile);
    free(h_calibration);
    free(h_test);
    free(h_dequantized_minmax);
    free(h_dequantized_percentile);

    return 0;
}
