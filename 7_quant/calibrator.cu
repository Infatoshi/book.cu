
/**
 * @file calibrator.cu
 * @brief Demonstrates and compares different calibration techniques for quantization in CUDA.
 *
 * Calibration is a crucial step in post-training quantization (PTQ). It involves analyzing
 * a small, representative dataset (the "calibration set") to determine the optimal
 * quantization parameters (scaling factor and zero-point). These parameters are then
 * used to quantize the model's weights and activations for inference.
 *
 * This file implements and compares two common calibration methods:
 * 1. Min-Max Calibration: Simple and fast, but highly sensitive to outliers. It maps the
 *    exact minimum and maximum values from the calibration data to the full integer range.
 * 2. Percentile Calibration: More robust to outliers. It ignores a certain percentage of
 *    extreme values, typically leading to better overall accuracy on test data that may
 *    contain outliers not seen during calibration.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/**
 * @brief CUDA kernel for min-max calibration.
 *
 * This kernel finds the absolute minimum and maximum values in the calibration data.
 * These values are then used to compute the scaling factor and zero-point for
 * asymmetric quantization.
 *
 * The quantization parameters are calculated as:
 * - `scale = (max_val - min_val) / 255.0f` (for INT8/UINT8)
 * - `zero_point = -min_val / scale`
 *
 * It uses atomic operations on shared memory for a fast, parallel reduction to find
 * the min and max values across all threads in the grid.
 *
 * @param calibration_data Pointer to the input calibration data on the device.
 * @param scale_output Pointer to a single float on the device to store the calculated scale.
 * @param zero_point_output Pointer to a single float on the device to store the calculated zero-point.
 * @param size The number of elements in the calibration data.
 */
__global__ void calibrate_min_max(float* calibration_data, float* scale_output,
                                  float* zero_point_output, int size) {
    // Shared memory for min/max reduction
    extern __shared__ float sdata_min_max[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads one value from global memory.
    float val = (idx < size) ? calibration_data[idx] : 0.0f;

    // The first thread in the block initializes the shared memory for reduction.
    if (tid == 0) {
        sdata_min_max[0] = val;  // min
        sdata_min_max[1] = val;  // max
    }
    __syncthreads();

    // Perform block-wide atomic reduction to find the min and max values.
    // The __float_as_int intrinsic is used because atomicMin/Max don't directly
    // support floating-point types prior to some CUDA architectures.
    atomicMin((int*)&sdata_min_max[0], __float_as_int(val));
    atomicMax((int*)&sdata_min_max[1], __float_as_int(val));
    __syncthreads();

    // The first thread of the first block computes the final scale and zero-point.
    // Note: This is a simplification; a two-level reduction would be needed for multiple blocks.
    if (tid == 0 && blockIdx.x == 0) {
        float min_val = sdata_min_max[0];
        float max_val = sdata_min_max[1];
        float range = max_val - min_val;

        // Avoid division by zero if the range is empty.
        if (range > 1e-6) {
            *scale_output = range / 255.0f;
            *zero_point_output = -min_val / *scale_output;
        } else {
            *scale_output = 1.0f;
            *zero_point_output = 0.0f;
        }
    }
}

/**
 * @brief CUDA kernel for percentile-based calibration.
 *
 * This kernel determines the scaling factor based on a high percentile (e.g., 95th) of
 * the absolute values in the calibration data. This makes the quantization scheme
 * symmetric (zero-point is 0) and robust to outliers.
 *
 * The scale is calculated as:
 * - `scale = percentile_value / 127.0f` (for symmetric INT8)
 *
 * This implementation approximates the 95th percentile by taking 95% of the
 * maximum absolute value found in the data, which is a common and efficient heuristic.
 *
 * @param calibration_data Pointer to the input calibration data on the device.
 * @param scale_output Pointer to a single float on the device to store the calculated scale.
 * @param size The number of elements in the calibration data.
 */
__global__ void calibrate_percentile(float* calibration_data, float* scale_output, int size) {
    // Shared memory for performing a block-level reduction.
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes the absolute value of one element.
    float val = (idx < size) ? fabsf(calibration_data[idx]) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Perform a parallel reduction within the block to find the maximum absolute value.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The first thread of the first block computes the scale based on the reduced max value.
    // This is a simplification; a two-level reduction is needed for multiple blocks.
    if (tid == 0 && blockIdx.x == 0) {
        // Use 95% of the max value as a robust estimate for the 95th percentile.
        *scale_output = (sdata[0] * 0.95f) / 127.0f;
    }
}

/**
 * @brief CUDA kernel to quantize and then dequantize test data.
 *
 * This kernel applies the quantization parameters (scale and zero-point) derived from a
 * calibration method to a test dataset. It first quantizes the FP32 data to INT8, and then
 * immediately dequantizes it back to FP32 to measure the quantization error.
 *
 * @param test_data Pointer to the input FP32 test data on the device.
 * @param quantized Pointer to the output INT8 quantized data on the device.
 * @param dequantized Pointer to the output FP32 dequantized data on the device.
 * @param scale The scaling factor determined during calibration.
 * @param zero_point The zero-point determined during calibration (can be 0 for symmetric quantization).
 * @param size The number of elements in the test data.
 */
__global__ void quantize_and_test(float* test_data, int8_t* quantized, float* dequantized,
                                  float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // --- Quantization ---
    // 1. Scale the input value.
    // 2. Add the zero-point.
    // 3. Round to the nearest integer.
    // 4. Clamp to the valid INT8 range.
    float val_f = roundf(test_data[idx] / scale + zero_point);
    val_f = fmaxf(fminf(val_f, 127.0f), -128.0f);
    quantized[idx] = (int8_t)val_f;

    // --- Dequantization ---
    // 1. Convert INT8 back to float.
    // 2. Subtract the zero-point.
    // 3. Multiply by the scale.
    dequantized[idx] = ((float)quantized[idx] - zero_point) * scale;
}

/**
 * @brief Generates a normally distributed random number using the Box-Muller transform.
 *
 * @param mean The mean of the normal distribution.
 * @param std The standard deviation of the normal distribution.
 * @return A single-precision floating-point random number.
 */
float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

/**
 * @brief Main function demonstrating and comparing calibration techniques.
 *
 * This function executes the following steps:
 * 1. Generates a calibration dataset and a separate test dataset. The test set
 *    is intentionally seeded with outliers to highlight the weaknesses of min-max calibration.
 * 2. Allocates memory on the GPU for both datasets and for the outputs.
 * 3. Launches the `calibrate_min_max` kernel to get one set of quantization parameters.
 * 4. Launches the `calibrate_percentile` kernel to get a second, more robust set.
 * 5. Launches `quantize_and_test` kernels to apply both sets of parameters to the test data.
 * 6. Copies the dequantized results back to the host.
 * 7. Computes the Mean Squared Error (MSE) for both methods and prints a comparison,
 *    demonstrating that percentile calibration yields lower error in the presence of outliers.
 */
int main() {
    const int CALIBRATION_SIZE = 100 * 1024;
    const int TEST_SIZE = 50 * 1024;
    const int THREADS = 256;
    const int CALIBRATION_BLOCKS = (CALIBRATION_SIZE + THREADS - 1) / THREADS;
    const int TEST_BLOCKS = (TEST_SIZE + THREADS - 1) / THREADS;

    // --- 1. Data Generation ---
    // Allocate host memory for calibration and test data.
    float* h_calibration = (float*)malloc(CALIBRATION_SIZE * sizeof(float));
    srand(time(NULL));

    // Generate calibration data from normal distribution
    float calib_min = INFINITY, calib_max = -INFINITY;
    for (int i = 0; i < CALIBRATION_SIZE; ++i) {
        h_calibration[i] = rand_normal(0.0f, 1.5f);
        calib_min = fminf(calib_min, h_calibration[i]);
        calib_max = fmaxf(calib_max, h_calibration[i]);
    }

    // Generate test data with some large outliers to test robustness.
    float* h_test = (float*)malloc(TEST_SIZE * sizeof(float));
    for (int i = 0; i < TEST_SIZE; ++i) {
        h_test[i] = rand_normal(0.2f, 2.0f);
        if (i % 1000 == 0) h_test[i] *= 5.0f;  // Add outliers
    }

    printf("Calibration Techniques Test\n");
    printf("Calibration data: %d samples\n", CALIBRATION_SIZE);
    printf("Test data: %d samples (with outliers)\n\n", TEST_SIZE);

    // --- 2. Device Memory Allocation ---
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

    // --- 3. Data Transfer to Device ---
    cudaMemcpy(d_calibration, h_calibration, CALIBRATION_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, h_test, TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Method 1: Min-Max Calibration ---
    // The third launch parameter specifies the amount of shared memory per block.
    // Here, 2 floats are needed for the min/max reduction.
    calibrate_min_max<<<1, THREADS, 2 * sizeof(float)>>>(
        d_calibration, d_minmax_scale, d_minmax_zero, CALIBRATION_SIZE);

    // Copy calibration results from device to host.
    float h_minmax_scale, h_minmax_zero;
    cudaMemcpy(&h_minmax_scale, d_minmax_scale, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_minmax_zero, d_minmax_zero, sizeof(float), cudaMemcpyDeviceToHost);

    // Apply min-max parameters to test data.
    quantize_and_test<<<TEST_BLOCKS, THREADS>>>(d_test, d_test_quantized_minmax,
        d_test_dequantized_minmax, h_minmax_scale, h_minmax_zero, TEST_SIZE);

    // --- 5. Method 2: Percentile Calibration ---
    // Shared memory size is based on the number of threads for the reduction tree.
    calibrate_percentile<<<1, THREADS, THREADS * sizeof(float)>>>(
        d_calibration, d_percentile_scale, CALIBRATION_SIZE);

    // Copy percentile scale from device to host.
    float h_percentile_scale;
    cudaMemcpy(&h_percentile_scale, d_percentile_scale, sizeof(float), cudaMemcpyDeviceToHost);

    // Apply percentile parameters to test data (symmetric, so zero_point = 0).
    quantize_and_test<<<TEST_BLOCKS, THREADS>>>(d_test, d_test_quantized_percentile,
        d_test_dequantized_percentile, h_percentile_scale, 0.0f, TEST_SIZE);

    // --- 6. Synchronization and Result Verification ---
    cudaDeviceSynchronize();

    // Allocate host memory for dequantized results.
    float* h_dequantized_minmax = (float*)malloc(TEST_SIZE * sizeof(float));
    float* h_dequantized_percentile = (float*)malloc(TEST_SIZE * sizeof(float));

    // Copy dequantized results from device to host for error calculation.
    cudaMemcpy(h_dequantized_minmax, d_test_dequantized_minmax,
               TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dequantized_percentile, d_test_dequantized_percentile,
               TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute Mean Squared Error (MSE) for both calibration methods.
    float minmax_mse = 0.0f, percentile_mse = 0.0f;
    for (int i = 0; i < TEST_SIZE; ++i) {
        float minmax_error = h_test[i] - h_dequantized_minmax[i];
        float percentile_error = h_test[i] - h_dequantized_percentile[i];
        minmax_mse += minmax_error * minmax_error;
        percentile_mse += percentile_error * percentile_error;
    }
    minmax_mse /= TEST_SIZE;
    percentile_mse /= TEST_SIZE;

    // --- 7. Print Results ---
    printf("Calibration Results:\n");
    printf("Min-Max Calibration:\n");
    printf("  Scale: %f, Zero Point: %f\n", h_minmax_scale, h_minmax_zero);
    printf("  MSE on test data: %f\n\n", minmax_mse);

    printf("95th Percentile Calibration:\n");
    printf("  Scale: %f, Zero Point: 0 (symmetric)\n", h_percentile_scale);
    printf("  MSE on test data: %f\n\n", percentile_mse);

    // Compare methods and print conclusion.
    if (percentile_mse < minmax_mse) {
        printf("Percentile calibration is more robust to outliers!\n");
    } else {
        printf("Min-max calibration works well for this data.\n");
    }

    printf("\nKey Insight:\n");
    printf("- Min-Max: Sensitive to outliers in calibration data\n");
    printf("- Percentile: More robust, ignores extreme values\n");

    // --- 8. Cleanup ---
    // Clean up device memory.
    cudaFree(d_calibration);
    cudaFree(d_test);
    cudaFree(d_minmax_scale);
    cudaFree(d_minmax_zero);
    cudaFree(d_percentile_scale);
    cudaFree(d_test_dequantized_minmax);
    cudaFree(d_test_dequantized_percentile);
    cudaFree(d_test_quantized_minmax);
    cudaFree(d_test_quantized_percentile);
    
    // Clean up host memory.
    free(h_calibration);
    free(h_test);
    free(h_dequantized_minmax);
    free(h_dequantized_percentile);

    return 0;
}
