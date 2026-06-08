#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file int8_uint8.cu
 * @brief Compares symmetric (INT8) and asymmetric (UINT8) quantization in CUDA.
 *
 * This file demonstrates the two primary modes of quantization for 8-bit integers:
 *
 * 1. Symmetric Quantization (using `int8_t`):
 *    - Maps floating-point values to a signed integer range, typically [-127, 127].
 *    - The mapping is symmetric around zero, meaning `quantize(x) = -quantize(-x)`.
 *    - A single `scale` parameter is used. The zero-point is implicitly 0.
 *    - `q = clamp(round(x / scale), -127, 127)`
 *    - `x' = q * scale`
 *    - Ideal for data that is naturally centered around zero (e.g., normalized weights).
 *
 * 2. Asymmetric Quantization (using `uint8_t`):
 *    - Maps floating-point values to an unsigned integer range, typically [0, 255].
 *    - Uses both a `scale` and a `zero_point` parameter to map the FP32 range to the
 *      full UINT8 range. The `zero_point` corresponds to the FP32 value 0.
 *    - `q = clamp(round(x / scale + zero_point_int), 0, 255)`
 *    - `x' = (q - zero_point_int) * scale`
 *    - More flexible and generally more accurate for data that is not centered around
 *      zero (e.g., activations after a ReLU function, which are always non-negative).
 *
 * This example shows that for non-zero-centered data, asymmetric quantization can
 * provide significantly better accuracy by utilizing the entire integer range more
 * effectively.
 */

/**
 * @brief CUDA kernel for symmetric quantization (FP32 to INT8).
 *
 * Quantizes FP32 values to INT8 using a single scaling factor. The mapping is
 * symmetric around zero.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 tensor on the device.
 * @param scale The symmetric quantization scaling factor.
 * @param size The number of elements in the tensor.
 */
__global__ void quantize_symmetric(float* input, int8_t* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Symmetric quantization: scale, clamp to [-127, 127], and round.
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

/**
 * @brief CUDA kernel for asymmetric quantization (FP32 to UINT8).
 *
 * Quantizes FP32 values to UINT8 using both a scale and a zero-point. This allows
 * the mapping to be shifted to best fit non-zero-centered data.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output UINT8 tensor on the device.
 * @param scale The asymmetric quantization scaling factor.
 * @param zero_point The floating-point value that maps to the integer zero-point.
 * @param size The number of elements in the tensor.
 */
__global__ void quantize_asymmetric(float* input, uint8_t* output, float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Asymmetric quantization:
    // 1. Shift the input by the zero-point.
    // 2. Scale the result.
    // 3. Clamp to the UINT8 range [0, 255] and round.
    // Note: A more precise formula is round(x/scale + zero_point_int). This is a simplification.
    float scaled = roundf(input[idx] / scale) + zero_point;
    scaled = fmaxf(fminf(scaled, 255.0f), 0.0f);
    output[idx] = (uint8_t)scaled;
}

/**
 * @brief CUDA kernel for symmetric dequantization (INT8 to FP32).
 *
 * Converts INT8 values back to FP32 by multiplying by the scale factor.
 *
 * @param input Pointer to the input INT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param scale The symmetric quantization scaling factor.
 * @param size The number of elements in the tensor.
 */
__global__ void dequantize_symmetric(int8_t* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = (float)input[idx] * scale;
}

/**
 * @brief CUDA kernel for asymmetric dequantization (UINT8 to FP32).
 *
 * Converts UINT8 values back to FP32 by first shifting by the zero-point and then
 * multiplying by the scale factor.
 *
 * @param input Pointer to the input UINT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param scale The asymmetric quantization scaling factor.
 * @param zero_point The zero-point used during quantization.
 * @param size The number of elements in the tensor.
 */
__global__ void dequantize_asymmetric(uint8_t* input, float* output, float scale, float zero_point, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Asymmetric dequantization: subtract zero-point and then multiply by scale.
    output[idx] = ((float)input[idx] - zero_point) * scale;
}

/**
 * @brief Generates a normally distributed random number.
 */
float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

/**
 * @brief Main function to compare symmetric and asymmetric quantization.
 *
 * This function does the following:
 * 1. Generates a host tensor with a non-zero-mean distribution to highlight the
 *    strengths of asymmetric quantization.
 * 2. Calculates the necessary `scale` and `zero_point` parameters for both methods.
 * 3. Allocates GPU memory and copies the input data.
 * 4. Launches kernels for both symmetric and asymmetric quantization/dequantization.
 * 5. Copies the results back to the host.
 * 6. Computes and prints the Mean Squared Error (MSE) and Mean Absolute Error (MAE)
 *    for both methods, demonstrating that asymmetric quantization is more accurate
 *    for this type of data.
 */
int main() {
    const int SIZE = 512 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    // --- 1. Data Generation ---
    // Use a non-zero mean to show the advantage of asymmetric quantization.
    float* h_input = (float*)malloc(SIZE * sizeof(float));
    srand(time(NULL));

    float min_val = INFINITY, max_val = -INFINITY;
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = rand_normal(2.0f, 1.5f);
        min_val = fminf(min_val, h_input[i]);
        max_val = fmaxf(max_val, h_input[i]);
    }

    // --- 2. Parameter Calculation ---
    // Symmetric: Scale is based on the max absolute value to cover the range symmetrically.
    float symmetric_scale = fmaxf(fabsf(min_val), fabsf(max_val)) / 127.0f;
    // Asymmetric: Scale and zero-point are calculated to map the exact [min, max] range
    // to the [0, 255] integer range.
    float asymmetric_scale = (max_val - min_val) / 255.0f;
    float zero_point = roundf(0.0f - min_val / asymmetric_scale); // Integer zero-point

    printf("INT8 vs UINT8 Quantization Test\n");
    printf("Data range: [%f, %f]\n", min_val, max_val);
    printf("Data mean: %f (non-zero)\n\n", (min_val + max_val) / 2.0f);

    printf("Symmetric (INT8):\n");
    printf("  Scale: %f\n", symmetric_scale);
    printf("  Zero point: 0 (fixed)\n");
    printf("  Range: [-127, 127]\n\n");

    printf("Asymmetric (UINT8):\n");
    printf("  Scale: %f\n", asymmetric_scale);
    printf("  Zero point: %f\n", zero_point);
    printf("  Range: [0, 255]\n\n");

    // --- 3. Device Memory Allocation and Transfer ---
    float *d_input, *d_symmetric_output, *d_asymmetric_output;
    int8_t *d_symmetric_quantized;
    uint8_t *d_asymmetric_quantized;

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_symmetric_output, SIZE * sizeof(float));
    cudaMalloc(&d_asymmetric_output, SIZE * sizeof(float));
    cudaMalloc(&d_symmetric_quantized, SIZE * sizeof(int8_t));
    cudaMalloc(&d_asymmetric_quantized, SIZE * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Execution ---
    // Symmetric path
    quantize_symmetric<<<BLOCKS, THREADS>>>(d_input, d_symmetric_quantized, symmetric_scale, SIZE);
    dequantize_symmetric<<<BLOCKS, THREADS>>>(d_symmetric_quantized, d_symmetric_output, symmetric_scale, SIZE);

    // Asymmetric path
    quantize_asymmetric<<<BLOCKS, THREADS>>>(d_input, d_asymmetric_quantized, asymmetric_scale, zero_point, SIZE);
    dequantize_asymmetric<<<BLOCKS, THREADS>>>(d_asymmetric_quantized, d_asymmetric_output, asymmetric_scale, zero_point, SIZE);

    cudaDeviceSynchronize();

    // --- 5. Result Verification ---
    float* h_symmetric_output = (float*)malloc(SIZE * sizeof(float));
    float* h_asymmetric_output = (float*)malloc(SIZE * sizeof(float));

    cudaMemcpy(h_symmetric_output, d_symmetric_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_asymmetric_output, d_asymmetric_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 6. Error Calculation ---
    float symmetric_mse = 0.0f, asymmetric_mse = 0.0f;
    float symmetric_mae = 0.0f, asymmetric_mae = 0.0f;

    for (int i = 0; i < SIZE; ++i) {
        float sym_error = h_input[i] - h_symmetric_output[i];
        float asym_error = h_input[i] - h_asymmetric_output[i];

        symmetric_mse += sym_error * sym_error;
        asymmetric_mse += asym_error * asym_error;
        symmetric_mae += fabsf(sym_error);
        asymmetric_mae += fabsf(asym_error);
    }

    symmetric_mse /= SIZE;
    asymmetric_mse /= SIZE;
    symmetric_mae /= SIZE;
    asymmetric_mae /= SIZE;

    printf("Accuracy Results:\n");
    printf("Symmetric (INT8):\n");
    printf("  MSE: %f\n", symmetric_mse);
    printf("  MAE: %f\n\n", symmetric_mae);

    printf("Asymmetric (UINT8):\n");
    printf("  MSE: %f\n", asymmetric_mse);
    printf("  MAE: %f\n\n", asymmetric_mae);

    printf("Asymmetric is %fx more accurate for this data\n\n", symmetric_mse / asymmetric_mse);

    printf("Key Insight:\n");
    printf("- Symmetric (INT8): Best for zero-mean data, simpler\n");
    printf("- Asymmetric (UINT8): Better for data with offset, uses full range\n");

    printf("\nSample comparisons:\n");
    printf("Original -> Symmetric -> Asymmetric\n");
    for (int i = 0; i < 3; ++i) {
        printf("%f -> %f -> %f\n",
               h_input[i], h_symmetric_output[i], h_asymmetric_output[i]);
    }

    // --- 7. Cleanup ---
    cudaFree(d_input);
    cudaFree(d_symmetric_output);
    cudaFree(d_asymmetric_output);
    cudaFree(d_symmetric_quantized);
    cudaFree(d_asymmetric_quantized);
    free(h_input);
    free(h_symmetric_output);
    free(h_asymmetric_output);

    return 0;
}
