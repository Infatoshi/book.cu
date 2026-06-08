#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>



/**
 * @file fp32_int8.cu
 * @brief Demonstrates basic symmetric quantization from FP32 to INT8 and back.
 *
 * Quantization is the process of converting a tensor with floating-point values
 * (like FP32) into a tensor with lower-precision integer values (like INT8). This
 * reduces the memory footprint and can significantly speed up computations on
 * hardware that supports low-precision arithmetic.
 *
 * This example uses symmetric quantization, where the range of floating-point values
 * is mapped symmetrically around zero. The key parameters are:
 *
 * - Scale: A positive floating-point number that defines the mapping between the
 *   floating-point and integer domains. It's typically calculated as:
 *   `scale = max(abs(original_values)) / 127.0f`.
 *
 * - Zero-point: In symmetric quantization, the zero-point is implicitly 0.
 *
 * The process involves:
 * 1. Quantization: `quantized_value = clamp(round(fp32_value / scale), -127, 127)`
 * 2. Dequantization: `fp32_value = quantized_value * scale`
 */

/**
 * @brief CUDA kernel to perform symmetric quantization from FP32 to INT8.
 *
 * Each thread processes one element from the input tensor. It scales the value,
 * clamps it to the valid range for a signed 8-bit integer [-127, 127], and rounds
 * it to the nearest integer.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 (`signed char`) tensor on the device.
 * @param scale The symmetric quantization scaling factor.
 * @param size The number of elements in the tensor.
 */
__global__ void quantize_fp32_to_int8(float* input, signed char* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Apply symmetric quantization:
    // 1. Divide by the scale to map the value to the integer range.
    // 2. Clamp the value to the representable range of INT8 [-127, 127].
    //    We use 127 to maintain symmetry and avoid using -128.
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    // 3. Round to the nearest integer and cast to signed char.
    output[idx] = (signed char)roundf(scaled);
}

/**
 * @brief CUDA kernel to dequantize from INT8 back to FP32.
 *
 * Each thread processes one element from the quantized INT8 tensor, converting it
 * back to a floating-point value by multiplying it by the scaling factor.
 *
 * @param input Pointer to the input INT8 (`signed char`) tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param scale The same scaling factor used during quantization.
 * @param size The number of elements in the tensor.
 */
__global__ void dequantize_int8_to_fp32(signed char* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Apply dequantization by multiplying the integer value by the scale.
    output[idx] = (float)input[idx] * scale;
}

/**
 * @brief Generates a normally distributed random number using the Box-Muller transform.
 *
 * @param mean The mean of the normal distribution.
 * @param std The standard deviation of the normal distribution.
 * @return A single-precision floating-point random number.
 */
float rand_normal(float mean, float std) {
    // Box-Muller transform for generating normally distributed random numbers.
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

/**
 * @brief Main function to orchestrate the FP32 <-> INT8 conversion and verification.
 *
 * This function performs the following steps:
 * 1. Generates a host tensor with normally distributed random FP32 values.
 * 2. Calculates the symmetric quantization `scale` factor by finding the maximum
 *    absolute value in the data and mapping it to 127.
 * 3. Allocates memory on the GPU for the input, quantized, and dequantized tensors.
 * 4. Copies the input data from host to device.
 * 5. Launches the `quantize_fp32_to_int8` kernel.
 * 6. Launches the `dequantize_int8_to_fp32` kernel.
 * 7. Copies the dequantized result back from device to host.
 * 8. Computes and prints error metrics (MSE, MAE, Max Error) to quantify the
 *    precision loss introduced by the quantization-dequantization cycle.
 */
int main() {
    const int SIZE = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    srand(time(NULL));

    printf("FP32 -> INT8 Quantization Test\n");

    // --- 1. Host Data Generation ---
    float* h_input = (float*)malloc(SIZE * sizeof(float));
    float* h_output = (float*)malloc(SIZE * sizeof(float));

    // Find the maximum absolute value to determine the quantization scale.
    float max_abs = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = rand_normal(0.0f, 2.0f);
        max_abs = fmaxf(max_abs, fabsf(h_input[i]));
    }

    // --- 2. Scale Calculation ---
    // The scale maps the maximum absolute value to the edge of the INT8 range.
    float scale = max_abs / 127.0f;
    printf("Scale: %f\n", scale);
    printf("Memory reduction: 4x (FP32 -> INT8)\n\n");

    // --- 3. Device Memory Allocation and Transfer ---
    float *d_input, *d_output;
    signed char *d_quantized;

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    cudaMalloc(&d_quantized, SIZE * sizeof(signed char));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Execution ---
    quantize_fp32_to_int8<<<BLOCKS, THREADS>>>(d_input, d_quantized, scale, SIZE);
    cudaDeviceSynchronize();

    dequantize_int8_to_fp32<<<BLOCKS, THREADS>>>(d_quantized, d_output, scale, SIZE);
    cudaDeviceSynchronize();

    // --- 5. Result Verification ---
    cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate error metrics to measure precision loss.
    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        float error = h_input[i] - h_output[i];
        mse += error * error;
        mae += fabsf(error);
        max_error = fmaxf(max_error, fabsf(error));
    }
    mse /= SIZE;
    mae /= SIZE;

    printf("Accuracy Results:\n");
    printf("  MSE: %f\n", mse);
    printf("  MAE: %f\n", mae);
    printf("  Max Error: %f\n", max_error);

    printf("\nSample comparisons:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  %f -> %f (error: %f)\n",
               h_input[i], h_output[i], h_input[i] - h_output[i]);
    }

    // --- 6. Cleanup ---
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    free(h_input);
    free(h_output);

    return 0;
}
