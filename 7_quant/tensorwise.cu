#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file tensorwise.cu
 * @brief Demonstrates tensor-wise symmetric quantization for CUDA.
 *
 * Tensor-wise quantization is the simplest form of quantization, where a single,
 * shared scaling factor is used for an entire tensor. This approach is memory-efficient
 * as it only requires storing one scale value per tensor.
 *
 * This example uses symmetric quantization to map FP32 values to INT8. The process is:
 * 1. Find the maximum absolute value (`max_abs`) in the entire tensor.
 * 2. Calculate a single scaling factor: `scale = max_abs / 127.0f`.
 * 3. Quantize each element: `q = clamp(round(x / scale), -127, 127)`.
 * 4. Dequantize each element: `x' = q * scale`.
 *
 * While simple and fast, tensor-wise quantization can suffer from reduced accuracy if
 * the tensor contains values with widely varying ranges, as the single scale must
 * accommodate all values, potentially causing smaller values to lose precision.
 */

/**
 * @brief CUDA kernel for tensor-wise symmetric quantization (FP32 to INT8).
 *
 * Each thread processes one element, quantizing it using a single, shared scaling
 * factor for the entire tensor.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 (`signed char`) tensor on the device.
 * @param scale The single, tensor-wise scaling factor.
 * @param size The number of elements in the tensor.
 */
__global__ void quantize_tensorwise(float* input, signed char* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Apply symmetric quantization using the single tensor-wise scale.
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (signed char)roundf(scaled);
}

/**
 * @brief CUDA kernel for tensor-wise symmetric dequantization (INT8 to FP32).
 *
 * Each thread processes one quantized element, converting it back to FP32 using
 * the same tensor-wise scaling factor used for quantization.
 *
 * @param input Pointer to the input INT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param scale The single, tensor-wise scaling factor.
 * @param size The number of elements in the tensor.
 */
__global__ void dequantize_tensorwise(signed char* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = (float)input[idx] * scale;
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
 * @brief Main function to demonstrate and verify tensor-wise quantization.
 *
 * This function performs the following steps:
 * 1. Generates a random FP32 host tensor.
 * 2. Calculates a single, tensor-wise `scale` factor from the maximum absolute
 *    value in the entire tensor.
 * 3. Allocates GPU memory and copies the input data to the device.
 * 4. Launches the `quantize_tensorwise` and `dequantize_tensorwise` kernels.
 * 5. Copies the dequantized result back to the host.
 * 6. Computes and prints quantization error metrics (MSE, MAE, Max Error) to
 *    evaluate the precision loss.
 */
int main() {
    const int TENSOR_SIZE = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (TENSOR_SIZE + THREADS - 1) / THREADS;

    srand(time(NULL));

    // --- 1. Host Data Generation ---
    float* h_tensor = (float*)malloc(TENSOR_SIZE * sizeof(float));
    float* h_output = (float*)malloc(TENSOR_SIZE * sizeof(float));

    // Find the maximum absolute value in the entire tensor to determine the scale.
    float tensor_max_abs = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        h_tensor[i] = rand_normal(0.0f, 2.0f);
        tensor_max_abs = fmaxf(tensor_max_abs, fabsf(h_tensor[i]));
    }

    // --- 2. Scale Calculation ---
    // A single scale is computed for the whole tensor.
    float scale = tensor_max_abs / 127.0f;

    printf("Tensor-wise Quantization Test\n");
    printf("Tensor size: %d elements\n", TENSOR_SIZE);
    printf("Single scale: %f (applied to entire tensor)\n", scale);
    printf("Memory reduction: 4x (FP32 -> INT8)\n\n");

    // --- 3. Device Memory Allocation and Transfer ---
    float *d_tensor, *d_output;
    signed char *d_quantized;

    cudaMalloc(&d_tensor, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_quantized, TENSOR_SIZE * sizeof(signed char));

    cudaMemcpy(d_tensor, h_tensor, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Execution ---
    quantize_tensorwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, scale, TENSOR_SIZE);
    cudaDeviceSynchronize();

    dequantize_tensorwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, scale, TENSOR_SIZE);
    cudaDeviceSynchronize();

    // --- 5. Result Verification ---
    cudaMemcpy(h_output, d_output, TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate error metrics to measure precision loss.
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

    // --- 6. Cleanup ---
    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_quantized);
    free(h_tensor);
    free(h_output);

    return 0;
}
