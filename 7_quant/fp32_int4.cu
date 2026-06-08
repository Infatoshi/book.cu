#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file fp32_int4.cu
 * @brief Demonstrates FP32 to INT4 quantization and dequantization with bit-packing.
 *
 * INT4 quantization offers a significant memory reduction (8x compared to FP32), making
 * it attractive for deploying large models on memory-constrained devices. A key
 * implementation detail is "packing," where two 4-bit integers are stored in a single
 * 8-bit byte (`uint8_t`) to fully realize the memory savings.
 *
 * This file contains two kernels:
 * 1. `quantize_fp32_to_int4_packed`: Converts FP32 values to INT4 and packs two INT4
 *    values into one `uint8_t`.
 * 2. `dequantize_int4_to_fp32_packed`: Unpacks the `uint8_t` back into two INT4 values
 *    and dequantizes them to FP32.
 *
 * A crucial step in dequantization is sign extension. Since INT4 can represent negative
 * numbers (e.g., in the range [-8, 7] or [-7, 7]), the 4-bit value must be correctly
 * converted to a wider signed type (like `int8_t` or `float`) to preserve its sign.
 */

/**
 * @brief CUDA kernel to quantize an FP32 tensor to a packed INT4 tensor.
 *
 * Each thread processes two FP32 elements and packs the resulting two INT4 values
 * into a single `uint8_t`. The quantization is symmetric, mapping floating-point
 * values to the signed integer range [-7, 7].
 *
 * The quantization formula for each element is:
 * `q = clamp(round(x / scale), -7, 7)`
 *
 * The packing logic places the first quantized value in the most significant 4 bits
 * and the second in the least significant 4 bits of a byte.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output packed `uint8_t` tensor on the device. Its size
 *               is `ceil(size / 2.0)`.
 * @param scale The symmetric quantization scaling factor, calculated as `max(abs(input)) / 7.0f`.
 * @param size The number of elements in the input FP32 tensor.
 */
__global__ void quantize_fp32_to_int4_packed(float* input, uint8_t* output, float scale, int size) {
    // Each thread is responsible for processing two FP32 elements and packing them.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size + 1) / 2) return;

    // Indices of the two FP32 elements to be quantized and packed.
    int elem1_idx = idx * 2;
    int elem2_idx = idx * 2 + 1;

    // --- Quantize the first element ---
    float scaled1 = input[elem1_idx] / scale;
    scaled1 = fmaxf(fminf(scaled1, 7.0f), -7.0f);  // Clamp to the INT4 range [-7, 7].
    int8_t quant1 = (int8_t)roundf(scaled1);

    // --- Quantize the second element ---
    // Handle the case where the tensor has an odd number of elements.
    int8_t quant2 = 0;
    if (elem2_idx < size) {
        float scaled2 = input[elem2_idx] / scale;
        scaled2 = fmaxf(fminf(scaled2, 7.0f), -7.0f);
        quant2 = (int8_t)roundf(scaled2);
    }

    // --- Pack the two 4-bit values into a single byte ---
    // `quant1` is stored in the upper 4 bits, `quant2` in the lower 4 bits.
    // The `& 0x0F` mask ensures we only take the lower 4 bits of the `int8_t` values.
    uint8_t packed = ((quant1 & 0x0F) << 4) | (quant2 & 0x0F);
    output[idx] = packed;
}

/**
 * @brief CUDA kernel to dequantize a packed INT4 tensor back to FP32.
 *
 * Each thread unpacks one `uint8_t` to produce two INT4 values, and then dequantizes
 * them back to FP32. A critical step is sign extension, which correctly interprets
 * the most significant bit of the 4-bit value as a sign bit.
 *
 * @param input Pointer to the input packed `uint8_t` tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param scale The scaling factor used during quantization.
 * @param size The total number of FP32 elements to be produced.
 */
__global__ void dequantize_int4_to_fp32_packed(uint8_t* input, float* output, float scale, int size) {
    // Each thread unpacks one byte to produce two FP32 values.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size + 1) / 2) return;

    // Indices for the two output FP32 elements.
    int elem1_idx = idx * 2;
    int elem2_idx = idx * 2 + 1;

    uint8_t packed = input[idx];
    
    // --- Unpack and sign-extend the two 4-bit values ---
    // Extract the upper 4 bits for the first value.
    int8_t quant1 = (packed >> 4);
    // Extract the lower 4 bits for the second value.
    int8_t quant2 = packed & 0x0F;

    // Sign extension: If the 4th bit (sign bit) is 1, the upper 4 bits of the `int8_t`
    // must be set to 1 to preserve the negative value.
    if (quant1 & 0x08) {
        quant1 |= 0xF0; // e.g., 0b1001 -> 0b11111001 (which is -7)
    }
    if (quant2 & 0x08) {
        quant2 |= 0xF0;
    }

    // --- Dequantize and store the results ---
    output[elem1_idx] = (float)quant1 * scale;
    if (elem2_idx < size) {
        output[elem2_idx] = (float)quant2 * scale;
    }
}

/**
 * @brief Main function to demonstrate and verify FP32 <-> INT4 conversion.
 *
 * This function performs the following steps:
 * 1. Generates a random FP32 host tensor.
 * 2. Calculates the symmetric quantization scale: `scale = max(abs(data)) / 7.0f`.
 * 3. Allocates memory on the GPU for the input, packed quantized, and output tensors.
 * 4. Copies the host data to the GPU.
 * 5. Launches the `quantize_fp32_to_int4_packed` kernel.
 * 6. Launches the `dequantize_int4_to_fp32_packed` kernel.
 * 7. Copies the dequantized result back to the host.
 * 8. Computes and prints quantization error metrics (MSE, MAE, Max Error).
 */
int main() {
    const int SIZE = 1024 * 1024;
    const int THREADS = 256;
    const int BLOCKS = (SIZE + THREADS - 1) / THREADS;

    // --- 1. Data Generation and Scale Calculation ---
    float* h_input = (float*)malloc(SIZE * sizeof(float));
    srand(time(NULL));

    float max_abs = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        max_abs = fmaxf(max_abs, fabsf(h_input[i]));
    }

    // For symmetric INT4 quantization, the scale is determined by the max absolute
    // value mapped to the edge of the integer range (7 in this case).
    float scale = max_abs / 7.0f;

    printf("FP32 -> INT4 Quantization Test\n");
    printf("Scale: %f\n", scale);
    printf("Memory reduction: 8x (FP32 -> INT4)\n");
    printf("Packing: 2 INT4 values per byte\n\n");

    // --- 2. Device Memory Allocation and Data Transfer ---
    float *d_input, *d_output;
    uint8_t *d_quantized;

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    // The quantized output buffer is half the size of the input, plus one byte
    // if the size is odd (though integer division handles this implicitly).
    cudaMalloc(&d_quantized, (SIZE + 1) / 2 * sizeof(uint8_t));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- 3. Kernel Execution ---
    quantize_fp32_to_int4_packed<<<BLOCKS, THREADS>>>(d_input, d_quantized, scale, SIZE);
    cudaDeviceSynchronize();

    dequantize_int4_to_fp32_packed<<<BLOCKS, THREADS>>>(d_quantized, d_output, scale, SIZE);
    cudaDeviceSynchronize();

    // --- 4. Result Verification ---
    float* h_output = (float*)malloc(SIZE * sizeof(float));
    cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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

    // --- 5. Cleanup ---
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_quantized);
    free(h_input);
    free(h_output);

    return 0;
}
