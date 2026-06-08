#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file dynamic_static.cu
 * @brief Compares dynamic and static quantization strategies in CUDA.
 *
 * This file demonstrates two fundamental approaches to determining quantization parameters
 * (specifically, the scaling factor) for activations in a neural network.
 *
 * 1. Static Quantization:
 *    - The scaling factor is determined offline using a representative calibration dataset.
 *    - The same, fixed scale is used for all inferences.
 *    - Pro: Fast, no runtime overhead to compute the scale.
 *    - Con: Can suffer from low accuracy if the runtime data distribution differs
 *           significantly from the calibration data (e.g., due to outliers or domain shift).
 *
 * 2. Dynamic Quantization:
 *    - The scaling factor is calculated on-the-fly for each batch of activations.
 *    - This involves finding the `max(abs(value))` of the current batch at runtime.
 *    - Pro: Highly accurate, as the scale is perfectly adapted to the current data's range.
 *    - Con: Incurs a performance penalty due to the runtime computation of the scale,
 *           which typically involves a reduction operation.
 *
 * This example shows that dynamic quantization maintains high accuracy even when data
 * distributions change, whereas static quantization's accuracy degrades.
 */

/**
 * @brief CUDA kernel for static quantization from FP32 to INT8.
 *
 * Each thread quantizes one element of the input tensor using a pre-computed,
 * fixed scaling factor (`static_scale`).
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 tensor on the device.
 * @param static_scale The fixed scaling factor determined offline.
 * @param size The number of elements in the tensor.
 */
__global__ void quantize_static(float* input, int8_t* output, float static_scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Quantize using the provided static scale.
    float scaled = input[idx] / static_scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

/**
 * @brief CUDA kernel for dynamic quantization from FP32 to INT8.
 *
 * This kernel first computes a scaling factor dynamically for the input batch and then
 * uses it to quantize the data. The scale is `max(abs(input)) / 127.0`.
 * The max value is found via a parallel reduction within the CUDA block.
 *
 * Note: This implementation assumes the entire tensor fits within a single block's
 * processing capability for the reduction. A multi-block version would require a
 * more complex, two-level reduction.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 tensor on the device.
 * @param size The number of elements in the tensor.
 */
__global__ void quantize_dynamic(float* input, int8_t* output, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Part 1: Dynamic Scale Calculation (Intra-Block Reduction) ---
    // Each thread loads the absolute value of one element into shared memory.
    float val = (idx < size) ? fabsf(input[idx]) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Perform a parallel reduction to find the maximum absolute value in the block.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The first thread calculates the dynamic scale for this batch.
    float dynamic_scale = sdata[0] / 127.0f;

    // --- Part 2: Quantization using the Dynamic Scale ---
    if (idx < size) {
        // All threads use the same dynamic_scale computed for this block.
        float scaled = input[idx] / dynamic_scale;
        scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
        output[idx] = (int8_t)roundf(scaled);
    }
}

/**
 * @brief CUDA kernel to dequantize an INT8 tensor back to FP32.
 *
 * This is a generic dequantization kernel that can be used with both static and
 * dynamic quantization, as it simply takes a scaling factor as input.
 *
 * @param input Pointer to the input INT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param scale The scaling factor used for dequantization.
 * @param size The number of elements in the tensor.
 */
__global__ void dequantize(int8_t* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    output[idx] = (float)input[idx] * scale;
}

/**
 * @brief Generates a normally distributed random number using the Box-Muller transform.
 */
float rand_normal(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.141592653589793f * u2);
    return z0 * std + mean;
}

/**
 * @brief Main function to compare static and dynamic quantization.
 *
 * This function performs the following steps:
 * 1. Generates multiple batches of data, each with a different standard deviation,
 *    to simulate a changing data distribution.
 * 2. Determines a single `static_scale` based only on the first batch, simulating a
 *    calibration process.
 * 3. For each batch:
 *    a. Performs static quantization using the fixed `static_scale`.
 *    b. Performs dynamic quantization, where the scale is computed on-the-fly.
 *    c. Dequantizes the results from both methods.
 *    d. Computes and compares the Mean Squared Error (MSE) for both, showing that
 *       dynamic quantization adapts better to changing data ranges.
 */
int main() {
    const int BATCH_SIZE = 256 * 1024;
    const int NUM_BATCHES = 4;
    const int THREADS = 256;
    const int BLOCKS = (BATCH_SIZE + THREADS - 1) / THREADS;

    // --- 1. Data Generation ---
    // Create multiple batches with different distributions.
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

    // --- 2. Static Scale Calculation ---
    // The static scale is determined only from the first batch (the "calibration" set).
    float static_max_abs = 0.0f;
    for (int i = 0; i < BATCH_SIZE; ++i) {
        static_max_abs = fmaxf(static_max_abs, fabsf(batches[0][i]));
    }
    float static_scale = static_max_abs / 127.0f;

    printf("Dynamic vs Static Quantization Test\n");
    printf("Static scale (from batch 0): %f\n\n", static_scale);

    // --- 3. Process Each Batch ---
    for (int b = 0; b < NUM_BATCHES; ++b) {
        printf("Batch %d:\n", b);

        float* h_input = batches[b];

        // Allocate device memory for the current batch.
        float *d_input, *d_static_output, *d_dynamic_output;
        int8_t *d_static_quantized, *d_dynamic_quantized;

        cudaMalloc(&d_input, BATCH_SIZE * sizeof(float));
        cudaMalloc(&d_static_output, BATCH_SIZE * sizeof(float));
        cudaMalloc(&d_dynamic_output, BATCH_SIZE * sizeof(float));
        cudaMalloc(&d_static_quantized, BATCH_SIZE * sizeof(int8_t));
        cudaMalloc(&d_dynamic_quantized, BATCH_SIZE * sizeof(int8_t));

        cudaMemcpy(d_input, h_input, BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        // --- Static Quantization Path ---
        quantize_static<<<BLOCKS, THREADS>>>(d_input, d_static_quantized, static_scale, BATCH_SIZE);
        dequantize<<<BLOCKS, THREADS>>>(d_static_quantized, d_static_output, static_scale, BATCH_SIZE);

        // --- Dynamic Quantization Path ---
        // The dynamic quantization kernel computes its own scale internally.
        quantize_dynamic<<<BLOCKS, THREADS, THREADS * sizeof(float)>>>(d_input, d_dynamic_quantized, BATCH_SIZE);

        // For dequantization, we need to know the scale that was computed dynamically.
        // Here, we re-compute it on the CPU for simplicity. In a real application,
        // the dynamic scale would be an output of the quantization kernel/step.
        float dynamic_max_abs = 0.0f;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            dynamic_max_abs = fmaxf(dynamic_max_abs, fabsf(h_input[i]));
        }
        float dynamic_scale = dynamic_max_abs / 127.0f;
        dequantize<<<BLOCKS, THREADS>>>(d_dynamic_quantized, d_dynamic_output, dynamic_scale, BATCH_SIZE);

        cudaDeviceSynchronize();

        // --- 4. Verification and Error Calculation ---
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

        // --- 5. Cleanup for the batch ---
        cudaFree(d_input);
        cudaFree(d_static_output);
        cudaFree(d_dynamic_output);
        cudaFree(d_static_quantized);
        cudaFree(d_dynamic_quantized);
        free(h_static_output);
        free(h_dynamic_output);
    }

    // --- 6. Final Cleanup ---
    for (int b = 0; b < NUM_BATCHES; ++b) {
        free(batches[b]);
    }
    free(batches);

    printf("Key Insight:\n");
    printf("- Static: Fast but uses fixed scale from calibration data\n");
    printf("- Dynamic: More accurate but slower (computes scale per batch)\n");

    return 0;
}
