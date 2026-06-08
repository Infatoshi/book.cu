#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file groupwise.cu
 * @brief Demonstrates group-wise quantization for CUDA.
 *
 * Group-wise quantization is an intermediate strategy between tensor-wise and
 * channel-wise quantization. It involves partitioning a flat tensor into contiguous,
 * equally-sized groups and calculating a unique scaling factor for each group.
 *
 * This approach provides a good trade-off between accuracy and overhead. It's more
 * accurate than using a single scale for the whole tensor, especially when data
 * distributions vary locally. It's also more parameter-efficient than per-channel
 * or per-token quantization if the groups are large. This technique is commonly used
 * in models like GGUF (GPT-Generated Unified Format) for representing weights.
 *
 * The key idea is to adapt the quantization range to local characteristics of the
 * tensor, improving precision where needed without storing an excessive number of
 * scaling factors.
 */






/**
 * @brief Generates a normally distributed random number using the Box-Muller transform.
 * @param mean The mean of the distribution.
 * @param stddev The standard deviation of the distribution.
 * @return A single-precision floating-point random number.
 */
float rand_normal(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * stddev + mean;
}


/**
 * @brief CUDA kernel to perform group-wise quantization from FP32 to INT8.
 *
 * Each thread processes one element of the input tensor. It determines the group
 * index for the element by dividing the element's linear index by the `group_size`.
 * It then fetches the corresponding group-specific scaling factor and uses it to
 * quantize the FP32 value to an INT8 value.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 tensor on the device.
 * @param group_scales Pointer to the array of FP32 scaling factors, one for each group.
 * @param group_size The number of elements in each group.
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void quantize_groupwise(float* input, int8_t* output,
                                   float* group_scales, int group_size, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Determine the group index for the current element.
    int group_idx = idx / group_size;
    float scale = group_scales[group_idx];

    // Quantize the value using the group-specific scale.
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

/**
 * @brief CUDA kernel to perform group-wise dequantization from INT8 to FP32.
 *
 * Each thread processes one quantized INT8 element. It calculates the group index,
 * fetches the corresponding group-specific scaling factor, and dequantizes the
 * value back to FP32.
 *
 * @param input Pointer to the input INT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param group_scales Pointer to the array of FP32 scaling factors for each group.
 * @param group_size The number of elements in each group.
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void dequantize_groupwise(int8_t* input, float* output,
                                     float* group_scales, int group_size, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Determine the group index and fetch the corresponding scale.
    int group_idx = idx / group_size;
    float scale = group_scales[group_idx];

    output[idx] = (float)input[idx] * scale;
}

/**
 * @brief Main function to set up and run the group-wise quantization demo.
 *
 * This function performs the following steps:
 * 1. Defines tensor and group dimensions.
 * 2. Creates a synthetic FP32 tensor where each group has a different standard
 *    deviation, simulating a tensor with locally varying data ranges.
 * 3. Calculates per-group scaling factors on the host. The scale for each group is
 *    `max(abs(value_in_group)) / 127.0`.
 * 4. Allocates memory on the GPU and copies the tensor and scales.
 * 5. Launches the quantization and dequantization kernels.
 * 6. Copies the result back and computes the group-wise quantization error (MSE).
 * 7. Computes the error for a simple tensor-wise quantization scheme on the same data
 *    to provide a baseline for comparison.
 * 8. Prints the results, highlighting the accuracy improvement of group-wise quantization.
 */
int main() {
    const int TENSOR_SIZE = 512 * 1024;
    const int GROUP_SIZE = 4096;
    const int NUM_GROUPS = (TENSOR_SIZE + GROUP_SIZE - 1) / GROUP_SIZE;
    const int THREADS = 256;
    const int BLOCKS = (TENSOR_SIZE + THREADS - 1) / THREADS;

    printf("Group-wise Quantization Test\n");
    printf("Tensor size: %d elements\n", TENSOR_SIZE);
    printf("Group size: %d elements per group\n", GROUP_SIZE);
    printf("Number of groups: %d\n\n", NUM_GROUPS);

    // --- 1. Host-side Tensor Initialization ---
    // Create a tensor where each group has a different data distribution.
    float* h_tensor = (float*)malloc(TENSOR_SIZE * sizeof(float));
    srand(time(NULL));

    for (int g = 0; g < NUM_GROUPS; ++g) {
        // Assign a different standard deviation to each group.
        float group_std = 1.0f + 0.5f * g;

        int start_idx = g * GROUP_SIZE;
        int end_idx = fminf((g + 1) * GROUP_SIZE, TENSOR_SIZE);

        for (int i = start_idx; i < end_idx; ++i) {
            h_tensor[i] = rand_normal(0.0f, group_std);
        }
    }

    // --- 2. Host-side Scale Calculation ---
    // For each group, find the max absolute value and compute its scaling factor.
    float* h_group_scales = (float*)malloc(NUM_GROUPS * sizeof(float));
    printf("Group scales:\n");
    for (int g = 0; g < NUM_GROUPS; ++g) {
        float group_max_abs = 0.0f;
        int start_idx = g * GROUP_SIZE;
        int end_idx = fminf((g + 1) * GROUP_SIZE, TENSOR_SIZE);

        for (int i = start_idx; i < end_idx; ++i) {
            group_max_abs = fmaxf(group_max_abs, fabsf(h_tensor[i]));
        }

        h_group_scales[g] = group_max_abs / 127.0f;

        if (g < 3 || g >= NUM_GROUPS - 3) {
            printf("  Group %d: %f\n", g, h_group_scales[g]);
        } else if (g == 3) {
            printf("  ... (%d more groups) ...\n", (NUM_GROUPS - 6));
        }
    }

    // --- 3. Device Memory Allocation and Data Transfer ---
    float *d_tensor, *d_output, *d_group_scales;
    int8_t *d_quantized;

    cudaMalloc(&d_tensor, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_group_scales, NUM_GROUPS * sizeof(float));
    cudaMalloc(&d_quantized, TENSOR_SIZE * sizeof(int8_t));

    // Copy host data to device.
    cudaMemcpy(d_tensor, h_tensor, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_scales, h_group_scales, NUM_GROUPS * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Execution ---
    // Launch the group-wise quantization and dequantization kernels.
    quantize_groupwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, d_group_scales, GROUP_SIZE, TENSOR_SIZE);
    cudaDeviceSynchronize();

    dequantize_groupwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, d_group_scales, GROUP_SIZE, TENSOR_SIZE);
    cudaDeviceSynchronize();

    // --- 5. Result Verification ---
    // Copy the dequantized output back to the host.
    float* h_output = (float*)malloc(TENSOR_SIZE * sizeof(float));
    cudaMemcpy(h_output, d_output, TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate overall error metrics for group-wise quantization.
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

    // --- 6. Comparison with Tensor-wise Quantization ---
    // Calculate the MSE for a simple tensor-wise scheme to show the improvement.
    float tensor_max_abs = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        tensor_max_abs = fmaxf(tensor_max_abs, fabsf(h_tensor[i]));
    }
    float tensor_scale = tensor_max_abs / 127.0f;

    // Perform tensor-wise quantization and dequantization on the host to get MSE.
    float tensor_mse = 0.0f;
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        float scaled = h_tensor[i] / tensor_scale;
        scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
        int8_t quantized = (int8_t)roundf(scaled);
        float dequantized = (float)quantized * tensor_scale;
        float error = h_tensor[i] - dequantized;
        tensor_mse += error * error;
    }
    tensor_mse /= TENSOR_SIZE;

    printf("\nComparison with Tensor-wise:\n");
    printf("  Tensor-wise MSE: %f\n", tensor_mse);
    printf("  Group-wise MSE: %f\n", mse);
    printf("  Group-wise is %fx more accurate\n\n", (tensor_mse / mse));

    printf("Key Properties of Group-wise Quantization:\n");
    printf("- Different scale for each group of %d elements\n", GROUP_SIZE);
    printf("- Better accuracy than tensor-wise when groups have different ranges\n");
    printf("- Trade-off: %dx more scale parameters to store\n", NUM_GROUPS);
    printf("- Useful for tensors with locally varying value distributions\n");

    // --- 7. Cleanup ---
    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_group_scales);
    cudaFree(d_quantized);

    free(h_tensor);
    free(h_group_scales);
    free(h_output);

    return 0;
}
