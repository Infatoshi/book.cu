#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file blockwise.cu
 * @brief Demonstrates block-wise quantization for CUDA.
 *
 * Block-wise quantization is a technique that strikes a balance between tensor-wise
 * and channel-wise (or group-wise) quantization. It divides a tensor into smaller,
 * equally-sized blocks and computes a unique scaling factor for each block. This
 * approach is particularly effective for data where value distributions vary
 * spatially, such as in images or intermediate feature maps in convolutional
 * neural networks.
 *
 * The main benefit is improved accuracy over tensor-wise quantization by adapting to
 * local data ranges, without the high memory overhead of storing a scale for every
 * small group of values.
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
 * @brief CUDA kernel to perform block-wise quantization from FP32 to INT8.
 *
 * Each thread processes one element of the input tensor. It identifies the block
 * the element belongs to, fetches the corresponding block-specific scaling factor,
 * and uses it to quantize the FP32 value to an INT8 value.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 tensor on the device.
 * @param block_scales Pointer to the array of FP32 scaling factors for each block.
 * @param block_height The height of each quantization block.
 * @param block_width The width of each quantization block.
 * @param tensor_height The height of the entire tensor.
 * @param tensor_width The width of the entire tensor.
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void quantize_blockwise(float* input, int8_t* output,
                                   float* block_scales, int block_height, int block_width,
                                   int tensor_height, int tensor_width, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Calculate the 2D coordinates of the current element.
    int row = idx / tensor_width;
    int col = idx % tensor_width;

    // Determine the block index for the current element.
    int block_row = row / block_height;
    int block_col = col / block_width;
    int blocks_per_row = (tensor_width + block_width - 1) / block_width;
    int block_idx = block_row * blocks_per_row + block_col;

    // Fetch the scale for this block.
    float scale = block_scales[block_idx];

    // Quantize the value: scale, clamp, and round to nearest integer.
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

/**
 * @brief CUDA kernel to perform block-wise dequantization from INT8 to FP32.
 *
 * Each thread processes one element of the quantized input tensor. It identifies the
 * block the element belongs to, fetches the corresponding block-specific scaling
 * factor, and uses it to dequantize the INT8 value back to an FP32 value.
 *
 * @param input Pointer to the input INT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param block_scales Pointer to the array of FP32 scaling factors for each block.
 * @param block_height The height of each quantization block.
 * @param block_width The width of each quantization block.
 * @param tensor_height The height of the entire tensor.
 * @param tensor_width The width of the entire tensor.
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void dequantize_blockwise(int8_t* input, float* output,
                                     float* block_scales, int block_height, int block_width,
                                     int tensor_height, int tensor_width, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Calculate the 2D coordinates of the current element.
    int row = idx / tensor_width;
    int col = idx % tensor_width;

    // Determine the block index for the current element.
    int block_row = row / block_height;
    int block_col = col / block_width;
    int blocks_per_row = (tensor_width + block_width - 1) / block_width;
    int block_idx = block_row * blocks_per_row + block_col;

    // Fetch the scale for this block and dequantize.
    float scale = block_scales[block_idx];
    output[idx] = (float)input[idx] * scale;
}

/**
 * @brief Main function to set up and run the block-wise quantization demo.
 *
 * This function performs the following steps:
 * 1. Initializes tensor and block dimensions.
 * 2. Creates a synthetic FP32 tensor on the host with a spatially varying distribution.
 * 3. Calculates the per-block scaling factors on the host. The scale for each block
 *    is `max(abs(value_in_block)) / 127.0`.
 * 4. Allocates memory on the GPU and copies the tensor and scales.
 * 5. Launches the quantization and dequantization kernels.
 * 6. Copies the dequantized result back to the host.
 * 7. Computes and prints quantization error metrics (MSE, MAE, Max Error).
 */
int main() {
    const int TENSOR_HEIGHT = 256;
    const int TENSOR_WIDTH = 256;
    const int TENSOR_SIZE = TENSOR_HEIGHT * TENSOR_WIDTH;

    const int BLOCK_HEIGHT = 32;
    const int BLOCK_WIDTH = 32;
    const int BLOCKS_PER_ROW = (TENSOR_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    const int BLOCKS_PER_COL = (TENSOR_HEIGHT + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    const int NUM_BLOCKS = BLOCKS_PER_ROW * BLOCKS_PER_COL;

    const int THREADS = 256;
    const int BLOCKS = (TENSOR_SIZE + THREADS - 1) / THREADS;

    printf("Block-wise Quantization Test\n");
    printf("Tensor size: %dx%d = %d elements\n", TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_SIZE);
    printf("Block size: %dx%d\n", BLOCK_HEIGHT, BLOCK_WIDTH);
    printf("Number of blocks: %d (%dx%d)\n\n", NUM_BLOCKS, BLOCKS_PER_COL, BLOCKS_PER_ROW);

    // --- 1. Host-side Tensor Initialization ---
    // Create a tensor with a non-uniform distribution to highlight the benefits
    // of block-wise quantization. The standard deviation is higher near the center.
    float* h_tensor = (float*)malloc(TENSOR_SIZE * sizeof(float));
    srand(time(NULL));

    for (int row = 0; row < TENSOR_HEIGHT; ++row) {
        for (int col = 0; col < TENSOR_WIDTH; ++col) {
            int idx = row * TENSOR_WIDTH + col;

            
            float center_row = row - TENSOR_HEIGHT / 2.0f;
            float center_col = col - TENSOR_WIDTH / 2.0f;
            float distance_from_center = sqrtf(center_row * center_row + center_col * center_col);

            
            float local_std = 1.0f + 2.0f * expf(-distance_from_center / 100.0f);

            h_tensor[idx] = rand_normal(0.0f, local_std);
        }
    }

    // --- 2. Host-side Scale Calculation ---
    // For each block, find the maximum absolute value and compute the scaling factor.
    // The scaling factor maps the block's maximum value to the INT8 range limit (127).
    float* h_block_scales = (float*)malloc(NUM_BLOCKS * sizeof(float));
    printf("Block scales (showing corner blocks):\n");
    for (int block_row = 0; block_row < BLOCKS_PER_COL; ++block_row) {
        for (int block_col = 0; block_col < BLOCKS_PER_ROW; ++block_col) {
            int block_idx = block_row * BLOCKS_PER_ROW + block_col;

            float block_max_abs = 0.0f;
            int start_row = block_row * BLOCK_HEIGHT;
            int end_row = fminf(start_row + BLOCK_HEIGHT, TENSOR_HEIGHT);
            int start_col = block_col * BLOCK_WIDTH;
            int end_col = fminf(start_col + BLOCK_WIDTH, TENSOR_WIDTH);

            for (int r = start_row; r < end_row; ++r) {
                for (int c = start_col; c < end_col; ++c) {
                    int idx = r * TENSOR_WIDTH + c;
                    block_max_abs = fmaxf(block_max_abs, fabsf(h_tensor[idx]));
                }
            }

            // The scale is the value that maps the max absolute value to 127.
            h_block_scales[block_idx] = block_max_abs / 127.0f;

            // Print scales for corner blocks for verification.
            if ((block_row == 0 || block_row == BLOCKS_PER_COL - 1) &&
                (block_col == 0 || block_col == BLOCKS_PER_ROW - 1)) {
                printf("  Block [%d,%d]: %f\n", block_row, block_col, h_block_scales[block_idx]);
            }
        }
    }

    // --- 3. Device Memory Allocation and Data Transfer ---
    float *d_tensor, *d_output, *d_block_scales;
    int8_t *d_quantized;

    cudaMalloc(&d_tensor, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_output, TENSOR_SIZE * sizeof(float));
    cudaMalloc(&d_block_scales, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_quantized, TENSOR_SIZE * sizeof(int8_t));

    // Copy host data to device.
    cudaMemcpy(d_tensor, h_tensor, TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_scales, h_block_scales, NUM_BLOCKS * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Execution ---
    // Launch the quantization kernel.
    quantize_blockwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, d_block_scales,
        BLOCK_HEIGHT, BLOCK_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_SIZE);
    cudaDeviceSynchronize();

    // Launch the dequantization kernel.
    dequantize_blockwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, d_block_scales,
        BLOCK_HEIGHT, BLOCK_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_SIZE);
    cudaDeviceSynchronize();

    // --- 5. Result Verification ---
    // Copy the dequantized output back to the host.
    float* h_output = (float*)malloc(TENSOR_SIZE * sizeof(float));
    cudaMemcpy(h_output, d_output, TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate error metrics.
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

    printf("\nKey Properties of Block-wise Quantization:\n");
    printf("- Different scale for each %dx%d block\n", BLOCK_HEIGHT, BLOCK_WIDTH);
    printf("- Ideal for spatially organized data (images, feature maps)\n");
    printf("- %d different scales for adaptive quantization\n", NUM_BLOCKS);
    printf("- Balances local accuracy with parameter overhead\n");
    printf("- Useful when different regions have different value ranges\n");

    // --- 6. Cleanup ---
    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_block_scales);
    cudaFree(d_quantized);

    free(h_tensor);
    free(h_block_scales);
    free(h_output);

    return 0;
}
