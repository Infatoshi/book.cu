#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/**
 * @file channelwise.cu
 * @brief Demonstrates channel-wise quantization for CUDA.
 *
 * Channel-wise (or per-channel) quantization is a common technique used in deep
 * learning to improve accuracy over simpler tensor-wise methods. Instead of using a
 * single scaling factor for an entire tensor, it computes a unique scaling factor for
 * each channel (typically the last dimension of a tensor).
 *
 * This is particularly effective for tensors like weights in convolutional or linear
 * layers, where different output channels can have vastly different value ranges. By
 * adapting the quantization range to each channel, it preserves more precision and
 * leads to better model performance.
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
 * @brief CUDA kernel to perform channel-wise quantization from FP32 to INT8.
 *
 * Each thread processes one element of the input tensor. It determines the channel
 * index of the element (using the modulo operator on the element's linear index),
 * fetches the corresponding per-channel scaling factor, and quantizes the value.
 *
 * @param input Pointer to the input FP32 tensor on the device.
 * @param output Pointer to the output INT8 tensor on the device.
 * @param channel_scales Pointer to the array of FP32 scaling factors for each channel.
 * @param num_channels The total number of channels (size of the last dimension).
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void quantize_channelwise(float* input, int8_t* output,
                                     float* channel_scales, int num_channels, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Determine the channel index for the current element.
    // This assumes a standard data layout like (B, T, C) where C is the channel dimension.
    int channel_idx = idx % num_channels;
    float scale = channel_scales[channel_idx];

    // Quantize the value: scale, clamp to INT8 range, and round.
    float scaled = input[idx] / scale;
    scaled = fmaxf(fminf(scaled, 127.0f), -127.0f);
    output[idx] = (int8_t)roundf(scaled);
}

/**
 * @brief CUDA kernel to perform channel-wise dequantization from INT8 to FP32.
 *
 * Each thread processes one element of the quantized INT8 tensor. It determines the
 * channel index, fetches the corresponding per-channel scaling factor, and uses it
 * to dequantize the INT8 value back to its FP32 approximation.
 *
 * @param input Pointer to the input INT8 tensor on the device.
 * @param output Pointer to the output FP32 tensor on the device.
 * @param channel_scales Pointer to the array of FP32 scaling factors for each channel.
 * @param num_channels The total number of channels.
 * @param num_elements Total number of elements in the tensor.
 */
__global__ void dequantize_channelwise(int8_t* input, float* output,
                                       float* channel_scales, int num_channels, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int channel_idx = idx % num_channels;
    float scale = channel_scales[channel_idx];

    output[idx] = (float)input[idx] * scale;
}

/**
 * @brief Main function to set up and run the channel-wise quantization demo.
 *
 * This function performs the following steps:
 * 1. Initializes a 3D tensor (Batch, Time, Channels).
 * 2. Creates synthetic FP32 data where each channel has a different standard deviation
 *    to simulate varying dynamic ranges.
 * 3. Calculates the per-channel scaling factors. For each channel, the scale is
 *    `max(abs(value_in_channel)) / 127.0`.
 * 4. Allocates GPU memory and copies the data and scales.
 * 5. Launches the `quantize_channelwise` and `dequantize_channelwise` kernels.
 * 6. Copies the result back to the host and computes error metrics (MSE, MAE).
 * 7. Calculates and prints per-channel MSE to show accuracy at a finer grain.
 */
int main() {
    
    const int BATCH_SIZE = 4;
    const int SEQ_LEN = 64;
    const int CHANNELS = 128;
    const int TOTAL_ELEMENTS = BATCH_SIZE * SEQ_LEN * CHANNELS;

    const int THREADS = 256;
    const int BLOCKS = (TOTAL_ELEMENTS + THREADS - 1) / THREADS;

    printf("Channel-wise Quantization Test\n");
    printf("Tensor shape: [%d, %d, %d] (B x T x C)\n", BATCH_SIZE, SEQ_LEN, CHANNELS);
    printf("Total elements: %d\n", TOTAL_ELEMENTS);
    printf("Different scale for each of %d channels\n\n", CHANNELS);

    // --- 1. Host-side Tensor Initialization ---
    // Create a tensor where each channel has a different distribution to
    // demonstrate the effectiveness of per-channel scaling.
    float* h_tensor = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));
    srand(time(NULL));

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int t = 0; t < SEQ_LEN; ++t) {
            for (int c = 0; c < CHANNELS; ++c) {
                int idx = (b * SEQ_LEN * CHANNELS) + (t * CHANNELS) + c;

                // The standard deviation of the random data varies with the channel index.
                float channel_scale = 0.5f + 1.5f * (float)c / CHANNELS;

                h_tensor[idx] = rand_normal(0.0f, channel_scale);
            }
        }
    }

    // --- 2. Host-side Scale Calculation ---
    // For each channel, find the maximum absolute value across the batch and time dimensions.
    // This value is used to compute the channel's specific scaling factor.
    float* h_channel_scales = (float*)malloc(CHANNELS * sizeof(float));
    printf("Channel scales (showing range):\n");

    float min_scale = INFINITY, max_scale = 0.0f;
    for (int c = 0; c < CHANNELS; ++c) {
        float channel_max_abs = 0.0f;

        // Iterate over all elements belonging to the current channel `c`.
        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int t = 0; t < SEQ_LEN; ++t) {
                int idx = (b * SEQ_LEN * CHANNELS) + (t * CHANNELS) + c;
                channel_max_abs = fmaxf(channel_max_abs, fabsf(h_tensor[idx]));
            }
        }

        // The scale maps the max absolute value in the channel to the INT8 limit (127).
        h_channel_scales[c] = channel_max_abs / 127.0f;
        min_scale = fminf(min_scale, h_channel_scales[c]);
        max_scale = fmaxf(max_scale, h_channel_scales[c]);
    }

    printf("  Scale range: %f to %f\n", min_scale, max_scale);
    printf("  Average scale: %f\n\n", (min_scale + max_scale) / 2.0f);

    // --- 3. Device Memory Allocation and Data Transfer ---
    float *d_tensor, *d_output, *d_channel_scales;
    int8_t *d_quantized;

    cudaMalloc(&d_tensor, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_channel_scales, CHANNELS * sizeof(float));
    cudaMalloc(&d_quantized, TOTAL_ELEMENTS * sizeof(int8_t));

    // Copy host data to device.
    cudaMemcpy(d_tensor, h_tensor, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channel_scales, h_channel_scales, CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Execution ---
    // Launch the quantization and dequantization kernels.
    quantize_channelwise<<<BLOCKS, THREADS>>>(d_tensor, d_quantized, d_channel_scales, CHANNELS, TOTAL_ELEMENTS);
    cudaDeviceSynchronize();

    dequantize_channelwise<<<BLOCKS, THREADS>>>(d_quantized, d_output, d_channel_scales, CHANNELS, TOTAL_ELEMENTS);
    cudaDeviceSynchronize();

    // --- 5. Result Verification ---
    // Copy the dequantized output back to the host.
    float* h_output = (float*)malloc(TOTAL_ELEMENTS * sizeof(float));
    cudaMemcpy(h_output, d_output, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate overall error metrics.
    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
        float error = h_tensor[i] - h_output[i];
        mse += error * error;
        mae += fabsf(error);
        max_error = fmaxf(max_error, fabsf(error));
    }
    mse /= TOTAL_ELEMENTS;
    mae /= TOTAL_ELEMENTS;

    printf("Quantization Accuracy:\n");
    printf("  MSE: %f\n", mse);
    printf("  MAE: %f\n", mae);
    printf("  Max Error: %f\n", max_error);
    printf("  Memory reduction: 4x (FP32 -> INT8)\n");

    // --- 6. Per-Channel Accuracy Analysis ---
    // Calculate MSE for a few individual channels to show fine-grained accuracy.
    printf("\nPer-channel accuracy (first 5 channels):\n");
    for (int c = 0; c < 5; ++c) {
        float channel_mse = 0.0f;
        int channel_elements = BATCH_SIZE * SEQ_LEN;

        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int t = 0; t < SEQ_LEN; ++t) {
                int idx = (b * SEQ_LEN * CHANNELS) + (t * CHANNELS) + c;
                float error = h_tensor[idx] - h_output[idx];
                channel_mse += error * error;
            }
        }
        channel_mse /= channel_elements;
        printf("  Channel %d MSE: %f\n", c, channel_mse);
    }

    printf("\nKey Properties of Channel-wise Quantization:\n");
    printf("- Different scale for each of %d channels\n", CHANNELS);
    printf("- Channels aggregated across batch and time dimensions\n");
    printf("- Essential when different features have different value ranges\n");
    printf("- Works for any tensor where last dimension represents 'channels'\n\n");

    printf("How to quantize across different dimensions:\n");
    printf("- Per-batch: Different scales for each batch item\n");
    printf("- Per-time: Different scales for each time step (sequence position)\n");
    printf("- Per-channel: Different scales for each feature/channel\n");
    printf("- Per-head: Different scales for each attention head\n");
    printf("- Per-layer: Different scales for each transformer layer\n");
    printf("\nChoose based on which dimension has the most variation in value ranges!\n");

    // --- 7. Cleanup ---
    cudaFree(d_tensor);
    cudaFree(d_output);
    cudaFree(d_channel_scales);
    cudaFree(d_quantized);

    free(h_tensor);
    free(h_channel_scales);
    free(h_output);

    return 0;
}
