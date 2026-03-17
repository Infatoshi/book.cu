/**
 * @file pipeline.cu
 * @brief Demonstrates and compares naive vs. stream-based pipeline parallelism for MLP inference.
 *
 * @section description
 * This program implements a multi-layer perceptron (MLP) inference pipeline distributed
 * across multiple GPUs using pipeline parallelism. It serves as a pedagogical tool to
 * highlight the performance difference between a naive, sequential implementation and an
 * optimized version that uses CUDA streams to overlap computation and data transfers.
 *
 * Pipeline Parallelism Concept:
 * In pipeline parallelism, the model is partitioned vertically, with different layers (or stages)
 * placed on different devices. Data flows through these stages sequentially. To improve
 * throughput, input data is split into micro-batches. While one GPU is processing micro-batch `k`
 * for stage `i`, the next GPU in the pipe can process micro-batch `k-1` for stage `i+1`.
 * This creates an execution pipeline, maximizing GPU utilization by overlapping work.
 *
 * Implementations Compared:
 * 1.  **Naive (Blocking) Pipeline**:
 *     - Uses the default CUDA stream (stream 0).
 *     - All operations (kernel launches, memory copies) are blocking within their stream.
 *     - Inter-GPU transfers are synchronous.
 *     - This leads to significant idle time ("pipeline bubbles") as each GPU must wait
 *       for the previous one to finish its *entire* batch before it can start.
 *
 * 2.  **Stream-based (Async) Pipeline**:
 *     - Utilizes multiple CUDA streams to manage concurrent operations.
 *     - Asynchronous memory copies (`cudaMemcpyAsync`) and kernel launches are used.
 *     - `cudaEvent`s are used to synchronize dependencies between stages (e.g., GPU 1
 *       cannot start processing a micro-batch until GPU 0's computation on it is complete
 *       and the data is transferred).
 *     - This approach allows for the overlapping of computation (on GPU `i`) with data
 *       transfers (between GPU `i-1` and `i`), which hides communication latency and
 *       dramatically improves throughput.
 *
 * @compilation
 * `nvcc -O3 -arch=sm_90 pipeline.cu -o pipeline -lcublas`
 * (Adjust `-arch` for your GPU, e.g., `sm_80` for A100).
 *
 * @usage
 * `./pipeline [num_gpus]`
 * Defaults to 2 GPUs if not specified.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// --- Configuration ---
#define BATCH_SIZE 2048
#define INPUT_DIM 4096
#define HIDDEN_DIM 4096
#define OUTPUT_DIM 1000
#define NUM_STREAMS_PER_GPU 4
#define MAX_GPUS 8
#define NUM_BATCHES 100
#define NUM_WARMUP 10

// --- Error Checking Macros ---

/**
 * @brief Macro to wrap CUDA API calls and check for errors.
 * @param call The CUDA API call to be executed.
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief Macro to wrap cuBLAS API calls and check for errors.
 * @param call The cuBLAS API call to be executed.
 */
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d - %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- CUDA Kernels for MLP Layers ---

/**
 * @brief Applies the ReLU activation function element-wise.
 * @param data Pointer to the data on which to apply ReLU.
 * @param size The total number of elements in the data tensor.
 */
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/**
 * @brief Adds a bias vector to a batch of activations.
 * @param data The input/output data (activations).
 * @param bias The bias vector to add.
 * @param batch_size The number of samples in the batch.
 * @param dim The dimension of the activation/bias vector.
 */
__global__ void bias_add_kernel(float* data, const float* bias, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int feature_idx = idx % dim;
        data[idx] += bias[feature_idx];
    }
}

/**
 * @brief First step of a two-pass softmax: finds the maximum value in each row of the batch.
 * This is a numerically stable approach to softmax.
 * @param input The input data.
 * @param max_vals Output buffer to store the max value for each row.
 * @param batch_size The number of samples in the batch.
 * @param dim The dimension of each sample.
 */
__global__ void softmax_max_kernel(const float* input, float* max_vals, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* row = input + batch_idx * dim;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    float thread_max = -INFINITY;
    for (int idx = tid; idx < dim; idx += blockDim.x) {
        thread_max = fmaxf(thread_max, row[idx]);
    }
    sdata[tid] = thread_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) max_vals[batch_idx] = sdata[0];
}

/**
 * @brief Second step of softmax: computes `exp(x - max)` and sums the results.
 * Subtracting the max value before `exp` prevents overflow for large inputs.
 * @param data The input data (will be overwritten with exponentiated values).
 * @param max_vals The max value for each row, computed in the first step.
 * @param sum_vals Output buffer to store the sum of exponentiated values for each row.
 * @param batch_size The number of samples in the batch.
 * @param dim The dimension of each sample.
 */
__global__ void softmax_exp_sum_kernel(float* data, const float* max_vals, float* sum_vals, 
                                       int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    float* row = data + batch_idx * dim;
    float max_val = max_vals[batch_idx];
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    float thread_sum = 0.0f;
    for (int idx = tid; idx < dim; idx += blockDim.x) {
        float exp_val = expf(row[idx] - max_val);
        row[idx] = exp_val;
        thread_sum += exp_val;
    }
    sdata[tid] = thread_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) sum_vals[batch_idx] = sdata[0];
}

/**
 * @brief Final step of softmax: normalizes each element by the sum of exponentiated values for its row.
 * @param data The exponentiated data (will be overwritten with normalized probabilities).
 * @param sum_vals The sum of exponentiated values for each row.
 * @param batch_size The number of samples in the batch.
 * @param dim The dimension of each sample.
 */
__global__ void softmax_normalize_kernel(float* data, const float* sum_vals, 
                                        int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int batch_idx = idx / dim;
        data[idx] /= sum_vals[batch_idx];
    }
}

/**
 * @struct GPULayer
 * @brief Manages all resources for a single layer of the MLP on a specific GPU.
 *
 * This struct encapsulates all the device pointers, handles, and stream/event objects
 * required for both the naive and the stream-based pipeline implementations.
 */
struct GPULayer {
    int gpu_id;
    int input_dim;
    int output_dim;
    bool has_relu;
    bool has_softmax;
    
    // --- Shared Resources ---
    float* d_weights;
    float* d_bias;
    
    // --- Naive Pipeline Resources ---
    float* d_input_naive;
    float* d_output_naive;
    float* d_max_vals_naive;
    float* d_sum_vals_naive;
    cublasHandle_t cublas_handle_naive;
    
    // --- Stream-based Pipeline Resources ---
    // Arrays of resources, one for each concurrent micro-batch (stream).
    float* d_input[NUM_STREAMS_PER_GPU];
    float* d_output[NUM_STREAMS_PER_GPU];
    float* d_max_vals[NUM_STREAMS_PER_GPU];
    float* d_sum_vals[NUM_STREAMS_PER_GPU];
    cudaStream_t streams[NUM_STREAMS_PER_GPU];
    cudaEvent_t events[NUM_STREAMS_PER_GPU];
    cublasHandle_t cublas_handles[NUM_STREAMS_PER_GPU];
};

// --- Forward Pass Functions ---

/**
 * @brief Performs the linear transformation (GEMM + bias) for a layer using the naive approach.
 * @param layer A pointer to the GPULayer struct.
 */
void forward_linear_naive(GPULayer* layer) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Matrix multiplication: Y = alpha * W * X + beta * Y
    CUBLAS_CHECK(cublasSgemm(layer->cublas_handle_naive, CUBLAS_OP_N, CUBLAS_OP_N,
        layer->output_dim, BATCH_SIZE, layer->input_dim,
        &alpha, layer->d_weights, layer->output_dim,
        layer->d_input_naive, layer->input_dim,
        &beta, layer->d_output_naive, layer->output_dim));
    
    // Add bias vector
    int total = BATCH_SIZE * layer->output_dim;
    bias_add_kernel<<<(total + 255) / 256, 256>>>(
        layer->d_output_naive, layer->d_bias, BATCH_SIZE, layer->output_dim);
}

/**
 * @brief Performs the linear transformation for a layer asynchronously on a specific stream.
 * @param layer A pointer to the GPULayer struct.
 * @param s The stream index to use for this operation.
 */
void forward_linear_stream(GPULayer* layer, int s) {
    const float alpha = 1.0f, beta = 0.0f;
    
    // Associate the cuBLAS call with the correct stream
    CUBLAS_CHECK(cublasSgemm(layer->cublas_handles[s], CUBLAS_OP_N, CUBLAS_OP_N,
        layer->output_dim, BATCH_SIZE, layer->input_dim,
        &alpha, layer->d_weights, layer->output_dim,
        layer->d_input[s], layer->input_dim,
        &beta, layer->d_output[s], layer->output_dim));
    
    // Launch the bias add kernel on the same stream
    int total = BATCH_SIZE * layer->output_dim;
    bias_add_kernel<<<(total + 255) / 256, 256, 0, layer->streams[s]>>>(
        layer->d_output[s], layer->d_bias, BATCH_SIZE, layer->output_dim);
}

/**
 * @brief Applies the ReLU activation for a layer using the naive approach.
 * @param layer A pointer to the GPULayer struct.
 */
void forward_relu_naive(GPULayer* layer) {
    int total = BATCH_SIZE * layer->output_dim;
    relu_kernel<<<(total + 255) / 256, 256>>>(layer->d_output_naive, total);
}

/**
 * @brief Applies the ReLU activation for a layer asynchronously on a specific stream.
 * @param layer A pointer to the GPULayer struct.
 * @param s The stream index to use.
 */
void forward_relu_stream(GPULayer* layer, int s) {
    int total = BATCH_SIZE * layer->output_dim;
    relu_kernel<<<(total + 255) / 256, 256, 0, layer->streams[s]>>>(
        layer->d_output[s], total);
}

/**
 * @brief Performs the three-step softmax activation for a layer using the naive approach.
 * @param layer A pointer to the GPULayer struct.
 */
void forward_softmax_naive(GPULayer* layer) {
    int threads = 256;
    int shared_mem = threads * sizeof(float);
    
    softmax_max_kernel<<<BATCH_SIZE, threads, shared_mem>>>(
        layer->d_output_naive, layer->d_max_vals_naive, BATCH_SIZE, layer->output_dim);
    softmax_exp_sum_kernel<<<BATCH_SIZE, threads, shared_mem>>>(
        layer->d_output_naive, layer->d_max_vals_naive, layer->d_sum_vals_naive,
        BATCH_SIZE, layer->output_dim);
    
    int total = BATCH_SIZE * layer->output_dim;
    softmax_normalize_kernel<<<(total + 255) / 256, 256>>>(
        layer->d_output_naive, layer->d_sum_vals_naive, BATCH_SIZE, layer->output_dim);
}

/**
 * @brief Performs the three-step softmax activation asynchronously on a specific stream.
 * @param layer A pointer to the GPULayer struct.
 * @param s The stream index to use.
 */
void forward_softmax_stream(GPULayer* layer, int s) {
    int threads = 256;
    int shared_mem = threads * sizeof(float);
    
    softmax_max_kernel<<<BATCH_SIZE, threads, shared_mem, layer->streams[s]>>>(
        layer->d_output[s], layer->d_max_vals[s], BATCH_SIZE, layer->output_dim);
    softmax_exp_sum_kernel<<<BATCH_SIZE, threads, shared_mem, layer->streams[s]>>>(
        layer->d_output[s], layer->d_max_vals[s], layer->d_sum_vals[s],
        BATCH_SIZE, layer->output_dim);
    
    int total = BATCH_SIZE * layer->output_dim;
    softmax_normalize_kernel<<<(total + 255) / 256, 256, 0, layer->streams[s]>>>(
        layer->d_output[s], layer->d_sum_vals[s], BATCH_SIZE, layer->output_dim);
}


/**
 * @brief Initializes a GPULayer struct.
 * Allocates all necessary device memory and creates handles/streams/events.
 * Initializes weights with a random distribution.
 * @param layer Pointer to the GPULayer to initialize.
 * @param gpu_id The ID of the GPU this layer will reside on.
 * @param input_dim The input dimension of the layer.
 * @param output_dim The output dimension of the layer.
 * @param has_relu Whether this layer has a ReLU activation.
 * @param has_softmax Whether this layer has a Softmax activation.
 */
void init_layer(GPULayer* layer, int gpu_id, int input_dim, int output_dim,
                bool has_relu, bool has_softmax) {
    layer->gpu_id = gpu_id;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->has_relu = has_relu;
    layer->has_softmax = has_softmax;
    
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    // Allocate shared weights and biases
    CUDA_CHECK(cudaMalloc(&layer->d_weights, output_dim * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias, output_dim * sizeof(float)));
    
    // Allocate memory and create handle for the naive implementation
    CUDA_CHECK(cudaMalloc(&layer->d_input_naive, BATCH_SIZE * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_output_naive, BATCH_SIZE * output_dim * sizeof(float)));
    if (has_softmax) {
        CUDA_CHECK(cudaMalloc(&layer->d_max_vals_naive, BATCH_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer->d_sum_vals_naive, BATCH_SIZE * sizeof(float)));
    }
    CUBLAS_CHECK(cublasCreate(&layer->cublas_handle_naive));
    
    // Allocate memory and create resources for the stream-based implementation
    for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
        CUDA_CHECK(cudaMalloc(&layer->d_input[s], BATCH_SIZE * input_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer->d_output[s], BATCH_SIZE * output_dim * sizeof(float)));
        if (has_softmax) {
            CUDA_CHECK(cudaMalloc(&layer->d_max_vals[s], BATCH_SIZE * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&layer->d_sum_vals[s], BATCH_SIZE * sizeof(float)));
        }
        CUDA_CHECK(cudaStreamCreate(&layer->streams[s]));
        CUDA_CHECK(cudaEventCreate(&layer->events[s]));
        CUBLAS_CHECK(cublasCreate(&layer->cublas_handles[s]));
        CUBLAS_CHECK(cublasSetStream(layer->cublas_handles[s], layer->streams[s]));
    }
    
    // Initialize weights and biases on the host
    float* h_weights = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* h_bias = (float*)malloc(output_dim * sizeof(float));
    
    float scale = sqrtf(2.0f / (input_dim + output_dim));
    for (int i = 0; i < output_dim * input_dim; i++) {
        h_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    for (int i = 0; i < output_dim; i++) {
        h_bias[i] = 0.0f;
    }
    
    // Copy weights and biases to the device
    CUDA_CHECK(cudaMemcpy(layer->d_weights, h_weights,
        output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer->d_bias, h_bias,
        output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_weights);
    free(h_bias);
}

/**
 * @brief Cleans up all resources used by a GPULayer.
 * Frees all device memory and destroys handles, streams, and events.
 * @param layer Pointer to the GPULayer to clean up.
 */
void cleanup_layer(GPULayer* layer) {
    CUDA_CHECK(cudaSetDevice(layer->gpu_id));
    CUDA_CHECK(cudaFree(layer->d_weights));
    CUDA_CHECK(cudaFree(layer->d_bias));
    CUDA_CHECK(cudaFree(layer->d_input_naive));
    CUDA_CHECK(cudaFree(layer->d_output_naive));
    if (layer->d_max_vals_naive) CUDA_CHECK(cudaFree(layer->d_max_vals_naive));
    if (layer->d_sum_vals_naive) CUDA_CHECK(cudaFree(layer->d_sum_vals_naive));
    CUBLAS_CHECK(cublasDestroy(layer->cublas_handle_naive));
    
    for (int s = 0; s < NUM_STREAMS_PER_GPU; s++) {
        CUDA_CHECK(cudaFree(layer->d_input[s]));
        CUDA_CHECK(cudaFree(layer->d_output[s]));
        if (layer->d_max_vals[s]) CUDA_CHECK(cudaFree(layer->d_max_vals[s]));
        if (layer->d_sum_vals[s]) CUDA_CHECK(cudaFree(layer->d_sum_vals[s]));
        CUDA_CHECK(cudaStreamDestroy(layer->streams[s]));
        CUDA_CHECK(cudaEventDestroy(layer->events[s]));
        CUBLAS_CHECK(cublasDestroy(layer->cublas_handles[s]));
    }
}

/**
 * @brief Enables peer-to-peer (P2P) memory access between all specified GPUs.
 * P2P allows a kernel on one GPU to directly access memory on another GPU,
 * which is essential for efficient data transfers in pipeline parallelism.
 * @param num_gpus The number of GPUs to enable P2P access among.
 */
void setup_p2p(int num_gpus) {
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    // Enable P2P access from GPU i to GPU j
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    // It's okay if it's already enabled, but other errors are fatal.
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,
                                cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError(); // Clear the "already enabled" error
                    }
                }
            }
        }
    }
}

/**
 * @brief Processes a single batch through the entire MLP pipeline using the naive, blocking approach.
 *
 * This function demonstrates the inefficiency of a simple pipeline. Operations are
 * serialized: compute on GPU 0, sync, copy to GPU 1, sync, compute on GPU 1, etc.
 * This leaves GPUs idle for significant periods.
 *
 * @param layers Array of GPULayer structs, one for each stage/GPU.
 * @param num_gpus The number of GPUs (and stages) in the pipeline.
 * @param h_input Pointer to the input data on the host.
 * @param h_output Pointer to the output buffer on the host.
 */
void process_batch_naive(GPULayer* layers, int num_gpus, float* h_input, float* h_output) {
    // 1. Copy input from Host to GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(layers[0].d_input_naive, h_input,
        BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    
    // 2. Process through each layer in the pipeline
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(layers[i].gpu_id));
        
        // Compute the forward pass for the current layer
        forward_linear_naive(&layers[i]);
        if (layers[i].has_relu) forward_relu_naive(&layers[i]);
        if (layers[i].has_softmax) forward_softmax_naive(&layers[i]);
        
        // --- Synchronization Point (MAJOR BOTTLENECK) ---
        // We must wait for the current GPU to finish before continuing.
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 3. If not the last layer, copy output to the next GPU's input buffer
        if (i < num_gpus - 1) {
            int next_dim = layers[i + 1].input_dim;
            // This is a blocking D2D copy.
            CUDA_CHECK(cudaMemcpy(layers[i + 1].d_input_naive, layers[i].d_output_naive,
                BATCH_SIZE * next_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            // --- Synchronization Point (MAJOR BOTTLENECK) ---
            // Another sync, ensuring the copy is done before the next GPU starts.
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    // 4. Copy final output from the last GPU back to the host
    int last = num_gpus - 1;
    CUDA_CHECK(cudaSetDevice(layers[last].gpu_id));
    CUDA_CHECK(cudaMemcpy(h_output, layers[last].d_output_naive,
        BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
}

/**
 * @brief Processes a single micro-batch through the pipeline using CUDA streams and events.
 *
 * This function is the core of the optimized pipeline. It enqueues all operations
 * for a single micro-batch into a specific stream. Dependencies are managed with events,
 * allowing the CUDA scheduler to overlap operations and hide communication latency.
 *
 * @param layers Array of GPULayer structs.
 * @param num_gpus The number of GPUs in the pipeline.
 * @param batch_id The ID of the current micro-batch, used to select a stream.
 * @param h_input_base Pointer to the start of the pinned host memory for all input micro-batches.
 * @param h_output_base Pointer to the start of the pinned host memory for all output micro-batches.
 */
void process_batch_async(GPULayer* layers, int num_gpus, int batch_id,
                        float* h_input_base, float* h_output_base) {
    // Select a stream based on the batch ID to process this micro-batch
    int s = batch_id % NUM_STREAMS_PER_GPU;
    // Calculate offsets into the large pinned host buffers for this specific micro-batch
    float* h_input = h_input_base + (s * BATCH_SIZE * INPUT_DIM);
    float* h_output = h_output_base + (s * BATCH_SIZE * OUTPUT_DIM);
    
    // 1. Asynchronously copy input from Host to GPU 0 for this micro-batch
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpyAsync(layers[0].d_input[s], h_input,
        BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice,
        layers[0].streams[s]));
    
    // 2. Launch operations for each layer in the pipeline
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(layers[i].gpu_id));
        
        // 3. If not the first layer, wait for the previous layer to finish its work on this micro-batch
        if (i > 0) {
            // Make the current stream (on GPU i) wait for the event recorded by the
            // previous stream (on GPU i-1). This ensures the input data is ready.
            CUDA_CHECK(cudaStreamWaitEvent(layers[i].streams[s],
                layers[i-1].events[s], 0));
            
            // 4. Asynchronously copy the activation from the previous GPU
            int transfer_dim = layers[i].input_dim;
            CUDA_CHECK(cudaMemcpyAsync(layers[i].d_input[s],
                layers[i-1].d_output[s], BATCH_SIZE * transfer_dim * sizeof(float),
                cudaMemcpyDeviceToDevice, layers[i].streams[s]));
        }
        
        // 5. Enqueue the forward pass computation for the current layer on its stream
        forward_linear_stream(&layers[i], s);
        if (layers[i].has_relu) forward_relu_stream(&layers[i], s);
        if (layers[i].has_softmax) forward_softmax_stream(&layers[i], s);
        
        // 6. Record an event on the current stream. This event marks the completion
        // of this layer's work for this micro-batch. The next GPU in the pipeline
        // will wait on this event.
        CUDA_CHECK(cudaEventRecord(layers[i].events[s], layers[i].streams[s]));
    }
    
    // 7. Asynchronously copy the final output from the last GPU back to the host
    int last = num_gpus - 1;
    CUDA_CHECK(cudaSetDevice(layers[last].gpu_id));
    CUDA_CHECK(cudaMemcpyAsync(h_output, layers[last].d_output[s],
        BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost,
        layers[last].streams[s]));
}


/**
 * @brief Main function to set up and run the pipeline parallelism comparison.
 */
int main(int argc, char** argv) {
    int num_gpus = 2; // Default to 2 GPUs for pipeline demo
    if (argc > 1) {
        num_gpus = atoi(argv[1]);
        if (num_gpus < 1 || num_gpus > MAX_GPUS) {
            fprintf(stderr, "Invalid number of GPUs. Must be 1-%d\n", MAX_GPUS);
            return EXIT_FAILURE;
        }
    }
    
    int available_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (num_gpus > available_gpus) {
        fprintf(stderr, "Requested %d GPUs but only %d available\n", num_gpus, available_gpus);
        return EXIT_FAILURE;
    }
    
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   Pipeline Parallel MLP Inference - Performance Comparison   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  GPUs: %d\n", num_gpus);
    printf("  Layers: %d (%dx Linear+ReLU, 1x Linear+Softmax)\n", num_gpus, num_gpus - 1);
    printf("  Input: %d, Hidden: %d, Output: %d\n", INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
    printf("  Batch Size: %d\n", BATCH_SIZE);
    printf("  Total Batches: %d (%d warmup + %d measured)\n\n",
        NUM_BATCHES, NUM_WARMUP, NUM_BATCHES - NUM_WARMUP);
    
    // Enable peer-to-peer access between all GPUs
    setup_p2p(num_gpus);
    
    // --- Initialization ---
    // Allocate and initialize layers, one per GPU
    GPULayer* layers = (GPULayer*)malloc(num_gpus * sizeof(GPULayer));
    for (int i = 0; i < num_gpus; i++) {
        int input_dim = (i == 0) ? INPUT_DIM : HIDDEN_DIM;
        int output_dim = (i == num_gpus - 1) ? OUTPUT_DIM : HIDDEN_DIM;
        bool has_relu = (i < num_gpus - 1);
        bool has_softmax = (i == num_gpus - 1);
        init_layer(&layers[i], i, input_dim, output_dim, has_relu, has_softmax);
    }
    
    // Allocate host memory for inputs and outputs.
    // Use pinned memory (`cudaMallocHost`) for faster async H2D/D2H transfers.
    float *h_input_naive, *h_output_naive, *h_input_stream, *h_output_stream;
    CUDA_CHECK(cudaMallocHost(&h_input_naive, BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output_naive, BATCH_SIZE * OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_input_stream, NUM_STREAMS_PER_GPU * BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output_stream, NUM_STREAMS_PER_GPU * BATCH_SIZE * OUTPUT_DIM * sizeof(float)));
    
    // Generate random input data
    for (int i = 0; i < BATCH_SIZE * INPUT_DIM; i++) {
        h_input_naive[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int i = 0; i < NUM_STREAMS_PER_GPU * BATCH_SIZE * INPUT_DIM; i++) {
        h_input_stream[i] = h_input_naive[i % (BATCH_SIZE * INPUT_DIM)];
    }
    
    
    // --- Naive Pipeline Benchmark ---
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ [1/2] Naive Sequential Pipeline (Blocking Operations)      │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    
    // Warmup runs
    for (int b = 0; b < NUM_WARMUP; b++) {
        process_batch_naive(layers, num_gpus, h_input_naive, h_output_naive);
    }
    // Synchronize all GPUs to ensure warmup is complete before timing.
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timed runs
    auto start_naive = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < NUM_BATCHES - NUM_WARMUP; b++) {
        process_batch_naive(layers, num_gpus, h_input_naive, h_output_naive);
    }
    // Synchronize all GPUs to ensure all work is finished.
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end_naive = std::chrono::high_resolution_clock::now();
    double time_naive = std::chrono::duration<double>(end_naive - start_naive).count();
    
    double throughput_naive = ((NUM_BATCHES - NUM_WARMUP) * BATCH_SIZE) / time_naive;
    
    printf("  ✓ Total Time: %.3f seconds\n", time_naive);
    printf("  ✓ Throughput: %.2f samples/sec (%.2f batches/sec)\n",
        throughput_naive, (NUM_BATCHES - NUM_WARMUP) / time_naive);
    printf("  ✓ Avg Latency: %.3f ms/batch\n\n", (time_naive * 1000.0) / (NUM_BATCHES - NUM_WARMUP));
    
    
    // --- Stream-based Pipeline Benchmark ---
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ [2/2] Pipelined with Streams (Async Operations)            │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    
    // Warmup runs
    for (int b = 0; b < NUM_WARMUP; b++) {
        process_batch_async(layers, num_gpus, b, h_input_stream, h_output_stream);
    }
    // Synchronize all GPUs.
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Timed runs
    auto start_stream = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < NUM_BATCHES - NUM_WARMUP; b++) {
        process_batch_async(layers, num_gpus, b, h_input_stream, h_output_stream);
    }
    // Final synchronization to ensure all enqueued micro-batches are finished.
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end_stream = std::chrono::high_resolution_clock::now();
    double time_stream = std::chrono::duration<double>(end_stream - start_stream).count();
    
    double throughput_stream = ((NUM_BATCHES - NUM_WARMUP) * BATCH_SIZE) / time_stream;
    
    printf("  ✓ Total Time: %.3f seconds\n", time_stream);
    printf("  ✓ Throughput: %.2f samples/sec (%.2f batches/sec)\n",
        throughput_stream, (NUM_BATCHES - NUM_WARMUP) / time_stream);
    printf("  ✓ Avg Latency: %.3f ms/batch\n\n", (time_stream * 1000.0) / (NUM_BATCHES - NUM_WARMUP));
    
    
    // --- Performance Analysis ---
    
    double speedup = throughput_stream / throughput_naive;
    double efficiency = (speedup / num_gpus) * 100.0;
    
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Performance Summary                       ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Speedup:     %.2fx                                           \n", speedup);
    printf("║  Efficiency:  %.1f%% (%.2fx / %d GPUs)                        \n", efficiency, speedup, num_gpus);
    printf("║  Time Saved:  %.1f%%                                          \n", (1.0 - time_stream/time_naive) * 100.0);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    // --- Validation ---
    // Compare the output of both implementations to ensure correctness.
    printf("Validating Correctness:\n");
    double sum_naive = 0.0, sum_stream = 0.0;
    double max_diff = 0.0;
    for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; i++) {
        sum_naive += h_output_naive[i];
        sum_stream += h_output_stream[i % (BATCH_SIZE * OUTPUT_DIM)];
        double diff = fabs(h_output_naive[i] - h_output_stream[i % (BATCH_SIZE * OUTPUT_DIM)]);
        if (diff > max_diff) max_diff = diff;
    }
    double mean_naive = sum_naive / (BATCH_SIZE * OUTPUT_DIM);
    double mean_stream = sum_stream / (BATCH_SIZE * OUTPUT_DIM);
    
    printf("  Naive output mean:  %.6f\n", mean_naive);
    printf("  Stream output mean: %.6f\n", mean_stream);
    printf("  Max difference:     %.2e\n", max_diff);
    
    if (max_diff < 1e-4) {
        printf("  ✓ Results match (within tolerance)\n\n");
    } else {
        printf("  ⚠ Results differ significantly!\n\n");
    }
    
    // --- Cleanup ---
    for (int i = 0; i < num_gpus; i++) {
        cleanup_layer(&layers[i]);
    }
    free(layers);
    CUDA_CHECK(cudaFreeHost(h_input_naive));
    CUDA_CHECK(cudaFreeHost(h_output_naive));
    CUDA_CHECK(cudaFreeHost(h_input_stream));
    CUDA_CHECK(cudaFreeHost(h_output_stream));
    
    return EXIT_SUCCESS;
}
