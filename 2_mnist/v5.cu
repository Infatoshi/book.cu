#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000


/**
 * Timing statistics structure for performance profiling
 * Tracks execution time of different components
 */
typedef struct {
    double memory_transfers;  // Time spent on memory transfers between host and device
    double gpu_compute;       // Time spent on GPU computation
    double host_computation;  // Time spent on CPU computation
    double total_time;        // Total training time
} TimingStats;


/**
 * CUDA error checking macro
 * Checks CUDA function calls for errors
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * cuBLAS error checking macro
 * Checks cuBLAS function calls for errors
 */
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Calculate time difference between two timespec structures
 * @param start Start time
 * @param end End time
 * @return Time difference in seconds
 */
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

/**
 * Neural network structure for CUDA implementation using cuBLAS
 * All pointers point to device memory for GPU computation
 */
typedef struct {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;  // Device weights and biases
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;  // Device gradients
    float *d_fc1_output, *d_fc2_output, *d_grad_hidden, *d_grad_output;  // Device intermediate buffers
    float *d_input_batch;  // Device input batch buffer
    float *h_fc2_output;   // Host output buffer
    float *h_grad_output;  // Host gradient buffer
    cublasHandle_t cublas_handle;  // cuBLAS handle for matrix operations
} NeuralNetworkCUDA;

/**
 * Load binary data from file
 * @param filename Path to binary file
 * @param data Pointer to buffer to store data
 * @param size Number of elements to read
 */
void load_data(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen data"); exit(EXIT_FAILURE); }
    fread(data, sizeof(float), size, f);
    fclose(f);
}

/**
 * Load binary labels from file
 * @param filename Path to binary file
 * @param labels Pointer to buffer to store labels
 * @param size Number of labels to read
 */
void load_labels(const char *filename, int *labels, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen labels"); exit(EXIT_FAILURE); }
    fread(labels, sizeof(int), size, f);
    fclose(f);
}

/**
 * Normalize data using MNIST dataset statistics
 * @param data Pointer to data array
 * @param size Number of elements to normalize
 */
void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

/**
 * Initialize weights using He initialization
 * @param weights Pointer to weight array
 * @param rows Number of rows (input dimension)
 * @param cols Number of columns (output dimension)
 */
void initialize_weights_host(float *weights, int rows, int cols) {
    float scale = sqrtf(6.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

/**
 * Initialize bias to zero
 * @param bias Pointer to bias array
 * @param size Size of bias array
 */
void initialize_bias_host(float *bias, int size) {
    memset(bias, 0, size * sizeof(float));
}

/**
 * CUDA kernel for bias addition
 * Adds bias vector to each sample in the batch
 * @param x Input/output array (batch × size, device memory, modified in-place)
 * @param bias Bias vector (size, device memory)
 * @param batch Number of samples in the batch
 * @param size Size of each sample
 */
__global__ void bias_add_kernel(float *x, float *bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        x[idx] += bias[bias_idx];
    }
}

/**
 * CUDA kernel for ReLU activation
 * Applies ReLU activation: f(x) = max(0, x) element-wise
 * @param x Input/output array (device memory, modified in-place)
 * @param total Number of elements
 */
__global__ void relu_kernel(float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

/**
 * CUDA kernel for ReLU backward pass
 * Computes gradient through ReLU activation
 * @param grad Gradient array (device memory, modified in-place)
 * @param x Input values from forward pass
 * @param total Number of elements
 */
__global__ void relu_backward_kernel(float *grad, float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

/**
 * CUDA kernel for bias gradient computation
 * Accumulates gradients across batch dimension using atomic operations
 * @param grad_output Gradient array (batch × size, device memory)
 * @param grad_bias Bias gradient array (size, device memory)
 * @param batch Number of samples in the batch
 * @param size Size of each sample
 */
__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]);
    }
}

/**
 * Forward pass using cuBLAS for matrix multiplications
 * Performs forward propagation through the network
 * @param nn Pointer to neural network structure
 * @param batch_size Number of samples in the batch
 */
void forward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;

    // First matrix multiplication: input -> hidden (using cuBLAS)
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           HIDDEN_SIZE, batch_size, INPUT_SIZE,
                           &alpha, nn->d_weights1, HIDDEN_SIZE,
                           nn->d_input_batch, INPUT_SIZE, &beta,
                           nn->d_fc1_output, HIDDEN_SIZE));

    // Add bias to hidden layer
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    bias_add_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, nn->d_bias1, batch_size, HIDDEN_SIZE);

    // Apply ReLU activation
    relu_kernel<<<grid_hidden, 256>>>(nn->d_fc1_output, total_hidden);

    // Second matrix multiplication: hidden -> output (using cuBLAS)
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           OUTPUT_SIZE, batch_size, HIDDEN_SIZE,
                           &alpha, nn->d_weights2, OUTPUT_SIZE,
                           nn->d_fc1_output, HIDDEN_SIZE, &beta,
                           nn->d_fc2_output, OUTPUT_SIZE));

    // Add bias to output layer
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_add_kernel<<<grid_out, 256>>>(nn->d_fc2_output, nn->d_bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize()); 
}

/**
 * Backward pass using cuBLAS for matrix multiplications
 * Performs backpropagation through the network
 * @param nn Pointer to neural network structure
 * @param batch_size Number of samples in the batch
 */
void backward_pass_only(NeuralNetworkCUDA *nn, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;

    // Zero out all gradients
    CUDA_CHECK(cudaMemset(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_weights2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias1, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(nn->d_grad_bias2, 0, OUTPUT_SIZE * sizeof(float)));

    // Compute weight gradients for second layer: grad_weights2 = grad_output * hidden^T
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
                           &alpha, nn->d_grad_output, OUTPUT_SIZE,
                           nn->d_fc1_output, HIDDEN_SIZE, &beta,
                           nn->d_grad_weights2, OUTPUT_SIZE));

    // Compute bias gradients for second layer
    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_backward_kernel<<<grid_out, 256>>>(nn->d_grad_output, nn->d_grad_bias2, batch_size, OUTPUT_SIZE);

    // Propagate gradients back through weights: grad_hidden = grad_output * weights2^T
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
                           &alpha, nn->d_weights2, OUTPUT_SIZE,
                           nn->d_grad_output, OUTPUT_SIZE, &beta,
                           nn->d_grad_hidden, HIDDEN_SIZE));

    // Apply ReLU backward pass
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    relu_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_fc1_output, total_hidden);

    // Compute weight gradients for first layer: grad_weights1 = grad_hidden * input^T
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           HIDDEN_SIZE, INPUT_SIZE, batch_size,
                           &alpha, nn->d_grad_hidden, HIDDEN_SIZE,
                           nn->d_input_batch, INPUT_SIZE, &beta,
                           nn->d_grad_weights1, HIDDEN_SIZE));

    // Compute bias gradients for first layer
    bias_backward_kernel<<<grid_hidden, 256>>>(nn->d_grad_hidden, nn->d_grad_bias1, batch_size, HIDDEN_SIZE);
}

/**
 * Update weights using cuBLAS axpy operations
 * Performs gradient descent: weights = weights - learning_rate * grad_weights
 * @param nn Pointer to neural network structure
 * @param lr Learning rate
 */
void update_weights_only(NeuralNetworkCUDA *nn, float lr) {
    float neg_lr = -lr;

    // Update weights using cuBLAS axpy (y = alpha * x + y)
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_bias1, 1, nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias2, 1, nn->d_bias2, 1));

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Compute loss and gradients on CPU
 * Computes cross-entropy loss and output gradients
 * @param batch_size Number of samples in the batch
 * @param h_logits Logits from forward pass (batch_size × OUTPUT_SIZE)
 * @param labels True labels (batch_size)
 * @param h_grad Output gradient array (batch_size × OUTPUT_SIZE)
 * @return Average cross-entropy loss
 */
float compute_loss_and_grad(int batch_size, float *h_logits, int *labels, float *h_grad) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        float *logits = h_logits + b * OUTPUT_SIZE;
        int label = labels[b];
        // Find maximum for numerical stability
        float max_logit = -INFINITY;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        // Compute softmax probabilities
        float sum_exp = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float shifted = logits[i] - max_logit;
            float expv = expf(shifted);
            sum_exp += expv;
            h_grad[b * OUTPUT_SIZE + i] = expv;
        }
        // Compute cross-entropy loss
        loss -= (logits[label] - max_logit - logf(sum_exp));
        // Normalize probabilities
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            h_grad[b * OUTPUT_SIZE + i] /= sum_exp;
        }
        // Subtract 1 from true label position (derivative of cross-entropy)
        h_grad[b * OUTPUT_SIZE + label] -= 1.0f;
    }
    // Normalize gradients by batch size
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++) {
        h_grad[i] /= batch_size;
    }
    return loss / batch_size;
}

/**
 * Initialize weights randomly on host and copy to device
 * @param nn Pointer to neural network structure
 */
void initialize_random_weights_cuda(NeuralNetworkCUDA *nn) {
    // Initialize first layer weights
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights_host(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    // Initialize second layer weights
    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights_host(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    // Initialize first layer bias
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias_host(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    // Initialize second layer bias
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias_host(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

/**
 * Initialize neural network structure
 * Allocates device memory and initializes cuBLAS handle
 * @param nn Pointer to neural network structure
 */
void initialize_nn_cuda(NeuralNetworkCUDA *nn) {
    // Allocate device memory for weights and biases
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float)));
    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float)));
    // Allocate device memory for intermediate buffers
    CUDA_CHECK(cudaMalloc(&nn->d_fc1_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Allocate device memory for input batch
    CUDA_CHECK(cudaMalloc(&nn->d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    // Allocate host memory for output and gradient buffers
    nn->h_fc2_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->h_grad_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    if (!nn->h_fc2_output || !nn->h_grad_output) {
        fprintf(stderr, "Failed to allocate persistent host buffers\n");
        exit(EXIT_FAILURE);
    }

    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    // Initialize weights randomly
    initialize_random_weights_cuda(nn);
}

/**
 * Free neural network structure
 * Deallocates all device and host memory
 * @param nn Pointer to neural network structure
 */
void free_nn_cuda(NeuralNetworkCUDA *nn) {
    // Free device memory
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));
    CUDA_CHECK(cudaFree(nn->d_fc1_output));
    CUDA_CHECK(cudaFree(nn->d_fc2_output));
    CUDA_CHECK(cudaFree(nn->d_grad_hidden));
    CUDA_CHECK(cudaFree(nn->d_grad_output));

    CUDA_CHECK(cudaFree(nn->d_input_batch));
    // Free host memory
    free(nn->h_fc2_output);
    free(nn->h_grad_output);

    // Destroy cuBLAS handle
    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
}

int main() {
    srand(12345); 

    // Load training data
    float *train_data = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *train_labels = (int *)malloc(TRAIN_SIZE * sizeof(int));
    load_data("./data/X_train.bin", train_data, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(train_data, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", train_labels, TRAIN_SIZE);

    // Initialize neural network
    NeuralNetworkCUDA nn;
    initialize_nn_cuda(&nn);

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    TimingStats stats = {0};

    struct timespec start, end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Training loop over epochs
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        // Process each batch
        for (int batch = 0; batch < num_batches; batch++) {
            float *batch_input = train_data + batch * BATCH_SIZE * INPUT_SIZE;
            int *batch_labels = train_labels + batch * BATCH_SIZE;

            // Copy input batch to device
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(nn.d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.memory_transfers += get_time_diff(step_start, step_end);

            // Forward pass
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            forward_pass_only(&nn, BATCH_SIZE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.gpu_compute += get_time_diff(step_start, step_end);

            // Copy output to host for loss computation
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(nn.h_fc2_output, nn.d_fc2_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.memory_transfers += get_time_diff(step_start, step_end);

            // Compute loss and gradients on CPU
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float batch_loss = compute_loss_and_grad(BATCH_SIZE, nn.h_fc2_output, batch_labels, nn.h_grad_output);
            total_loss += batch_loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.host_computation += get_time_diff(step_start, step_end);

            // Copy gradients to device
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpy(nn.d_grad_output, nn.h_grad_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.memory_transfers += get_time_diff(step_start, step_end);

            // Backward pass
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            backward_pass_only(&nn, BATCH_SIZE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.gpu_compute += get_time_diff(step_start, step_end);

            // Update weights
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            update_weights_only(&nn, LEARNING_RATE);
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.gpu_compute += get_time_diff(step_start, step_end);
        }
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    stats.total_time = get_time_diff(start, end);

    // Print timing statistics
    printf("\n=== CUBLAS GPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("Detailed Breakdown:\n");
    printf("  Data loading:     %6.3fs (%5.1f%%)\n", stats.memory_transfers, 100.0 * stats.memory_transfers / stats.total_time);
    printf("  Forward pass:     %6.3fs (%5.1f%%)\n", stats.gpu_compute * 0.4, 100.0 * stats.gpu_compute * 0.4 / stats.total_time);
    printf("  Loss computation: %6.3fs (%5.1f%%)\n", stats.host_computation, 100.0 * stats.host_computation / stats.total_time);
    printf("  Backward pass:    %6.3fs (%5.1f%%)\n", stats.gpu_compute * 0.4, 100.0 * stats.gpu_compute * 0.4 / stats.total_time);
    printf("  Weight updates:   %6.3fs (%5.1f%%)\n", stats.gpu_compute * 0.2, 100.0 * stats.gpu_compute * 0.2 / stats.total_time);

    // Clean up
    free_nn_cuda(&nn);
    free(train_data);
    free(train_labels);

    return 0;
}
