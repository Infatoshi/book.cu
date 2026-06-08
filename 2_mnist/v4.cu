#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000


/**
 * MNIST Neural Network Implementation with CUDA
 * Implements a two-layer fully connected network for MNIST digit classification
 * Demonstrates forward pass, backward pass, and weight updates using custom CUDA kernels
 */

/**
 * Timing statistics structure for performance profiling
 * Tracks execution time of each component in the neural network
 */
typedef struct {
    double data_loading;      // Time spent loading data from host to device
    double fwd_matmul1;      // Time for first matrix multiplication (input -> hidden)
    double fwd_bias1;        // Time for first bias addition
    double fwd_relu;         // Time for ReLU activation
    double fwd_matmul2;      // Time for second matrix multiplication (hidden -> output)
    double fwd_bias2;        // Time for second bias addition
    double fwd_softmax;      // Time for softmax activation
    double cross_entropy;    // Time for cross-entropy loss computation
    double bwd_output_grad;  // Time for output gradient computation
    double bwd_matmul2;      // Time for backward matrix multiplication
    double bwd_bias2;        // Time for backward bias gradient
    double bwd_relu;         // Time for ReLU backward pass
    double bwd_matmul1;      // Time for backward matrix multiplication
    double bwd_bias1;        // Time for backward bias gradient
    double weight_updates;   // Time for weight updates
    double total_time;       // Total training time
} TimingStats;

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
 * CUDA error checking macro
 * @param call CUDA function call to check
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Neural network structure containing all weights and gradients
 * All pointers point to device memory for GPU computation
 */
typedef struct {
    float *weights1;      // First layer weights (INPUT_SIZE × HIDDEN_SIZE)
    float *weights2;      // Second layer weights (HIDDEN_SIZE × OUTPUT_SIZE)
    float *bias1;         // First layer bias (HIDDEN_SIZE)
    float *bias2;         // Second layer bias (OUTPUT_SIZE)
    float *grad_weights1; // First layer weight gradients
    float *grad_weights2; // Second layer weight gradients
    float *grad_bias1;    // First layer bias gradients
    float *grad_bias2;    // Second layer bias gradients
} NeuralNetwork;

/**
 * Load binary data from file
 * @param filename Path to binary file
 * @param data Pointer to buffer to store data
 * @param size Number of elements to read
 */
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

/**
 * Load binary labels from file
 * @param filename Path to binary file
 * @param labels Pointer to buffer to store labels
 * @param size Number of labels to read
 */
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

/**
 * Initialize weights using He initialization
 * @param weights Pointer to weight array
 * @param input_size Input dimension
 * @param output_size Output dimension
 */
void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(6.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

/**
 * Initialize bias to zero
 * @param bias Pointer to bias array
 * @param size Size of bias array
 */
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
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
 * Matrix multiplication kernel: C = A * B
 * Each thread computes one element of the output matrix
 * 
 * @param A Input matrix A (m×n, device memory)
 * @param B Input matrix B (n×k, device memory)
 * @param C Output matrix C (m×k, device memory)
 * @param m Number of rows in A and C
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B and C
 */
__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in output matrix

    // Bounds check to ensure we don't access out-of-range elements
    if (row < m && col < k) {
        float sum = 0.0f;
        // Compute dot product of row A[row,:] and column B[:,col]
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        // Store result in output matrix
        C[row * k + col] = sum;
    }
}

/**
 * Matrix multiplication kernel: C = A * B^T
 * Computes matrix multiplication with B transposed
 * Used in backward pass for gradient computation
 * 
 * @param A Input matrix A (m×n, device memory)
 * @param B Input matrix B (k×n, device memory) - accessed as B^T
 * @param C Output matrix C (m×k, device memory)
 * @param m Number of rows in A and C
 * @param n Number of columns in A and B
 * @param k Number of rows in B and columns in C
 */
__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in output matrix

    // Bounds check
    if (row < m && col < k) {
        float sum = 0.0f;
        // Compute dot product of row A[row,:] and row B[col,:] (transpose)
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        // Store result in output matrix
        C[row * k + col] = sum;
    }
}

/**
 * Matrix multiplication kernel: C = A^T * B
 * Computes matrix multiplication with A transposed
 * Used in backward pass for weight gradient computation
 * 
 * @param A Input matrix A (m×n, device memory) - accessed as A^T
 * @param B Input matrix B (m×k, device memory)
 * @param C Output matrix C (n×k, device memory)
 * @param m Number of rows in A and B
 * @param n Number of columns in A and rows in C
 * @param k Number of columns in B and C
 */
__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    // Calculate 2D coordinates from thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // Row index in output matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Column index in output matrix

    // Bounds check
    if (row < n && col < k) {
        float sum = 0.0f;
        // Compute dot product of column A[:,row] and column B[:,col]
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + row] * B[i * k + col];
        }
        // Store result in output matrix
        C[row * k + col] = sum;
    }
}

/**
 * ReLU activation function kernel
 * Applies ReLU activation: f(x) = max(0, x) element-wise
 * 
 * @param x Input/output array (device memory, modified in-place)
 * @param size Number of elements in the array
 */
__global__ void relu_forward_kernel(float *x, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < size) {
        // Apply ReLU activation: f(x) = max(0, x)
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

/**
 * Bias addition kernel
 * Adds bias vector to each sample in the batch
 * 
 * @param x Input/output array (batch_size × size, device memory, modified in-place)
 * @param bias Bias vector (size, device memory)
 * @param batch_size Number of samples in the batch
 * @param size Size of each sample (feature dimension)
 */
__global__ void bias_forward_kernel(float *x, float *bias, int batch_size, int size) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Extract batch and feature indices
    int b = idx / size;  // Batch index
    int i = idx % size;  // Feature index

    // Bounds check
    if (b < batch_size && i < size) {
        // Add bias to corresponding element
        x[idx] += bias[i];
    }
}

/**
 * Softmax activation kernel
 * Applies softmax normalization to each sample in the batch
 * Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * @param x Input/output array (batch_size × size, device memory, modified in-place)
 * @param batch_size Number of samples in the batch
 * @param size Size of each sample (number of classes)
 */
__global__ void softmax_kernel(float *x, int batch_size, int size) {
    // Each block processes one sample in the batch
    int b = blockIdx.x;
    
    if (b < batch_size) {
        // Step 1: Find maximum value for numerical stability
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        // Step 2: Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        // Step 3: Normalize to get probabilities
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

/**
 * Zero out gradient arrays kernel
 * Sets all gradient values to zero
 * @param grad Gradient array (device memory, modified in-place)
 * @param size Number of elements
 */
__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

/**
 * Compute output gradients from softmax and labels
 * Computes gradient of cross-entropy loss with respect to output
 * @param grad_output Output gradient array (batch_size × OUTPUT_SIZE, device memory)
 * @param output Output probabilities (batch_size × OUTPUT_SIZE, device memory)
 * @param labels True labels (batch_size, device memory)
 * @param batch_size Number of samples in the batch
 */
__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        // Initialize gradients with softmax probabilities
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        // Subtract 1 from the true label position (derivative of cross-entropy)
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;

        // Normalize by batch size
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] /= batch_size;
        }
    }
}

/**
 * ReLU backward pass kernel
 * Computes gradient through ReLU activation
 * @param grad Gradient array (device memory, modified in-place)
 * @param x Input values from forward pass
 * @param size Number of elements
 */
__global__ void relu_backward_kernel(float *grad, float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ReLU derivative: 1 if x > 0, else 0
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

/**
 * Bias gradient computation kernel
 * Accumulates gradients across batch dimension
 * @param grad_bias Bias gradient array (size, device memory)
 * @param grad Gradient array (batch_size × size, device memory)
 * @param batch_size Number of samples in the batch
 * @param size Size of each sample
 */
__global__ void bias_backward_kernel(float *grad_bias, float *grad, int batch_size, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Sum gradients across batch dimension
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad[b * size + i];
        }
        grad_bias[i] = sum;
    }
}

/**
 * Weight update kernel
 * Performs gradient descent: weights = weights - learning_rate * grad_weights
 * @param weights Weight array (device memory, modified in-place)
 * @param grad_weights Gradient array (device memory)
 * @param size Number of elements
 */
__global__ void weight_update_kernel(float *weights, float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }
}

/**
 * Forward pass with timing
 * Performs forward propagation through the network
 * @param nn Pointer to neural network structure
 * @param input Input batch (batch_size × INPUT_SIZE, device memory)
 * @param hidden Hidden layer output (batch_size × HIDDEN_SIZE, device memory)
 * @param output Output layer output (batch_size × OUTPUT_SIZE, device memory)
 * @param batch_size Number of samples in the batch
 * @param stats Pointer to timing statistics structure
 */
void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    dim3 block_size(32, 32);

    // First matrix multiplication: input -> hidden
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_size1((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size1, block_size>>>(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);

    // First bias addition
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);

    // ReLU activation
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    // Second matrix multiplication: hidden -> output
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_size2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_b_kernel<<<grid_size2, block_size>>>(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);

    // Second bias addition
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);

    // Softmax activation
    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax_kernel<<<batch_size, 1>>>(output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

/**
 * Compute cross-entropy loss
 * @param output Output probabilities (batch_size × OUTPUT_SIZE)
 * @param labels True labels (batch_size)
 * @param batch_size Number of samples in the batch
 * @return Average cross-entropy loss
 */
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

/**
 * Backward pass with timing
 * Performs backpropagation through the network
 * @param nn Pointer to neural network structure
 * @param input Input batch (batch_size × INPUT_SIZE, device memory)
 * @param hidden Hidden layer output from forward pass (batch_size × HIDDEN_SIZE, device memory)
 * @param output Output layer output from forward pass (batch_size × OUTPUT_SIZE, device memory)
 * @param labels True labels (batch_size, device memory)
 * @param batch_size Number of samples in the batch
 * @param stats Pointer to timing statistics structure
 */
void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    dim3 block_size(32, 32);

    // Zero out all gradients
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);

    // Allocate temporary device memory for gradients
    float *grad_output, *dX2, *d_ReLU_out;
    CUDA_CHECK(cudaMalloc(&grad_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ReLU_out, batch_size * HIDDEN_SIZE * sizeof(float)));

    // Compute output gradients
    clock_gettime(CLOCK_MONOTONIC, &start);
    compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(grad_output, output, labels, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    // Backward pass through second layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_weights2((OUTPUT_SIZE + block_size.x - 1) / block_size.x, (HIDDEN_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_weights2, block_size>>>(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul2 += get_time_diff(start, end);

    // Compute bias gradients for second layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias2 += get_time_diff(start, end);

    // Propagate gradients back through weights: grad_hidden = grad_output * weights2^T
    dim3 grid_hidden((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
    matmul_a_bt_kernel<<<grid_hidden, block_size>>>(grad_output, nn->weights2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    // Apply ReLU backward pass
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_backward_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(dX2, hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(d_ReLU_out, dX2, batch_size * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_relu += get_time_diff(start, end);

    // Backward pass through first layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    dim3 grid_weights1((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (INPUT_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_weights1, block_size>>>(input, d_ReLU_out, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul1 += get_time_diff(start, end);

    // Compute bias gradients for first layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->grad_bias1, d_ReLU_out, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias1 += get_time_diff(start, end);

    // Free temporary device memory
    CUDA_CHECK(cudaFree(grad_output));
    CUDA_CHECK(cudaFree(dX2));
    CUDA_CHECK(cudaFree(d_ReLU_out));
}

/**
 * Update weights using gradient descent
 * @param nn Pointer to neural network structure
 * @param stats Pointer to timing statistics structure
 */
void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    // Update all weights and biases using gradient descent
    weight_update_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256>>>(nn->weights1, nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    weight_update_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256>>>(nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    weight_update_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(nn->bias1, nn->grad_bias1, HIDDEN_SIZE);
    weight_update_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(nn->bias2, nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->weight_updates += get_time_diff(start, end);
}

/**
 * Train the neural network
 * @param nn Pointer to neural network structure
 * @param X_train Training data (TRAIN_SIZE × INPUT_SIZE, host memory)
 * @param y_train Training labels (TRAIN_SIZE, host memory)
 */
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train) {
    // Allocate device memory for intermediate activations
    float *d_hidden, *d_output, *d_input_batch;
    int *d_labels_batch;

    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels_batch, BATCH_SIZE * sizeof(int)));

    // Allocate host memory for output (used for loss computation)
    float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    TimingStats stats = {0};

    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    // Training loop over epochs
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        // Process each batch
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;

            // Load batch data and copy to device
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];

            CUDA_CHECK(cudaMemcpy(d_input_batch, batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_labels_batch, batch_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading += get_time_diff(step_start, step_end);

            // Forward pass
            forward_timed(nn, d_input_batch, d_hidden, d_output, BATCH_SIZE, &stats);

            // Copy output to host for loss computation
            CUDA_CHECK(cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            // Compute loss
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(h_output, batch_labels, BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start, step_end);

            // Backward pass
            backward_timed(nn, d_input_batch, d_hidden, d_output, d_labels_batch, BATCH_SIZE, &stats);

            // Weight update
            update_weights_timed(nn, &stats);
        }

        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);

    // Print timing statistics
    printf("\n=== CUDA GPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);

    printf("Detailed Breakdown:\n");
    printf("  Data loading:     %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading / stats.total_time);
    double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
    printf("  Forward pass:     %6.3fs (%5.1f%%)\n", forward_pass, 100.0 * forward_pass / stats.total_time);
    printf("  Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy / stats.total_time);
    double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 + stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 + stats.bwd_bias1;
    printf("  Backward pass:    %6.3fs (%5.1f%%)\n", backward_pass, 100.0 * backward_pass / stats.total_time);
    printf("  Weight updates:   %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates / stats.total_time);

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_batch));
    CUDA_CHECK(cudaFree(d_labels_batch));
    free(h_output);
}

/**
 * Initialize weights randomly on host and copy to device
 * @param nn Pointer to neural network structure
 */
void initialize_random_weights(NeuralNetwork *nn) {
    // Allocate host memory for weights
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases
    initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(h_bias1, HIDDEN_SIZE);
    initialize_bias(h_bias2, OUTPUT_SIZE);

    // Copy to device
    CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Free host memory
    free(h_weights1);
    free(h_weights2);
    free(h_bias1);
    free(h_bias2);
}

/**
 * Initialize neural network structure
 * Allocates device memory for all weights and gradients
 * @param nn Pointer to neural network structure
 */
void initialize_neural_network(NeuralNetwork *nn) {
    // Allocate device memory for weights and biases
    CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));
    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

    // Initialize weights randomly
    initialize_random_weights(nn);
}

int main() {
    srand(time(NULL));  

    // Initialize neural network
    NeuralNetwork nn;
    initialize_neural_network(&nn);

    // Allocate host memory for training and test data
    float *X_train = (float *)malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = (int *)malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = (int *)malloc(TEST_SIZE * sizeof(int));

    // Load and normalize training data
    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", y_train, TRAIN_SIZE);
    
    // Load and normalize test data
    load_data("./data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    normalize_data(X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", y_test, TEST_SIZE);

    // Train the network
    train_timed(&nn, X_train, y_train);

    // Clean up device memory
    CUDA_CHECK(cudaFree(nn.weights1));
    CUDA_CHECK(cudaFree(nn.weights2));
    CUDA_CHECK(cudaFree(nn.bias1));
    CUDA_CHECK(cudaFree(nn.bias2));
    CUDA_CHECK(cudaFree(nn.grad_weights1));
    CUDA_CHECK(cudaFree(nn.grad_weights2));
    CUDA_CHECK(cudaFree(nn.grad_bias1));
    CUDA_CHECK(cudaFree(nn.grad_bias2));
    
    // Clean up host memory
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
