#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000

/**
 * MNIST Neural Network Implementation in C
 * Implements a two-layer fully connected network for MNIST digit classification
 * Demonstrates forward pass, backward pass, and weight updates on CPU
 */

/**
 * Timing statistics structure for performance profiling
 * Tracks execution time of each component in the neural network
 */
typedef struct {
    double data_loading;      // Time spent loading data
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
 * Neural network structure containing all weights and gradients
 * All pointers point to host memory for CPU computation
 */
typedef struct {
    float *weights1;      // First layer weights (INPUT_SIZE × HIDDEN_SIZE)
    float *weights2;     // Second layer weights (HIDDEN_SIZE × OUTPUT_SIZE)
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
 * Compute softmax activation function
 * Applies softmax normalization to each sample in the batch
 * @param x Input/output array (batch_size × size, modified in-place)
 * @param batch_size Number of samples in the batch
 * @param size Size of each sample (number of classes)
 */
void softmax(float *x, int batch_size, int size) {
    // Process each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        // Step 1: Find maximum value for numerical stability
        float max = x[b * size];
        for (int i = 1; i < size; i++) {
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        // Step 2: Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max);
            sum += x[b * size + i];
        }
        // Step 3: Normalize to get probabilities
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

/**
 * Matrix multiplication: C = A * B
 * @param A Input matrix A (m×n)
 * @param B Input matrix B (n×k)
 * @param C Output matrix C (m×k)
 * @param m Number of rows in A and C
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B and C
 */
void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

/**
 * Matrix multiplication: C = A * B^T
 * Computes matrix multiplication with B transposed
 * @param A Input matrix A (m×n)
 * @param B Input matrix B (k×n) - accessed as B^T
 * @param C Output matrix C (m×k)
 * @param m Number of rows in A and C
 * @param n Number of columns in A and B
 * @param k Number of rows in B and columns in C
 */
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

/**
 * Matrix multiplication: C = A^T * B
 * Computes matrix multiplication with A transposed
 * @param A Input matrix A (m×n) - accessed as A^T
 * @param B Input matrix B (m×k)
 * @param C Output matrix C (n×k)
 * @param m Number of rows in A and B
 * @param n Number of columns in A and rows in C
 * @param k Number of columns in B and C
 */
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++) {
                C[i * k + j] += A[l * n + i] * B[l * k + j];
            }
        }
    }
}

/**
 * ReLU activation function: f(x) = max(0, x)
 * @param x Input/output array (modified in-place)
 * @param size Number of elements
 */
void relu_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

/**
 * Add bias to each sample in the batch
 * @param x Input/output array (batch_size × size, modified in-place)
 * @param bias Bias vector (size)
 * @param batch_size Number of samples in the batch
 * @param size Size of each sample
 */
void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

/**
 * Forward pass with timing
 * Performs forward propagation through the network
 * @param nn Pointer to neural network structure
 * @param input Input batch (batch_size × INPUT_SIZE)
 * @param hidden Hidden layer output (batch_size × HIDDEN_SIZE)
 * @param output Output layer output (batch_size × OUTPUT_SIZE)
 * @param batch_size Number of samples in the batch
 * @param stats Pointer to timing statistics structure
 */
void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    
    // First matrix multiplication: input -> hidden
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);
    
    // First bias addition
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);
    
    // ReLU activation
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward(hidden, batch_size * HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    // Second matrix multiplication: hidden -> output
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);

    // Second bias addition
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);
    
    // Softmax activation
    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax(output, batch_size, OUTPUT_SIZE);
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
 * Zero out gradient arrays
 * @param grad Pointer to gradient array
 * @param size Size of gradient array
 */
void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

/**
 * ReLU backward pass
 * Computes gradient through ReLU activation
 * @param grad Gradient array (modified in-place)
 * @param x Input values from forward pass
 * @param size Number of elements
 */
void relu_backward(float *grad, float *x, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] *= (x[i] > 0);
    }
}

/**
 * Compute bias gradients
 * Accumulates gradients across batch dimension
 * @param grad_bias Bias gradient array (size)
 * @param grad Gradient array (batch_size × size)
 * @param batch_size Number of samples in the batch
 * @param size Size of each sample
 */
void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
    for (int i = 0; i < size; i++) {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad[b * size + i];
        }
    }
}

/**
 * Compute output gradients from softmax and labels
 * Computes gradient of cross-entropy loss with respect to output
 * @param grad_output Output gradient array (batch_size × OUTPUT_SIZE)
 * @param output Output probabilities (batch_size × OUTPUT_SIZE)
 * @param labels True labels (batch_size)
 * @param batch_size Number of samples in the batch
 */
void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size) {
    // Initialize gradients with softmax probabilities
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        // Subtract 1 from the true label position (derivative of cross-entropy)
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
    // Normalize by batch size
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++) {
        grad_output[i] /= batch_size;
    }
}

/**
 * Update gradients for a layer
 * Computes weight and bias gradients
 * @param grad_weights Weight gradient array (curr_size × prev_size)
 * @param grad_bias Bias gradient array (curr_size)
 * @param grad_layer Layer gradient (batch_size × curr_size)
 * @param prev_layer Previous layer output (batch_size × prev_size)
 * @param batch_size Number of samples in the batch
 * @param prev_size Size of previous layer
 * @param curr_size Size of current layer
 */
void update_gradients(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    // Compute weight gradients: grad_weights = grad_layer^T * prev_layer
    for (int i = 0; i < curr_size; i++) {
        for (int j = 0; j < prev_size; j++) {
            for (int b = 0; b < batch_size; b++) {
                grad_weights[i * prev_size + j] += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
            }
        }
        // Compute bias gradients: grad_bias = sum of grad_layer across batch
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad_layer[b * curr_size + i];
        }
    }
}

/**
 * Backward pass with timing
 * Performs backpropagation through the network
 * @param nn Pointer to neural network structure
 * @param input Input batch (batch_size × INPUT_SIZE)
 * @param hidden Hidden layer output from forward pass (batch_size × HIDDEN_SIZE)
 * @param output Output layer output from forward pass (batch_size × OUTPUT_SIZE)
 * @param labels True labels (batch_size)
 * @param batch_size Number of samples in the batch
 * @param stats Pointer to timing statistics structure
 */
void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    
    // Zero out all gradients
    zero_grad(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    // Compute output gradients
    clock_gettime(CLOCK_MONOTONIC, &start);
    float *grad_output = malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    // Backward pass through second layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_at_b(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul2 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias2 += get_time_diff(start, end);

    // Propagate gradients back through ReLU
    float *dX2 = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    matmul_a_bt(grad_output, nn->weights2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    clock_gettime(CLOCK_MONOTONIC, &start);
    float *d_ReLU_out = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < batch_size * HIDDEN_SIZE; i++) {
        d_ReLU_out[i] = dX2[i] * (hidden[i] > 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_relu += get_time_diff(start, end);
    
    // Backward pass through first layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_at_b(input, d_ReLU_out, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul1 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward(nn->grad_bias1, d_ReLU_out, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias1 += get_time_diff(start, end);

    // Free temporary memory
    free(grad_output);
    free(dX2);
    free(d_ReLU_out);
}

/**
 * Update weights using gradient descent
 * @param nn Pointer to neural network structure
 * @param stats Pointer to timing statistics structure
 */
void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Update weights and biases using gradient descent
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        nn->weights1[i] -= LEARNING_RATE * nn->grad_weights1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->weights2[i] -= LEARNING_RATE * nn->grad_weights2[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->bias1[i] -= LEARNING_RATE * nn->grad_bias1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->bias2[i] -= LEARNING_RATE * nn->grad_bias2[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->weight_updates += get_time_diff(start, end);
}

/**
 * Train the neural network
 * @param nn Pointer to neural network structure
 * @param X_train Training data (TRAIN_SIZE × INPUT_SIZE)
 * @param y_train Training labels (TRAIN_SIZE)
 */
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train) {
    // Allocate memory for intermediate activations
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

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
            
            // Load batch data
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading += get_time_diff(step_start, step_end);
            
            // Forward pass
            forward_timed(nn, batch_input, hidden, output, BATCH_SIZE, &stats);

            // Compute loss
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(output, batch_labels, BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start, step_end);

            // Backward pass
            backward_timed(nn, batch_input, hidden, output, batch_labels, BATCH_SIZE, &stats);
            
            // Weight update
            update_weights_timed(nn, &stats);
        }
        
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);
    
    // Print timing statistics
    printf("\n=== C CPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);
    
    printf("Detailed Breakdown:\n");
    printf("  Data loading:     %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading / stats.total_time);
    double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
    printf("  Forward pass:     %6.3fs (%5.1f%%)\n", forward_pass, 100.0 * forward_pass / stats.total_time);
    printf("  Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy / stats.total_time);
    double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 + stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 + stats.bwd_bias1;
    printf("  Backward pass:    %6.3fs (%5.1f%%)\n", backward_pass, 100.0 * backward_pass / stats.total_time);
    printf("  Weight updates:   %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates / stats.total_time);
    
    free(hidden);
    free(output);
}

/**
 * Initialize weights randomly
 * @param nn Pointer to neural network structure
 */
void initialize_random_weights(NeuralNetwork *nn) {
    initialize_weights(nn->weights1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(nn->weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}

/**
 * Initialize neural network structure
 * Allocates memory for all weights and gradients
 * @param nn Pointer to neural network structure
 */
void initialize_neural_network(NeuralNetwork *nn) {
    nn->weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weights1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_weights2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_random_weights(nn);
}

int main() {
    srand(time(NULL));  

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = malloc(TEST_SIZE * sizeof(int));

    load_data("./data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("./data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    normalize_data(X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", y_test, TEST_SIZE);


    train_timed(&nn, X_train, y_train);

    free(nn.weights1);
    free(nn.weights2);
    free(nn.bias1);
    free(nn.bias2);
    free(nn.grad_weights1);
    free(nn.grad_weights2);
    free(nn.grad_bias1);
    free(nn.grad_bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
