
typedef struct {
    double data_loading;
    double fwd_matmul1;
    double fwd_bias1;
    double fwd_relu;
    double fwd_matmul2;
    double fwd_bias2;
    double fwd_softmax;
    double cross_entropy;
    double bwd_output_grad;
    double bwd_matmul2;
    double bwd_bias2;
    double bwd_relu;
    double bwd_matmul1;
    double bwd_bias1;
    double weight_updates;
    double total_time;
} TimingStats;

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}


typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;


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

void initialize_weights(float *weights, int input_size, int output_size) {
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void softmax(float *x, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        float max = x[b * size];
        for (int i = 1; i < size; i++) {
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[b * size + i] = expf(x[b * size + i] - max);
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; i++) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

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

void relu_forward(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

void bias_forward(float *x, float *bias, int batch_size, int size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            x[b * size + i] += bias[i];
        }
    }
}

void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward(hidden, batch_size * HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax(output, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}


void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

void relu_backward(float *grad, float *x, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] *= (x[i] > 0);
    }
}

void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
    for (int i = 0; i < size; i++) {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad[b * size + i];
        }
    }
}

void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++) {
        grad_output[i] /= batch_size;
    }
}

void update_gradients(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    for (int i = 0; i < curr_size; i++) {
        for (int j = 0; j < prev_size; j++) {
            for (int b = 0; b < batch_size; b++) {
                grad_weights[i * prev_size + j] += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
            }
        }
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad_layer[b * curr_size + i];
        }
    }
}

void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    
    zero_grad(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    clock_gettime(CLOCK_MONOTONIC, &start);
    float *grad_output = malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_at_b(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul2 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias2 += get_time_diff(start, end);

    float *dX2 = malloc(batch_size * HIDDEN_SIZE * sizeof(float));

    matmul_a_bt(grad_output, nn->weights2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    clock_gettime(CLOCK_MONOTONIC, &start);
    float *d_ReLU_out = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < batch_size * HIDDEN_SIZE; i++) {
        d_ReLU_out[i] = dX2[i] * (hidden[i] > 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_relu += get_time_diff(start, end);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_at_b(input, d_ReLU_out, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul1 += get_time_diff(start, end);

    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward(nn->grad_bias1, d_ReLU_out, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias1 += get_time_diff(start, end);

    free(grad_output);
    free(dX2);
    free(d_ReLU_out);
}

void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
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

void train_timed(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    TimingStats stats = {0};
    
    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading += get_time_diff(step_start, step_end);
            
            forward_timed(nn, batch_input, hidden, output, BATCH_SIZE, &stats);

            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(output, batch_labels, BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start, step_end);

            backward_timed(nn, batch_input, hidden, output, batch_labels, BATCH_SIZE, &stats);
            
            update_weights_timed(nn, &stats);
        }
        
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);
    
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

void initialize_random_weights(NeuralNetwork *nn) {
    initialize_weights(nn->weights1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(nn->weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}

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
