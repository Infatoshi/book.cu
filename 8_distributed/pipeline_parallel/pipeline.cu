/**
 * Pipeline Parallel MLP Inference - Unified Implementation
 * 
 * Demonstrates the critical importance of CUDA streams for pipeline parallelism
 * by running both naive (sequential) and optimized (stream-based) implementations
 * and comparing their performance.
 * 
 * Compilation: nvcc -O3 -arch=sm_90 pipeline.cu -o pipeline -lcublas
 * Usage: ./pipeline [num_gpus]
 */



    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d - %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void bias_add_kernel(float* data, const float* bias, int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int feature_idx = idx % dim;
        data[idx] += bias[feature_idx];
    }
}

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

__global__ void softmax_normalize_kernel(float* data, const float* sum_vals, 
                                        int batch_size, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * dim) {
        int batch_idx = idx / dim;
        data[idx] /= sum_vals[batch_idx];
    }
}


struct GPULayer {
    int gpu_id;
    int input_dim;
    int output_dim;
    bool has_relu;
    bool has_softmax;
    
    
    float* d_weights;
    float* d_bias;
    
    
    float* d_input_naive;
    float* d_output_naive;
    float* d_max_vals_naive;
    float* d_sum_vals_naive;
    cublasHandle_t cublas_handle_naive;
    
    
    float* d_input[NUM_STREAMS_PER_GPU];
    float* d_output[NUM_STREAMS_PER_GPU];
    float* d_max_vals[NUM_STREAMS_PER_GPU];
    float* d_sum_vals[NUM_STREAMS_PER_GPU];
    cudaStream_t streams[NUM_STREAMS_PER_GPU];
    cudaEvent_t events[NUM_STREAMS_PER_GPU];
    cublasHandle_t cublas_handles[NUM_STREAMS_PER_GPU];
};


void forward_linear_naive(GPULayer* layer) {
    const float alpha = 1.0f, beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgemm(layer->cublas_handle_naive, CUBLAS_OP_N, CUBLAS_OP_N,
        layer->output_dim, BATCH_SIZE, layer->input_dim,
        &alpha, layer->d_weights, layer->output_dim,
        layer->d_input_naive, layer->input_dim,
        &beta, layer->d_output_naive, layer->output_dim));
    
    int total = BATCH_SIZE * layer->output_dim;
    bias_add_kernel<<<(total + 255) / 256, 256>>>(
        layer->d_output_naive, layer->d_bias, BATCH_SIZE, layer->output_dim);
}

void forward_linear_stream(GPULayer* layer, int s) {
    const float alpha = 1.0f, beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgemm(layer->cublas_handles[s], CUBLAS_OP_N, CUBLAS_OP_N,
        layer->output_dim, BATCH_SIZE, layer->input_dim,
        &alpha, layer->d_weights, layer->output_dim,
        layer->d_input[s], layer->input_dim,
        &beta, layer->d_output[s], layer->output_dim));
    
    int total = BATCH_SIZE * layer->output_dim;
    bias_add_kernel<<<(total + 255) / 256, 256, 0, layer->streams[s]>>>(
        layer->d_output[s], layer->d_bias, BATCH_SIZE, layer->output_dim);
}

void forward_relu_naive(GPULayer* layer) {
    int total = BATCH_SIZE * layer->output_dim;
    relu_kernel<<<(total + 255) / 256, 256>>>(layer->d_output_naive, total);
}

void forward_relu_stream(GPULayer* layer, int s) {
    int total = BATCH_SIZE * layer->output_dim;
    relu_kernel<<<(total + 255) / 256, 256, 0, layer->streams[s]>>>(
        layer->d_output[s], total);
}

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


void init_layer(GPULayer* layer, int gpu_id, int input_dim, int output_dim,
                bool has_relu, bool has_softmax) {
    layer->gpu_id = gpu_id;
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    layer->has_relu = has_relu;
    layer->has_softmax = has_softmax;
    
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    
    CUDA_CHECK(cudaMalloc(&layer->d_weights, output_dim * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias, output_dim * sizeof(float)));
    
    
    CUDA_CHECK(cudaMalloc(&layer->d_input_naive, BATCH_SIZE * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_output_naive, BATCH_SIZE * output_dim * sizeof(float)));
    if (has_softmax) {
        CUDA_CHECK(cudaMalloc(&layer->d_max_vals_naive, BATCH_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer->d_sum_vals_naive, BATCH_SIZE * sizeof(float)));
    }
    CUBLAS_CHECK(cublasCreate(&layer->cublas_handle_naive));
    
    
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
    
    
    float* h_weights = (float*)malloc(output_dim * input_dim * sizeof(float));
    float* h_bias = (float*)malloc(output_dim * sizeof(float));
    
    float scale = sqrtf(2.0f / (input_dim + output_dim));
    for (int i = 0; i < output_dim * input_dim; i++) {
        h_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    for (int i = 0; i < output_dim; i++) {
        h_bias[i] = 0.0f;
    }
    
    CUDA_CHECK(cudaMemcpy(layer->d_weights, h_weights,
        output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer->d_bias, h_bias,
        output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_weights);
    free(h_bias);
}

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


void setup_p2p(int num_gpus) {
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,
                                cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError();  
                    }
                }
            }
        }
    }
}


void process_batch_naive(GPULayer* layers, int num_gpus, float* h_input, float* h_output) {
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(layers[0].d_input_naive, h_input,
        BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(layers[i].gpu_id));
        
        forward_linear_naive(&layers[i]);
        if (layers[i].has_relu) forward_relu_naive(&layers[i]);
        if (layers[i].has_softmax) forward_softmax_naive(&layers[i]);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if (i < num_gpus - 1) {
            int next_dim = layers[i + 1].input_dim;
            CUDA_CHECK(cudaMemcpy(layers[i + 1].d_input_naive, layers[i].d_output_naive,
                BATCH_SIZE * next_dim * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    int last = num_gpus - 1;
    CUDA_CHECK(cudaSetDevice(layers[last].gpu_id));
    CUDA_CHECK(cudaMemcpy(h_output, layers[last].d_output_naive,
        BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
}


void process_batch_async(GPULayer* layers, int num_gpus, int batch_id,
                        float* h_input_base, float* h_output_base) {
    int s = batch_id % NUM_STREAMS_PER_GPU;
    float* h_input = h_input_base + (s * BATCH_SIZE * INPUT_DIM);
    float* h_output = h_output_base + (s * BATCH_SIZE * OUTPUT_DIM);
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpyAsync(layers[0].d_input[s], h_input,
        BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice,
        layers[0].streams[s]));
    
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(layers[i].gpu_id));
        
        if (i > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(layers[i].streams[s],
                layers[i-1].events[s], 0));
            
            int transfer_dim = layers[i].input_dim;
            CUDA_CHECK(cudaMemcpyAsync(layers[i].d_input[s],
                layers[i-1].d_output[s], BATCH_SIZE * transfer_dim * sizeof(float),
                cudaMemcpyDeviceToDevice, layers[i].streams[s]));
        }
        
        forward_linear_stream(&layers[i], s);
        if (layers[i].has_relu) forward_relu_stream(&layers[i], s);
        if (layers[i].has_softmax) forward_softmax_stream(&layers[i], s);
        
        CUDA_CHECK(cudaEventRecord(layers[i].events[s], layers[i].streams[s]));
    }
    
    int last = num_gpus - 1;
    CUDA_CHECK(cudaSetDevice(layers[last].gpu_id));
    CUDA_CHECK(cudaMemcpyAsync(h_output, layers[last].d_output[s],
        BATCH_SIZE * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost,
        layers[last].streams[s]));
}


int main(int argc, char** argv) {
    int num_gpus = 1;
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
    
    setup_p2p(num_gpus);
    
    
    GPULayer* layers = (GPULayer*)malloc(num_gpus * sizeof(GPULayer));
    for (int i = 0; i < num_gpus; i++) {
        int input_dim = (i == 0) ? INPUT_DIM : HIDDEN_DIM;
        int output_dim = (i == num_gpus - 1) ? OUTPUT_DIM : HIDDEN_DIM;
        bool has_relu = (i < num_gpus - 1);
        bool has_softmax = (i == num_gpus - 1);
        init_layer(&layers[i], i, input_dim, output_dim, has_relu, has_softmax);
    }
    
    
    float *h_input_naive, *h_output_naive, *h_input_stream, *h_output_stream;
    CUDA_CHECK(cudaMallocHost(&h_input_naive, BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output_naive, BATCH_SIZE * OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_input_stream, NUM_STREAMS_PER_GPU * BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output_stream, NUM_STREAMS_PER_GPU * BATCH_SIZE * OUTPUT_DIM * sizeof(float)));
    
    
    for (int i = 0; i < BATCH_SIZE * INPUT_DIM; i++) {
        h_input_naive[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int i = 0; i < NUM_STREAMS_PER_GPU * BATCH_SIZE * INPUT_DIM; i++) {
        h_input_stream[i] = h_input_naive[i % (BATCH_SIZE * INPUT_DIM)];
    }
    
    
    
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ [1/2] Naive Sequential Pipeline (Blocking Operations)      │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    
    
    for (int b = 0; b < NUM_WARMUP; b++) {
        process_batch_naive(layers, num_gpus, h_input_naive, h_output_naive);
    }
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    
    auto start_naive = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < NUM_BATCHES - NUM_WARMUP; b++) {
        process_batch_naive(layers, num_gpus, h_input_naive, h_output_naive);
    }
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
    
    
    
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ [2/2] Pipelined with Streams (Async Operations)            │\n");
    printf("└─────────────────────────────────────────────────────────────┘\n");
    
    
    for (int b = 0; b < NUM_WARMUP; b++) {
        process_batch_async(layers, num_gpus, b, h_input_stream, h_output_stream);
    }
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    
    auto start_stream = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < NUM_BATCHES - NUM_WARMUP; b++) {
        process_batch_async(layers, num_gpus, b, h_input_stream, h_output_stream);
    }
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
    
    
    
    
    double speedup = throughput_stream / throughput_naive;
    double efficiency = (speedup / num_gpus) * 100.0;
    
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    Performance Summary                       ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Speedup:     %.2fx                                           \n", speedup);
    printf("║  Efficiency:  %.1f%% (%.2fx / %d GPUs)                        \n", efficiency, speedup, num_gpus);
    printf("║  Time Saved:  %.1f%%                                          \n", (1.0 - time_stream/time_naive) * 100.0);
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    
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
