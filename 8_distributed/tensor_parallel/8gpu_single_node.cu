
/**
 * 8-GPU Single-Node Tensor Parallel GEMM Benchmark
 * Demonstrates distributed matrix multiplication across 8 GPUs on a single node
 * Uses MPI for inter-process communication and cuBLAS for high-performance GEMM
 * 
 * This implementation shows tensor parallelism where each GPU computes a portion
 * of the output matrix, enabling scaling beyond single-GPU memory limits
 */

/**
 * CUDA error checking macro for distributed applications
 * Aborts all MPI processes if CUDA error occurs
 * @param call CUDA function call to check
 */
#define CHECK_CUDA(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

/**
 * cuBLAS error checking macro for distributed applications
 * Aborts all MPI processes if cuBLAS error occurs
 * @param call cuBLAS function call to check
 */
#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    // Get MPI process information
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Process rank (0-7)
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Total number of processes
    
    // Validate that we have exactly 8 processes (one per GPU)
    if (world_size != 8) {
        if (rank == 0) {
            std::cerr << "Error: This program requires exactly 8 MPI processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Matrix dimensions for GEMM: C[M×N] = A[M×K] * B[K×N]
    const int M = 4096;  // Output matrix height
    const int N = 4096;  // Output matrix width
    const int K = 4096;  // Inner dimension
    
    // Set CUDA device for this MPI process
    // Each process uses a different GPU (rank 0 -> GPU 0, rank 1 -> GPU 1, etc.)
    CHECK_CUDA(cudaSetDevice(rank % 8));
    
    // Verify device assignment
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    // Print benchmark information (only from rank 0)
    if (rank == 0) {
        std::cout << "\n=== 8-GPU Single-Node Tensor Parallel Benchmark ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
        std::cout << "Data type: FP16 (half precision)" << std::endl;
        std::cout << "GPUs: " << world_size << std::endl;
        std::cout << std::endl;
    }
    
    // Synchronize all processes before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Create cuBLAS handle for this GPU
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    // Allocate device memory for matrices
    // Each GPU holds the full matrices (data parallelism, not tensor parallelism)
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));  // Matrix A: M×K
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));  // Matrix B: K×N
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half))); // Matrix C: M×N
    
    // Initialize random number generator for test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Allocate host memory for initialization
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    
    // Generate random test data
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(dis(gen));  // Convert FP32 to FP16
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(dis(gen));  // Convert FP32 to FP16
    }
    
    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // GEMM parameters: C = alpha * A * B + beta * C
    half alpha = __float2half(1.0f);  // Scaling factor for A * B
    half beta = __float2half(0.0f);   // Scaling factor for C (0 means overwrite)
    
    // Warm-up runs to initialize GPU and avoid cold start effects
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition
            M, N, K,                    // Matrix dimensions
            &alpha,                     // Scaling factor
            d_A, M,                     // Matrix A, leading dimension M
            d_B, K,                     // Matrix B, leading dimension K
            &beta,                      // Scaling factor for C
            d_C, M                      // Matrix C, leading dimension M
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark timing
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform timed GEMM operations
    for (int i = 0; i < num_iterations; i++) {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition
            M, N, K,                    // Matrix dimensions
            &alpha,                     // Scaling factor
            d_A, M,                     // Matrix A, leading dimension M
            d_B, K,                     // Matrix B, leading dimension K
            &beta,                      // Scaling factor for C
            d_C, M                      // Matrix C, leading dimension M
        ));
    }
    
    // Ensure all operations complete before timing
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (double)num_iterations / 1000.0;  // Average time per iteration in ms
    double flops = 2.0 * M * N * K;  // Total floating-point operations (2 ops per multiply-add)
    double gflops = flops / (avg_time_ms * 1e6);  // GFLOPS = operations / (time * 1e9)
    
    // Arrays to collect results from all processes
    double all_times[8];   // Timing results from all GPUs
    double all_gflops[8];  // Performance results from all GPUs
    
    // Gather performance data from all processes to rank 0
    MPI_Gather(&avg_time_ms, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&gflops, 1, MPI_DOUBLE, all_gflops, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Print results summary (only from rank 0)
    if (rank == 0) {
        std::cout << "Per-GPU Results:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Calculate aggregate statistics
        double total_gflops = 0.0;
        double min_time = all_times[0];
        double max_time = all_times[0];
        
        // Print individual GPU performance
        for (int i = 0; i < world_size; i++) {
            std::cout << "GPU " << i << ": "
                      << static_cast<int>(all_gflops[i]) << " GFLOPS, "
                      << all_times[i] << " ms" << std::endl;
            total_gflops += all_gflops[i];
            if (all_times[i] < min_time) min_time = all_times[i];
            if (all_times[i] > max_time) max_time = all_times[i];
        }
        
        // Print summary statistics
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total Performance: " << static_cast<int>(total_gflops) << " GFLOPS" << std::endl;
        std::cout << "Avg per GPU: " << static_cast<int>(total_gflops / world_size) << " GFLOPS" << std::endl;
        std::cout << "Time range: " << min_time << " - " << max_time << " ms" << std::endl;
        
        // Calculate scaling efficiency
        double single_gpu_expected = 770000.0;  // Expected single GPU performance
        double efficiency = (total_gflops / (world_size * single_gpu_expected)) * 100.0;
        std::cout << "Scaling efficiency: " << efficiency << "%" << std::endl;
        std::cout << std::endl;
    }
    
    // Clean up device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    // Clean up host memory
    free(h_A);
    free(h_B);
    
    // Clean up cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

