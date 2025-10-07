
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 8) {
        if (rank == 0) {
            std::cerr << "Error: This program requires exactly 8 MPI processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    
    CHECK_CUDA(cudaSetDevice(rank % 8));
    
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    if (rank == 0) {
        std::cout << "\n=== 8-GPU Single-Node Tensor Parallel Benchmark ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
        std::cout << "Data type: FP16 (half precision)" << std::endl;
        std::cout << "GPUs: " << world_size << std::endl;
        std::cout << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(dis(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(dis(gen));
    }
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M
        ));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (double)num_iterations / 1000.0;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (avg_time_ms * 1e6);
    
    double all_times[8];
    double all_gflops[8];
    
    MPI_Gather(&avg_time_ms, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&gflops, 1, MPI_DOUBLE, all_gflops, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Per-GPU Results:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        double total_gflops = 0.0;
        double min_time = all_times[0];
        double max_time = all_times[0];
        
        for (int i = 0; i < world_size; i++) {
            std::cout << "GPU " << i << ": "
                      << static_cast<int>(all_gflops[i]) << " GFLOPS, "
                      << all_times[i] << " ms" << std::endl;
            total_gflops += all_gflops[i];
            if (all_times[i] < min_time) min_time = all_times[i];
            if (all_times[i] > max_time) max_time = all_times[i];
        }
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total Performance: " << static_cast<int>(total_gflops) << " GFLOPS" << std::endl;
        std::cout << "Avg per GPU: " << static_cast<int>(total_gflops / world_size) << " GFLOPS" << std::endl;
        std::cout << "Time range: " << min_time << " - " << max_time << " ms" << std::endl;
        
        double single_gpu_expected = 770000.0;
        double efficiency = (total_gflops / (world_size * single_gpu_expected)) * 100.0;
        std::cout << "Scaling efficiency: " << efficiency << "%" << std::endl;
        std::cout << std::endl;
    }
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    MPI_Finalize();
    return 0;
}

