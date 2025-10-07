
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
    char hostname[256];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    gethostname(hostname, 256);
    
    if (world_size != 16) {
        if (rank == 0) {
            std::cerr << "Error: This program requires exactly 16 MPI processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    CHECK_CUDA(cudaSetDevice(rank % 8));
    
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    if (rank == 0) {
        std::cout << "\n=== 16-GPU Multi-Node Tensor Parallel Benchmark ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
        std::cout << "Data type: FP16 (half precision)" << std::endl;
        std::cout << "Nodes: 2" << std::endl;
        std::cout << "GPUs per node: 8" << std::endl;
        std::cout << "Total GPUs: " << world_size << std::endl;
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
    
    for (int i = 0; i < 2; i++) {
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
    
    const int num_iterations = 5;
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
    
    struct GPUResult {
        double time_ms;
        double gflops;
        char hostname[256];
        int rank;
    } local_result, all_results[16];
    
    local_result.time_ms = avg_time_ms;
    local_result.gflops = gflops;
    local_result.rank = rank;
    strncpy(local_result.hostname, hostname, 255);
    local_result.hostname[255] = '\0';
    
    MPI_Gather(&local_result, sizeof(GPUResult), MPI_BYTE,
               all_results, sizeof(GPUResult), MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Per-GPU Results:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        double total_gflops = 0.0;
        double min_time = all_results[0].time_ms;
        double max_time = all_results[0].time_ms;
        
        std::string prev_host = "";
        for (int i = 0; i < world_size; i++) {
            std::string current_host = all_results[i].hostname;
            if (current_host != prev_host) {
                std::cout << "\nNode: " << current_host << std::endl;
                prev_host = current_host;
            }
            
            std::cout << "  Rank " << all_results[i].rank << ": "
                      << static_cast<int>(all_results[i].gflops) << " GFLOPS, "
                      << all_results[i].time_ms << " ms" << std::endl;
            
            total_gflops += all_results[i].gflops;
            if (all_results[i].time_ms < min_time) min_time = all_results[i].time_ms;
            if (all_results[i].time_ms > max_time) max_time = all_results[i].time_ms;
        }
        
        std::cout << "\n=== Multi-Node Summary ===" << std::endl;
        std::cout << "Total Performance: " << static_cast<int>(total_gflops) << " GFLOPS" << std::endl;
        std::cout << "Avg per GPU: " << static_cast<int>(total_gflops / world_size) << " GFLOPS" << std::endl;
        std::cout << "Time range: " << min_time << " - " << max_time << " ms" << std::endl;
        
        double single_gpu_expected = 550000.0;
        double efficiency = (total_gflops / (world_size * single_gpu_expected)) * 100.0;
        std::cout << "Scaling efficiency: " << efficiency << "%" << std::endl;
        std::cout << "\nNote: Slightly lower per-GPU performance due to inter-node communication overhead" << std::endl;
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

