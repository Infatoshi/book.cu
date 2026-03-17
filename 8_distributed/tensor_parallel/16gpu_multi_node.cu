/**
 * @file 16gpu_multi_node.cu
 * @brief A benchmark for multi-node, multi-GPU tensor parallelism using MPI and NCCL.
 *
 * @section description
 * This program demonstrates a foundational concept in large-scale model training:
 * tensor parallelism across multiple nodes. It uses MPI to launch processes and
 * NCCL for high-speed, inter-GPU communication. Although this specific example
 * does not implement a true tensor-parallel operation (which would involve splitting
 * matrices and using collective operations like All-Reduce), it establishes the
 * necessary communication framework and performance baseline.
 *
 * Each of the 16 MPI processes is bound to a specific GPU. It performs a local
 * SGEMM (Single-precision General Matrix Multiplication) operation and measures its
 * performance in GFLOPS. The results from all GPUs are then gathered to the root
 * process (rank 0) using `MPI_Gather` for aggregation and reporting.
 *
 * Key Concepts Illustrated:
 * 1.  **MPI for Process Management**: `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`,
 *     and `MPI_Finalize` are used to manage the lifecycle of the distributed application.
 *     The program is launched with `mpirun`, which starts 16 processes across the
 *     nodes defined in the hostfile.
 *
 * 2.  **GPU Affinity**: Each MPI rank is assigned to a specific GPU using
 *     `cudaSetDevice(rank % 8)`. This ensures that computations for a given rank
 *     happen on a dedicated GPU.
 *
 * 3.  **Inter-Process Communication (IPC) via MPI**: `MPI_Gather` is used as a
 *     simple collective communication primitive. It collects a piece of data
 *     (the `GPUResult` struct) from every process and delivers all pieces to the
 *     root process. This is a common pattern for collecting statistics or results.
 *     In a real tensor parallel setup, you would use `ncclAllReduce` or `ncclAllGather`.
 *
 * 4.  **Performance Baselining**: By measuring the performance of an identical,
 *     independent operation on each GPU, we establish a baseline. Any deviation
 *     from near-linear scaling in a real tensor-parallel operation can be attributed
 *     to communication overhead.
 *
 * @compilation
 * See the accompanying Makefile. It links against CUDA, cuBLAS, and MPI libraries.
 * `make 16gpu`
 *
 * @usage
 * The executable is launched via an `mpirun` command, typically from a script like `run_all.sh`.
 * `mpirun -np 16 --hostfile ../hosts ./16gpu_multi_node`
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <cuda_fp16.h>

/**
 * @brief Macro to wrap CUDA API calls for robust error checking.
 * If a CUDA call fails, it prints the error, file, and line number,
 * and then aborts the MPI job.
 * @param call The CUDA API call.
 */
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

/**
 * @brief Macro to wrap cuBLAS API calls for robust error checking.
 * If a cuBLAS call fails, it prints the error, file, and line number,
 * and then aborts the MPI job.
 * @param call The cuBLAS API call.
 */
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

/**
 * @brief Main entry point for the 16-GPU multi-node benchmark.
 */
int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    char hostname[256];
    // Get the rank (ID) of the current process and the total number of processes (world size)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the hostname of the node this process is running on
    gethostname(hostname, 256);
    
    // This benchmark is specifically designed for 16 GPUs.
    if (world_size != 16) {
        if (rank == 0) { // Only have the root process print the error
            std::cerr << "Error: This program requires exactly 16 MPI processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Define matrix dimensions for the GEMM operation (M x K) * (K x N) = (M x N)
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    
    // Assign each MPI rank to a GPU. With 16 ranks and 8 GPUs per node,
    // ranks 0-7 map to GPUs 0-7 on the first node, and ranks 8-15 map to
    // GPUs 0-7 on the second node.
    CHECK_CUDA(cudaSetDevice(rank % 8));
    
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    // Rank 0 is responsible for printing the header and summary information.
    if (rank == 0) {
        std::cout << "\n=== 16-GPU Multi-Node Tensor Parallel Benchmark ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << " x " << N << " x " << K << std::endl;
        std::cout << "Data type: FP16 (half precision)" << std::endl;
        std::cout << "Nodes: 2" << std::endl;
        std::cout << "GPUs per node: 8" << std::endl;
        std::cout << "Total GPUs: " << world_size << std::endl;
        std::cout << std::endl;
    }
    
    // Synchronize all processes before starting the benchmark.
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Resource Initialization ---
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    // Allocate memory for matrices A, B, and C on the GPU.
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));
    
    // Initialize matrices A and B on the host with random data.
    // In a real application, data would be loaded or generated differently.
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
    
    // Copy the host matrices to their respective device memory.
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // Define scaling factors for the GEMM operation.
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);
    
    // --- Warm-up Runs ---
    // Perform a few iterations to warm up the GPU and ensure accurate timing.
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
    
    // Synchronize all MPI processes after the warm-up.
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Timed Benchmark ---
    const int num_iterations = 5;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        // Perform the half-precision matrix multiplication.
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N, // No transpose on A or B
            M, N, K,                 // Dimensions
            &alpha,                  // Scaling factor for A*B
            d_A, M,                  // Matrix A and its leading dimension
            d_B, K,                  // Matrix B and its leading dimension
            &beta,                   // Scaling factor for C
            d_C, M                   // Matrix C and its leading dimension
        ));
    }
    
    // Block until all previously issued CUDA calls on this device are complete.
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // --- Result Calculation and Gathering ---
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (double)num_iterations / 1000.0;
    // FLOPS = 2 * M * N * K for a standard GEMM
    double flops = 2.0 * M * N * K;
    // GFLOPS = (FLOPS / 1e9) / (time_in_seconds)
    double gflops = flops / (avg_time_ms * 1e6);
    
    // A struct to hold the results from each GPU.
    struct GPUResult {
        double time_ms;
        double gflops;
        char hostname[256];
        int rank;
    } local_result, all_results[16];
    
    // Populate the local result struct.
    local_result.time_ms = avg_time_ms;
    local_result.gflops = gflops;
    local_result.rank = rank;
    strncpy(local_result.hostname, hostname, 255);
    local_result.hostname[255] = '\0';
    
    // Gather the results from all processes to the root process (rank 0).
    // Each process sends its `local_result`, and rank 0 receives them into
    // the `all_results` array.
    MPI_Gather(&local_result, sizeof(GPUResult), MPI_BYTE,
               all_results, sizeof(GPUResult), MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    // --- Reporting ---
    if (rank == 0) {
        std::cout << "Per-GPU Results:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        double total_gflops = 0.0;
        double min_time = all_results[0].time_ms;
        double max_time = all_results[0].time_ms;
        
        std::string prev_host = "";
        for (int i = 0; i < world_size; i++) {
            // Print a header for each new node.
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
        
        // Print the aggregated summary.
        std::cout << "\n=== Multi-Node Summary ===" << std::endl;
        std::cout << "Total Performance: " << static_cast<int>(total_gflops) << " GFLOPS" << std::endl;
        std::cout << "Avg per GPU: " << static_cast<int>(total_gflops / world_size) << " GFLOPS" << std::endl;
        std::cout << "Time range: " << min_time << " - " << max_time << " ms" << std::endl;
        
        // A placeholder for theoretical single-GPU performance to calculate efficiency.
        double single_gpu_expected_tflops = 55.0; // Theoretical TFLOPS
        double efficiency = (total_gflops / (world_size * single_gpu_expected_tflops * 1000));
        std::cout << "Approx. Scaling Efficiency vs Peak: " << efficiency * 100.0 << "%" << std::endl;
        std::cout << "\nNote: This benchmark measures independent GEMM performance. True tensor" << std::endl;
        std::cout << "parallelism would introduce communication overhead (e.g., All-Reduce), " << std::endl;
        std::cout << "resulting in lower per-GPU throughput." << std::endl;
        std::cout << std::endl;
    }
    
    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

