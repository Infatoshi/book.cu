
/**
 * @file 8gpu_single_node.cu
 * @brief A benchmark for single-node, 8-GPU data parallelism using MPI and cuBLAS.
 *
 * @section description
 * This program serves as a foundational benchmark for multi-GPU computing on a single
 * node. It uses MPI to launch 8 processes, with each process managing one GPU.
 * Each GPU independently performs an identical, large GEMM (General Matrix Multiplication)
 * operation. The primary purpose is to measure the baseline performance of each GPU
 * and the aggregate throughput of the node under a full load.
 *
 * Although the directory is named "tensor_parallel", this specific example demonstrates
 * a **data parallel** pattern, not a true tensor parallel operation.
 *
 * Key Concepts Illustrated:
 * 1.  **Data Parallelism**: Each of the 8 GPUs runs an identical copy of the task (the
 *     GEMM operation) on identical data. This is a common strategy for tasks that
 *     can be easily divided, but it doesn't allow a single operation (like a matrix
 *     multiplication) to exceed the memory of a single GPU.
 *
 * 2.  **Tensor Parallelism (for context)**: In contrast, true tensor parallelism would
 *     involve splitting the matrices themselves across the 8 GPUs. For example, in
 *     `C = A * B`, matrix `B` could be split column-wise across the GPUs. Each GPU
 *     would compute a slice of `C`, `C_i = A * B_i`. Afterwards, a collective
 *     communication operation like `All-Gather` would be needed to assemble the full
 *     `C` matrix on every GPU. This is more complex but allows for training models
 *     that are too large to fit on a single GPU.
 *
 * 3.  **MPI for Process Management**: `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`,
 *     and `MPI_Finalize` are used to manage the 8 processes on the single node.
 *
 * 4.  **GPU Affinity**: Each MPI rank is pinned to a specific GPU using
 *     `cudaSetDevice(rank)`, ensuring no two processes compete for the same device.
 *
 * 5.  **Result Aggregation**: `MPI_Gather` is used to collect performance results from
 *     each of the 8 processes and send them to the root process (rank 0) for display.
 *
 * @compilation
 * See the accompanying Makefile. It links against CUDA, cuBLAS, and MPI libraries.
 * `make 8gpu`
 *
 * @usage
 * The executable is launched via an `mpirun` command, typically from a script like `run_all.sh`.
 * `mpirun -np 8 ./8gpu_single_node`
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <cuda_fp16.h>

/**
 * @brief CUDA error checking macro for distributed applications.
 * If a CUDA call fails, it prints a detailed error message and then calls
 * `MPI_Abort` to terminate all processes in the MPI communicator gracefully.
 * @param call The CUDA API function call to be checked.
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
 * @brief cuBLAS error checking macro for distributed applications.
 * If a cuBLAS call fails, it prints a detailed error message and then calls
 * `MPI_Abort` to terminate all processes in the MPI communicator gracefully.
 * @param call The cuBLAS API function call to be checked.
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

/**
 * @brief Main entry point for the 8-GPU single-node benchmark.
 */
int main(int argc, char* argv[]) {
    // Initialize the MPI execution environment
    MPI_Init(&argc, &argv);
    
    // Get the rank of the process and the total number of processes
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // The ID of the current process (0 to 7)
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Total number of processes
    
    // This benchmark is hardcoded for 8 GPUs on a single node.
    if (world_size != 8) {
        if (rank == 0) { // Only the root process should print the error
            std::cerr << "Error: This program requires exactly 8 MPI processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Define the dimensions for the GEMM operation: C[M,N] = A[M,K] * B[K,N]
    const int M = 4096;  // Rows of A and C
    const int N = 4096;  // Columns of B and C
    const int K = 4096;  // Columns of A and Rows of B
    
    // Pin the current MPI process to a specific GPU.
    // Rank `i` gets assigned to GPU `i`.
    CHECK_CUDA(cudaSetDevice(rank));
    
    // Verify the device assignment for sanity checking.
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    // Rank 0 is designated to print headers and final summaries.
    if (rank == 0) {
        std::cout << "\n=== 8-GPU Single-Node Data Parallel Benchmark ===" << std::endl;
        std::cout << "Matrix dimensions (M, N, K): " << M << " x " << N << " x " << K << std::endl;
        std::cout << "Data type: FP16 (half precision)" << std::endl;
        std::cout << "Total GPUs: " << world_size << std::endl;
        std::cout << "(Note: This is a data-parallel benchmark, not tensor-parallel.)" << std::endl;
        std::cout << std::endl;
    }
    
    // Use a barrier to synchronize all MPI processes before starting the main work.
    // This ensures that all GPUs start the benchmark at roughly the same time.
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Resource Initialization ---
    
    // Create a handle for cuBLAS operations. Each process/GPU needs its own handle.
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    
    // Allocate memory on the GPU for the matrices.
    // In this data-parallel setup, each GPU allocates memory for the full matrices.
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));  // Matrix A
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));  // Matrix B
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(half)));  // Result Matrix C
    
    // Create random matrices on the host (CPU).
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    
    // Populate host matrices with random values converted to half precision.
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(dis(gen));
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(dis(gen));
    }
    
    // Copy the input matrices from host memory to device memory.
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // Define GEMM scaling factors: C = alpha * A * B + beta * C
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);   // beta = 0 effectively means C = A * B
    
    // --- Warm-up Runs ---
    // Perform a few untimed iterations to warm up the GPU, JIT compile kernels, etc.
    for (int i = 0; i < 3; i++) {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition on input matrices
            M, N, K,
            &alpha,
            d_A, M,                     // Matrix A and its leading dimension
            d_B, K,                     // Matrix B and its leading dimension
            &beta,
            d_C, M                      // Result Matrix C and its leading dimension
        ));
    }
    // Synchronize the device to ensure all warm-up operations are complete.
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Synchronize all MPI processes again before starting the timed benchmark.
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Timed Benchmark ---
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform the main batch of GEMM operations.
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
    
    // Block the CPU thread until all previously issued commands on the device are complete.
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // --- Performance Calculation ---
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (double)num_iterations / 1000.0;
    // Theoretical FLOPS for one GEMM operation
    double flops = 2.0 * M * N * K;
    // GFLOPS = (Total Operations / 1e9) / time_in_seconds
    double gflops = flops / (avg_time_ms * 1e6);
    
    // --- Result Aggregation ---
    // Arrays on rank 0 to store the results gathered from all processes.
    double all_times[8];
    double all_gflops[8];
    
    // Gather the timing and GFLOPS data from each process to the root process (rank 0).
    MPI_Gather(&avg_time_ms, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&gflops, 1, MPI_DOUBLE, all_gflops, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // --- Reporting ---
    if (rank == 0) {
        std::cout << "Per-GPU Results:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Calculate aggregate statistics on rank 0.
        double total_gflops = 0.0;
        double min_time = all_times[0];
        double max_time = all_times[0];
        
        // Print individual GPU performance.
        for (int i = 0; i < world_size; i++) {
            std::cout << "  GPU " << i << ": "
                      << static_cast<int>(all_gflops[i]) << " GFLOPS, "
                      << all_times[i] << " ms" << std::endl;
            total_gflops += all_gflops[i];
            if (all_times[i] < min_time) min_time = all_times[i];
            if (all_times[i] > max_time) max_time = all_times[i];
        }
        
        // Print the final summary.
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Total Aggregate Performance: " << static_cast<int>(total_gflops) << " GFLOPS" << std::endl;
        std::cout << "Average Performance per GPU: " << static_cast<int>(total_gflops / world_size) << " GFLOPS" << std::endl;
        std::cout << "Fastest/Slowest Time Range: " << min_time << " - " << max_time << " ms" << std::endl;
        
        // A placeholder for theoretical single-GPU performance to calculate efficiency.
        double single_gpu_peak_gflops = 77000.0; // Example peak for H100 FP16
        double efficiency = (total_gflops / (world_size * single_gpu_peak_gflops)) * 100.0;
        std::cout << "Approx. Scaling Efficiency vs Peak: " << efficiency << "%" << std::endl;
        std::cout << std::endl;
    }
    
    // --- Cleanup ---
    
    // Free the memory allocated on the device.
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    // Free the memory allocated on the host.
    free(h_A);
    free(h_B);
    
    // Destroy the cuBLAS handle.
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    
    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}

