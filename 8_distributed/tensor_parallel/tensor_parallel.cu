/**
 * @file tensor_parallel.cu
 * @brief Tensor parallelism: sharded GEMM with NCCL AllReduce.
 *
 * @section description
 * This program implements the tensor-parallel matrix multiplication described
 * in chapter 10. The weight matrix B is split by rows (the K dimension) across
 * P GPUs, and the input matrix A is split by columns to match. Each GPU
 * computes a partial product
 *
 *     C_partial = A_r (M x K/P) * B_r (K/P x N)
 *
 * which is a full-size M x N matrix containing 1/P of every output element's
 * inner product. ncclAllReduce with ncclSum then combines the partial products
 * so that every GPU ends up with the complete C. This is the "row-parallel
 * linear layer" pattern used by Megatron-style tensor parallelism.
 *
 * Key concepts illustrated:
 * 1.  **Sharded storage**: Each GPU allocates only its A and B shards
 *     (1/P of each input matrix), not full copies.
 * 2.  **Partial products + AllReduce**: The GEMM produces partial sums, and a
 *     single NCCL collective combines them across all GPUs.
 * 3.  **MPI + NCCL integration**: MPI launches one process per GPU and
 *     broadcasts the NCCL unique id; NCCL handles the GPU-to-GPU data path
 *     (NVLink on an SXM node).
 * 4.  **Verification**: Rank 0 computes the same product in FP32 with
 *     cublasSgemm and compares both the tensor-parallel result and a
 *     single-GPU FP16 reference against it, so you can see that the
 *     distributed result carries the same FP16-level error as the
 *     single-GPU one.
 * 5.  **Honest scaling metrics**: The program times the single-GPU full GEMM
 *     as the baseline, then reports tensor-parallel step time with and
 *     without communication, speedup, efficiency, and the AllReduce share
 *     of the step.
 *
 * @compilation
 * See the accompanying Makefile. Links against CUDA, cuBLAS, NCCL, and MPI.
 * `make`
 *
 * @usage
 * Single node:  `mpirun -np 8 ./tensor_parallel`
 * Multi node:   `mpirun -np 16 --hostfile hosts ./tensor_parallel`
 * Any rank count that divides K = 4096 works (2, 4, 8, 16, ...); each rank
 * needs its own GPU on its node. NCCL routes over NVLink within a node and
 * InfiniBand/Ethernet between nodes automatically.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <nccl.h>
#include <mpi.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(error) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CHECK_NCCL(call) \
do { \
    ncclResult_t result = call; \
    if (result != ncclSuccess) { \
        std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << ncclGetErrorString(result) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Matrix dimensions: C[M,N] = A[M,K] * B[K,N], stored column-major
    // (cuBLAS's native layout). Defaults to 4096^3; pass M N K to override,
    // e.g. `mpirun -np 8 ./tensor_parallel 4096 4096 32768`.
    const int M = (argc > 1) ? atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? atoi(argv[2]) : 4096;
    const int K = (argc > 3) ? atoi(argv[3]) : 4096;

    // Ranks on the same node share that node's GPUs, so device selection
    // must use the node-local rank, not the global one. MPI_Comm_split_type
    // groups ranks by shared-memory domain (i.e., by node).
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (world_size < 2 || local_size > device_count || K % world_size != 0) {
        if (rank == 0) {
            std::cerr << "Error: need >= 2 ranks, one GPU per rank on each "
                         "node, and K divisible by the rank count" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    const int Kp = K / world_size;  // K-dimension slice owned by this rank

    // Pin each rank to its node-local GPU.
    CHECK_CUDA(cudaSetDevice(local_rank));

    // NCCL initialization: rank 0 creates the unique id, MPI broadcasts it,
    // and every rank joins the communicator.
    ncclUniqueId nccl_id;
    if (rank == 0) CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));

    if (rank == 0) {
        std::cout << "\n=== " << world_size
                  << "-GPU single-node tensor parallel (K-split + AllReduce) ==="
                  << std::endl;
        std::cout << "Matrix dimensions (M, N, K): " << M << " x " << N
                  << " x " << K << std::endl;
        std::cout << "Shard per GPU: A " << M << " x " << Kp << ", B " << Kp
                  << " x " << N << std::endl;
        std::cout << "Data type: FP16 inputs, FP16 output, AllReduce in FP16"
                  << std::endl << std::endl;
    }

    // Every rank generates the same full matrices from the same seed, then
    // keeps only its shard. (A real system would load shards from disk; the
    // shared seed just keeps this example self-contained.)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> h_A_full(static_cast<size_t>(M) * K);
    std::vector<float> h_B_full(static_cast<size_t>(K) * N);
    for (auto& v : h_A_full) v = dis(gen);
    for (auto& v : h_B_full) v = dis(gen);

    // Column-major shard extraction.
    // A_r = columns [rank*Kp, (rank+1)*Kp) of A: contiguous, lda = M.
    // B_r = rows    [rank*Kp, (rank+1)*Kp) of B: strided copy, ldb = Kp.
    std::vector<half> h_A_shard(static_cast<size_t>(M) * Kp);
    std::vector<half> h_B_shard(static_cast<size_t>(Kp) * N);
    const size_t a_off = static_cast<size_t>(rank) * Kp * M;
    for (size_t i = 0; i < h_A_shard.size(); i++) {
        h_A_shard[i] = __float2half(h_A_full[a_off + i]);
    }
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < Kp; k++) {
            h_B_shard[static_cast<size_t>(n) * Kp + k] =
                __float2half(h_B_full[static_cast<size_t>(n) * K + rank * Kp + k]);
        }
    }

    // Device allocations: only the shards plus the full-size output buffer
    // that AllReduce fills in. Input memory per GPU shrinks by a factor of P.
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, static_cast<size_t>(M) * Kp * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(Kp) * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, static_cast<size_t>(M) * N * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_shard.data(),
                          h_A_shard.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_shard.data(),
                          h_B_shard.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // One tensor-parallel step: local partial GEMM, then AllReduce the
    // partial products so every GPU holds the complete C.
    auto tp_step = [&]() {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, Kp,
            &alpha,
            d_A, M,
            d_B, Kp,
            &beta,
            d_C, M
        ));
        CHECK_NCCL(ncclAllReduce(d_C, d_C, static_cast<size_t>(M) * N,
                                 ncclHalf, ncclSum, nccl_comm, stream));
    };

    // --- Correctness check (once, before timing) ---
    tp_step();
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (rank == 0) {
        // FP32 reference on rank 0's GPU.
        float *d_A32, *d_B32, *d_C32;
        CHECK_CUDA(cudaMalloc(&d_A32, h_A_full.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B32, h_B_full.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C32, static_cast<size_t>(M) * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A32, h_A_full.data(),
                              h_A_full.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B32, h_B_full.data(),
                              h_B_full.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        float alpha32 = 1.0f, beta32 = 0.0f;
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K, &alpha32, d_A32, M, d_B32, K,
                                 &beta32, d_C32, M));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        std::vector<float> h_C_ref(static_cast<size_t>(M) * N);
        std::vector<half> h_C_tp(static_cast<size_t>(M) * N);
        CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C32,
                              h_C_ref.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_C_tp.data(), d_C,
                              h_C_tp.size() * sizeof(half),
                              cudaMemcpyDeviceToHost));

        double max_abs_err = 0.0, max_ref = 0.0;
        for (size_t i = 0; i < h_C_ref.size(); i++) {
            double err = std::fabs(__half2float(h_C_tp[i]) - h_C_ref[i]);
            if (err > max_abs_err) max_abs_err = err;
            double mag = std::fabs(h_C_ref[i]);
            if (mag > max_ref) max_ref = mag;
        }
        std::cout << "Verification vs FP32 reference:" << std::endl;
        std::cout << "  max |C| = " << max_ref
                  << ", max abs error = " << max_abs_err
                  << " (" << (100.0 * max_abs_err / max_ref)
                  << "% of max |C|)" << std::endl;
        // FP16 has ~3 decimal digits; partial sums pass through FP16 once
        // more than the single-GPU path, so allow 1% of the output range.
        if (max_abs_err > 0.01 * max_ref) {
            std::cerr << "FAILED: tensor-parallel result diverges from FP32 "
                         "reference" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "  PASSED" << std::endl << std::endl;

        CHECK_CUDA(cudaFree(d_A32));
        CHECK_CUDA(cudaFree(d_B32));
        CHECK_CUDA(cudaFree(d_C32));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Single-GPU baseline (rank 0): the full, unsharded GEMM ---
    double baseline_ms = 0.0;
    if (rank == 0) {
        half *d_Af, *d_Bf, *d_Cf;
        CHECK_CUDA(cudaMalloc(&d_Af, static_cast<size_t>(M) * K * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_Bf, static_cast<size_t>(K) * N * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_Cf, static_cast<size_t>(M) * N * sizeof(half)));
        std::vector<half> h_tmp(h_A_full.size());
        for (size_t i = 0; i < h_A_full.size(); i++) h_tmp[i] = __float2half(h_A_full[i]);
        CHECK_CUDA(cudaMemcpy(d_Af, h_tmp.data(), h_tmp.size() * sizeof(half),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < h_B_full.size(); i++) h_tmp[i] = __float2half(h_B_full[i]);
        CHECK_CUDA(cudaMemcpy(d_Bf, h_tmp.data(), h_tmp.size() * sizeof(half),
                              cudaMemcpyHostToDevice));

        auto full_gemm = [&]() {
            CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     M, N, K, &alpha, d_Af, M, d_Bf, K,
                                     &beta, d_Cf, M));
        };
        for (int i = 0; i < 3; i++) full_gemm();
        CHECK_CUDA(cudaStreamSynchronize(stream));

        cudaEvent_t ev_start, ev_stop;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));
        const int iters = 20;
        CHECK_CUDA(cudaEventRecord(ev_start, stream));
        for (int i = 0; i < iters; i++) full_gemm();
        CHECK_CUDA(cudaEventRecord(ev_stop, stream));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));
        float total_ms;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
        baseline_ms = total_ms / iters;
        double tflops = 2.0 * M * N * K / (baseline_ms * 1e-3) / 1e12;
        std::cout << "Single-GPU baseline (full " << M << "x" << N << "x" << K
                  << " FP16 GEMM on GPU 0): " << baseline_ms << " ms, "
                  << tflops << " TFLOPS" << std::endl << std::endl;
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));
        CHECK_CUDA(cudaFree(d_Af));
        CHECK_CUDA(cudaFree(d_Bf));
        CHECK_CUDA(cudaFree(d_Cf));
    }
    MPI_Bcast(&baseline_ms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Timed tensor-parallel runs ---
    const int iters = 20;
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    // (a) local partial GEMM only
    for (int i = 0; i < 3; i++) tp_step();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, Kp, &alpha, d_A, M, d_B, Kp,
                                 &beta, d_C, M));
    }
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float gemm_total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&gemm_total_ms, ev_start, ev_stop));
    double gemm_ms = gemm_total_ms / iters;

    // (b) full step: partial GEMM + AllReduce
    MPI_Barrier(MPI_COMM_WORLD);
    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; i++) tp_step();
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    float step_total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&step_total_ms, ev_start, ev_stop));
    double step_ms = step_total_ms / iters;

    // The step completes when the slowest rank finishes.
    double gemm_ms_max, step_ms_max;
    MPI_Reduce(&gemm_ms, &gemm_ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&step_ms, &step_ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double per_rank_tflops = 2.0 * M * N * Kp / (gemm_ms * 1e-3) / 1e12;
    std::vector<double> all_tflops(world_size);
    MPI_Gather(&per_rank_tflops, 1, MPI_DOUBLE, all_tflops.data(), 1,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Per-GPU local GEMM throughput (shard "
                  << M << "x" << N << "x" << Kp << "):" << std::endl;
        for (int i = 0; i < world_size; i++) {
            std::cout << "  GPU " << i << ": " << all_tflops[i] << " TFLOPS"
                      << std::endl;
        }
        double comm_ms = step_ms_max - gemm_ms_max;
        double speedup = baseline_ms / step_ms_max;
        double efficiency = 100.0 * speedup / world_size;
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Local partial GEMM (slowest rank): " << gemm_ms_max
                  << " ms" << std::endl;
        std::cout << "Full TP step, GEMM + AllReduce (slowest rank): "
                  << step_ms_max << " ms" << std::endl;
        std::cout << "AllReduce share of step: "
                  << (100.0 * comm_ms / step_ms_max) << "%" << std::endl;
        std::cout << "Speedup vs single GPU: " << speedup << "x on "
                  << world_size << " GPUs" << std::endl;
        std::cout << "Scaling efficiency: " << efficiency << "%" << std::endl;
        std::cout << std::endl;
    }

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(nccl_comm));
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
