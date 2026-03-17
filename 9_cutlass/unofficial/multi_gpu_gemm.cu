/**
 * @file multi_gpu_gemm.cu
 * @brief A multi-GPU GEMM scaling benchmark using NVIDIA's CUTLASS library.
 *
 * @details This example demonstrates how to run multiple independent GEMM computations across
 *          several GPUs to measure scaling performance. It uses the same high-performance
 *          CUTLASS kernel as the single-GPU example, optimized for the Hopper architecture.
 *
 *          **Important**: This is NOT a distributed GEMM implementation that solves a single large
 *          problem across multiple GPUs. Instead, it runs a separate, identical GEMM on each
 * a         vailable GPU and aggregates the performance. This is useful for testing the throughput
 *          of a system with multiple GPUs.
 *
 *          This file showcases:
 *          - Initialization of multiple GPUs.
 *          - Use of NCCL for basic communicator setup (though not used for communication in the GEMM itself).
 *          - A loop to launch the same CUTLASS GEMM kernel on each device.
 *          - Calculation of aggregate performance and scaling efficiency.
 *
 *          The CUTLASS kernel configuration is identical to the single-GPU example.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.h"
#include "cutlass/epilogue/collective/collective_builder.h"
#include "cutlass/epilogue/fusion/linear_combination.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "nccl.h"

using namespace cute;

//
// CUTLASS GEMM Configuration (Identical to single-GPU example)
//
using ArchTag = cutlass::arch::Sm90;
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementC = ElementOutput;
using ElementD = ElementOutput;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;

static constexpr int AlignmentA = 16 / sizeof(ElementInput);
static constexpr int AlignmentB = 16 / sizeof(ElementInput);
static constexpr int AlignmentC = 16 / sizeof(ElementOutput);
static constexpr int AlignmentD = 16 / sizeof(ElementOutput);

using ElementCompute = float;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementCompute, ElementOutput, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,
    ElementInput, LayoutA, AlignmentA,
    ElementInput, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define NCCL_CHECK(call) \
    do { \
        ncclResult_t err = call; \
        if (err != ncclSuccess) {
            std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << ncclGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper functions (to_float, from_float, cpu_gemm, etc.) are identical to single_gpu_gemm.cu
// and are included here for completeness.

template<typename T>
float to_float(T val) {
    if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return float(val);
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        return float(val);
    } else {
        return static_cast<float>(val);
    }
}

template<typename T>
T from_float(float val) {
    if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return cutlass::half_t(val);
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        return cutlass::float_e4m3_t(val);
    } else {
        return static_cast<T>(val);
    }
}

template<typename InType, typename OutType>
void cpu_gemm(const std::vector<InType>& A, const std::vector<InType>& B,
              std::vector<OutType>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = to_float(A[i * K + k]);
                float b_val = to_float(B[k * N + j]);
                sum += a_val * b_val;
            }
            C[i * N + j] = from_float<OutType>(sum);
        }
    }
}

template<typename T>
void initialize_random(std::vector<T>& data, int seed = 42) {
    std::srand(seed);
    for (auto& val : data) {
        val = from_float<T>((std::rand() / float(RAND_MAX)) * 2.0f - 1.0f);
    }
}

template<typename T>
bool verify_results(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result,
                    float tolerance = 0.5f) {
    if (gpu_result.size() != cpu_result.size()) return false;
    
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int non_zero_count = 0;
    
    for (size_t i = 0; i < gpu_result.size(); i++) {
        float gpu_val = to_float(gpu_result[i]);
        float cpu_val = to_float(cpu_result[i]);
        
        if (std::abs(cpu_val) > 1e-5f) {
            float rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);
            max_error = std::max(max_error, rel_error);
            avg_error += rel_error;
            non_zero_count++;
        }
    }
    
    avg_error /= non_zero_count > 0 ? non_zero_count : 1;
    
    std::cout << "Max relative error: " << max_error 
              << " (over " << non_zero_count << " non-zero elements)" << std::endl;
    std::cout << "Average relative error: " << avg_error 
              << " (" << (avg_error * 100.0f) << "%)" << std::endl;
    
    bool passed = avg_error < tolerance;
    std::cout << (passed ? "✓" : "✗") << " CPU verification " 
              << (passed ? "PASSED" : "FAILED") 
              << " (average error < " << (tolerance * 100.0f) << "%)" << std::endl;
    
    return passed;
}

/**
 * @brief Runs a single GEMM operation on the currently selected GPU and returns its performance.
 * @return Performance in TFLOPS.
 */
template<typename InType, typename OutType>
double run_gemm(const std::vector<InType>& h_A, const std::vector<InType>& h_B,
              std::vector<OutType>& h_C, int M, int N, int K,
              int warmup_iters = 5, int bench_iters = 10) {
    
    InType *d_A, *d_B;
    OutType *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(InType)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(InType)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(OutType)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(InType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(InType), cudaMemcpyHostToDevice));
    
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    
    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {d_A, stride_a, d_B, stride_b},
        {{1.0f, 0.0f}, d_C, stride_c, d_C, stride_d}
    };
    
    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(args);
    
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_ptr, workspace_size));
    }
    
    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM initialization failed" << std::endl;
        exit(1);
    }
    
    for (int i = 0; i < warmup_iters; i++) {
        gemm_op.run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        gemm_op.run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / bench_iters;
    double tflops = (2.0 * M * N * K) / (time_ms * 1e9);
    
    std::cout << "Average time: " << time_ms << " ms | Performance: " << tflops << " TFLOPS" << std::endl;
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(OutType), cudaMemcpyDeviceToHost));
    
    if (workspace_ptr) cudaFree(workspace_ptr);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return tflops;
}

int main(int argc, char** argv) {
    std::cout << "\n=== CUTLASS Multi-GPU GEMM Scaling Benchmark ===" << std::endl;
    
    // Get the number of available GPUs.
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    std::cout << "Detected " << num_devices << " GPU(s)" << std::endl;
    
    if (num_devices < 2) {
        std::cout << "This benchmark is for multi-GPU scaling. Only 1 GPU detected. "
                  << "Running a single-GPU benchmark instead." << std::endl;
    }
    
    // Initialize NCCL communicators if we have more than one GPU.
    // NCCL is a library for multi-GPU and multi-node collective communication.
    ncclComm_t* comms = nullptr;
    if (num_devices > 1) {
        comms = new ncclComm_t[num_devices];
        // Create a unique ID for the NCCL communicator group.
        ncclUniqueId nccl_id;
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        
        // Initialize one communicator per GPU.
        for (int i = 0; i < num_devices; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            NCCL_CHECK(ncclCommInitRank(&comms[i], num_devices, nccl_id, i));
        }
    }
    
    std::cout << std::endl;
    
    //
    // Part 1: Verification (on a single GPU)
    //
    // Before benchmarking, we run a smaller problem on a single GPU (device 0) to verify
    // the correctness of the CUTLASS kernel against a CPU reference.
    //
    {
        std::cout << "=== CPU Verification (1024³) on GPU 0 ===" << std::endl;
        const int M = 1024, N = 1024, K = 1024;
        
        std::vector<ElementInput> h_A(M * K);
        std::vector<ElementInput> h_B(K * N);
        std::vector<ElementOutput> h_C_gpu(M * N);
        std::vector<ElementOutput> h_C_cpu(M * N);
        
        initialize_random(h_A, 42);
        initialize_random(h_B, 43);
        
        std::cout << "Running CPU GEMM for verification..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        std::cout << "CPU GEMM time: " << cpu_time << " ms" << std::endl;
        
        // Set the active device to GPU 0 for the verification run.
        CUDA_CHECK(cudaSetDevice(0));
        run_gemm(h_A, h_B, h_C_gpu, M, N, K, 1, 1);
        
        bool passed = verify_results(h_C_gpu, h_C_cpu);
        
        if (!passed) {
            std::cerr << "\n✗ Verification failed! Not proceeding to benchmark." << std::endl;
            if (comms) {
                for (int i = 0; i < num_devices; i++) {
                    ncclCommDestroy(comms[i]);
                }
                delete[] comms;
            }
            return 1;
        }
        std::cout << std::endl;
    }
    
    //
    // Part 2: Multi-GPU Scaling Benchmark
    //
    // This section runs an independent GEMM benchmark on each available GPU.
    //
    {
        std::cout << "=== Multi-GPU Scaling Benchmark (8192³) ===" << std::endl;
        const int M = 8192, N = 8192, K = 8192;
        
        // Host data for each GPU.
        std::vector<std::vector<ElementInput>> h_A(num_devices);
        std::vector<std::vector<ElementInput>> h_B(num_devices);
        std::vector<std::vector<ElementOutput>> h_C(num_devices);
        
        // Initialize data for each GPU's benchmark.
        for (int device = 0; device < num_devices; device++) {
            h_A[device].resize(M * K);
            h_B[device].resize(K * N);
            h_C[device].resize(M * N);
            
            initialize_random(h_A[device], 42 + device);
            initialize_random(h_B[device], 43 + device);
        }
        
        std::vector<double> tflops_per_gpu(num_devices);
        
        // Loop over each device and run the benchmark.
        for (int device = 0; device < num_devices; device++) {
            CUDA_CHECK(cudaSetDevice(device));
            std::cout << "Running benchmark on GPU " << device << "..." << std::endl;
            tflops_per_gpu[device] = run_gemm(h_A[device], h_B[device], h_C[device], M, N, K, 5, 10);
        }
        
        // If more than one GPU was used, calculate and print scaling results.
        if (num_devices > 1) {
            double total_tflops = std::accumulate(tflops_per_gpu.begin(), tflops_per_gpu.end(), 0.0);
            double single_gpu_tflops = tflops_per_gpu[0];
            double ideal_tflops = num_devices * single_gpu_tflops;
            double scaling_efficiency = (total_tflops / ideal_tflops) * 100.0;
            
            std::cout << "\n=== Scaling Summary ===" << std::endl;
            std::cout << "Total aggregate performance: " << total_tflops << " TFLOPS" << std::endl;
            std::cout << "Ideal performance: " << ideal_tflops << " TFLOPS (based on GPU 0)" << std::endl;
            std::cout << "Scaling efficiency: " << scaling_efficiency << "%" << std::endl;
        }
    }
    
    // Clean up NCCL communicators.
    if (comms) {
        for (int i = 0; i < num_devices; i++) {
            ncclCommDestroy(comms[i]);
        }
        delete[] comms;
    }
    
    std::cout << "\n✓ Complete!" << std::endl;
    return 0;
}