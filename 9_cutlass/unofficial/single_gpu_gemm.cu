

/**
 * @file single_gpu_gemm.cu
 * @brief Demonstrates a high-performance single-GPU GEMM using NVIDIA's CUTLASS library.
 *
 * @details This example implements a General Matrix-Matrix Multiplication (GEMM) for a single GPU
 *          using the CUTLASS 3.x API. CUTLASS is a template library for building high-performance
 *          GEMM-like computations. Instead of providing pre-compiled kernels, it provides C++
 *          abstractions that can be composed to generate efficient, custom kernels.
 *
 *          This implementation is optimized for the NVIDIA Hopper architecture (SM90) and uses
 *          Tensor Cores for FP16 matrix multiplication with accumulation in FP32 for precision.
 *
 *          The key components configured in this file are:
 *          - **Data Types**: FP16 for inputs/output, FP32 for accumulation.
 *          - **Layouts**: Row-major layout for all matrices.
 *          - **Tiling Strategy**: Defines the shape of thread block tiles (`TileShape`) and
 *            inter-thread-block clusters (`ClusterShape`) to partition the GEMM problem.
 *          - **Kernel Schedule**: `KernelScheduleAuto` allows CUTLASS to automatically determine
 *            the best way to schedule work.
 *          - **Mainloop**: The `CollectiveMainloop` performs the core matrix multiplication. It's
 *            the part of the kernel that computes `A * B`.
 *          - **Epilogue**: The `CollectiveEpilogue` handles operations after the mainloop, such as
 *            scaling, clamping, or other element-wise transformations.
 *
 *          The file also includes a CPU reference implementation for verification and a simple
 *          benchmarking setup to measure performance.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

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

using namespace cute;

//
// CUTLASS GEMM Configuration
//
// This section defines the building blocks of the CUTLASS GEMM kernel through a series of C++
// type aliases. Each `using` statement configures a specific aspect of the kernel, from the
// target architecture and data types to the tiling strategy and memory layouts.
//

//
// Architecture and Data Type Configuration
//
// These types define the target GPU architecture and the precision of the data involved in
// the GEMM computation (A, B, C, and the accumulator).
//
using ArchTag = cutlass::arch::Sm90;           // Target NVIDIA Hopper architecture (e.g., H100 GPU).
                                               // CUTLASS uses this tag to generate architecture-specific instructions.
using ElementInput = cutlass::half_t;          // Data type for input matrices A and B (FP16).
                                               // FP16 allows leveraging Tensor Cores for high throughput.
using ElementOutput = cutlass::half_t;         // Data type for the output matrix D (FP16).
using ElementAccumulator = float;              // Data type for accumulation (FP32). Using a higher precision
                                               // for accumulation helps maintain numerical stability.
using ElementC = ElementOutput;                // Data type for the C matrix (source accumulator).
using ElementD = ElementOutput;                // Data type for the D matrix (destination).

//
// Memory Layout Configuration
//
// Defines how matrices are stored in memory. Row-major is standard in C/C++.
//
using LayoutA = cutlass::layout::RowMajor;     // Row-major layout for matrix A.
using LayoutB = cutlass::layout::RowMajor;     // Row-major layout for matrix B.
using LayoutC = cutlass::layout::RowMajor;     // Row-major layout for matrix C.
using LayoutD = cutlass::layout::RowMajor;     // Row-major layout for matrix D.

//
// Tiling and Scheduling Configuration
//
// These parameters define how the GEMM problem is partitioned and scheduled across the GPU.
//
// - TileShape: The shape of the matrix multiplication performed by a single thread block (M, N, K).
// - ClusterShape: Groups thread blocks into clusters for locality. For single-GPU, this is less critical
//   but is part of the modern CUTLASS API.
// - KernelSchedule: The policy for dispatching thread blocks. 'Auto' lets CUTLASS decide.
//
using TileShape = Shape<_128, _256, _64>;      // Thread block tile shape: 128 rows of A, 256 columns of B, and 64 inner dimension K.
                                               // This is a common choice for high-performance GEMM.
using ClusterShape = Shape<_2, _1, _1>;        // Defines a 2x1x1 cluster of thread blocks. This is more
                                               // relevant for multi-GPU or very large problems.
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto; // CUTLASS automatically selects the kernel schedule.

//
// Memory Alignment
//
// Specifies the memory alignment for matrices to ensure efficient memory access.
// Aligned memory access patterns are critical for maximizing bandwidth.
//
static constexpr int AlignmentA = 16 / sizeof(ElementInput);  // 128-bit alignment for matrix A.
static constexpr int AlignmentB = 16 / sizeof(ElementInput);  // 128-bit alignment for matrix B.
static constexpr int AlignmentC = 16 / sizeof(ElementOutput); // 128-bit alignment for matrix C.
static constexpr int AlignmentD = 16 / sizeof(ElementOutput); // 128-bit alignment for matrix D.

//
// Epilogue Configuration
//
// The epilogue is responsible for finalizing the computation after the main GEMM loop.
// It can perform operations like scaling, clamping, or other element-wise transformations.
//
using ElementCompute = float;                  // Data type for epilogue computations, typically float.

// Defines the epilogue operation itself. Here, it's a linear combination:
// D = alpha * accumulator + beta * C
// This is the standard GEMM formula.
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementCompute, ElementOutput, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// The CollectiveEpilogue builder composes the epilogue. It takes the architecture, tile shapes,
// and the epilogue operation to create a hardware-optimized epilogue implementation.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,     // Target architecture and instruction class (Tensor Cores).
    TileShape, ClusterShape,                      // GEMM tile and cluster shapes.
    cutlass::epilogue::collective::EpilogueTileAuto, // Automatic tiling for the epilogue.
    ElementAccumulator, ElementCompute,          // Accumulator and compute element types.
    ElementC, LayoutC, AlignmentC,               // C matrix configuration.
    ElementD, LayoutD, AlignmentD,               // D matrix configuration.
    cutlass::epilogue::collective::EpilogueScheduleAuto, // Automatic scheduling for the epilogue.
    EpilogueOp                                   // The epilogue operation defined above.
>::CollectiveOp;

//
// Mainloop Configuration
//
// The mainloop is the core of the GEMM computation, responsible for iterating over the K dimension
// and performing the multiply-accumulate operations.
//
// The CollectiveMainloop builder assembles the mainloop based on the configuration.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,     // Target architecture and instruction class.
    ElementInput, LayoutA, AlignmentA,           // A matrix configuration.
    ElementInput, LayoutB, AlignmentB,           // B matrix configuration.
    ElementAccumulator,                          // Accumulator element type.
    TileShape, ClusterShape,                      // GEMM tile and cluster shapes.
    cutlass::gemm::collective::StageCountAutoCarveout< // Automatically configures shared memory usage
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, // by considering epilogue's needs.
    KernelSchedule                               // Kernel scheduling policy.
>::CollectiveOp;

//
// Final GEMM Kernel Definition
//
// The GemmKernel combines the mainloop and the epilogue into a single universal kernel.
//
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,                        // Problem shape (M, N, K) will be specified at runtime.
    CollectiveMainloop,                          // The main computation loop.
    CollectiveEpilogue                           // The output processing stage.
>;

// The GemmUniversalAdapter provides a user-friendly, device-level API to launch the kernel.
// It handles argument packing, kernel launch configuration, and workspace management.
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

/**
 * CUDA error checking macro
 * @param call CUDA function call to check
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * Convert CUTLASS data types to float for computation
 * @param val Value to convert
 * @return Float representation
 */
template<typename T>
float to_float(T val) {
    if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return float(val);  // Convert FP16 to FP32
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        return float(val);  // Convert FP8 to FP32
    } else {
        return static_cast<float>(val);  // Generic conversion
    }
}

/**
 * Convert float to CUTLASS data types
 * @param val Float value to convert
 * @return Converted value
 */
template<typename T>
T from_float(float val) {
    if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return cutlass::half_t(val);  // Convert FP32 to FP16
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        return cutlass::float_e4m3_t(val);  // Convert FP32 to FP8
    } else {
        return static_cast<T>(val);  // Generic conversion
    }
}

/**
 * CPU reference implementation of GEMM for verification
 * Computes C = A * B where A is M×K, B is K×N, C is M×N
 * 
 * @param A Input matrix A (M×K, row-major)
 * @param B Input matrix B (K×N, row-major)
 * @param C Output matrix C (M×N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
template<typename InType, typename OutType>
void cpu_gemm(const std::vector<InType>& A, const std::vector<InType>& B,
              std::vector<OutType>& C, int M, int N, int K) {
    // Standard triple-nested loop GEMM implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            // Compute dot product of row A[i,:] and column B[:,j]
            for (int k = 0; k < K; k++) {
                float a_val = to_float(A[i * K + k]);      // Convert to float for computation
                float b_val = to_float(B[k * N + j]);      // Convert to float for computation
                sum += a_val * b_val;                      // Accumulate multiply-add
            }
            C[i * N + j] = from_float<OutType>(sum);       // Convert back to output type
        }
    }
}

/**
 * Initialize vector with random values in range [-1, 1]
 * @param data Vector to initialize
 * @param seed Random seed for reproducibility
 */
template<typename T>
void initialize_random(std::vector<T>& data, int seed = 42) {
    std::srand(seed);
    for (auto& val : data) {
        // Generate random float in [-1, 1] and convert to target type
        val = from_float<T>((std::rand() / float(RAND_MAX)) * 2.0f - 1.0f);
    }
}

/**
 * Verify GPU results against CPU reference implementation
 * Computes relative error statistics and determines if verification passes
 * 
 * @param gpu_result GPU computation results
 * @param cpu_result CPU reference results
 * @param tolerance Maximum allowed average relative error (default: 0.5%)
 * @return true if verification passes, false otherwise
 */
template<typename T>
bool verify_results(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result,
                    float tolerance = 0.5f) {
    // Check size compatibility
    if (gpu_result.size() != cpu_result.size()) return false;
    
    float max_error = 0.0f;      // Maximum relative error
    float avg_error = 0.0f;      // Average relative error
    int non_zero_count = 0;      // Count of non-zero elements
    
    // Compute error statistics
    for (size_t i = 0; i < gpu_result.size(); i++) {
        float gpu_val = to_float(gpu_result[i]);
        float cpu_val = to_float(cpu_result[i]);
        
        // Only compute relative error for non-zero elements
        if (std::abs(cpu_val) > 1e-5f) {
            float rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);
            max_error = std::max(max_error, rel_error);
            avg_error += rel_error;
            non_zero_count++;
        }
    }
    
    avg_error /= non_zero_count;
    
    // Print error statistics
    std::cout << "Max relative error: " << max_error 
              << " (over " << non_zero_count << " non-zero elements)" << std::endl;
    std::cout << "Average relative error: " << avg_error 
              << " (" << (avg_error * 100.0f) << "%)" << std::endl;
    
    // Determine if verification passes
    bool passed = avg_error < tolerance;
    std::cout << (passed ? "✓" : "✗") << " CPU verification " 
              << (passed ? "PASSED" : "FAILED") 
              << " (average error < " << (tolerance * 100.0f) << "%)" << std::endl;
    
    return passed;
}

/**
 * @brief Executes the CUTLASS GEMM kernel and measures its performance.
 *
 * @details This function orchestrates the entire GPU-side GEMM computation. It handles:
 *          1. Allocating device memory for matrices A, B, and C.
 *          2. Copying input matrices A and B from host to device.
 *          3. Defining the stride layout for each matrix using CuTe's utilities.
 *          4. Configuring the GEMM kernel arguments, including problem size, pointers, strides,
 *             and the epilogue operation (alpha=1.0, beta=0.0).
 *          5. Initializing the CUTLASS `Gemm` object, which prepares the kernel for execution
 *             and calculates any required workspace memory.
 *          6. Running warmup iterations to stabilize GPU clocks.
 *          7. Benchmarking the kernel execution time.
 *          8. Copying the result matrix C from device back to host.
 *          9. Freeing all allocated device memory.
 *
 * @param h_A Host vector for matrix A.
 * @param h_B Host vector for matrix B.
 * @param h_C Host vector to store the resulting matrix C.
 * @param M, N, K The dimensions of the GEMM problem.
 * @param warmup_iters Number of warmup iterations.
 * @param bench_iters Number of benchmarking iterations.
 */
template<typename InType, typename OutType>
void run_gemm(const std::vector<InType>& h_A, const std::vector<InType>& h_B,
              std::vector<OutType>& h_C, int M, int N, int K,
              int warmup_iters = 5, int bench_iters = 10) {
    
    InType *d_A, *d_B;
    OutType *d_C;
    
    // Allocate memory on the GPU for matrices A, B, and C.
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(InType)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(InType)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(OutType)));
    
    // Copy input matrices A and B from host (CPU) to device (GPU).
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(InType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(InType), cudaMemcpyHostToDevice));
    
    // Define the memory layout (strides) of the matrices. CUTLASS 3.x uses CuTe for this,
    // which provides a flexible way to describe tensor layouts. For a standard row-major
    // layout, the strides are packed.
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    
    // Create stride objects for packed row-major matrices.
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    
    // Set up the arguments for the GEMM kernel. This struct packages all the information
    // the kernel needs to run: problem size, pointers to matrices, strides, and epilogue parameters.
    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm, // Specifies a standard GEMM operation.
        {M, N, K}, // Problem dimensions.
        {d_A, stride_a, d_B, stride_b}, // Pointers and strides for A and B.
        // Epilogue parameters: {alpha, beta}, C_ptr, C_stride, D_ptr, D_stride
        // We perform D = 1.0 * (A*B) + 0.0 * C, which is a simple copy.
        {{1.0f, 0.0f}, d_C, stride_c, d_C, stride_d}
    };
    
    // Create an instance of the GEMM operator.
    Gemm gemm_op;
    // Query the workspace size required by the kernel. Some CUTLASS kernels need temporary
    // storage (workspace) for intermediate results.
    size_t workspace_size = Gemm::get_workspace_size(args);
    
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_ptr, workspace_size));
    }
    
    // Initialize the GEMM operator. This step configures the kernel launch parameters
    // (grid size, block size) and prepares any internal state.
    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM initialization failed" << std::endl;
        exit(1);
    }
    
    // Run warmup iterations. This helps ensure the GPU is running at a stable clock frequency
    // before we start benchmarking.
    for (int i = 0; i < warmup_iters; i++) {
        gemm_op.run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark the kernel.
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        // Run the GEMM kernel.
        gemm_op.run();
    }
    // Synchronize the device to ensure all kernel computations are complete before stopping the timer.
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics.
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / bench_iters;
    // TFLOPS = 2 * M * N * K (for GEMM) / time / 1e12
    double tflops = (2.0 * M * N * K) / (time_ms * 1e9);
    
    std::cout << "Average time: " << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    
    // Copy the result matrix D from device to host.
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(OutType), cudaMemcpyDeviceToHost));
    
    // Free all allocated GPU memory.
    if (workspace_ptr) cudaFree(workspace_ptr);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char** argv) {
    std::cout << "\n=== CUTLASS Single-GPU GEMM ===" << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << prop.minor << ")" << std::endl;
    std::cout << std::endl;
    
    //
    // Part 1: Verification
    //
    // This section runs a smaller GEMM problem (1024x1024x1024) and compares the GPU's output
    // against a simple, triple-nested-loop CPU implementation to verify correctness.
    //
    {
        std::cout << "=== CPU Verification (1024³) ===" << std::endl;
        const int M = 1024, N = 1024, K = 1024;
        
        std::vector<ElementInput> h_A(M * K);
        std::vector<ElementInput> h_B(K * N);
        std::vector<ElementOutput> h_C_gpu(M * N);
        std::vector<ElementOutput> h_C_cpu(M * N);
        
        // Initialize host matrices with random data.
        initialize_random(h_A, 42);
        initialize_random(h_B, 43);
        
        // Run the reference GEMM on the CPU.
        std::cout << "Running CPU GEMM for verification..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        std::cout << "CPU GEMM time: " << cpu_time << " ms" << std::endl;
        
        // Run the CUTLASS GEMM on the GPU.
        run_gemm(h_A, h_B, h_C_gpu, M, N, K, 1, 1);
        
        // Verify the GPU result against the CPU result.
        bool passed = verify_results(h_C_gpu, h_C_cpu);
        
        if (!passed) {
            std::cerr << "\n✗ Verification failed! Not proceeding to benchmark." << std::endl;
            return 1;
        }
        std::cout << std::endl;
    }
    
    //
    // Part 2: Benchmarking
    //
    // This section runs a larger GEMM problem (8192x8192x8192) to measure the performance
    // of the CUTLASS kernel in TFLOPS (TeraFLoating-point Operations Per Second).
    //
    {
        std::cout << "=== GPU Benchmark (8192³) ===" << std::endl;
        const int M = 8192, N = 8192, K = 8192;
        
        std::vector<ElementInput> h_A(M * K);
        std::vector<ElementInput> h_B(K * N);
        std::vector<ElementOutput> h_C(M * N);
        
        // Initialize host matrices.
        initialize_random(h_A, 42);
        initialize_random(h_B, 43);
        
        // Run the benchmark.
        run_gemm(h_A, h_B, h_C, M, N, K, 5, 10);
    }
    
    std::cout << "\n✓ Complete!" << std::endl;
    return 0;
}