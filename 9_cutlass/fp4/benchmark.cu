/***************************************************************************************************
 * @file benchmark.cu
 * @brief A benchmark for an FP4 GEMM kernel using CUTLASS on NVIDIA Blackwell architecture.
 *
 * @details This educational example provides a focused benchmark for a CUTLASS GEMM kernel that
 *          utilizes 4-bit floating-point (FP4) inputs. FP4 is a new data type introduced with the
 *          NVIDIA Blackwell architecture (SM100+) to dramatically increase throughput and reduce
 *          memory footprint for deep learning inference.
 *
 *          This benchmark specifically targets:
 *          - **Architecture**: NVIDIA Blackwell (SM100), using its new block-scaled FP4 Tensor Cores.
 *          - **Data Types**:
 *            - `A` and `B` matrices: FP4 (`cutlass::float_e2m1_t`), a 2-exponent, 1-mantissa format.
 *            - `C` and `D` matrices: BFloat16 (`bfloat16_t`).
 *            - Accumulation: FP32 (`float`) for numerical stability.
 *          - **Kernel**: A CUTLASS 3.x-style kernel constructed using `CollectiveMainloop` and
 *            `CollectiveEpilogue`.
 *          - **Block-Scaling**: Demonstrates the use of `OpClassBlockScaledTensorOp`, where a block
 *            of FP4 values shares a single scaling factor to maintain dynamic range.
 *
 *          The benchmark measures only the kernel execution time, excluding data transfer, to isolate
 *          the performance of the GEMM computation itself.
 **************************************************************************************************/

#include <iostream>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/**
 * @def CUDA_CHECK(status)
 * @brief Macro for checking the return value of CUDA API calls.
 *
 * @details If the CUDA call returns an error, this macro prints the error message,
 *          the file name, and the line number where the error occurred, and then
 *          terminates the program.
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

/**
 * @def CUTLASS_CHECK(status)
 * @brief Macro for checking the return value of CUTLASS API calls.
 *
 * @details Similar to CUDA_CHECK, but for functions that return a `cutlass::Status`.
 */
#define CUTLASS_CHECK(status)                                           \
  {                                                                     \
    cutlass::Status error = status;                                     \
    if (error != cutlass::Status::kSuccess) {                           \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(error)   \
                << " at: " << __LINE__ << std::endl;                    \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

/***************************************************************************************************
 *                                  CUTLASS Kernel Configuration
 **************************************************************************************************/

//
// This section configures the CUTLASS GEMM kernel by defining a set of type aliases.
// These types specify everything from the data precision and memory layout to the
// hardware target and tiling strategy.
//

// A matrix (left-hand side) configuration.
// `nv_float4_t` is a packed type holding two FP4 values. `float_e2m1_t` is the E2M1 FP4 format.
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32; // Alignment in bytes for efficient memory access.

// B matrix (right-hand side) configuration.
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor; // Using column-major for B can sometimes improve performance.
constexpr int AlignmentB = 32;

// C/D matrix (output) configuration.
// Output is in BFloat16, a common choice for mixed-precision training and inference.
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Accumulator and target architecture configuration.
using ElementAccumulator = float; // Use FP32 for accumulation to preserve precision.
using ArchTag = cutlass::arch::Sm100; // Target NVIDIA Blackwell architecture.
// This is the key operator class for Blackwell's FP4/INT4 support. It indicates that the
// kernel will use the block-scaled MMA instructions.
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shape configuration.
// - MmaTileShape: The size of the GEMM handled by a single thread block (M, N, K).
// - ClusterShape: The 2D grid of thread blocks that work together on the problem.
using MmaTileShape = Shape<_256,_256,_256>;
using ClusterShape = Shape<_2,_4,_1>; // A 2x4 cluster of thread blocks.

// Build the Epilogue.
// The epilogue handles writing the final result to memory after the main computation.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Build the Mainloop.
// The mainloop performs the core matrix multiply-accumulate operation (A * B).
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    // Automatically configure shared memory based on what the epilogue needs.
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// The final GEMM kernel, composed of the mainloop and epilogue.
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Problem shape (M, N, K, Batch) is specified at runtime.
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

// The device-level adapter, providing a user-friendly API for launching the kernel.
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

//
// Layout and Stride types for Block-Scaled FP4
//
// For block-scaled formats, we need layouts not only for the data matrices (A, B, C, D)
// but also for the scaling factor matrices (SFA, SFB).
//
using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
// Layout for the scaling factors of matrix A.
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
// Layout for the scaling factors of matrix B.
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

/***************************************************************************************************
 *                                    Benchmark Infrastructure
 **************************************************************************************************/

/**
 * @brief A simple struct to hold and parse benchmark configuration from the command line.
 */
struct BenchmarkConfig {
  int m, n, k, batch;
  int warmup_iterations;
  int timing_iterations;
  
  BenchmarkConfig() : m(8192), n(8192), k(8192), batch(8), 
                      warmup_iterations(5), timing_iterations(20) {}
  
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("batch", batch);
    cmd.get_cmd_line_argument("warmup", warmup_iterations);
    cmd.get_cmd_line_argument("iters", timing_iterations);
  }
  
  /**
   * @brief Calculates the performance of the GEMM in PetaFLOPs/s.
   * @param time_ms The average execution time in milliseconds.
   * @return The performance in PFLOPS.
   */
  double compute_petaflops(double time_ms) const {
    // FLOPs = 2 * M * N * K * batch (multiply-add counts as 2 ops)
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * 
                   static_cast<double>(k) * static_cast<double>(batch);
    double time_s = time_ms / 1000.0;
    double petaflops = (flops / time_s) / 1e15;
    return petaflops;
  }
};

/***************************************************************************************************
 *                                        Helper Functions
 **************************************************************************************************/

/**
 * @brief Initializes a tensor with random data.
 */
template <typename Element, typename Layout>
bool initialize_tensor(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double scope_max = 2.0;
  double scope_min = -2.0;
  
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);
  
  return true;
}

/***************************************************************************************************
 *                                    Core Benchmark Function
 **************************************************************************************************/

/**
 * @brief Runs the main GEMM benchmark.
 * @param config The benchmark configuration.
 */
int benchmark_gemm(BenchmarkConfig const& config) {
  // Get the block-scaled configuration from the mainloop. This is specific to SM100+.
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  
  std::cout << "\n=== nvFP4 GEMM Benchmark ===" << std::endl;
  std::cout << "Problem size: " << config.m << " x " << config.n << " x " << config.k 
            << " (batch=" << config.batch << ")" << std::endl;
  std::cout << "Architecture: SM100 (Blackwell)" << std::endl;
  std::cout << "Precision: FP4 (A, B) x BF16 (C, D)" << std::endl;
  
  // Setup strides and layouts for all matrices, including the scale factors.
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {config.m, config.k, config.batch});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {config.n, config.k, config.batch});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {config.m, config.n, config.batch});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {config.m, config.n, config.batch});
  
  // Create layout objects from shapes and strides.
  auto layout_A = make_layout(make_shape(config.m, config.k, config.batch), stride_A);
  auto layout_B = make_layout(make_shape(config.n, config.k, config.batch), stride_B);
  auto layout_C = make_layout(make_shape(config.m, config.n, config.batch), stride_C);
  auto layout_D = make_layout(make_shape(config.m, config.n, config.batch), stride_D);
  
  // For block-scaled formats, the layout of the scale factor matrix (SFA, SFB) is
  // derived from the problem shape and the kernel's internal tiling structure.
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(config.m, config.n, config.k, config.batch));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
    cute::make_shape(config.m, config.n, config.k, config.batch));
  
  // Allocate host-side tensors for A, B, C, D, and the scale factors SFA, SFB.
  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
  
  // Reset tensors to the correct size based on their layouts.
  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  // The size of the scale factor tensor is smaller than the data tensor.
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  
  std::cout << "Initializing tensors..." << std::endl;
  // Initialize all tensors with random data.
  initialize_tensor(block_A.host_view(), 2021);
  initialize_tensor(block_B.host_view(), 2022);
  initialize_tensor(block_C.host_view(), 2023);
  initialize_tensor(block_SFA.host_view(), 2024);
  initialize_tensor(block_SFB.host_view(), 2025);
  
  // Transfer data from host to device. This operation is NOT included in the benchmark timing.
  std::cout << "Transferring data to GPU..." << std::endl;
  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  
  // Setup the arguments for the CUTLASS kernel.
  // This struct packages all the necessary pointers, strides, and problem dimensions.
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {config.m, config.n, config.k, config.batch},
    // Mainloop arguments: data and scale factor pointers and layouts for A and B.
    {
      block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    // Epilogue arguments: alpha/beta scalars, and pointers/strides for C and D.
    {
      {1.0f, 0.0f},  // alpha, beta
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };
  
  // Initialize the CUTLASS GEMM object.
  Gemm gemm;
  // Allocate workspace memory if the kernel requires it.
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
  // Check if the kernel can be configured for the given problem size.
  CUTLASS_CHECK(gemm.can_implement(arguments));
  // Initialize the kernel launch parameters (e.g., grid/block dimensions).
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  
  // Run warmup iterations to ensure the GPU is at a stable clock frequency.
  std::cout << "Running " << config.warmup_iterations << " warmup iterations..." << std::endl;
  for (int i = 0; i < config.warmup_iterations; ++i) {
    CUTLASS_CHECK(gemm.run());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Time the kernel execution using CUDA events for high precision.
  std::cout << "Running " << config.timing_iterations << " timed iterations..." << std::endl;
  
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  // Record start event.
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < config.timing_iterations; ++i) {
    CUTLASS_CHECK(gemm.run());
  }
  // Record stop event and synchronize.
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float elapsed_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  
  // Calculate average time and performance.
  float avg_time_ms = elapsed_ms / config.timing_iterations;
  double petaflops = config.compute_petaflops(avg_time_ms);
  
  // Report results.
  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "Average kernel time: " << avg_time_ms << " ms" << std::endl;
  std::cout << "Performance: " << petaflops << " PFLOPS" << std::endl;
  std::cout << "Performance: " << (petaflops * 1000.0) << " TFLOPS" << std::endl;
  
  // Clean up CUDA events.
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  
  return 0;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  // Check for minimum required CUDA version (12.8 for this kernel).
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This benchmark requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }
  
  // Check that the GPU has the correct compute capability (SM100/Blackwell).
  cudaDeviceProp props;
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
  
  std::cout << "GPU: " << props.name << std::endl;
  std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
  
  if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
    std::cerr << "This benchmark requires SM100, SM101, or SM103 (Blackwell architecture)." << std::endl;
    return 0;
  }
  
  // Only run the benchmark if CUTLASS was compiled with SM100 support.
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  BenchmarkConfig config;
  config.parse(argc, args);
  return benchmark_gemm(config);
#else
  std::cerr << "CUTLASS was not compiled with SM100 support." << std::endl;
  return 1;
#endif
}

