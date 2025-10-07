/***************************************************************************************************
 * nvFP4 GEMM Benchmark - Educational Example
 * 
 * This file contains an isolated benchmark for the CUTLASS 72a nvFP4 kernel.
 * It focuses on kernel execution time only (not data movement).
 * Designed for SM100 architecture (B200 GPU).
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

// Macro for CUDA error checking
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define CUTLASS_CHECK(status)                                           \
  {                                                                     \
    cutlass::Status error = status;                                     \
    if (error != cutlass::Status::kSuccess) {                           \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(error)   \
                << " at: " << __LINE__ << std::endl;                    \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

/////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Configuration (Same as 72a)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration - FP4
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

// B matrix configuration - FP4
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// C/D matrix configuration - BF16
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Accumulator and architecture
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shapes
using MmaTileShape = Shape<_256,_256,_256>;
using ClusterShape = Shape<_2,_4,_1>;

// Build the collective operations
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Layout types
using StrideA = typename Gemm::GemmKernel::StrideA;
using LayoutA = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using LayoutB = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using LayoutC = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutD = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

/////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark Structure
/////////////////////////////////////////////////////////////////////////////////////////////////

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
  
  // Compute petaflops
  double compute_petaflops(double time_ms) const {
    // FLOPs = 2 * M * N * K * batch (multiply-add counts as 2 ops)
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * 
                   static_cast<double>(k) * static_cast<double>(batch);
    double time_s = time_ms / 1000.0;
    double petaflops = (flops / time_s) / 1e15;
    return petaflops;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Initialize tensors
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Layout>
bool initialize_tensor(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double scope_max = 2.0;
  double scope_min = -2.0;
  
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);
  
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Main benchmark function
/////////////////////////////////////////////////////////////////////////////////////////////////

int benchmark_gemm(BenchmarkConfig const& config) {
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  
  std::cout << "\n=== nvFP4 GEMM Benchmark ===" << std::endl;
  std::cout << "Problem size: " << config.m << " x " << config.n << " x " << config.k 
            << " (batch=" << config.batch << ")" << std::endl;
  std::cout << "Architecture: SM100 (Blackwell)" << std::endl;
  std::cout << "Precision: FP4 (A, B) x BF16 (C, D)" << std::endl;
  
  // Setup strides and layouts
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {config.m, config.k, config.batch});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {config.n, config.k, config.batch});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {config.m, config.n, config.batch});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {config.m, config.n, config.batch});
  
  auto layout_A = make_layout(make_shape(config.m, config.k, config.batch), stride_A);
  auto layout_B = make_layout(make_shape(config.n, config.k, config.batch), stride_B);
  auto layout_C = make_layout(make_shape(config.m, config.n, config.batch), stride_C);
  auto layout_D = make_layout(make_shape(config.m, config.n, config.batch), stride_D);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(config.m, config.n, config.k, config.batch));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
    cute::make_shape(config.m, config.n, config.k, config.batch));
  
  // Allocate host tensors
  cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
  
  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  
  std::cout << "Initializing tensors..." << std::endl;
  initialize_tensor(block_A.host_view(), 2021);
  initialize_tensor(block_B.host_view(), 2022);
  initialize_tensor(block_C.host_view(), 2023);
  initialize_tensor(block_SFA.host_view(), 2024);
  initialize_tensor(block_SFB.host_view(), 2025);
  
  // Transfer to device - THIS IS NOT TIMED
  std::cout << "Transferring data to GPU..." << std::endl;
  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  
  // Setup kernel arguments
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {config.m, config.n, config.k, config.batch},
    {
      block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    {
      {1.0f, 0.0f},  // alpha, beta
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };
  
  // Initialize CUTLASS kernel
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  
  // Warmup iterations
  std::cout << "Running " << config.warmup_iterations << " warmup iterations..." << std::endl;
  for (int i = 0; i < config.warmup_iterations; ++i) {
    CUTLASS_CHECK(gemm.run());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Timing iterations - ONLY KERNEL TIME
  std::cout << "Running " << config.timing_iterations << " timed iterations..." << std::endl;
  
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < config.timing_iterations; ++i) {
    CUTLASS_CHECK(gemm.run());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float elapsed_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  
  float avg_time_ms = elapsed_ms / config.timing_iterations;
  double petaflops = config.compute_petaflops(avg_time_ms);
  
  // Report results
  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "Average kernel time: " << avg_time_ms << " ms" << std::endl;
  std::cout << "Performance: " << petaflops << " PFLOPS" << std::endl;
  std::cout << "Performance: " << (petaflops * 1000.0) << " TFLOPS" << std::endl;
  
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  
  return 0;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  // Check CUDA version
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This benchmark requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }
  
  // Check GPU compute capability
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
  
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  BenchmarkConfig config;
  config.parse(argc, args);
  return benchmark_gemm(config);
#else
  std::cerr << "CUTLASS was not compiled with SM100 support." << std::endl;
  return 1;
#endif
}

