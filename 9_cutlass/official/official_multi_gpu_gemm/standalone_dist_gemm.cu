/***************************************************************************************************
 * @file standalone_dist_gemm.cu
 * @brief A standalone performance benchmark for a CUTLASS 3.x Distributed GEMM.
 *
 * @details This example showcases how to use CUTLASS to perform a single, large
 *          General Matrix-Matrix Multiplication (GEMM) distributed across multiple GPUs.
 *          Unlike a simple scaling test that runs independent GEMMs, this implementation
 *          has multiple GPUs collaborating on one problem, communicating intermediate
 *          results to produce a final, globally correct output matrix.
 *
 *          This benchmark is based on the official CUTLASS Example 65 and is adapted for
 *          clarity and educational purposes.
 *
 *          Key Concepts Demonstrated:
 *          - **Distributed GEMM**: A single GEMM problem (C = A * B) is partitioned and
 *            computed across a user-specified number of GPUs (e.g., 2, 4, or 8).
 *          - **Distribution Schedule**: The logic for data partitioning and communication is
 *            encapsulated in a `DistSchedule`. This example uses `AllGather1D_TilingCD_RotatingA`,
 *            which splits the N dimension of the problem across GPUs and uses an All-Gather
 *            collective for communication.
 *          - **`DistributedGemm` Adapter**: The `cutlass::distributed::device::DistributedGemmUniversalAdapter`
 *            is a device-level wrapper that extends a standard CUTLASS GEMM kernel with the
 *            necessary logic for distributed execution, including communication and synchronization.
 *          - **Peer-to-Peer Access**: The benchmark enables direct GPU-to-GPU memory access,
 *            which is crucial for efficient communication between the distributed GEMM participants.
 *
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <chrono>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/distributed/device/gemm_universal_adapter.hh"
#include "cutlass/distributed/kernel/distributed_gemm.h"
#include "cutlass/distributed/collective/consumer/all_gather_consumer.h"
#include "cutlass/distributed/collective/producer/all_gather_producer.h"
#include "cutlass/distributed/collective/merger/all_gather_merger.h"
#include "cutlass/distributed/collective/visitors.hpp"
#include "cutlass/distributed/schedules/all_gather_1d_tiling_cd_rotating_a.h"

// Common CUDA checks
#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                 \
                << " at line " << __LINE__ << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(error)          \
                << " at line " << __LINE__ << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }


using namespace cute;

/***************************************************************************************************
 *                                CUTLASS Distributed GEMM Configuration
 **************************************************************************************************/

/**
 * @brief Defines the configuration for the distributed GEMM kernel.
 *
 * @tparam TP_ The tensor parallelism degree (i.e., the number of GPUs).
 *
 * This struct templatizes the entire kernel configuration on the number of participating GPUs.
 * It defines the distribution strategy, data types, layouts, tiling, and ultimately composes
 * the `DistributedGemm` object.
 */
template<int TP_>
struct DistGemmConfig {
  using TP = cute::Int<TP_>;
  static constexpr int TP_val = TP_;

  //
  // Distributed Schedule Configuration
  //
  // This is the core of the distributed logic. It defines how the GEMM problem is partitioned
  // across GPUs and what communication pattern is used.
  //
  // `AllGather1D_TilingCD_RotatingA<TP>`:
  // - The problem is tiled in 1D across the N-dimension of the output matrices C and D.
  // - Each GPU computes a full M x (N/TP) slice of the output.
  // - To do this, each GPU needs the full A matrix but only a slice of the B matrix.
  // - Matrix A is "rotated" through the GPUs. Each GPU starts with a different slice of A
  //   and receives the next slice in a pipelined fashion.
  // - An All-Gather collective is used to assemble the final output C/D from the partial results.
  //
  using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;

  // Data Types
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t; // Using half for accumulator can be faster but less precise
  using ElementCompute = cutlass::half_t;

  // Memory Layouts
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor; // Column-major for B is often paired with row-major A
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  // Memory Alignments
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture and Tiling Configuration
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<_128, _256, _64>;
  using ClusterShape = Shape<_1, _2, _1>;

  // Kernel and Epilogue Scheduling Policies
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  //
  // Building the CUTLASS Kernel Components
  //
  // Note: These are the same building blocks as a single-GPU kernel.
  // The distributed logic will wrap these base components.
  //

  // Epilogue: Handles the final stage of writing data to memory.
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule
    >::CollectiveOp;

  // Mainloop: The core matrix multiply-accumulate logic.
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      KernelSchedule
    >::CollectiveOp;

  // The base single-GPU GEMM kernel.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  // The single-GPU device-level adapter.
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  //
  // Building the Distributed GEMM
  //
  // These wrappers add the distributed logic around the base `GemmKernel`.
  //

  // The distributed kernel wrapper, which injects the `DistSchedule` logic.
  using DistGemmKernel = cutlass::distributed::kernel::DistributedGemmKernelWrapper<
    GemmKernel,
    DistSchedule
  >;

  // The device-level adapter for the distributed GEMM. This is the main entry point for the user.
  using DistGemm = cutlass::distributed::device::DistributedGemmUniversalAdapter<DistGemmKernel>;

  // Helper type aliases for strides and host tensors.
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  
  using HostTensorA = cutlass::HostTensor<ElementA, LayoutA>;
  using HostTensorB = cutlass::HostTensor<ElementB, LayoutB>;
  using HostTensorC = cutlass::HostTensor<ElementC, LayoutC>;
  using HostTensorD = cutlass::HostTensor<ElementD, LayoutD>;
};


/**
 * @brief Parses and holds command-line options for the benchmark.
 */
struct Options {
  int num_gpus = 2;
  int m = 8192, n = 8192, k = 8192, l = 1;
  int iterations = 100;
  int warmup_iterations = 10;
  float alpha = 1.0f, beta = 0.0f;
  bool help = false;
  
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    
    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    
    cmd.get_cmd_line_argument("num-gpus", num_gpus);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmup-iterations", warmup_iterations);
  }
  
  void print_usage(std::ostream &out) const {
    out << "Standalone Distributed GEMM Benchmark\n\n"
        << "Options:\n"
        << "  --help                      Display this help message\n"
        << "  --num-gpus=<int>            Number of GPUs to use (default: 2)\n"
        << "  --m=<int>                   M dimension (default: 8192)\n"
        << "  --n=<int>                   N dimension (default: 8192)\n"
        << "  --k=<int>                   K dimension (default: 8192)\n"
        << "  --l=<int>                   Batch count (default: 1)\n"
        << "  --alpha=<float>             Alpha scalar (default: 1.0)\n"
        << "  --beta=<float>              Beta scalar (default: 0.0)\n"
        << "  --iterations=<int>          Benchmark iterations (default: 100)\n"
        << "  --warmup-iterations=<int>   Warmup iterations (default: 10)\n\n"
        << "Example:\n"
        << "  ./standalone_dist_gemm --num-gpus=4 --m=16384 --n=16384 --k=16384\n";
  }
  
  /**
   * @brief Calculates theoretical TFLOPS.
   * @param runtime_s The total runtime in seconds.
   * @return The performance in TFLOPS.
   */
  double tflops(double runtime_s) const {
    // Note: For distributed GEMM, the total FLOPs are divided by the number of GPUs,
    // as each GPU computes a fraction of the total work.
    uint64_t flop = uint64_t(2) * m * n * k * l / num_gpus;
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }
};


/**
 * @brief The main function for running the distributed GEMM benchmark.
 *
 * @tparam TP_ The number of GPUs (tensor parallelism degree).
 * @param options The command-line options.
 */
template<int TP_>
int run_benchmark(const Options& options) {
  using Config = DistGemmConfig<TP_>;
  using DistGemm = typename Config::DistGemm;
  using DistSchedule = typename Config::DistSchedule;
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using ElementCompute = typename Config::ElementCompute;
  using StrideA = typename Config::StrideA;
  using StrideB = typename Config::StrideB;
  using StrideC = typename Config::StrideC;
  using StrideD = typename Config::StrideD;
  using HostTensorA = typename Config::HostTensorA;
  using HostTensorB = typename Config::HostTensorB;
  using HostTensorC = typename Config::HostTensorC;
  using HostTensorD = typename Config::HostTensorD;
  
  constexpr int TP = TP_;
  
  std::cout << "\n==========================================================================\n";
  std::cout << "Distributed GEMM: " << TP << " GPUs\n";
  std::cout << "Problem: " << options.m << " x " << options.n << " x " << options.k << " x " << options.l << "\n";
  std::cout << "==========================================================================\n\n";
  
  
  // Verify that the required number of GPUs are available.
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices < TP) {
    std::cerr << "Error: Requested " << TP << " GPUs but only " << num_devices << " available\n";
    return -1;
  }
  
  int primary_device_idx;
  CUDA_CHECK(cudaGetDevice(&primary_device_idx));
  
  // Define the global problem shape.
  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);
  
  
  // Define the shapes of the global matrices A, B, and C.
  auto shape_A = cute::select<0, 2, 3>(problem_shape);
  auto shape_B = cute::select<1, 2, 3>(problem_shape);
  auto shape_C = cute::select<0, 1, 3>(problem_shape);
  auto shape_D = cute::select<0, 1, 3>(problem_shape);
  
  // Create packed strides for the global matrices.
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_C);
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_D);
  
  
  // Allocate host tensors for the global matrices. These will hold the initial data.
  auto a_coord = cutlass::make_Coord(size(shape_A), 1);
  auto b_coord = cutlass::make_Coord(size(shape_B), 1);
  auto c_coord = cutlass::make_Coord(size(shape_C), 1);
  
  HostTensorA tensor_A(a_coord);
  HostTensorB tensor_B(b_coord);
  HostTensorC tensor_C(c_coord);
  
  
  // Initialize the global host tensors with random data on the device.
  uint64_t seed = 2024;
  cutlass::reference::device::TensorFillRandomUniform(
    tensor_A.device_view(), seed, ElementA(2), ElementA(-2), 0);
  cutlass::reference::device::TensorFillRandomUniform(
    tensor_B.device_view(), seed + 1, ElementB(2), ElementB(-2), 0);
  cutlass::reference::device::TensorFillRandomUniform(
    tensor_C.device_view(), seed + 2, ElementC(2), ElementC(-2), 0);
  
  
  //
  // Per-GPU (Local) Data Setup
  //
  // Determine the shape of the matrix slices each GPU will own.
  auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
  auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
  auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);
  
  // Coordinates for allocating local device memory.
  auto a_coord_device = cutlass::make_Coord(size(local_shape_A), 1);
  auto b_coord_device = cutlass::make_Coord(size(local_shape_B), 1);
  auto c_coord_device = cutlass::make_Coord(size(local_shape_C), 1);
  
  // Arrays of host tensors to hold the local data for each GPU.
  HostTensorA tensor_A_arr[TP];
  HostTensorB tensor_B_arr[TP];
  HostTensorC tensor_C_arr[TP];
  HostTensorD tensor_D_arr[TP];
  
  
  // Enable peer-to-peer (P2P) access between all GPUs. This is essential for the distributed
  // kernel to allow one GPU's SMs to directly read from/write to another GPU's memory.
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    for (int peer_idx = 0; peer_idx < TP; ++peer_idx) {
      if (peer_idx != device_idx) {
        int can_access;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_idx, peer_idx));
        if (!can_access) {
          std::cerr << "Error: Device " << device_idx << " cannot access device " << peer_idx << "\n";
          return -1;
        }
        cudaError_t err = cudaDeviceEnablePeerAccess(peer_idx, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CHECK(err);
        } else {
          cudaGetLastError();
        }
      }
    }
    
    // Allocate local device memory for each GPU's slice of the matrices.
    tensor_A_arr[device_idx].resize(a_coord_device);
    tensor_B_arr[device_idx].resize(b_coord_device);
    tensor_C_arr[device_idx].resize(c_coord_device);
    tensor_D_arr[device_idx].resize(c_coord_device);
  }
  CUDA_CHECK(cudaSetDevice(primary_device_idx));
  
  
  // Create one CUDA stream for each GPU.
  cudaStream_t stream_arr[TP];
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamCreate(&stream_arr[device_idx]));
  }
  
  
  // Arrays to hold the per-GPU distributed GEMM objects and their workspaces.
  DistGemm dist_gemm_arr[TP];
  cutlass::device_memory::allocation<uint8_t> workspace_arr[TP];
  cutlass::device_memory::allocation<uint8_t> exclusive_workspace_arr[TP];
  void* workspace_ptr_arr[TP];
  void* exclusive_workspace_ptr_arr[TP];
  typename DistGemm::Arguments arguments_[TP];
  
  
  //
  // Initialize Arguments and Copy Data for Each GPU
  //
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    
    // Create tensors representing the full global matrices in device memory.
    auto global_A = cute::make_tensor(tensor_A.device_data(),
        cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
    auto global_B = cute::make_tensor(tensor_B.device_data(),
        cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
    auto global_C = cute::make_tensor(tensor_C.device_data(),
        cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));
    
    // Get the slice of the global matrices that corresponds to the current device.
    auto global_A_device_slice = DistSchedule::get_device_slice_A(global_A, device_idx);
    auto global_B_device_slice = DistSchedule::get_device_slice_B(global_B, device_idx);
    auto global_C_device_slice = DistSchedule::get_device_slice_C(global_C, device_idx);
    
    // Create strides for the local, per-GPU matrices.
    auto local_stride_A = cutlass::make_cute_packed_stride(StrideA{}, local_shape_A);
    auto local_stride_B = cutlass::make_cute_packed_stride(StrideB{}, local_shape_B);
    auto local_stride_C = cutlass::make_cute_packed_stride(StrideC{}, local_shape_C);
    auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);
    
    // Create tensors for the local matrices in this GPU's memory.
    auto local_A = cute::make_tensor(tensor_A_arr[device_idx].device_data(),
        make_layout(local_shape_A, local_stride_A));
    auto local_B = cute::make_tensor(tensor_B_arr[device_idx].device_data(),
        make_layout(local_shape_B, local_stride_B));
    auto local_C = cute::make_tensor(tensor_C_arr[device_idx].device_data(),
        make_layout(local_shape_C, local_stride_C));
    auto local_D = cute::make_tensor(tensor_D_arr[device_idx].device_data(),
        make_layout(local_shape_D, local_stride_D));
    
    // Copy the device's slice of the global data into its local memory.
    cutlass::device_copy(global_A_device_slice, local_A, stream_arr[device_idx]);
    cutlass::device_copy(global_B_device_slice, local_B, stream_arr[device_idx]);
    cutlass::device_copy(global_C_device_slice, local_C, stream_arr[device_idx]);
    
    // Set up the kernel arguments for this GPU.
    arguments_[device_idx] = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {
        reinterpret_cast<const ElementA*>(local_A.data()),
        local_A.stride(),
        reinterpret_cast<const ElementB*>(local_B.data()),
        local_B.stride()
      },
      {
        {static_cast<ElementCompute>(options.alpha), static_cast<ElementCompute>(options.beta)},
        reinterpret_cast<const ElementC*>(local_C.data()),
        local_C.stride(),
        reinterpret_cast<ElementD*>(local_D.data()),
        local_D.stride()
      },
      {},
      {}
    };
    
    // Allocate shared and exclusive workspaces.
    // - Shared Workspace: Accessible by all GPUs. Used for communication primitives.
    // - Exclusive Workspace: Private to each GPU.
    size_t workspace_size = DistGemm::get_workspace_size(arguments_[device_idx]);
    size_t exclusive_workspace_size = DistGemm::get_exclusive_workspace_size();
    
    workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    exclusive_workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(exclusive_workspace_size);
    
    workspace_ptr_arr[device_idx] = workspace_arr[device_idx].get();
    exclusive_workspace_ptr_arr[device_idx] = exclusive_workspace_arr[device_idx].get();
    
    // It's good practice to zero out the workspace before use.
    cudaMemsetAsync(exclusive_workspace_ptr_arr[device_idx], 0, exclusive_workspace_size, stream_arr[device_idx]);
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  
  // Initialize the distributed GEMM object on each GPU.
  // This step is collective; all GPUs must participate. It sets up the communication
  // channels and pre-computes kernel launch parameters.
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUTLASS_CHECK(dist_gemm_arr[device_idx].can_implement(arguments_[device_idx]));
    
    // This file does not use the PDL (Persistent Distributed Latch) launch mode.
    bool launch_with_pdl = false;
    
    CUTLASS_CHECK(dist_gemm_arr[device_idx].initialize(
      arguments_,                     // Array of arguments for all ranks
      workspace_ptr_arr,              // Array of shared workspace pointers
      exclusive_workspace_ptr_arr,    // Array of exclusive workspace pointers
      device_idx,                     // The rank of the current GPU
      stream_arr[device_idx],         // The stream for this GPU's kernel
      launch_with_pdl
    ));
    
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  std::cout << "Initialization complete. Running warmup...\n";
  
  
  // Run warmup iterations to stabilize GPU clocks.
  for (int warmup = 0; warmup < options.warmup_iterations; ++warmup) {
    for (int device_idx = 0; device_idx < TP; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
    }
  }
  
  // Synchronize all streams to ensure warmup is complete.
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  std::cout << "Warmup complete. Running benchmark...\n";
  
  
  //
  // Benchmark the distributed GEMM execution.
  //
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch the kernel on all GPUs.
    for (int device_idx = 0; device_idx < TP; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
    }
  }
  
  // Wait for all GPUs to finish.
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  
  // Calculate and print performance results.
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double avg_time_ms = elapsed_ms / options.iterations;
  double avg_time_s = avg_time_ms / 1000.0;
  double tflops = options.tflops(avg_time_s);
  
  
  // Clean up resources.
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamDestroy(stream_arr[device_idx]));
  }
  
  CUDA_CHECK(cudaSetDevice(primary_device_idx));
  
  
  std::cout << "\n==========================================================================\n";
  std::cout << "RESULTS\n";
  std::cout << "==========================================================================\n";
  std::cout << "GPUs:             " << TP << "\n";
  std::cout << "Problem size:     " << options.m << " x " << options.n << " x " << options.k << "\n";
  std::cout << "Avg time:         " << avg_time_ms << " ms\n";
  std::cout << "Performance:      " << tflops << " TFLOPS\n";
  std::cout << "Per-GPU:          " << (tflops / TP) << " TFLOPS\n";
  std::cout << "==========================================================================\n\n";
  
  return 0;
}


int main(int argc, char const **args) {
  Options options;
  options.parse(argc, args);
  
  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }
  
  
  // Check for minimum required CUDA version.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 6)) {
    std::cerr << "This program requires CUDA 12.6 or newer.\n";
    return 0;
  }
  
  
  // Use a switch statement to instantiate the correct benchmark function
  // based on the number of GPUs requested.
  switch (options.num_gpus) {
    case 2:
      return run_benchmark<2>(options);
    case 4:
      return run_benchmark<4>(options);
    case 8:
      return run_benchmark<8>(options);
    default:
      std::cerr << "Error: --num-gpus must be 2, 4, or 8\n";
      return -1;
  }
}

