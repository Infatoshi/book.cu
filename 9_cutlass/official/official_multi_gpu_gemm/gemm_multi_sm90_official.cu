/***************************************************************************************************
 * @file gemm_multi_sm90_official.cu
 * @brief An official CUTLASS 3.x Distributed GEMM kernel for 2 GPUs on NVIDIA Hopper (SM90),
 *        wrapped for use with PyTorch.
 *
 * @details This file implements a distributed GEMM operation that splits a single large matrix
 *          multiplication problem across two GPUs. It is based on the official CUTLASS Example 65
 *          and demonstrates how to integrate a complex, multi-GPU CUTLASS kernel into the
 *          PyTorch ecosystem via C++ extensions.
 *
 *          This implementation is specifically hardcoded for a Tensor Parallelism (TP) degree of 2.
 *
 *          Key Concepts:
 *          - **2-GPU Distributed GEMM**: Solves one `C = A * B` problem by having two GPUs
 *            collaborate. Each GPU computes part of the result and communicates with the other.
 *          - **Distribution Schedule**: Uses `AllGather1D_TilingCD_RotatingA`, the same schedule as
 *            the standalone example, to partition the work and manage communication.
 *          - **PyTorch Integration**: The kernel is wrapped in a C++ function that accepts and
 *            operates on `torch::Tensor` objects. This involves handling data layouts (e.g.,
 *            transposing from row-major to column-major) and memory management.
 *          - **Peer-to-Peer (P2P) Communication**: Explicitly enables P2P memory access between
 *            the two participating GPUs, which is required for the communication collectives
 *            within the distributed kernel.
 **************************************************************************************************/

#include <torch/extension.h>
#include <stdexcept>
#include <string>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/distributed/device/gemm_universal_adapter.hh"
#include "cutlass/distributed/kernel/distributed_gemm.h"
#include "cutlass/distributed/schedules/all_gather_1d_tiling_cd_rotating_a.h"

// Common CUDA checks
#define CUDA_CHECK(status)                                                     \
  {                                                                            \
    cudaError_t error = status;                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)                 \
                << " at line " << __LINE__ << std::endl;                       \
      throw std::runtime_error("CUDA error");                                  \
    }                                                                          \
  }
#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(error)          \
                << " at line " << __LINE__ << std::endl;                       \
      throw std::runtime_error("CUTLASS error");                               \
    }                                                                          \
  }

using namespace cute;

/***************************************************************************************************
 *                          CUTLASS Distributed GEMM Configuration (2-GPU)
 **************************************************************************************************/

// Fixed Tensor Parallelism degree of 2.
using TP = _2;
static constexpr int TP_ = TP{};

// The distribution schedule, identical to the standalone benchmark.
// It splits the problem along the N dimension and uses an All-Gather for communication.
using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;

// Data type, layout, and alignment configuration.
using         ElementA    = cutlass::half_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 8;

using         ElementB    = cutlass::half_t;
using         LayoutB     = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 8;

using         ElementC    = cutlass::half_t;
using         LayoutC     = cutlass::layout::ColumnMajor;
constexpr int AlignmentC  = 8;

using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

using ElementAccumulator  = cutlass::half_t;
using ElementCompute      = cutlass::half_t;
using ArchTag             = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
using TileShape           = Shape<_128,_256,_64>;
using ClusterShape        = Shape<_1,_2,_1>;

using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

// Assemble the collective epilogue and mainloop.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

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

// The base, non-distributed GEMM kernel.
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Adapter for the base kernel.
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Wrap the base kernel with the distributed schedule.
using DistGemmKernel = cutlass::distributed::kernel::DistributedGemmKernelWrapper<
  GemmKernel,
  DistSchedule
>;
// The final device-level adapter for launching the distributed kernel.
using DistGemm = cutlass::distributed::device::DistributedGemmUniversalAdapter<DistGemmKernel>;

// Stride types for tensor manipulation.
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;



/**
 * @brief The C++ function, exposed to Python, that executes the 2-GPU distributed GEMM.
 *
 * @param A Input tensor (M x K), expected to be row-major.
 * @param B Input tensor (K x N), expected to be row-major.
 * @param C Output tensor (M x N), which will be overwritten with the result.
 */
void cutlass_gemm_multi_sm90_official(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // Get problem dimensions from the input tensors.
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    int L = 1; // Batch size is 1

    
    // Ensure tensors have the correct layout for the CUTLASS kernel.
    // The kernel expects A to be RowMajor, and B/C to be ColumnMajor.
    // PyTorch tensors are row-major by default, so we need to transpose B and C.
    auto A_row_major = A.contiguous();
    auto B_column_major = B.transpose(0, 1).contiguous();
    auto C_column_major = C.transpose(0, 1).contiguous();

    
    // Store the primary device ID to restore it later.
    int primary_device_idx;
    CUDA_CHECK(cudaGetDevice(&primary_device_idx));

    
    // Enable peer-to-peer access between the 2 GPUs.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        for (int peer_idx = 0; peer_idx < TP_; ++peer_idx) {
            if (peer_idx != device_idx) {
                int can_access;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_idx, peer_idx));
                if (!can_access) {
                    throw std::runtime_error("Device " + std::to_string(device_idx) + 
                                           " can't access device " + std::to_string(peer_idx));
                }
                cudaError_t err = cudaDeviceEnablePeerAccess(peer_idx, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_CHECK(err);
                } else {
                    cudaGetLastError(); // Clear the 'already enabled' error
                }
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(primary_device_idx));

    // Define the global problem shape for CUTLASS.
    auto problem_shape = cute::make_tuple(M, N, K, L);

    
    // Define the shapes and strides for the global tensors.
    auto shape_A = cute::select<0,2,3>(problem_shape);
    auto shape_B = cute::select<1,2,3>(problem_shape);
    auto shape_C = cute::select<0,1,3>(problem_shape);
    auto shape_D = cute::select<0,1,3>(problem_shape);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_C);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_D);

    
    // Get raw pointers to the global tensor data on the device.
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A_row_major.data_ptr<at::Half>());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B_column_major.data_ptr<at::Half>());
    ElementD* C_ptr = reinterpret_cast<ElementD*>(C_column_major.data_ptr<at::Half>());

    
    // Create a CUDA stream for each GPU.
    cudaStream_t stream_arr[TP_];
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaStreamCreate(&stream_arr[device_idx]));
    }

    
    // Get the shapes of the local, per-GPU matrix slices.
    auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
    auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
    auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
    auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);

    // Create strides for the local tensors.
    auto local_stride_A = cutlass::make_cute_packed_stride(StrideA{}, local_shape_A);
    auto local_stride_B = cutlass::make_cute_packed_stride(StrideB{}, local_shape_B);
    auto local_stride_C = cutlass::make_cute_packed_stride(StrideC{}, local_shape_C);
    auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);

    
    // Allocate local device memory for each GPU's matrix slices.
    ElementA* local_A_arr[TP_];
    ElementB* local_B_arr[TP_];
    ElementC* local_C_arr[TP_];
    ElementD* local_D_arr[TP_];

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaMalloc(&local_A_arr[device_idx], cute::size(local_shape_A) * sizeof(ElementA)));
        CUDA_CHECK(cudaMalloc(&local_B_arr[device_idx], cute::size(local_shape_B) * sizeof(ElementB)));
        CUDA_CHECK(cudaMalloc(&local_C_arr[device_idx], cute::size(local_shape_C) * sizeof(ElementC)));
        CUDA_CHECK(cudaMalloc(&local_D_arr[device_idx], cute::size(local_shape_D) * sizeof(ElementD)));
    }

    
    // Create CuTe tensor objects for the global matrices.
    auto global_A = cute::make_tensor(A_ptr,
        cute::make_layout(cute::make_shape(M, K, L), stride_A));
    auto global_B = cute::make_tensor(B_ptr,
        cute::make_layout(cute::make_shape(N, K, L), stride_B));
    auto global_C = cute::make_tensor(C_ptr,
        cute::make_layout(cute::make_shape(M, N, L), stride_C));

    
    // For each GPU, copy its slice of the global data into its local memory.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));

        // Use the DistSchedule to determine which part of the global tensor belongs to this device.
        auto global_A_device_slice = DistSchedule::get_device_slice_A(global_A, device_idx);
        auto global_B_device_slice = DistSchedule::get_device_slice_B(global_B, device_idx);
        auto global_C_device_slice = DistSchedule::get_device_slice_C(global_C, device_idx);

        // Create CuTe tensors for the local, per-GPU memory buffers.
        auto local_A = cute::make_tensor(local_A_arr[device_idx],
            make_layout(local_shape_A, local_stride_A));
        auto local_B = cute::make_tensor(local_B_arr[device_idx],
            make_layout(local_shape_B, local_stride_B));
        auto local_C = cute::make_tensor(local_C_arr[device_idx],
            make_layout(local_shape_C, local_stride_C));

        // Perform the async copies.
        cudaMemcpyAsync(local_A_arr[device_idx], global_A_device_slice.data(),
            sizeof(ElementA) * cute::size(local_shape_A), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
        cudaMemcpyAsync(local_B_arr[device_idx], global_B_device_slice.data(),
            sizeof(ElementB) * cute::size(local_shape_B), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
        cudaMemcpyAsync(local_C_arr[device_idx], global_C_device_slice.data(),
            sizeof(ElementC) * cute::size(local_shape_C), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
    }

    
    // Synchronize to ensure all data copies are complete before proceeding.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    // Setup for distributed GEMM launch.
    DistGemm dist_gemm_arr[TP_];

    
    // Allocate shared and exclusive workspaces for each GPU.
    cutlass::device_memory::allocation<uint8_t> workspace_arr[TP_];
    cutlass::device_memory::allocation<uint8_t> exclusive_workspace_arr[TP_];

    void* workspace_ptr_arr[TP_];
    void* exclusive_workspace_ptr_arr[TP_];

    
    // Array to hold the kernel arguments for each rank.
    typename DistGemm::Arguments arguments_[TP_];

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));

        // Populate the arguments struct for the current GPU.
        arguments_[device_idx] = {
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_shape,
            {
                reinterpret_cast<const ElementA*>(local_A_arr[device_idx]),
                local_stride_A,
                reinterpret_cast<const ElementB*>(local_B_arr[device_idx]),
                local_stride_B
            },
            {
                {static_cast<ElementCompute>(1.0f), static_cast<ElementCompute>(0.0f)},
                reinterpret_cast<const ElementC*>(local_C_arr[device_idx]),
                local_stride_C,
                reinterpret_cast<ElementD*>(local_D_arr[device_idx]),
                local_stride_D
            },
            {},
            {}
        };

        // Allocate workspace memory.
        size_t workspace_size = DistGemm::get_workspace_size(arguments_[device_idx]);
        size_t exclusive_workspace_size = DistGemm::get_exclusive_workspace_size();

        workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(workspace_size);
        exclusive_workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(exclusive_workspace_size);

        workspace_ptr_arr[device_idx] = workspace_arr[device_idx].get();
        exclusive_workspace_ptr_arr[device_idx] = exclusive_workspace_arr[device_idx].get();

        cudaMemsetAsync(exclusive_workspace_ptr_arr[device_idx], 0, exclusive_workspace_size, stream_arr[device_idx]);
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    // Initialize the distributed GEMM objects on all GPUs. This is a collective operation.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));

        CUTLASS_CHECK(dist_gemm_arr[device_idx].can_implement(arguments_[device_idx]));

        // PDL launch mode is not used in this example.
        bool launch_with_pdl = false;

        CUTLASS_CHECK(dist_gemm_arr[device_idx].initialize(
            arguments_,
            workspace_ptr_arr,
            exclusive_workspace_ptr_arr,
            device_idx,
            stream_arr[device_idx],
            launch_with_pdl
        ));

        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    // Launch the distributed kernel on all GPUs.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
    }

    
    // Create a CuTe tensor for the final global output matrix D.
    auto global_D = cute::make_tensor(C_ptr,
        cute::make_layout(cute::make_shape(M, N, L), stride_D));

    // For each GPU, copy its local result slice back to the correct location in the global output tensor.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));

        
        // Get the slice of the global output tensor for this device.
        auto global_D_device_slice = DistSchedule::get_device_slice_D(global_D, device_idx);
        // This copy reassembles the final result.
        cudaMemcpyAsync(global_D_device_slice.data(), local_D_arr[device_idx],
            sizeof(ElementD) * cute::size(local_shape_D), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
    }

    
    // Synchronize to ensure all result copies are finished.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    // The result is now in `C_column_major`. Transpose it back to row-major and copy to the original `C` tensor.
    C.copy_(C_column_major.transpose(0, 1));

    
    // Clean up all allocated device memory and streams.
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaFree(local_A_arr[device_idx]));
        CUDA_CHECK(cudaFree(local_B_arr[device_idx]));
        CUDA_CHECK(cudaFree(local_C_arr[device_idx]));
        CUDA_CHECK(cudaFree(local_D_arr[device_idx]));
        CUDA_CHECK(cudaStreamDestroy(stream_arr[device_idx]));
    }

    // Restore the original device context.
    CUDA_CHECK(cudaSetDevice(primary_device_idx));
}

/**
 * @brief Pybind11 module definition to expose the C++ function to Python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &cutlass_gemm_multi_sm90_official, "Official Multi-GPU FP16 GEMM (2 GPUs, Distributed)");
}

