/***************************************************************************************************
 * @file gemm_sm90_official.cu
 * @brief An official CUTLASS 3.x single-GPU GEMM kernel for NVIDIA Hopper (SM90),
 *        wrapped for use with PyTorch.
 *
 * @details This file demonstrates the modern CUTLASS 3.x API for building a high-performance,
 *          single-GPU GEMM kernel specifically targeting the NVIDIA Hopper architecture (SM90).
 *          It uses Tensor Cores for FP16 computation with FP32 accumulation.
 *
 *          Key CUTLASS 3.x Concepts Demonstrated:
 *          - **`CollectiveBuilder`**: A unified builder for creating both the `CollectiveMainloop`
 *            (the core `A*B` computation) and the `CollectiveEpilogue` (writing the result).
 *          - **`GemmUniversalAdapter`**: A device-level wrapper that simplifies launching the kernel,
 *            handling argument packing and workspace management.
 *          - **CuTe Integration**: Uses `cute::Shape` for tiling (`TileShape`, `ClusterShape`) and
 *            `cute::make_cute_packed_stride` for defining memory layouts.
 *          - **PyTorch Integration**: The kernel is wrapped in a C++ function that accepts
 *            `torch::Tensor` objects, making it easy to call from Python.
 **************************************************************************************************/

#include <torch/extension.h>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

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
 *                            CUTLASS 3.x GEMM Configuration for Hopper
 **************************************************************************************************/

//
// Data Type, Layout, and Alignment
//
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float; // Use FP32 for accumulation to maintain precision

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// Memory alignment is crucial for performance. 128-bit (16-byte) alignment is standard.
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

//
// Tiling and Scheduling Configuration
//
// - TileShape: The size of the GEMM problem solved by one thread block (M, N, K).
// - ClusterShape: The 2D arrangement of thread blocks. A 2x1 cluster can improve locality.
//
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;

//
// Mainloop and Epilogue Construction
//
// CUTLASS 3.x uses a `CollectiveBuilder` to compose the main computation loop and the epilogue.
//

// The epilogue handles writing the accumulated result to the output tensor D.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// The mainloop performs the core multiply-accumulate operations (A * B).
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    // Automatically configure shared memory staging based on the epilogue's requirements.
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

//
// Final Kernel and Adapter
//

// The `GemmUniversal` kernel combines the mainloop and epilogue into a single entity.
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>, // Problem shape is specified at runtime
    CollectiveMainloop,
    CollectiveEpilogue
>;

// The `GemmUniversalAdapter` provides a user-friendly, device-level API to the kernel.
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

/**
 * @brief C++ function that executes the CUTLASS Hopper GEMM on PyTorch tensors.
 *
 * @param A Input tensor (M x K).
 * @param B Input tensor (K x N).
 * @param C Output tensor (M x N), which will be overwritten.
 */
void cutlass_gemm_official_sm90(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    // Extract problem dimensions from the input tensors.
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Get raw data pointers from the PyTorch tensors.
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A.data_ptr<at::Half>());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B.data_ptr<at::Half>());
    ElementD* C_ptr = reinterpret_cast<ElementD*>(C.data_ptr<at::Half>());

    // Define stride types from the kernel.
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    // Create stride objects for packed row-major layouts.
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Populate the kernel arguments struct. This includes the problem size,
    // pointers and strides for the matrices, and the epilogue scalars (alpha, beta).
    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A_ptr, stride_a, B_ptr, stride_b},
        // Perform D = 1.0 * (A*B) + 0.0 * C. The output is written to C_ptr.
        {{1.0f, 0.0f}, C_ptr, stride_c, C_ptr, stride_d}
    };

    // Instantiate the GEMM adapter.
    Gemm gemm_op;
    // Allocate workspace memory if required by the kernel.
    size_t workspace_size = Gemm::get_workspace_size(args);
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_ptr, workspace_size));
    }

    // Initialize the kernel. This configures launch parameters and checks if the kernel
    // can implement the requested problem size.
    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        throw std::runtime_error("Official SM90 GEMM initialization failed");
    }

    // Launch the kernel.
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        throw std::runtime_error("Official SM90 GEMM execution failed");
    }

    // Free the workspace memory.
    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
}

/**
 * @brief Pybind11 module definition to expose the C++ function to Python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &cutlass_gemm_official_sm90, "Official CUTLASS Hopper FP16 GEMM");
}

