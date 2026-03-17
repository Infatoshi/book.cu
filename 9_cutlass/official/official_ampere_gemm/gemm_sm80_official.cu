/***************************************************************************************************
 * @file gemm_sm80_official.cu
 * @brief An official CUTLASS 2.x GEMM kernel for NVIDIA Ampere (SM80) architecture,
 *        wrapped for use with PyTorch.
 *
 * @details This file demonstrates the CUTLASS 2.x API for creating a high-performance
 *          GEMM kernel. CUTLASS is a C++ template library for building GEMM-like primitives.
 *          This specific implementation is tailored for the NVIDIA Ampere architecture (SM80)
 *          and leverages its Tensor Core capabilities for FP16 computation.
 *
 *          Key CUTLASS 2.x concepts demonstrated:
 *          - **Tiling Hierarchy**: The GEMM problem is explicitly partitioned into a three-level
 *            hierarchy: `ThreadblockShape`, `WarpShape`, and `InstructionShape`. This defines how
 *            the work is distributed among thread blocks, warps within a block, and the Tensor
 *            Core instructions themselves.
 *          - **`cutlass::gemm::device::Gemm`**: The primary device-level template for creating a
 *            GEMM kernel. Its template arguments configure every aspect of the operation.
 *          - **Epilogue**: A `LinearCombination` epilogue is used to perform the final
 *            `D = alpha * A*B + beta * C` operation.
 *          - **PyTorch Integration**: The C++ function `cutlass_gemm_official_sm80` is exposed
 *            to Python using Pybind11 and the Torch C++ extension mechanism, allowing it to be
 *            called directly on `torch::Tensor` objects.
 **************************************************************************************************/

#include <torch/extension.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

//
// CUTLASS 2.x GEMM Configuration
//

//
// Data Type and Layout Configuration
//
// Defines the precision for the input, output, and accumulator matrices.
//
using ElementA = cutlass::half_t;           // Input A matrix element type (FP16)
using ElementB = cutlass::half_t;           // Input B matrix element type (FP16)
using ElementC = cutlass::half_t;           // Output C/D matrix element type (FP16)
using ElementAccumulator = float;         // Accumulation type (FP32 for precision)
using ElementCompute = float;             // Type for epilogue computations (e.g., alpha/beta scaling)

// Defines the memory layout of the matrices (RowMajor is standard for C-style arrays).
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

//
// Tiling Hierarchy Configuration
//
// This is a core concept in CUTLASS 2.x. The GEMM problem is tiled at three levels:
// 1. ThreadblockShape: The size of the GEMM tile processed by a single thread block. (M, N, K)
// 2. WarpShape: The size of the tile processed by a single warp within the thread block.
// 3. InstructionShape: The size of the MMA (Matrix-Multiply-Accumulate) instruction executed
//    by the Tensor Cores.
//
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;  // 128x256 tile per thread block, with a K-dimension of 32
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;         // 64x64 tile per warp
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // 16x8x16 MMA instruction shape for SM80 FP16 Tensor Cores

//
// Epilogue Configuration
//
// The epilogue operation is performed after the main matrix multiplication.
// It combines the accumulated result with the source matrix C and writes to the output matrix D.
// D = alpha * (A * B) + beta * C
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                                     // Output element type
    128 / cutlass::sizeof_bits<ElementC>::value,  // Number of elements per memory access
    ElementAccumulator,                           // Accumulator type
    ElementCompute                                // Type for alpha/beta scaling
>;

//
// Complete GEMM Kernel Definition
//
// The `cutlass::gemm::device::Gemm` class template assembles the final kernel by taking all the
// configuration types defined above as template parameters.
//
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,                              // Input A matrix description
    ElementB, LayoutB,                              // Input B matrix description
    ElementC, LayoutC,                              // Output C matrix description
    ElementAccumulator,                             // Accumulator data type
    cutlass::arch::OpClassTensorOp,                 // Indicates use of Tensor Cores
    cutlass::arch::Sm80,                            // Target NVIDIA Ampere architecture
    ThreadblockShape,                               // Tiling configuration
    WarpShape,                                      //
    InstructionShape,                               //
    EpilogueOp,                                     // The epilogue operation
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Threadblock scheduling strategy
    3                                               // Number of stages for pipeline (controls shared memory usage)
>;

/**
 * @brief C++ function that executes the CUTLASS GEMM kernel on PyTorch tensors.
 *
 * @param A The first input tensor (M x K).
 * @param B The second input tensor (K x N).
 * @param C The output tensor (M x N). The result is written into this tensor.
 */
void cutlass_gemm_official_sm80(
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
    ElementC* C_ptr = reinterpret_cast<ElementC*>(C.data_ptr<at::Half>());

    // Define the problem size for CUTLASS.
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Create CUTLASS TensorRef objects. These are lightweight wrappers around the raw pointers
    // that include stride information, allowing CUTLASS to work with different memory layouts.
    // The second argument to the constructor is the leading dimension (stride).
    cutlass::TensorRef<ElementA, LayoutA> ref_A(A_ptr, K);
    cutlass::TensorRef<ElementB, LayoutB> ref_B(B_ptr, N);
    cutlass::TensorRef<ElementC, LayoutC> ref_C(C_ptr, N); // Source C matrix
    cutlass::TensorRef<ElementC, LayoutC> ref_D(C_ptr, N); // Destination D matrix (in-place)

    // Define the arguments for the GEMM operation. This includes the problem size,
    // tensor references, and the alpha/beta scaling factors for the epilogue.
    // Here, we perform D = 1.0 * (A * B) + 0.0 * C, which is a simple matrix multiplication.
    typename Gemm::Arguments args(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        {ElementCompute(1.0f), ElementCompute(0.0f)} // {alpha, beta}
    );

    // Instantiate the GEMM operator.
    Gemm gemm_op;

    // Check if the CUTLASS kernel can be configured for the given problem size.
    // This is an important step as some kernels have alignment or size restrictions.
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Official SM80 GEMM cannot implement this problem size");
    }

    // Initialize the GEMM operator. This step pre-computes some internal parameters
    // and determines the kernel launch configuration (grid and block size).
    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Official SM80 GEMM initialization failed");
    }

    // Launch the GEMM kernel. The `operator()` is overloaded to execute the kernel.
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Official SM80 GEMM execution failed");
    }
}

/**
 * @brief Pybind11 module definition.
 *
 * @details This macro uses Pybind11 to create a Python module that exposes the C++
 *          `cutlass_gemm_official_sm80` function. The module will be named whatever
 *          `TORCH_EXTENSION_NAME` is defined as during the build process.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &cutlass_gemm_official_sm80, "Official CUTLASS Ampere FP16 GEMM");
}

