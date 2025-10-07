
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;  
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;     

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementCompute
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3  
>;

void cutlass_gemm_official_sm80(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A.data_ptr<at::Half>());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B.data_ptr<at::Half>());
    ElementC* C_ptr = reinterpret_cast<ElementC*>(C.data_ptr<at::Half>());
    
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    cutlass::TensorRef<ElementA, LayoutA> ref_A(A_ptr, K);
    cutlass::TensorRef<ElementB, LayoutB> ref_B(B_ptr, N);
    cutlass::TensorRef<ElementC, LayoutC> ref_C(C_ptr, N);
    cutlass::TensorRef<ElementC, LayoutC> ref_D(C_ptr, N);
    
    typename Gemm::Arguments args(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        {ElementCompute(1.0f), ElementCompute(0.0f)}
    );
    
    Gemm gemm_op;
    
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Official SM80 GEMM cannot implement this problem size");
    }
    
    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Official SM80 GEMM initialization failed");
    }
    
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Official SM80 GEMM execution failed");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &cutlass_gemm_official_sm80, "Official CUTLASS Ampere FP16 GEMM");
}

