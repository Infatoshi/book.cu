
using namespace cute;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using TileShape = Shape<_128, _256, _64>;  
using ClusterShape = Shape<_2, _1, _1>;    

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void cutlass_gemm_official_sm90(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A.data_ptr<at::Half>());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B.data_ptr<at::Half>());
    ElementD* C_ptr = reinterpret_cast<ElementD*>(C.data_ptr<at::Half>());
    
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
        {A_ptr, stride_a, B_ptr, stride_b},
        {{1.0f, 0.0f}, C_ptr, stride_c, C_ptr, stride_d}
    };
    
    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(args);
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) cudaMalloc(&workspace_ptr, workspace_size);
    
    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        throw std::runtime_error("Official SM90 GEMM initialization failed");
    }
    
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        if (workspace_ptr) cudaFree(workspace_ptr);
        throw std::runtime_error("Official SM90 GEMM execution failed");
    }
    
    if (workspace_ptr) cudaFree(workspace_ptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &cutlass_gemm_official_sm90, "Official CUTLASS Hopper FP16 GEMM");
}

