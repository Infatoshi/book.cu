/***************************************************************************************************
 * Official Distributed GEMM for 2 GPUs (Hopper SM90)
 * Adapted from CUTLASS Example 65
 **************************************************************************************************/








using namespace cute;


using TP = _2;
static constexpr int TP_ = TP{};

using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;


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

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using DistGemmKernel = cutlass::distributed::kernel::DistributedGemmKernelWrapper<
  GemmKernel,
  DistSchedule
>;
using DistGemm = cutlass::distributed::device::DistributedGemmUniversalAdapter<DistGemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;



void cutlass_gemm_multi_sm90_official(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);
    int L = 1; 

    
    auto A_row_major = A.contiguous();
    auto B_column_major = B.transpose(0, 1).contiguous();
    auto C_column_major = C.transpose(0, 1).contiguous();

    
    int primary_device_idx;
    CUDA_CHECK(cudaGetDevice(&primary_device_idx));

    
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
                    cudaGetLastError(); 
                }
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(primary_device_idx));

    auto problem_shape = cute::make_tuple(M, N, K, L);

    
    auto shape_A = cute::select<0,2,3>(problem_shape);
    auto shape_B = cute::select<1,2,3>(problem_shape);
    auto shape_C = cute::select<0,1,3>(problem_shape);
    auto shape_D = cute::select<0,1,3>(problem_shape);

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_C);
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_D);

    
    ElementA* A_ptr = reinterpret_cast<ElementA*>(A_row_major.data_ptr<at::Half>());
    ElementB* B_ptr = reinterpret_cast<ElementB*>(B_column_major.data_ptr<at::Half>());
    ElementD* C_ptr = reinterpret_cast<ElementD*>(C_column_major.data_ptr<at::Half>());

    
    cudaStream_t stream_arr[TP_];
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaStreamCreate(&stream_arr[device_idx]));
    }

    
    auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
    auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
    auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
    auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);

    auto local_stride_A = cutlass::make_cute_packed_stride(StrideA{}, local_shape_A);
    auto local_stride_B = cutlass::make_cute_packed_stride(StrideB{}, local_shape_B);
    auto local_stride_C = cutlass::make_cute_packed_stride(StrideC{}, local_shape_C);
    auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);

    
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

    
    auto global_A = cute::make_tensor(A_ptr,
        cute::make_layout(cute::make_shape(M, K, L), stride_A));
    auto global_B = cute::make_tensor(B_ptr,
        cute::make_layout(cute::make_shape(N, K, L), stride_B));
    auto global_C = cute::make_tensor(C_ptr,
        cute::make_layout(cute::make_shape(M, N, L), stride_C));

    
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));

        auto global_A_device_slice = DistSchedule::get_device_slice_A(global_A, device_idx);
        auto global_B_device_slice = DistSchedule::get_device_slice_B(global_B, device_idx);
        auto global_C_device_slice = DistSchedule::get_device_slice_C(global_C, device_idx);

        auto local_A = cute::make_tensor(local_A_arr[device_idx],
            make_layout(local_shape_A, local_stride_A));
        auto local_B = cute::make_tensor(local_B_arr[device_idx],
            make_layout(local_shape_B, local_stride_B));
        auto local_C = cute::make_tensor(local_C_arr[device_idx],
            make_layout(local_shape_C, local_stride_C));

        cudaMemcpyAsync(local_A_arr[device_idx], global_A_device_slice.data(),
            sizeof(ElementA) * cute::size(local_shape_A), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
        cudaMemcpyAsync(local_B_arr[device_idx], global_B_device_slice.data(),
            sizeof(ElementB) * cute::size(local_shape_B), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
        cudaMemcpyAsync(local_C_arr[device_idx], global_C_device_slice.data(),
            sizeof(ElementC) * cute::size(local_shape_C), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
    }

    
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    DistGemm dist_gemm_arr[TP_];

    
    cutlass::device_memory::allocation<uint8_t> workspace_arr[TP_];
    cutlass::device_memory::allocation<uint8_t> exclusive_workspace_arr[TP_];

    void* workspace_ptr_arr[TP_];
    void* exclusive_workspace_ptr_arr[TP_];

    
    typename DistGemm::Arguments arguments_[TP_];

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));

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

        size_t workspace_size = DistGemm::get_workspace_size(arguments_[device_idx]);
        size_t exclusive_workspace_size = DistGemm::get_exclusive_workspace_size();

        workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(workspace_size);
        exclusive_workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(exclusive_workspace_size);

        workspace_ptr_arr[device_idx] = workspace_arr[device_idx].get();
        exclusive_workspace_ptr_arr[device_idx] = exclusive_workspace_arr[device_idx].get();

        cudaMemsetAsync(exclusive_workspace_ptr_arr[device_idx], 0, exclusive_workspace_size, stream_arr[device_idx]);
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));

        CUTLASS_CHECK(dist_gemm_arr[device_idx].can_implement(arguments_[device_idx]));

        bool launch_with_pdl = true;
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

    
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
    }

    
    auto global_D = cute::make_tensor(C_ptr,
        cute::make_layout(cute::make_shape(M, N, L), stride_D));

    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));

        
        auto global_D_device_slice = DistSchedule::get_device_slice_D(global_D, device_idx);
        auto local_D = cute::make_tensor(local_D_arr[device_idx],
            make_layout(local_shape_D, local_stride_D));
        cudaMemcpyAsync(global_D_device_slice.data(), local_D_arr[device_idx],
            sizeof(ElementD) * cute::size(local_shape_D), cudaMemcpyDeviceToDevice, stream_arr[device_idx]);
    }

    
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
    }

    
    C.copy_(C_column_major.transpose(0, 1));

    
    for (int device_idx = 0; device_idx < TP_; ++device_idx) {
        CUDA_CHECK(cudaSetDevice(device_idx));
        CUDA_CHECK(cudaFree(local_A_arr[device_idx]));
        CUDA_CHECK(cudaFree(local_B_arr[device_idx]));
        CUDA_CHECK(cudaFree(local_C_arr[device_idx]));
        CUDA_CHECK(cudaFree(local_D_arr[device_idx]));
        CUDA_CHECK(cudaStreamDestroy(stream_arr[device_idx]));
    }

    CUDA_CHECK(cudaSetDevice(primary_device_idx));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &cutlass_gemm_multi_sm90_official, "Official Multi-GPU FP16 GEMM (2 GPUs, Distributed)");
}

