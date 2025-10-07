/***************************************************************************************************
 * Standalone Distributed GEMM Performance Benchmark
 * Based on CUTLASS Example 65
 * 
 * Compile:
 *   nvcc -std=c++17 -O3 --use_fast_math \
 *        -gencode=arch=compute_90a,code=sm_90a \
 *        -I../../cutlass/include \
 *        -I../../cutlass/tools/util/include \
 *        -I../../cutlass/examples \
 *        -DCUTLASS_ENABLE_GDC_FOR_SM90=1 \
 *        standalone_dist_gemm.cu -o standalone_dist_gemm
 * 
 * Run:
 *   ./standalone_dist_gemm --num-gpus=2 --m=8192 --n=8192 --k=8192 --iterations=100
 **************************************************************************************************/





using namespace cute;


template<int TP_>
struct DistGemmConfig {
  using TP = cute::Int<TP_>;
  static constexpr int TP_val = TP_;
  
  
  using DistSchedule = cutlass::distributed::schedules::AllGather1D_TilingCD_RotatingA<TP>;
  
  
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
  
  
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  
  
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  
  
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<_128, _256, _64>;
  using ClusterShape = Shape<_1, _2, _1>;
  
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  
  
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
      Shape<int, int, int, int>,
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
  
  using HostTensorA = cutlass::HostTensor<ElementA, LayoutA>;
  using HostTensorB = cutlass::HostTensor<ElementB, LayoutB>;
  using HostTensorC = cutlass::HostTensor<ElementC, LayoutC>;
  using HostTensorD = cutlass::HostTensor<ElementD, LayoutD>;
};


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
  
  double tflops(double runtime_s) const {
    uint64_t flop = uint64_t(2) * m * n * k * l / num_gpus;
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }
};


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
  
  
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  if (num_devices < TP) {
    std::cerr << "Error: Requested " << TP << " GPUs but only " << num_devices << " available\n";
    return -1;
  }
  
  int primary_device_idx;
  CUDA_CHECK(cudaGetDevice(&primary_device_idx));
  
  auto problem_shape = cute::make_tuple(options.m, options.n, options.k, options.l);
  
  
  auto shape_A = cute::select<0, 2, 3>(problem_shape);
  auto shape_B = cute::select<1, 2, 3>(problem_shape);
  auto shape_C = cute::select<0, 1, 3>(problem_shape);
  auto shape_D = cute::select<0, 1, 3>(problem_shape);
  
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, shape_C);
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, shape_D);
  
  
  auto a_coord = cutlass::make_Coord(size(shape_A), 1);
  auto b_coord = cutlass::make_Coord(size(shape_B), 1);
  auto c_coord = cutlass::make_Coord(size(shape_C), 1);
  
  HostTensorA tensor_A(a_coord);
  HostTensorB tensor_B(b_coord);
  HostTensorC tensor_C(c_coord);
  
  
  uint64_t seed = 2024;
  cutlass::reference::device::TensorFillRandomUniform(
    tensor_A.device_view(), seed, ElementA(2), ElementA(-2), 0);
  cutlass::reference::device::TensorFillRandomUniform(
    tensor_B.device_view(), seed + 1, ElementB(2), ElementB(-2), 0);
  cutlass::reference::device::TensorFillRandomUniform(
    tensor_C.device_view(), seed + 2, ElementC(2), ElementC(-2), 0);
  
  
  auto local_shape_A = DistSchedule::get_local_a_shape(problem_shape);
  auto local_shape_B = DistSchedule::get_local_b_shape(problem_shape);
  auto local_shape_C = DistSchedule::get_local_c_shape(problem_shape);
  auto local_shape_D = DistSchedule::get_local_d_shape(problem_shape);
  
  auto a_coord_device = cutlass::make_Coord(size(local_shape_A), 1);
  auto b_coord_device = cutlass::make_Coord(size(local_shape_B), 1);
  auto c_coord_device = cutlass::make_Coord(size(local_shape_C), 1);
  
  HostTensorA tensor_A_arr[TP];
  HostTensorB tensor_B_arr[TP];
  HostTensorC tensor_C_arr[TP];
  HostTensorD tensor_D_arr[TP];
  
  
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
    
    tensor_A_arr[device_idx].resize(a_coord_device);
    tensor_B_arr[device_idx].resize(b_coord_device);
    tensor_C_arr[device_idx].resize(c_coord_device);
    tensor_D_arr[device_idx].resize(c_coord_device);
  }
  CUDA_CHECK(cudaSetDevice(primary_device_idx));
  
  
  cudaStream_t stream_arr[TP];
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamCreate(&stream_arr[device_idx]));
  }
  
  
  DistGemm dist_gemm_arr[TP];
  cutlass::device_memory::allocation<uint8_t> workspace_arr[TP];
  cutlass::device_memory::allocation<uint8_t> exclusive_workspace_arr[TP];
  void* workspace_ptr_arr[TP];
  void* exclusive_workspace_ptr_arr[TP];
  typename DistGemm::Arguments arguments_[TP];
  
  
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    
    auto global_A = cute::make_tensor(tensor_A.device_data(),
        cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
    auto global_B = cute::make_tensor(tensor_B.device_data(),
        cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
    auto global_C = cute::make_tensor(tensor_C.device_data(),
        cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));
    
    auto global_A_device_slice = DistSchedule::get_device_slice_A(global_A, device_idx);
    auto global_B_device_slice = DistSchedule::get_device_slice_B(global_B, device_idx);
    auto global_C_device_slice = DistSchedule::get_device_slice_C(global_C, device_idx);
    
    auto local_stride_A = cutlass::make_cute_packed_stride(StrideA{}, local_shape_A);
    auto local_stride_B = cutlass::make_cute_packed_stride(StrideB{}, local_shape_B);
    auto local_stride_C = cutlass::make_cute_packed_stride(StrideC{}, local_shape_C);
    auto local_stride_D = cutlass::make_cute_packed_stride(StrideD{}, local_shape_D);
    
    auto local_A = cute::make_tensor(tensor_A_arr[device_idx].device_data(),
        make_layout(local_shape_A, local_stride_A));
    auto local_B = cute::make_tensor(tensor_B_arr[device_idx].device_data(),
        make_layout(local_shape_B, local_stride_B));
    auto local_C = cute::make_tensor(tensor_C_arr[device_idx].device_data(),
        make_layout(local_shape_C, local_stride_C));
    auto local_D = cute::make_tensor(tensor_D_arr[device_idx].device_data(),
        make_layout(local_shape_D, local_stride_D));
    
    cutlass::device_copy(global_A_device_slice, local_A, stream_arr[device_idx]);
    cutlass::device_copy(global_B_device_slice, local_B, stream_arr[device_idx]);
    cutlass::device_copy(global_C_device_slice, local_C, stream_arr[device_idx]);
    
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
    
    size_t workspace_size = DistGemm::get_workspace_size(arguments_[device_idx]);
    size_t exclusive_workspace_size = DistGemm::get_exclusive_workspace_size();
    
    workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    exclusive_workspace_arr[device_idx] = cutlass::device_memory::allocation<uint8_t>(exclusive_workspace_size);
    
    workspace_ptr_arr[device_idx] = workspace_arr[device_idx].get();
    exclusive_workspace_ptr_arr[device_idx] = exclusive_workspace_arr[device_idx].get();
    
    cudaMemsetAsync(exclusive_workspace_ptr_arr[device_idx], 0, exclusive_workspace_size, stream_arr[device_idx]);
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
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
  
  std::cout << "Initialization complete. Running warmup...\n";
  
  
  for (int warmup = 0; warmup < options.warmup_iterations; ++warmup) {
    for (int device_idx = 0; device_idx < TP; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
    }
  }
  
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  std::cout << "Warmup complete. Running benchmark...\n";
  
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int iter = 0; iter < options.iterations; ++iter) {
    for (int device_idx = 0; device_idx < TP; ++device_idx) {
      CUDA_CHECK(cudaSetDevice(device_idx));
      CUTLASS_CHECK(dist_gemm_arr[device_idx].run(stream_arr[device_idx]));
    }
  }
  
  for (int device_idx = 0; device_idx < TP; ++device_idx) {
    CUDA_CHECK(cudaSetDevice(device_idx));
    CUDA_CHECK(cudaStreamSynchronize(stream_arr[device_idx]));
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double avg_time_ms = elapsed_ms / options.iterations;
  double avg_time_s = avg_time_ms / 1000.0;
  double tflops = options.tflops(avg_time_s);
  
  
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
  
  
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 6)) {
    std::cerr << "This program requires CUDA 12.6 or newer.\n";
    return 0;
  }
  
  
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

