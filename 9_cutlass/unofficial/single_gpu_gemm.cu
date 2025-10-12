

/**
 * CUTLASS Single-GPU GEMM Implementation
 * Demonstrates high-performance matrix multiplication using NVIDIA's CUTLASS library
 * Optimized for Hopper architecture (SM90) with FP16 precision and Tensor Cores
 * 
 * This implementation showcases:
 * - Modern CUTLASS API with CuTe tensor abstractions
 * - Automatic kernel scheduling and optimization
 * - FP16 input/output with FP32 accumulation for numerical stability
 * - Comprehensive verification and benchmarking
 */

using namespace cute;

// Architecture and data type configuration
using ArchTag = cutlass::arch::Sm90;           // Hopper architecture (H100, RTX 4090, etc.)
using ElementInput = cutlass::half_t;          // FP16 input data type
using ElementOutput = cutlass::half_t;         // FP16 output data type
using ElementAccumulator = float;              // FP32 accumulation for numerical stability
using ElementC = ElementOutput;                // C matrix element type
using ElementD = ElementOutput;                // D matrix element type

// Memory layout configuration (row-major for all matrices)
using LayoutA = cutlass::layout::RowMajor;     // A matrix layout
using LayoutB = cutlass::layout::RowMajor;     // B matrix layout
using LayoutC = cutlass::layout::RowMajor;     // C matrix layout
using LayoutD = cutlass::layout::RowMajor;     // D matrix layout

// Tile configuration for optimal performance
using TileShape = Shape<_128, _256, _64>;      // Thread block tile: 128×256×64
using ClusterShape = Shape<_2, _1, _1>;        // Cluster shape for multi-GPU coordination
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto; // Automatic scheduling

// Memory alignment requirements for optimal performance
static constexpr int AlignmentA = 16 / sizeof(ElementInput);  // A matrix alignment
static constexpr int AlignmentB = 16 / sizeof(ElementInput);  // B matrix alignment
static constexpr int AlignmentC = 16 / sizeof(ElementOutput);  // C matrix alignment
static constexpr int AlignmentD = 16 / sizeof(ElementOutput);  // D matrix alignment

// Compute type for epilogue operations
using ElementCompute = float;

// Epilogue operation: D = alpha * C + beta * D (where alpha=1, beta=0 for simple copy)
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementCompute, ElementOutput, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// Collective epilogue: handles output processing and memory writes
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,     // Architecture and operation class
    TileShape, ClusterShape,                      // Tile and cluster shapes
    cutlass::epilogue::collective::EpilogueTileAuto, // Automatic epilogue tiling
    ElementAccumulator, ElementCompute,          // Accumulator and compute types
    ElementC, LayoutC, AlignmentC,               // C matrix configuration
    ElementD, LayoutD, AlignmentD,               // D matrix configuration
    cutlass::epilogue::collective::EpilogueScheduleAuto, // Automatic scheduling
    EpilogueOp                                   // Epilogue operation
>::CollectiveOp;

// Collective mainloop: handles the core GEMM computation
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, cutlass::arch::OpClassTensorOp,     // Architecture and operation class
    ElementInput, LayoutA, AlignmentA,           // A matrix configuration
    ElementInput, LayoutB, AlignmentB,           // B matrix configuration
    ElementAccumulator,                          // Accumulator type
    TileShape, ClusterShape,                      // Tile and cluster shapes
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>, // Shared memory management
    KernelSchedule                               // Kernel scheduling strategy
>::CollectiveOp;

// GEMM kernel: combines mainloop and epilogue
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,                        // Problem shape type
    CollectiveMainloop,                          // Main computation loop
    CollectiveEpilogue                           // Output processing
>;

// GEMM device adapter: high-level interface to the kernel
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

/**
 * CUDA error checking macro
 * @param call CUDA function call to check
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * Convert CUTLASS data types to float for computation
 * @param val Value to convert
 * @return Float representation
 */
template<typename T>
float to_float(T val) {
    if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return float(val);  // Convert FP16 to FP32
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        return float(val);  // Convert FP8 to FP32
    } else {
        return static_cast<float>(val);  // Generic conversion
    }
}

/**
 * Convert float to CUTLASS data types
 * @param val Float value to convert
 * @return Converted value
 */
template<typename T>
T from_float(float val) {
    if constexpr (std::is_same_v<T, cutlass::half_t>) {
        return cutlass::half_t(val);  // Convert FP32 to FP16
    } else if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>) {
        return cutlass::float_e4m3_t(val);  // Convert FP32 to FP8
    } else {
        return static_cast<T>(val);  // Generic conversion
    }
}

/**
 * CPU reference implementation of GEMM for verification
 * Computes C = A * B where A is M×K, B is K×N, C is M×N
 * 
 * @param A Input matrix A (M×K, row-major)
 * @param B Input matrix B (K×N, row-major)
 * @param C Output matrix C (M×N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
template<typename InType, typename OutType>
void cpu_gemm(const std::vector<InType>& A, const std::vector<InType>& B,
              std::vector<OutType>& C, int M, int N, int K) {
    // Standard triple-nested loop GEMM implementation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            // Compute dot product of row A[i,:] and column B[:,j]
            for (int k = 0; k < K; k++) {
                float a_val = to_float(A[i * K + k]);      // Convert to float for computation
                float b_val = to_float(B[k * N + j]);      // Convert to float for computation
                sum += a_val * b_val;                      // Accumulate multiply-add
            }
            C[i * N + j] = from_float<OutType>(sum);       // Convert back to output type
        }
    }
}

/**
 * Initialize vector with random values in range [-1, 1]
 * @param data Vector to initialize
 * @param seed Random seed for reproducibility
 */
template<typename T>
void initialize_random(std::vector<T>& data, int seed = 42) {
    std::srand(seed);
    for (auto& val : data) {
        // Generate random float in [-1, 1] and convert to target type
        val = from_float<T>((std::rand() / float(RAND_MAX)) * 2.0f - 1.0f);
    }
}

/**
 * Verify GPU results against CPU reference implementation
 * Computes relative error statistics and determines if verification passes
 * 
 * @param gpu_result GPU computation results
 * @param cpu_result CPU reference results
 * @param tolerance Maximum allowed average relative error (default: 0.5%)
 * @return true if verification passes, false otherwise
 */
template<typename T>
bool verify_results(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result,
                    float tolerance = 0.5f) {
    // Check size compatibility
    if (gpu_result.size() != cpu_result.size()) return false;
    
    float max_error = 0.0f;      // Maximum relative error
    float avg_error = 0.0f;      // Average relative error
    int non_zero_count = 0;      // Count of non-zero elements
    
    // Compute error statistics
    for (size_t i = 0; i < gpu_result.size(); i++) {
        float gpu_val = to_float(gpu_result[i]);
        float cpu_val = to_float(cpu_result[i]);
        
        // Only compute relative error for non-zero elements
        if (std::abs(cpu_val) > 1e-5f) {
            float rel_error = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);
            max_error = std::max(max_error, rel_error);
            avg_error += rel_error;
            non_zero_count++;
        }
    }
    
    avg_error /= non_zero_count;
    
    // Print error statistics
    std::cout << "Max relative error: " << max_error 
              << " (over " << non_zero_count << " non-zero elements)" << std::endl;
    std::cout << "Average relative error: " << avg_error 
              << " (" << (avg_error * 100.0f) << "%)" << std::endl;
    
    // Determine if verification passes
    bool passed = avg_error < tolerance;
    std::cout << (passed ? "✓" : "✗") << " CPU verification " 
              << (passed ? "PASSED" : "FAILED") 
              << " (average error < " << (tolerance * 100.0f) << "%)" << std::endl;
    
    return passed;
}

template<typename InType, typename OutType>
void run_gemm(const std::vector<InType>& h_A, const std::vector<InType>& h_B,
              std::vector<OutType>& h_C, int M, int N, int K,
              int warmup_iters = 5, int bench_iters = 10) {
    
    InType *d_A, *d_B;
    OutType *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(InType)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(InType)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(OutType)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(InType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(InType), cudaMemcpyHostToDevice));
    
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
        {d_A, stride_a, d_B, stride_b},
        {{1.0f, 0.0f}, d_C, stride_c, d_C, stride_d}
    };
    
    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(args);
    
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_ptr, workspace_size));
    }
    
    cutlass::Status status = gemm_op.initialize(args, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM initialization failed" << std::endl;
        exit(1);
    }
    
    for (int i = 0; i < warmup_iters; i++) {
        gemm_op.run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        gemm_op.run();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / bench_iters;
    double tflops = (2.0 * M * N * K) / (time_ms * 1e9);
    
    std::cout << "Average time: " << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(OutType), cudaMemcpyDeviceToHost));
    
    if (workspace_ptr) cudaFree(workspace_ptr);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char** argv) {
    std::cout << "\n=== CUTLASS Single-GPU GEMM ===" << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << prop.minor << ")" << std::endl;
    std::cout << std::endl;
    
    {
        std::cout << "=== CPU Verification (1024³) ===" << std::endl;
        const int M = 1024, N = 1024, K = 1024;
        
        std::vector<ElementInput> h_A(M * K);
        std::vector<ElementInput> h_B(K * N);
        std::vector<ElementOutput> h_C_gpu(M * N);
        std::vector<ElementOutput> h_C_cpu(M * N);
        
        initialize_random(h_A, 42);
        initialize_random(h_B, 43);
        
        std::cout << "Running CPU GEMM for verification..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        std::cout << "CPU GEMM time: " << cpu_time << " ms" << std::endl;
        
        run_gemm(h_A, h_B, h_C_gpu, M, N, K, 1, 1);
        
        bool passed = verify_results(h_C_gpu, h_C_cpu);
        
        if (!passed) {
            std::cerr << "\n✗ Verification failed! Not proceeding to benchmark." << std::endl;
            return 1;
        }
        std::cout << std::endl;
    }
    
    {
        std::cout << "=== GPU Benchmark (8192³) ===" << std::endl;
        const int M = 8192, N = 8192, K = 8192;
        
        std::vector<ElementInput> h_A(M * K);
        std::vector<ElementInput> h_B(K * N);
        std::vector<ElementOutput> h_C(M * N);
        
        initialize_random(h_A, 42);
        initialize_random(h_B, 43);
        
        run_gemm(h_A, h_B, h_C, M, N, K, 5, 10);
    }
    
    std::cout << "\n✓ Complete!" << std::endl;
    return 0;
}