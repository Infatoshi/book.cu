/**
 * @file wgmma_basic_fp16.cuh
 * @brief Basic WGMMA (Warp-Group Matrix-Multiply-Accumulate) GEMM implementation using Tensor Cores
 * 
 * This file implements a high-performance matrix multiplication kernel using NVIDIA's WGMMA instructions
 * introduced in Hopper architecture (sm_90+). WGMMA extends the concept of Tensor Cores by allowing
 * multiple warps (a "warp group") to cooperatively perform matrix operations.
 * 
 * Key Concepts:
 * 
 * ## Tensor Cores Overview
 * Tensor Cores are specialized compute units in modern NVIDIA GPUs designed to accelerate matrix
 * operations, particularly dense matrix multiplication (GEMM). They provide:
 * - Massive throughput: Up to 4x faster than standard CUDA cores for matrix operations
 * - Reduced precision support: FP16/BF16 inputs with FP32 accumulation for numerical stability
 * - Fixed-size operations: Typically 16x16x16 or 64x64x16 tile sizes depending on architecture
 * 
 * ## WGMMA vs WMMA/MMA
 * - WMMA (Warp Matrix Multiply Accumulate): Volta/Turing architecture, single warp (32 threads)
 * - MMA (Matrix Multiply Accumulate): Ampere architecture, single warp operations
 * - WGMMA (Warp-Group Matrix Multiply Accumulate): Hopper architecture, multiple warps (128 threads)
 * 
 * WGMMA advantages:
 * - Larger tile sizes: Can process 64x64x16 or 64x256x16 tiles per warp group
 * - Better memory bandwidth utilization: Coordinated access patterns across warps
 * - Asynchronous execution: Can overlap computation with memory operations
 * 
 * ## Architecture Requirements
 * - Requires NVIDIA Hopper architecture (sm_90+) or later
 * - Uses TMA (Tensor Memory Accelerator) for efficient global-to-shared memory transfers
 * - Requires column-major layout for matrix B (transposed from standard row-major)
 */

namespace WGMMA_Basic_fp16 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

/**
 * @brief Encodes a matrix descriptor field for WGMMA instructions
 * @param x Input value to encode
 * @return Encoded descriptor field (lower 14 bits of x >> 4)
 * 
 * WGMMA instructions require matrix descriptors that encode:
 * - Shared memory address
 * - Matrix dimensions and strides
 * - Data type information
 */
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

/**
 * @brief Creates a shared memory descriptor for WGMMA operations
 * @param ptr Pointer to shared memory matrix data
 * @return 64-bit descriptor encoding matrix location and properties
 * 
 * The descriptor encodes:
 * - Bits 0-13: Shared memory address (aligned to 16 bytes)
 * - Bits 16-29: Leading dimension (16 elements = 32 bytes for FP16)
 * - Bits 32-45: Stride information (1024 bytes for this configuration)
 * - Bit 62: Memory space indicator (1 = shared memory)
 * 
 * This descriptor is used by WGMMA instructions to locate matrix data in shared memory.
 */
__device__ uint64_t make_smem_desc(fp16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; 
    return desc;
  }

/**
 * @brief Synchronizes warp group before WGMMA operations
 * 
 * Issues a fence instruction that ensures all previous memory operations are visible
 * to the Tensor Cores before beginning WGMMA operations. This is required for correct
 * execution ordering when using asynchronous memory loads.
 */
__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Commits a batch of WGMMA operations for execution
 * 
 * Signals that all WGMMA operations in the current batch have been issued.
 * The Tensor Cores will begin processing these operations asynchronously.
 * Must be called after issuing all WGMMA instructions in a batch.
 */
__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Waits for WGMMA operations to complete
 * @tparam N Number of WGMMA batches to wait for (0-7)
 * 
 * Blocks until the specified number of WGMMA batches have completed.
 * This is necessary because WGMMA operations execute asynchronously,
 * allowing computation to overlap with memory operations.
 * 
 * Typical usage pattern:
 * 1. Issue async memory loads
 * 2. Issue WGMMA operations (they start asynchronously)
 * 3. Call warpgroup_commit_batch()
 * 4. Call warpgroup_wait<0>() to wait for completion
 */
template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

/**
 * @brief Creates a Tensor Map for TMA (Tensor Memory Accelerator) operations
 * @tparam BlockMajorSize Major dimension of each tile (height for A, width for B)
 * @tparam BlockMinorSize Minor dimension of each tile (depth K for both A and B)
 * @param tma_map Output tensor map descriptor
 * @param gmem_ptr Pointer to global memory matrix data
 * @param blocks_height Number of blocks in major dimension
 * @param blocks_width Number of blocks in minor dimension
 * 
 * TMA (Tensor Memory Accelerator) is a hardware unit in Hopper GPUs that efficiently transfers
 * 2D tiles from global memory to shared memory. This function creates a descriptor that tells
 * TMA how to interpret and transfer the matrix data.
 * 
 * The tensor map encodes:
 * - Global memory layout: Tiled structure with specified strides
 * - Shared memory layout: Contiguous tiles ready for WGMMA
 * - Transfer size: Each tile is BlockMajorSize x BlockMinorSize elements
 * - Swizzling: 128B swizzling for optimal memory bank access
 * 
 * This enables cp_async_bulk_tensor_2d_global_to_shared() to efficiently load data.
 */
template <int BlockMajorSize, int BlockMinorSize>
void create_tensor_map(CUtensorMap *tma_map, fp16* gmem_ptr, int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(fp16), sizeof(fp16) * BlockMinorSize*blocks_width, 0, 0, 0};
    uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, gmem_address, gmem_prob_shape,
        gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(result == CUDA_SUCCESS);
}

/**
 * @brief Allocates device memory and creates a tensor map on the device
 * @tparam BlockMajorSize Major dimension of each tile
 * @tparam BlockMinorSize Minor dimension of each tile
 * @param src Pointer to source matrix in global memory
 * @param blocks_height Number of blocks in major dimension
 * @param blocks_width Number of blocks in minor dimension
 * @return Pointer to device-allocated tensor map descriptor
 * 
 * Creates a tensor map on the host, then copies it to device memory.
 * The device-side tensor map is used by TMA operations in kernel code.
 */
template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(fp16* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

// Global tensor map descriptors for matrices A and B (cached for reuse)
CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m=0, _prev_n=0, _prev_k=0;
const fp16* _prev_A_ptr = nullptr;
const fp16* _prev_B_ptr = nullptr;

/**
 * @brief Ensures tensor maps are created and cached for the given matrix dimensions
 * @tparam BM Block tile size in M dimension
 * @tparam BN Block tile size in N dimension
 * @tparam BK Block tile size in K dimension
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in matrix B and C
 * @param K Number of columns in A and rows in B
 * @param A Pointer to matrix A in global memory
 * @param B Pointer to matrix B in global memory
 * 
 * Creates tensor maps if they don't exist or if matrix dimensions/pointers have changed.
 * Caches the maps to avoid expensive reallocation on repeated calls with same dimensions.
 */
template <int BM, int BN, int BK>
void ensure_tensor_maps(int M, int N, int K, fp16* A, fp16* B) {
    if (!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k ||
        A != _prev_A_ptr || B != _prev_B_ptr) {
        if (d_tma_map_A) cudaFree(d_tma_map_A);
        if (d_tma_map_B) cudaFree(d_tma_map_B);
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
        _prev_A_ptr = A;
        _prev_B_ptr = B;
    }
}

/**
 * @brief Performs WGMMA operation: 64x64x16 matrix multiply-accumulate
 * @tparam ScaleD Scale factor for accumulator (D matrix)
 * @tparam ScaleA Scale factor for matrix A
 * @tparam ScaleB Scale factor for matrix B
 * @tparam TransA Whether to transpose A (0 = no transpose)
 * @tparam TransB Whether to transpose B (0 = no transpose)
 * @param d Accumulator register array [4][8] storing FP32 results
 * @param sA Pointer to matrix A tile in shared memory (64x16 FP16)
 * @param sB Pointer to matrix B tile in shared memory (64x16 FP16, column-major)
 * 
 * This function issues a WGMMA instruction that computes:
 *   d = ScaleD * d + ScaleA * ScaleB * (A @ B)
 * 
 * Operation details:
 * - Input: A (64x16 FP16), B (64x16 FP16)
 * - Output: Accumulates into d (64x64 FP32)
 * - Precision: FP16 inputs, FP32 accumulation (for numerical stability)
 * - Execution: Asynchronous - starts immediately, completes later
 * 
 * Register layout:
 * - d[4][8] represents a 64x64 result matrix
 * - Each row of d corresponds to 16 rows of the output (4 rows * 16 = 64)
 * - Each column of d corresponds to 8 columns of the output (8 columns * 8 elements = 64)
 * 
 * The operation is performed by a warp group (128 threads = 4 warps) cooperatively.
 * After calling this, you must:
 * 1. Call warpgroup_commit_batch() after issuing all WGMMA operations
 * 2. Call warpgroup_wait<0>() before using the results
 */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @brief Main GEMM kernel using WGMMA Tensor Cores
 * @tparam BM Block tile size in M dimension (typically 64)
 * @tparam BN Block tile size in N dimension (typically 64)
 * @tparam BK Block tile size in K dimension (typically 64)
 * @tparam WGMMA_M WGMMA tile size in M dimension (64 for this kernel)
 * @tparam WGMMA_N WGMMA tile size in N dimension (64 for this kernel)
 * @tparam WGMMA_K WGMMA tile size in K dimension (16, fixed by hardware)
 * @tparam NUM_THREADS Number of threads per block (128 = 1 warp group)
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param C Output matrix C (M×N, FP16, device memory, column-major)
 * @param tensorMapA TMA descriptor for matrix A
 * @param tensorMapB TMA descriptor for matrix B
 * 
 * This kernel implements the basic WGMMA GEMM pattern:
 * 
 * 1. **Block-level tiling**: Each thread block computes a BM×BN tile of C
 * 2. **K-dimension iteration**: Iterates over K/BK tiles to accumulate partial results
 * 3. **Asynchronous data loading**: Uses TMA to load tiles from global to shared memory
 * 4. **WGMMA computation**: Uses Tensor Cores to multiply-accumulate tiles
 * 5. **Result storage**: Writes accumulated results back to global memory
 * 
 * Memory hierarchy:
 * - Global memory: Input matrices A, B; output matrix C
 * - Shared memory: Tile buffers sA (BM×BK) and sB (BK×BN)
 * - Registers: Accumulator array d storing FP32 intermediate results
 * 
 * Execution flow per K-tile:
 * 1. Thread 0 initiates TMA async loads for A and B tiles
 * 2. All threads wait for loads to complete (barrier synchronization)
 * 3. Warp group synchronizes before WGMMA operations
 * 4. Issue 4 WGMMA operations (BK=64 split into 4×16 chunks)
 * 5. Commit batch and wait for completion
 * 6. Repeat for next K-tile
 * 
 * After processing all K-tiles, accumulated results in registers are written to C.
 */
template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) matmulKernel2(int M, int N, int K, fp16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB) {
    // Shared memory tile buffers (128-byte aligned for optimal access)
    __shared__ alignas(128) fp16 sA[BM*BK];  // A tile: BM rows × BK columns
    __shared__ alignas(128) fp16 sB[BK*BN];  // B tile: BK rows × BN columns
    
    // Register accumulator array: WGMMA_N/16 rows × 8 columns
    // Each element is FP32 for numerical precision during accumulation
    float d[WGMMA_N/16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    // Calculate number of K-dimension tiles to process
    const int num_blocks_k = K / BK;
    
    // Determine which tile this block computes
    int num_block_n = blockIdx.x % (N / BN);  // Block position in N dimension
    int num_block_m = blockIdx.x / (N / BN);  // Block position in M dimension
    
    // Barriers for coordinating asynchronous memory loads
    __shared__ barrier barA;
    __shared__ barrier barB;

    // Initialize barriers (one thread per block)
    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // Main loop: iterate over K-dimension tiles
    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        
        // Phase 1: Asynchronous data loading using TMA
        // Thread 0 initiates TMA bulk transfers from global to shared memory
        if (threadIdx.x == 0) {
            // Load A tile: BM×BK elements from global memory
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            
            // Load B tile: BK×BN elements from global memory
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            // Other threads participate in barrier but don't initiate transfers
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        
        // Wait for async loads to complete before using shared memory
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();
    
        // Phase 2: WGMMA computation using Tensor Cores
        // Synchronize warp group before issuing WGMMA operations
        warpgroup_arrive();
        
        // Issue 4 WGMMA operations to process BK=64 in chunks of 16
        // Each WGMMA computes: d += A[64×16] @ B[64×16]
        wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
        
        // Commit batch and wait for Tensor Core operations to complete
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    
    // Phase 3: Store accumulated results to global memory
    // Each thread writes its portion of the accumulator array to C
    {
        int tid = threadIdx.x;
        int lane = tid % 32;      // Lane ID within warp (0-31)
        int warp = tid / 32;      // Warp ID within block
        uint32_t row = warp*16 + lane / 4;  // Row index this thread handles
        
        // Calculate base pointer for this block's output tile
        fp16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

        // Iterate over WGMMA tiles within this block
        for (int m_it = 0; m_it < BM/WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < BN/WGMMA_N; ++n_it) {
                // Each accumulator element d[w] stores 8 FP32 values
                // representing a 16×8 sub-tile of the result
                for (int w = 0; w < WGMMA_N/16; ++w) {
                    int col = 16*w + 2*(tid % 4);  // Column index this thread handles
                    int row_base = (m_it * WGMMA_M + row);
                    int col_base = (n_it * WGMMA_N + col);

                    // Helper lambda to convert FP32 accumulator to FP16 and store
                    // Matrix C is stored in column-major layout
                    auto store = [&](int r_offset, int c_offset, float value) {
                        int g_row = row_base + r_offset;
                        int g_col = col_base + c_offset;
                        block_C[g_col * M + g_row] = __float2half(value);
                    };

                    // Store 8 accumulator values in a 2×4 pattern
                    // Layout: d[w][0-7] maps to positions:
                    //   [0,0] [0,1]    [8,0] [8,1]
                    //   [0,8] [0,9]    [8,8] [8,9]
                    store(0, 0, d[w][0]);
                    store(0, 1, d[w][1]);
                    store(8, 0, d[w][2]);
                    store(8, 1, d[w][3]);

                    store(0, 8, d[w][4]);
                    store(0, 9, d[w][5]);
                    store(8, 8, d[w][6]);
                    store(8, 9, d[w][7]);
                }
            }
        }
    }
}


/**
 * @brief Host function to launch the basic WGMMA GEMM kernel
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, FP16, device memory, column-major)
 * @param B Input matrix B (K×N, FP16, device memory, column-major)
 * @param C Output matrix C (M×N, FP16, device memory, column-major)
 * 
 * This function sets up and launches the WGMMA kernel with optimized tile sizes:
 * - Block tiles: 64×64×64 (BM×BN×BK)
 * - WGMMA tiles: 64×64×16 (M×N×K)
 * - Threads per block: 128 (1 warp group = 4 warps)
 * 
 * The kernel computes C = A @ B using Hopper architecture Tensor Cores.
 * 
 * Requirements:
 * - M, N, K must be multiples of 64
 * - Matrices A and B must be in column-major layout
 * - Requires Hopper GPU (sm_90+) with TMA support
 */
void runKernel_fp16(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
    constexpr int BM = 64;  // Block tile height
    constexpr int BN = 64;  // Block tile width
    constexpr int BK = 64;  // Block tile depth
    constexpr int NUM_THREADS = 128;  // One warp group

    // Ensure TMA tensor maps are created and cached
    ensure_tensor_maps<BM, BN, BK>(M, N, K, A, B);
    
    // Launch kernel with appropriate grid dimensions
    matmulKernel2<
    /*BM*/ BM,
    /*BN*/ BN,
    /*BK*/ BK,
    /*WGMMA_M*/ 64,
    /*WGMMA_N*/ 64,
    /*WGMMA_K*/ 16,
    /*NUM_THREADS*/ NUM_THREADS>
    <<<(M/BM) * (N/BN), NUM_THREADS>>>(M, N, K, C, d_tma_map_A, d_tma_map_B);
}

} 

using WGMMA_Basic_fp16::runKernel_fp16;
