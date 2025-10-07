
namespace WGMMA_Basic_fp16 {

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(fp16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
    desc |= 1llu << 62; 
    return desc;
  }


__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

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

template <int BlockMajorSize, int BlockMinorSize>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(fp16* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m=0, _prev_n=0, _prev_k=0;
const fp16* _prev_A_ptr = nullptr;
const fp16* _prev_B_ptr = nullptr;

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

template<int BM, int BN, int BK, int WGMMA_M, int WGMMA_N, int WGMMA_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) matmulKernel2(int M, int N, int K, fp16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB) {
    __shared__ alignas(128) fp16 sA[BM*BK];
    __shared__ alignas(128) fp16 sB[BK*BN];
    float d[WGMMA_N/16][8];
    static_assert(sizeof(d) * 128 == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);
    
    __shared__ barrier barA;
    __shared__ barrier barB;

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token tokenA, tokenB;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, sizeof(sB));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();
    
        
        warpgroup_arrive();
        wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
        wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }

    
    {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int warp = tid / 32;
        uint32_t row = warp*16 + lane / 4;
        fp16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

        for (int m_it = 0; m_it < BM/WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < BN/WGMMA_N; ++n_it) {
                for (int w = 0; w < WGMMA_N/16; ++w) {
                    int col = 16*w + 2*(tid % 4);
                    int row_base = (m_it * WGMMA_M + row);
                    int col_base = (n_it * WGMMA_N + col);

                    auto store = [&](int r_offset, int c_offset, float value) {
                        int g_row = row_base + r_offset;
                        int g_col = col_base + c_offset;
                        block_C[g_col * M + g_row] = __float2half(value);
                    };

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


void runKernel_fp16(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    ensure_tensor_maps<BM, BN, BK>(M, N, K, A, B);
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
