
/**
 * @file wgmma_larger_tiles_fp16.cuh
 * @brief WGMMA GEMM implementation with larger block tiles (128×128×64)
 * 
 * This implementation extends the basic WGMMA kernel by using larger block tiles
 * to improve memory bandwidth utilization and reduce synchronization overhead.
 * 
 * Key Differences from Basic WGMMA:
 * - **Larger block tiles**: 128×128×64 (vs 64×64×64)
 * - **Multiple WGMMA tile sizes**: Supports 32, 64, 128, 192, and 256 in N dimension
 * - **Better throughput**: Larger tiles reduce global memory traffic per computation
 * - **Multiple warp groups**: Can use multiple warp groups per block for better parallelism
 * 
 * WGMMA Tile Size Selection:
 * - WGMMA supports different output tile sizes: m64n{16,32,64,128,192,256}k16
 * - Larger N dimensions increase register pressure but improve compute efficiency
 * - This kernel uses variable WGMMA_N based on block configuration
 * 
 * Performance Characteristics:
 * - Better memory bandwidth utilization due to larger tiles
 * - Reduced synchronization overhead (fewer iterations over K dimension)
 * - Higher register pressure may limit occupancy
 * - Best for larger matrices where memory bandwidth is the bottleneck
 */

namespace WGMMA_LargerTiles_fp16 {

using fp16 = __half;

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

CUtensorMap *d_tma_map_A = 0;
CUtensorMap *d_tma_map_B = 0;
int _prev_m=0, _prev_n=0, _prev_k=0;
const fp16* _prev_a_ptr = nullptr;
const fp16* _prev_b_ptr = nullptr;

template<int st_rows, int st_cols>
__host__ static inline CUtensorMap* allocate_and_create_tensor_map(fp16* src, int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map<st_rows, st_cols>(&tma_map_host, src, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

/**
 * @brief Performs WGMMA operation: 64x256x16 matrix multiply-accumulate
 * @tparam ScaleD Scale factor for accumulator
 * @tparam ScaleA Scale factor for matrix A
 * @tparam ScaleB Scale factor for matrix B
 * @tparam TransA Whether to transpose A
 * @tparam TransB Whether to transpose B
 * @param d Accumulator register array [16][8] storing FP32 results (64×256 output)
 * @param sA Pointer to matrix A tile in shared memory (64×16 FP16)
 * @param sB Pointer to matrix B tile in shared memory (256×16 FP16, column-major)
 * 
 * This is the largest WGMMA tile size supported, processing 64×256 output per operation.
 * Requires more registers (16×8 = 128 FP32 values) but maximizes Tensor Core utilization.
 */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma256(float d[16][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @brief Performs WGMMA operation: 64x128x16 matrix multiply-accumulate
 * @param d Accumulator register array [8][8] storing FP32 results (64×128 output)
 * @param sA Pointer to matrix A tile (64×16 FP16)
 * @param sB Pointer to matrix B tile (128×16 FP16, column-major)
 * 
 * Medium-sized WGMMA tile, balancing register usage and throughput.
 */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma128(float d[8][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
            "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
            "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
            "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
            "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
            "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
            "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @brief Performs WGMMA operation: 64x192x16 matrix multiply-accumulate
 * @param d Accumulator register array [12][8] storing FP32 results (64×192 output)
 * @param sA Pointer to matrix A tile (64×16 FP16)
 * @param sB Pointer to matrix B tile (192×16 FP16, column-major)
 * 
 * Intermediate tile size between 128 and 256, useful for specific block configurations.
 */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma192(float d[12][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},  "
        " %96,"
        " %97,"
        " %98,    %99,  %100,  %101,  %102;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @brief Performs WGMMA operation: 64x64x16 matrix multiply-accumulate
 * @param d Accumulator register array [4][8] storing FP32 results (64×64 output)
 * @param sA Pointer to matrix A tile (64×16 FP16)
 * @param sB Pointer to matrix B tile (64×16 FP16, column-major)
 * 
 * Same as basic WGMMA kernel - standard 64×64 output tile.
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
 * @brief Performs WGMMA operation: 64x32x16 matrix multiply-accumulate
 * @param d Accumulator register array [2][8] storing FP32 results (64×32 output)
 * @param sA Pointer to matrix A tile (64×16 FP16)
 * @param sB Pointer to matrix B tile (32×16 FP16, column-major)
 * 
 * Smaller tile size, uses fewer registers. Useful when register pressure is high.
 */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma32(float d[2][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},  "
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @brief Performs WGMMA operation: 64x16x16 matrix multiply-accumulate
 * @param d Accumulator register array [1][8] storing FP32 results (64×16 output)
 * @param sA Pointer to matrix A tile (64×16 FP16)
 * @param sB Pointer to matrix B tile (16×16 FP16, column-major)
 * 
 * Smallest WGMMA tile size, minimizes register usage.
 */
template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma16(float d[1][8], fp16* sA, fp16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
        " %8,"
        " %9,"
        " %10, %11, %12, %13, %14;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
            "+f"(d[0][6]), "+f"(d[0][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}

/**
 * @brief Template wrapper to select appropriate WGMMA tile size
 * @tparam WGMMA_N N dimension of WGMMA operation (32, 64, 128, 192, or 256)
 * @param d Accumulator array sized appropriately for WGMMA_N
 * @param sA Pointer to matrix A tile in shared memory
 * @param sB Pointer to matrix B tile in shared memory
 * 
 * This template function dispatches to the appropriate WGMMA implementation
 * based on the WGMMA_N template parameter. Allows compile-time selection of
 * optimal tile size based on block configuration and register availability.
 */
template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ inline void wgmma(float d[WGMMA_N/16][8], fp16* sA, fp16* sB) {
    static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 192 || WGMMA_N == 256);
    if  constexpr (WGMMA_N == 256)
        wgmma256<1, 1, 1, 0, 0>(d, sA, sB);
    if  constexpr (WGMMA_N == 192)
        wgmma192<1, 1, 1, 0, 0>(d, sA, sB);
    if  constexpr (WGMMA_N == 128)
        wgmma128<1, 1, 1, 0, 0>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64<1, 1, 1, 0, 0>(d, sA, sB);
    if constexpr (WGMMA_N == 32)
        wgmma32<1, 1, 1, 0, 0>(d, sA, sB);
}

template <int BM, int BN, int BK>
struct SMem {
    alignas(128) fp16 A[BM*BK];
    alignas(128) fp16 B[BK*BN];
};

/**
 * @brief GEMM kernel with larger tiles and multiple warp groups
 * @tparam BM Block tile size in M dimension (128)
 * @tparam BN Block tile size in N dimension (128)
 * @tparam BK Block tile size in K dimension (64)
 * @tparam NUM_THREADS Number of threads per block (128 = 1 warp group)
 * @tparam DBG Whether to collect debug timing information
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param C Output matrix C (M×N, FP16, device memory, column-major)
 * @param tensorMapA TMA descriptor for matrix A
 * @param tensorMapB TMA descriptor for matrix B
 * @param DB Debug buffer for timing data (if DBG=true)
 * 
 * This kernel uses larger block tiles (128×128×64) compared to the basic kernel.
 * The larger tiles improve memory bandwidth utilization by:
 * - Reducing global memory accesses per computation
 * - Better L2 cache utilization
 * - Fewer synchronization barriers
 * 
 * Key features:
 * - Uses dynamic shared memory for tile buffers
 * - Supports optional debug timing collection
 * - Variable WGMMA_N based on block configuration
 */
template<int BM, int BN, int BK, int NUM_THREADS, bool DBG>
__global__ void __launch_bounds__(NUM_THREADS) matmulKernel3(int M, int N, int K, fp16* C, const CUtensorMap* tensorMapA, const CUtensorMap* tensorMapB, int *DB) {
    constexpr int WGMMA_M = 64, WGMMA_K = 16, WGMMA_N=BN;
    constexpr int B_WG_M = BM / (NUM_THREADS / 128);
    extern __shared__ SMem<BM, BN, BK> s;
    fp16 *sA = s.A;
    fp16 *sB = s.B;
    
    
    __shared__ barrier barA, barB;
    float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];
    static_assert(sizeof(d) * NUM_THREADS == BM * BN * sizeof(float));
    memset(d, 0, sizeof(d));

    const int num_blocks_k = K / BK;
    int num_block_n = blockIdx.x % (N / BN);
    int num_block_m = blockIdx.x / (N / BN);

    if (threadIdx.x == 0) {
        init(&barA, blockDim.x);
        init(&barB, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    int wg_idx = threadIdx.x / 128;

    barrier::arrival_token tokenA, tokenB;
    int sumLoad = 0, cntLoad = 0;
    int sumCompute = 0, cntCompute = 0;
    int sumStore = 0, cntStore = 0;
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter) {
        clock_t start = clock();
        
        if (threadIdx.x == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
            tokenA = cuda::device::barrier_arrive_tx(barA, 1, BK*BM*sizeof(fp16));
            cde::cp_async_bulk_tensor_2d_global_to_shared(&sB[0], tensorMapB, block_k_iter*BK, num_block_n*BN, barB);
            tokenB = cuda::device::barrier_arrive_tx(barB, 1, BK*BN*sizeof(fp16));
        } else {
            tokenA = barA.arrive();
            tokenB = barB.arrive();
        }
        barA.wait(std::move(tokenA));
        barB.wait(std::move(tokenB));
        __syncthreads();
        if constexpr (DBG) {
            sumLoad += clock() - start;
            cntLoad++;
            start = clock();
        }
    
        
        warpgroup_arrive();
        
        for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
            fp16 *wgmma_sA = sA + BK*(m_it + wg_idx*B_WG_M/WGMMA_M)*WGMMA_M;
            
            for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
                wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &sB[k_it*WGMMA_K]);
            }
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        
        if constexpr (DBG) {
            sumCompute += clock() - start;
            cntCompute++;
        }
    }

    
    {
        clock_t start = clock();

        uint32_t tid = threadIdx.x % 128;
        uint32_t lane = tid & 31;
        uint32_t warp = tid / 32;
        uint32_t row = warp*16 + lane / 4;

        fp16 *block_C = C + num_block_n*BN*M + num_block_m*BM;

        
        for (uint32_t m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
            int yo = m_it*WGMMA_M + wg_idx*B_WG_M;
            
            for (uint32_t w = 0; w < WGMMA_N/16; ++w) {
                int col = 16*w + 2*(tid % 4);
                

                block_C[IDX(row, col)] = __float2half(d[m_it][w][0]);
                block_C[IDX(row, col+1)] = __float2half(d[m_it][w][1]);
                block_C[IDX(row+8, col)] = __float2half(d[m_it][w][2]);
                block_C[IDX(row+8, col+1)] = __float2half(d[m_it][w][3]);
                block_C[IDX(row, col+8)] = __float2half(d[m_it][w][4]);
                block_C[IDX(row, col+9)] = __float2half(d[m_it][w][5]);
                block_C[IDX(row+8, col+8)] = __float2half(d[m_it][w][6]);
                block_C[IDX(row+8, col+9)] = __float2half(d[m_it][w][7]);
                
                
            }
        }
        if constexpr (DBG) {
            sumStore += clock() - start;
            cntStore++;
            if (threadIdx.x == 63) {
                int i = blockIdx.x*6;
                DB[i] = sumLoad; DB[i + 1] = cntLoad;
                DB[i + 2] = sumCompute; DB[i + 3] = cntCompute;
                DB[i + 4] = sumStore; DB[i + 5] = cntStore;
            }
        }
    }
}


void runKernel_fp16(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int NUM_THREADS = 128;

    if (!d_tma_map_A || M != _prev_m || N != _prev_n || K != _prev_k ||
        A != _prev_a_ptr || B != _prev_b_ptr) {
        if (d_tma_map_A) cudaFree(d_tma_map_A);
        if (d_tma_map_B) cudaFree(d_tma_map_B);
        d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(A, M / BM, K / BK);
        d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(B, N / BN, K / BK);
        _prev_m = M;
        _prev_n = N;
        _prev_k = K;
        _prev_a_ptr = A;
        _prev_b_ptr = B;
    }
    auto* kernel = DB ? matmulKernel3<BM, BN, BK, NUM_THREADS, true>
            : matmulKernel3<BM, BN, BK, NUM_THREADS, false>;
    size_t sMemSize = sizeof(SMem<BM, BN, BK>);
    cudaCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    kernel<<<(M/BM) * (N/BN), NUM_THREADS, sMemSize>>>(M, N, K, C, d_tma_map_A, d_tma_map_B, DB);
}

} 

using WGMMA_LargerTiles_fp16::runKernel_fp16;
