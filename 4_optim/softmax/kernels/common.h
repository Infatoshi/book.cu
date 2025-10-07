


static __device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

static __device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<typename T>
static __device__ __forceinline__ void blockReduceSum(T val, T* smem) {
    int tid = threadIdx.x;
    int warp_size = 32;
    val = warpReduceSum(val);
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    if (tid < CEIL_DIV(blockDim.x, warp_size)) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }
    if (tid / warp_size == 0) {
        val = warpReduceSum(val);
    }
    if (tid == 0) smem[0] = val;
    __syncthreads();
}

template<typename T>
static __device__ __forceinline__ void blockReduceMax(T val, T* smem, T identity) {
    int tid = threadIdx.x;
    int warp_size = 32;
    val = warpReduceMax(val);
    if (tid % warp_size == 0) smem[tid / warp_size] = val;
    __syncthreads();
    if (tid < CEIL_DIV(blockDim.x, warp_size)) {
        val = smem[tid];
    } else {
        val = identity;
    }
    if (tid / warp_size == 0) {
        val = warpReduceMax(val);
    }
    if (tid == 0) smem[0] = val;
    __syncthreads();
}

