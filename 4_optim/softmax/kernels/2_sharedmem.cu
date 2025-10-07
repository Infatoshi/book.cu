

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


/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
In this, we handle each row with a block where the threads within one block work together
to process one row (max and norm factor). Each thread will process some elements
and will contains its local max and local norm in shared memory. Then, we perform reduction
operations to compute the final max and norm factor. Also, we compute maxes and norms
in one pass itself.
*/
__global__ void softmax_kernel_2(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    
    
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    
    
    smem[tid] = local_max;
    __syncthreads();

    
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        
        __syncthreads();
    }

    
    
    float row_max = smem[0];
    __syncthreads();

    
    
    
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/*
Runs the online softmax kernel: `id = 2`
*/
void run_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(M);

    softmax_kernel_2<<<grid_size, block_size>>>(matd, resd, M, N);
}