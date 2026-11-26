// Softmax Vectorized Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0 license)
// https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Ceiling division macro
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

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
Instead of accessing shared memory and having sync barrier overhead, we will use warp-level primitives (then
block-level) for performing max and sum reductions. The benefit is: it is faster than shared
memory access and also does not need syncing since each warp (group of 32 threads) execute
an instuction parallely on GPU so no chance of race conditions.

We will also use vectorized loads and stores.
*/
__global__ void softmax_kernel_4(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    
    int n_float4s = N / 4;
    int tail = N % 4;
    float4* input_row_vec = reinterpret_cast<float4*>(input_row);
    float4* output_row_vec = reinterpret_cast<float4*>(output_row);
    float maxval = -INFINITY;

    
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];

        maxval = fmaxf(maxval, elem.x);
        maxval = fmaxf(maxval, elem.y);
        maxval = fmaxf(maxval, elem.z);
        maxval = fmaxf(maxval, elem.w);
        if (maxval > local_max) {
            local_norm *= __expf(local_max - maxval);
            local_max = maxval;
        }
        local_norm += __expf(elem.x - maxval);
        local_norm += __expf(elem.y - maxval);
        local_norm += __expf(elem.z - maxval);
        local_norm += __expf(elem.w - maxval);
    }

    
    if (tail && tid < tail) {
        float val = input_row[n_float4s * 4 + tid];
        if (val > local_max) {
            local_norm *= __expf(local_max - val);
            local_max = val;
        }
        local_norm += __expf(val - local_max);
    }
    __syncthreads();

    
    
    
    
    
    blockReduceMax<float>(local_max, smem, -INFINITY);
    __syncthreads();

    
    float row_max = smem[0];
    __syncthreads();

    
    
    
    
    float val = local_norm * expf(local_max - row_max);
    blockReduceSum<float>(val, smem);
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    
    
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        elem.x = __expf(elem.x - row_max) / row_norm;
        elem.y = __expf(elem.y - row_max) / row_norm;
        elem.z = __expf(elem.z - row_max) / row_norm;
        elem.w = __expf(elem.w - row_max) / row_norm;

        output_row_vec[i] = elem;
    }
    
    if (tail && tid < tail)
    {
        float val = input_row[n_float4s * 4 + tid];
        output_row[n_float4s * 4 + tid] = __expf(val - row_max) / row_norm;
    }
}

/*
Runs the online softmax kernel: `id = 4`
*/
void run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(M);

    int warp_size = 32;
    size_t smem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    softmax_kernel_4<<<grid_size, block_size, smem_size>>>(matd, resd, M, N);
}