
/*
Warp-parallel TopK kernel
Uses warp-level primitives to find top K elements in parallel
Each iteration finds the max across all threads using warp reduction
*/

    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__device__ __forceinline__ ValueIndex warp_reduce_max_with_idx(ValueIndex val) {
    
    for (int offset = 16; offset > 0; offset >>= 1) {
        ValueIndex other;
        other.value = __shfl_down_sync(0xffffffff, val.value, offset);
        other.index = __shfl_down_sync(0xffffffff, val.index, offset);
        if (other.value > val.value) {
            val = other;
        }
    }
    return val;
}

__device__ __forceinline__ ValueIndex block_reduce_max_with_idx(ValueIndex val, ValueIndex* smem) {
    int tid = threadIdx.x;
    int warp_size = 32;
    int warp_id = tid / warp_size;
    int lane = tid % warp_size;
    
    
    val = warp_reduce_max_with_idx(val);
    
    
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    
    
    if (warp_id == 0) {
        ValueIndex warp_max = (lane < blockDim.x / warp_size) ? smem[lane] : (ValueIndex){-INFINITY, -1};
        warp_max = warp_reduce_max_with_idx(warp_max);
        if (lane == 0) {
            smem[0] = warp_max;
        }
    }
    __syncthreads();
    
    return smem[0];
}
__global__ void topk_kernel_2(
    float* __restrict__ input,
    int* __restrict__ indices,
    float* __restrict__ values,
    int N, int K
) {
    int tid = threadIdx.x;
    
    extern __shared__ char shared_mem[];
    bool* selected = (bool*)shared_mem;
    ValueIndex* reduce_smem = (ValueIndex*)(shared_mem + N * sizeof(bool));
    
    
    for (int i = tid; i < N; i += blockDim.x) {
        selected[i] = false;
    }
    __syncthreads();
    
    
    for (int k = 0; k < K; k++) {
        ValueIndex local;
        local.value = -INFINITY;
        local.index = -1;
        
        
        for (int i = tid; i < N; i += blockDim.x) {
            if (!selected[i] && input[i] > local.value) {
                local.value = input[i];
                local.index = i;
            }
        }
        
        
        ValueIndex global_max = block_reduce_max_with_idx(local, reduce_smem);
        
        
        if (tid == 0) {
            selected[global_max.index] = true;
            values[k] = global_max.value;
            indices[k] = global_max.index;
        }
        __syncthreads();
    }
}

void run_kernel_2(float* input, int* indices, float* values, int N, int K) {
    dim3 block_size(256);
    dim3 grid_size(1);
    int num_warps = (block_size.x + 31) / 32;
    int smem_size = N * sizeof(bool) + num_warps * sizeof(ValueIndex);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    topk_kernel_2<<<grid_size, block_size, smem_size>>>(input, indices, values, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
