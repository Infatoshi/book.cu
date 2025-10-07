
/*
Heap-based TopK kernel
Uses a min-heap of size K to track top K elements
More efficient than naive selection: O(N log K) vs O(NK)
*/

    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__device__ void heapify_down(ValueIndex* heap, int k, int size) {
    int smallest = k;
    int left = 2 * k + 1;
    int right = 2 * k + 2;
    
    if (left < size && heap[left] < heap[smallest]) smallest = left;
    if (right < size && heap[right] < heap[smallest]) smallest = right;
    
    if (smallest != k) {
        ValueIndex temp = heap[k];
        heap[k] = heap[smallest];
        heap[smallest] = temp;
        heapify_down(heap, smallest, size);
    }
}

__global__ void topk_kernel_1(
    float* __restrict__ input,
    int* __restrict__ indices,
    float* __restrict__ values,
    int N, int K
) {
    int tid = threadIdx.x;
    
    extern __shared__ ValueIndex heap[];
    
    
    if (tid < K) {
        heap[tid].value = input[tid];
        heap[tid].index = tid;
    }
    __syncthreads();
    
    
    if (tid == 0) {
        for (int i = K/2 - 1; i >= 0; i--) {
            heapify_down(heap, i, K);
        }
    }
    __syncthreads();
    
    
    if (tid == 0) {
        for (int i = K; i < N; i++) {
            if (input[i] > heap[0].value) {
                heap[0].value = input[i];
                heap[0].index = i;
                heapify_down(heap, 0, K);
            }
        }
    }
    __syncthreads();
    
    
    if (tid < K) {
        values[tid] = heap[tid].value;
        indices[tid] = heap[tid].index;
    }
}

void run_kernel_1(float* input, int* indices, float* values, int N, int K) {
    dim3 block_size(256);
    dim3 grid_size(1);
    int smem_size = K * sizeof(ValueIndex);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    topk_kernel_1<<<grid_size, block_size, smem_size>>>(input, indices, values, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
