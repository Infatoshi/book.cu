

/*
This kernel implements a naive softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, and thus this kernel will be the slowest
since we aren't exploiting parallelism capabilities of GPUs that much.
We are only parallelizing over the rows.
*/
__global__ void softmax_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        
        float m = -1 * INFINITY;
        
        float L = 0.0f;

        
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            m = max(m, matd[i]);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            L += expf(matd[i] - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}

/*
Runs the naive softmax kernel: `id = 0`
*/
void run_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    softmax_kernel_0<<<grid_size, block_size>>>(matd, resd, M, N);
}