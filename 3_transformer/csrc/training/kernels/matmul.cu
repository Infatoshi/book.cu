
__global__ void matmul_fwd_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_bwd_A_kernel(const float* grad_C, const float* B, float* grad_A,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += grad_C[row * N + n] * B[col * N + n];
        }
        grad_A[row * K + col] = sum;
    }
}

__global__ void matmul_bwd_B_kernel(const float* A, const float* grad_C, float* grad_B,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_C[m * N + col];
        }
        grad_B[row * N + col] = sum;
    }
}

__global__ void batched_matmul_fwd_kernel(const float* A, const float* B, float* C,
                                         int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            int a_idx = batch * M * K + row * K + k;
            int b_idx = batch * K * N + k * N + col;
            sum += A[a_idx] * B[b_idx];
        }
        int c_idx = batch * M * N + row * N + col;
        C[c_idx] = sum;
    }
}

__global__ void batched_matmul_bwd_A_kernel(const float* grad_C, const float* B, float* grad_A,
                                           int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            int grad_c_idx = batch * M * N + row * N + n;
            int b_idx = batch * K * N + col * N + n;  
            sum += grad_C[grad_c_idx] * B[b_idx];
        }
        int grad_a_idx = batch * M * K + row * K + col;
        grad_A[grad_a_idx] = sum;
    }
}

__global__ void batched_matmul_bwd_B_kernel(const float* A, const float* grad_C, float* grad_B,
                                           int batch_size, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            int a_idx = batch * M * K + m * K + row;  
            int grad_c_idx = batch * M * N + m * N + col;  
            sum += A[a_idx] * grad_C[grad_c_idx];
        }
        int grad_b_idx = batch * K * N + row * N + col;  
        grad_B[grad_b_idx] = sum;
    }
}

void matmul_fwd_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    matmul_fwd_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
}

void matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                    float* grad_A, float* grad_B, int M, int N, int K) {
    dim3 threads_A(16, 16);
    dim3 blocks_A((K + threads_A.x - 1) / threads_A.x, (M + threads_A.y - 1) / threads_A.y);
    matmul_bwd_A_kernel<<<blocks_A, threads_A>>>(grad_C, B, grad_A, M, N, K);

    dim3 threads_B(16, 16);
    dim3 blocks_B((N + threads_B.x - 1) / threads_B.x, (K + threads_B.y - 1) / threads_B.y);
    matmul_bwd_B_kernel<<<blocks_B, threads_B>>>(A, grad_C, grad_B, M, N, K);
}

void batched_matmul_fwd_cuda(const float* A, const float* B, float* C,
                            int batch_size, int M, int N, int K) {
    dim3 threads(8, 8);  
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                batch_size);
    batched_matmul_fwd_kernel<<<blocks, threads>>>(A, B, C, batch_size, M, N, K);
}

void batched_matmul_bwd_cuda(const float* A, const float* B, const float* grad_C,
                            float* grad_A, float* grad_B,
                            int batch_size, int M, int N, int K) {
    dim3 threads_A(8, 8);  
    dim3 blocks_A((K + threads_A.x - 1) / threads_A.x,
                  (M + threads_A.y - 1) / threads_A.y,
                  batch_size);
    batched_matmul_bwd_A_kernel<<<blocks_A, threads_A>>>(grad_C, B, grad_A, batch_size, M, N, K);

    dim3 threads_B(8, 8);  
    dim3 blocks_B((N + threads_B.x - 1) / threads_B.x,
                  (K + threads_B.y - 1) / threads_B.y,
                  batch_size);
    batched_matmul_bwd_B_kernel<<<blocks_B, threads_B>>>(A, grad_C, grad_B, batch_size, M, N, K);
}
