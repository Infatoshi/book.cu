
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


/*
CuBLAS matrix vector multiplication for the baseline scores.
We simply run the Sgemv function that cuBLAS provides.

Note: We use a static handle to avoid the overhead of creating/destroying
the handle on every call (~0.4ms overhead). The handle is created once
on first use and reused for all subsequent calls.
*/
void run_kernel_0(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    
    
    static cublasHandle_t handle = nullptr;
    
    if (handle == nullptr) {
        cublasCreate(&handle);
    }

    
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, matd, N, vecd, 1, &beta, resd, 1);
}
