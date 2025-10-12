
/**
 * Flash Attention 2.5 Implementation with WMMA Tensor Cores
 * High-performance attention mechanism using NVIDIA's Tensor Cores
 * Optimized for memory efficiency and computational throughput
 */

using namespace nvcuda;  // For WMMA API

/**
 * CUDA error checking macro
 * @param ans CUDA function call to check
 */
#define CUDA_CHECK(ans) {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }

/**
 * CUDA error assertion function
 * @param code CUDA error code
 * @param file Source file name
 * @param line Line number
 */
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}


/**
 * Flash Attention 2.5 Kernel with WMMA Tensor Cores
 * 
 * Key optimizations:
 * 1. Uses WMMA for Q @ K^T computation (FP16 → FP32)
 * 2. Uses WMMA for S @ V computation (FP16 × FP16 → FP32)
 * 3. Implements online softmax for numerical stability
 * 4. Memory-efficient tiling strategy
 * 
 * Tile sizes: Br=16, Bc=16 (matches WMMA 16x16x16)
 * Compatible with: Volta, Turing, Ampere, Ada, Hopper (SM_70+)
 * 
 * @param Br Block size for rows (must be 16 for WMMA)
 * @param Bc Block size for columns (must be 16 for WMMA)
 * @param Q Query matrix (device memory, FP16)
 * @param K Key matrix (device memory, FP16)
 * @param V Value matrix (device memory, FP16)
 * @param N Sequence length
 * @param d Head dimension (must be multiple of 16)
 * @param scale Scaling factor (1/sqrt(d))
 * @param O Output matrix (device memory, FP32)
 */
template <const int Br, const int Bc>
__global__ void flash_attn_2_5_kernel(
    half* Q, half* K, half* V, 
    int N, int d,
    float scale, 
    float* O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;  
    int by = blockIdx.y;  

    
    int qkv_off = (bx * gridDim.y * N * d) + (by * N * d);

    
    extern __shared__ char smem_raw[];
    half* Qi = reinterpret_cast<half*>(smem_raw);
    half* Kj = Qi + Br * d;
    half* Vj = Kj + Bc * d;
    float* Sij_fp32 = reinterpret_cast<float*>(Vj + Bc * d);
    half* Sij_fp16 = reinterpret_cast<half*>(Sij_fp32 + Br * Bc);
    float* Oi = reinterpret_cast<float*>(Sij_fp16 + Br * Bc);
    float* mi = Oi + Br * d;
    float* mi_new = mi + Br;
    float* li = mi_new + Br;

    int Tc = CEIL_DIV(N, Bc);
    int Tr = CEIL_DIV(N, Br);

    
    int s_row = tx / Bc;
    int s_col = tx % Bc;

    
    for (int i = 0; i < Tr; i++) {
        int row_offset = i * Br;
        
        
        for (int idx = tx; idx < Br * d; idx += blockDim.x) {
            int r = idx / d;
            int c = idx % d;
            if (row_offset + r < N) {
                Qi[r * d + c] = __ldg(&Q[qkv_off + (row_offset + r) * d + c]);
            } else {
                Qi[r * d + c] = __float2half(0.0f);
            }
        }

        
        for (int idx = tx; idx < Br * d; idx += blockDim.x) {
            Oi[idx] = 0.0f;
        }
        if (tx < Br) {
            mi[tx] = -INFINITY;
            mi_new[tx] = -INFINITY;
            li[tx] = 0.0f;
        }
        __syncthreads();

        
        for (int j = 0; j < Tc; j++) {
            int col_offset = j * Bc;
            
            
            for (int idx = tx; idx < Bc * d; idx += blockDim.x) {
                int r = idx / d;
                int c = idx % d;
                if (col_offset + r < N) {
                    Kj[r * d + c] = __ldg(&K[qkv_off + (col_offset + r) * d + c]);
                    Vj[r * d + c] = __ldg(&V[qkv_off + (col_offset + r) * d + c]);
                } else {
                    Kj[r * d + c] = __float2half(0.0f);
                    Vj[r * d + c] = __float2half(0.0f);
                }
            }
            __syncthreads();

            
            
            
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                
                for (int k = 0; k < d; k += 16) {
                    wmma::load_matrix_sync(a_frag, Qi + k, d);
                    wmma::load_matrix_sync(b_frag, Kj + k, d);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                
                wmma::store_matrix_sync(Sij_fp32, c_frag, Bc, wmma::mem_row_major);
            }
            __syncthreads();

            
            for (int idx = tx; idx < Br * Bc; idx += blockDim.x) {
                Sij_fp32[idx] *= scale;
            }
            __syncthreads();

            
            
            
            if (s_col == 0 && s_row < Br) {
                
                mi[s_row] = mi_new[s_row];
                
                
                float row_max = -INFINITY;
                for (int c = 0; c < Bc; c++) {
                    row_max = fmaxf(row_max, Sij_fp32[s_row * Bc + c]);
                }
                
                
                float new_max = fmaxf(mi[s_row], row_max);
                mi_new[s_row] = new_max;
                
                
                float row_sum = 0.0f;
                for (int c = 0; c < Bc; c++) {
                    float exp_val = expf(Sij_fp32[s_row * Bc + c] - new_max);
                    Sij_fp32[s_row * Bc + c] = exp_val;
                    row_sum += exp_val;
                }
                
                
                float correction = (mi[s_row] == -INFINITY) ? 0.0f : expf(mi[s_row] - new_max);
                li[s_row] = correction * li[s_row] + row_sum;
            }
            __syncthreads();

            
            for (int idx = tx; idx < Br * Bc; idx += blockDim.x) {
                Sij_fp16[idx] = __float2half(Sij_fp32[idx]);
            }
            __syncthreads();

            
            
            
            
            __shared__ float temp_pv[Br * 64];
            
            
            for (int tile_col = 0; tile_col < d / 16; tile_col++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                
                for (int k = 0; k < Bc; k += 16) {
                    wmma::load_matrix_sync(a_frag, Sij_fp16 + k, Bc);
                    wmma::load_matrix_sync(b_frag, Vj + k * d + tile_col * 16, d);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                
                wmma::store_matrix_sync(temp_pv + tile_col * 16, c_frag, d, wmma::mem_row_major);
            }
            __syncthreads();
            
            
            for (int idx = tx; idx < Br * d; idx += blockDim.x) {
                int r = idx / d;
                int c = idx % d;
                
                float correction = (mi[r] == -INFINITY || mi_new[r] == -INFINITY) 
                    ? 0.0f 
                    : expf(mi[r] - mi_new[r]);
                
                Oi[r * d + c] = correction * Oi[r * d + c] + temp_pv[r * d + c];
            }
            __syncthreads();
        }

        
        for (int col = s_col; col < d; col += Bc) {
            int global_row = row_offset + s_row;
            if (s_row < Br && global_row < N) {
                O[qkv_off + global_row * d + col] = Oi[s_row * d + col] / li[s_row];
            }
        }
        __syncthreads();
    }
}


/**
 * PyTorch wrapper for Flash Attention 2.5 forward pass
 * 
 * @param Q Query tensor [B, nh, N, d] (FP32)
 * @param K Key tensor [B, nh, N, d] (FP32)
 * @param V Value tensor [B, nh, N, d] (FP32)
 * @return Output tensor [B, nh, N, d] (FP32)
 */
torch::Tensor fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Convert inputs to FP16 for Tensor Core computation
    auto Q_fp16 = Q.to(torch::kFloat16);
    auto K_fp16 = K.to(torch::kFloat16);
    auto V_fp16 = V.to(torch::kFloat16);

    // Extract tensor dimensions
    int B = Q.size(0);   // Batch size
    int nh = Q.size(1);  // Number of heads
    int N = Q.size(2);   // Sequence length
    int d = Q.size(3);   // Head dimension

    // Tile dimensions (must match WMMA fragment size)
    const int Br = 16;  // Row tile size
    const int Bc = 16;  // Column tile size

    // Validate head dimension for WMMA compatibility
    assert(d % 16 == 0 && "Head dimension must be multiple of 16 for WMMA");

    // Compute softmax scaling factor
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(d));

    // Allocate output tensor (FP32 for accumulation)
    auto O = torch::zeros({B, nh, N, d}, torch::dtype(torch::kFloat32).device(Q.device()));

    // Calculate shared memory requirements
    size_t smem_size = (
        Br * d * sizeof(half) +      // Qi tile
        Bc * d * sizeof(half) +      // Kj tile
        Bc * d * sizeof(half) +      // Vj tile
        Br * Bc * sizeof(float) +    // Sij_fp32
        Br * Bc * sizeof(half) +     // Sij_fp16
        Br * d * sizeof(float) +     // Oi accumulator
        3 * Br * sizeof(float)        // mi, mi_new, li arrays
    );

    // Configure kernel launch parameters
    dim3 grid_size(B, nh);      // One block per batch and head
    dim3 block_size(Br * Bc);   // 256 threads per block (16*16)

    // Launch Flash Attention kernel
    flash_attn_2_5_kernel<16, 16><<<grid_size, block_size, smem_size>>>(
        reinterpret_cast<half*>(Q_fp16.data_ptr()),
        reinterpret_cast<half*>(K_fp16.data_ptr()),
        reinterpret_cast<half*>(V_fp16.data_ptr()),
        N, d,
        softmax_scale,
        O.data_ptr<float>()
    );

    // Check for errors and synchronize
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Convert output back to input dtype
    return O.to(Q.dtype());
}
