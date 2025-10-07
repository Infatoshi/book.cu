
using namespace nvcuda;

    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}


/*
 * FA2.5: Flash Attention with WMMA Tensor Cores
 * 
 * Uses WMMA for:
 * 1. Q @ K^T computation (FP16 → FP32)
 * 2. S @ V computation (FP16 × FP16 → FP32)
 * 
 * Tile sizes: Br=16, Bc=16 (matches WMMA 16x16x16)
 * Works on: Volta, Turing, Ampere, Ada, Hopper (SM_70+)
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


torch::Tensor fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    
    auto Q_fp16 = Q.to(torch::kFloat16);
    auto K_fp16 = K.to(torch::kFloat16);
    auto V_fp16 = V.to(torch::kFloat16);

    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    
    const int Br = 16;
    const int Bc = 16;

    assert(d % 16 == 0 && "Head dimension must be multiple of 16 for WMMA");

    float softmax_scale = 1.0f / sqrtf(static_cast<float>(d));

    
    auto O = torch::zeros({B, nh, N, d}, torch::dtype(torch::kFloat32).device(Q.device()));

    
    size_t smem_size = (
        Br * d * sizeof(half) +      
        Bc * d * sizeof(half) +      
        Bc * d * sizeof(half) +      
        Br * Bc * sizeof(float) +    
        Br * Bc * sizeof(half) +     
        Br * d * sizeof(float) +     
        3 * Br * sizeof(float)       
    );

    dim3 grid_size(B, nh);
    dim3 block_size(Br * Bc);

    
    flash_attn_2_5_kernel<16, 16><<<grid_size, block_size, smem_size>>>(
        reinterpret_cast<half*>(Q_fp16.data_ptr()),
        reinterpret_cast<half*>(K_fp16.data_ptr()),
        reinterpret_cast<half*>(V_fp16.data_ptr()),
        N, d,
        softmax_scale,
        O.data_ptr<float>()
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return O.to(Q.dtype());
}
