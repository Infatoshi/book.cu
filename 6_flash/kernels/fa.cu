
/**
 * @file fa.cu
 * @brief This file contains a custom implementation of FlashAttention-2.5 using
 *        WMMA (Wavefront Matrix Multiply-Accumulate) tensor cores.
 *
 * @details
 * This implementation is designed for high performance on NVIDIA GPUs with
 * tensor core support (Volta architecture and later). It leverages several
 * key optimizations to address the memory bandwidth bottleneck of standard
 * attention mechanisms.
 *
 * Core Concepts of FlashAttention:
 * 1. Tiling: Instead of processing the entire Q, K, and V matrices at once,
 *    FlashAttention divides them into smaller blocks (tiles). This allows the
 *    computation to be performed in smaller, faster on-chip shared memory,
 *    reducing the need for slow HBM (High Bandwidth Memory) access.
 *
 * 2. Online Softmax: Standard softmax requires the entire attention score matrix
 *    to be computed before normalization. This is memory-intensive. FlashAttention
 *    uses an "online" or "one-pass" softmax algorithm. As each tile of the
 *    attention scores is computed, it is immediately scaled and normalized using
 *    running statistics (max value and sum of exponentials). This avoids
 *    materializing the full, large attention matrix in HBM.
 *
 * 3. WMMA Tensor Cores: The matrix multiplications (Q @ K^T and S @ V) are the
 *    most computationally expensive parts of attention. This implementation uses
 *    the nvcuda::wmma API to accelerate these operations on NVIDIA's tensor cores,
 *    which are specialized for high-throughput matrix math (FP16/FP32 mixed-precision).
 *
 * This combination of tiling, online softmax, and tensor core acceleration allows
 * FlashAttention to be significantly faster and more memory-efficient than standard
 * attention, especially for long sequences.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cassert>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

using namespace nvcuda;

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
 * @brief The main kernel for FlashAttention-2.5 using WMMA tensor cores.
 *
 * @tparam Br Block size for rows of the attention matrix tile (must be 16 for WMMA).
 * @tparam Bc Block size for columns of the attention matrix tile (must be 16 for WMMA).
 *
 * @param Q Pointer to the Query matrix in device memory (FP16).
 * @param K Pointer to the Key matrix in device memory (FP16).
 * @param V Pointer to the Value matrix in device memory (FP16).
 * @param N The sequence length of the input.
 * @param d The dimension of the attention heads.
 * @param scale The scaling factor for the attention scores (typically 1/sqrt(d)).
 * @param O Pointer to the Output matrix in device memory (FP32 for accumulation).
 *
 * @details
 * This kernel computes the attention output for a single attention head. It is
 * launched with a grid of blocks, where each block is responsible for computing
 * a `Br x N` slice of the final output matrix `O`.
 *
 * The computation is broken down into two main loops:
 * 1. Outer Loop (over `i`): Iterates through the rows of `O` in blocks of size `Br`.
 *    - A tile of Query `Qi` is loaded into shared memory.
 *
 * 2. Inner Loop (over `j`): Iterates through the Key/Value matrices in blocks of size `Bc`.
 *    - Tiles of Key `Kj` and Value `Vj` are loaded into shared memory.
 *    - The attention scores `Sij = Qi @ Kj^T` are computed using WMMA.
 *    - The online softmax calculation is performed:
 *      - The running max `mi` and sum `li` are updated.
 *      - The previous output `Oi` is rescaled based on the new statistics.
 *    - The scaled attention scores `Pij` are multiplied with the Value tile `Vj`
 *      (`Pij @ Vj`) using WMMA, and the result is accumulated into `Oi`.
 *
 * After the inner loop completes, the final `Oi` tile is normalized with the
 * final `li` value and written to global memory.
 */
template <const int Br, const int Bc>
__global__ void flash_attn_2_5_kernel(
    half* Q, half* K, half* V, 
    int N, int d,
    float scale, 
    float* O
) {
    // Thread and block indices for navigating the grid.
    int tx = threadIdx.x; // Thread index within the block
    int bx = blockIdx.x;  // Block index for batch dimension
    int by = blockIdx.y;  // Block index for head dimension

    // Offset to select the correct Q, K, V matrices for the current batch and head.
    int qkv_off = (bx * gridDim.y * N * d) + (by * N * d);

    // Shared memory layout for tiles and intermediate results.
    // This is a single contiguous block of shared memory, partitioned using pointers.
    extern __shared__ char smem_raw[];
    half* Qi = reinterpret_cast<half*>(smem_raw);          // Tile of Query matrix (Br x d)
    half* Kj = Qi + Br * d;                                // Tile of Key matrix (Bc x d)
    half* Vj = Kj + Bc * d;                                // Tile of Value matrix (Bc x d)
    float* Sij_fp32 = reinterpret_cast<float*>(Vj + Bc * d); // Attention scores (Br x Bc), FP32 for precision
    half* Sij_fp16 = reinterpret_cast<half*>(Sij_fp32 + Br * Bc); // Attention scores, converted to FP16 for S@V matmul
    float* Oi = reinterpret_cast<float*>(Sij_fp16 + Br * Bc);  // Accumulated output tile (Br x d)
    float* temp_pv = Oi + Br * d;                              // Temporary storage for Pij @ Vj result (Br x d)
    float* mi = temp_pv + Br * d;                              // Running max for online softmax (Br)
    float* mi_new = mi + Br;                                   // New max for online softmax update (Br)
    float* li = mi_new + Br;                                   // Running sum of exps for online softmax (Br)

    // Calculate the number of tiles needed for the sequence length.
    int Tc = CEIL_DIV(N, Bc); // Number of column tiles
    int Tr = CEIL_DIV(N, Br); // Number of row tiles

    // Thread mapping for row-major access within a tile.
    int s_row = tx / Bc;
    int s_col = tx % Bc;

    // Outer loop: Iterate over the query matrix in row blocks (tiles).
    // Each block processes one `Br x N` portion of the attention map.
    for (int i = 0; i < Tr; i++) {
        int row_offset = i * Br;
        
        // Load a `Br x d` tile of the Query matrix (Qi) from HBM into shared memory.
        // Threads cooperate to load the tile efficiently.
        for (int idx = tx; idx < Br * d; idx += blockDim.x) {
            int r = idx / d;
            int c = idx % d;
            if (row_offset + r < N) {
                Qi[r * d + c] = __ldg(&Q[qkv_off + (row_offset + r) * d + c]);
            } else {
                Qi[r * d + c] = __float2half(0.0f);
            }
        }

        // Initialize the output accumulator (Oi) and online softmax statistics (mi, li) for this row tile.
        for (int idx = tx; idx < Br * d; idx += blockDim.x) {
            Oi[idx] = 0.0f;
        }
        if (tx < Br) {
            mi[tx] = -INFINITY;     // Initialize current max to negative infinity.
            mi_new[tx] = -INFINITY; // Initialize new max to negative infinity.
            li[tx] = 0.0f;          // Initialize sum of exps to zero.
        }
        __syncthreads();

        // Inner loop: Iterate over the key/value matrices in column blocks (tiles).
        for (int j = 0; j < Tc; j++) {
            int col_offset = j * Bc;
            
            // Load `Bc x d` tiles of Key (Kj) and Value (Vj) from HBM into shared memory.
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

            // --- Step 1: Compute Sij = Qi @ Kj^T ---
            // This is the core matrix multiplication for attention scores.
            // It's accelerated using WMMA tensor cores.
            // Each warp computes a 16x16 tile of the Sij matrix.
            {
                // Define WMMA fragments for matrices A (Qi), B (Kj), and C (Sij).
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag; // K must be column major for Q @ K^T
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                // Iterate over the head dimension `d` in chunks of 16 (WMMA size).
                for (int k = 0; k < d; k += 16) {
                    // Load 16x16 sub-matrices of Qi and Kj into fragments.
                    wmma::load_matrix_sync(a_frag, Qi + k, d);
                    wmma::load_matrix_sync(b_frag, Kj + k, d);
                    // Perform matrix multiply-accumulate: c_frag += a_frag * b_frag.
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                // Store the resulting 16x16 Sij tile from accumulator to shared memory.
                wmma::store_matrix_sync(Sij_fp32, c_frag, Bc, wmma::mem_row_major);
            }
            __syncthreads();

            // Apply the attention scaling factor.
            for (int idx = tx; idx < Br * Bc; idx += blockDim.x) {
                Sij_fp32[idx] *= scale;
            }
            __syncthreads();

            // --- Step 2: Online Softmax Calculation ---
            // This step updates the running statistics (max and sum) and rescales
            // the intermediate output `Oi` based on the new `Sij` tile.
            // This is done per-row of Sij, so only a subset of threads do this work.
            if (s_col == 0 && s_row < Br) {
                
                // Persist the max from the previous iteration.
                mi[s_row] = mi_new[s_row];
                
                // Find the maximum value in the current row of Sij.
                float row_max = -INFINITY;
                for (int c = 0; c < Bc; c++) {
                    row_max = fmaxf(row_max, Sij_fp32[s_row * Bc + c]);
                }
                
                // Calculate the new overall maximum.
                float new_max = fmaxf(mi[s_row], row_max);
                mi_new[s_row] = new_max;
                
                // Calculate the sum of exponentials for the current Sij row, scaled by the new max.
                float row_sum = 0.0f;
                for (int c = 0; c < Bc; c++) {
                    float exp_val = expf(Sij_fp32[s_row * Bc + c] - new_max);
                    Sij_fp32[s_row * Bc + c] = exp_val; // Store the scaled exponentiated value back
                    row_sum += exp_val;
                }
                
                // Update the overall sum of exponentials (li).
                // The existing `li` is rescaled by `exp(old_max - new_max)`.
                float correction = (mi[s_row] == -INFINITY) ? 0.0f : expf(mi[s_row] - new_max);
                li[s_row] = correction * li[s_row] + row_sum;
            }
            __syncthreads();

            // Convert the FP32 softmax scores (Pij) to FP16 to prepare for the S@V matmul.
            for (int idx = tx; idx < Br * Bc; idx += blockDim.x) {
                Sij_fp16[idx] = __float2half(Sij_fp32[idx]);
            }
            __syncthreads();

            // --- Step 3: Compute Oi += Pij @ Vj ---
            // This step multiplies the just-computed softmax scores `Pij` with the
            // Value tile `Vj` and accumulates the result into the output tile `Oi`.
            // This is also accelerated using WMMA tensor cores.
            
            // Loop over the columns of Vj, since d can be > 16.
            for (int tile_col = 0; tile_col < d / 16; tile_col++) {
                // Define WMMA fragments.
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                // Iterate over the inner dimension (Bc) in chunks of 16.
                for (int k = 0; k < Bc; k += 16) {
                    // Load 16x16 sub-matrices of Pij (Sij_fp16) and Vj.
                    wmma::load_matrix_sync(a_frag, Sij_fp16 + k, Bc);
                    wmma::load_matrix_sync(b_frag, Vj + k * d + tile_col * 16, d);
                    // Perform matrix multiply-accumulate: c_frag += a_frag * b_frag.
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                // Store the result tile to temporary shared memory.
                wmma::store_matrix_sync(temp_pv + tile_col * 16, c_frag, d, wmma::mem_row_major);
            }
            __syncthreads();
            
            // Rescale the existing Oi and accumulate the new result.
            // This is the second part of the online softmax update.
            for (int idx = tx; idx < Br * d; idx += blockDim.x) {
                int r = idx / d;
                int c = idx % d;
                
                // Calculate the correction factor for the previous Oi value.
                float correction = (mi[r] == -INFINITY || mi_new[r] == -INFINITY) 
                    ? 0.0f 
                    : expf(mi[r] - mi_new[r]);
                
                // Apply correction and add the new value.
                Oi[r * d + c] = correction * Oi[r * d + c] + temp_pv[r * d + c];
            }
            __syncthreads();
        }

        // --- Step 4: Final Normalization and Write to HBM ---
        // After iterating through all key/value tiles, normalize the accumulated
        // output `Oi` with the final `li` and write it to global memory.
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
 * @brief PyTorch wrapper for the forward pass of FlashAttention-2.5.
 *
 * @param Q Query tensor with shape [B, nh, N, d], where B is batch size,
 *          nh is number of heads, N is sequence length, and d is head dimension. (FP32)
 * @param K Key tensor with the same shape as Q. (FP32)
 * @param V Value tensor with the same shape as Q. (FP32)
 * @return torch::Tensor The output tensor with the same shape as the inputs. (FP32)
 *
 * @details
 * This function serves as the C++/CUDA interface for the FlashAttention kernel,
 * allowing it to be called from Python using PyTorch.
 *
 * It performs the following steps:
 * 1. Converts the input FP32 tensors (Q, K, V) to FP16, which is required
 *    for the WMMA tensor core operations in the kernel.
 * 2. Extracts tensor dimensions (batch size, number of heads, etc.).
 * 3. Defines the tile dimensions (Br, Bc), which must match the kernel template.
 * 4. Calculates the total shared memory required by the kernel.
 * 5. Configures the kernel launch parameters (grid size and block size).
 *    - The grid is `B x nh`, meaning one CUDA block handles one attention head.
 * 6. Launches the `flash_attn_2_5_kernel`.
 * 7. Performs error checking and synchronizes the device.
 * 8. Converts the FP32 output tensor back to the original input data type.
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

    // Calculate shared memory requirements for the kernel.
    // This must match the layout defined inside the kernel.
    size_t smem_size = (
        Br * d * sizeof(half) +      // Qi tile
        Bc * d * sizeof(half) +      // Kj tile
        Bc * d * sizeof(half) +      // Vj tile
        Br * Bc * sizeof(float) +    // Sij_fp32
        Br * Bc * sizeof(half) +     // Sij_fp16
        Br * d * sizeof(float) +     // Oi accumulator
        Br * d * sizeof(float) +     // temp_pv (Pij @ Vj result)
        3 * Br * sizeof(float)        // mi, mi_new, li arrays for online softmax
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

    CUDA_CHECK(cudaGetLastError());
    
    // Convert output back to input dtype
    return O.to(Q.dtype());
}
