
/**
 * WMMA (Warp Matrix Multiply Accumulate) GEMM implementation using Tensor Cores
 * Demonstrates high-performance matrix multiplication using NVIDIA's Tensor Cores
 * with FP16 precision and advanced tiling strategies
 */

typedef __half fp16;  // Half precision floating point

using namespace nvcuda;  // For WMMA API

/**
 * High-performance GEMM kernel using WMMA Tensor Cores
 * Implements advanced tiling with multiple levels of hierarchy:
 * - Block-level tiling (BM × BN)
 * - Warp-level tiling (WARP_TILE_M × WARP_TILE_N) 
 * - WMMA-level tiling (WMMA_M × WMMA_N × WMMA_K)
 * 
 * Template parameters:
 * @param WMMA_M WMMA fragment height (typically 16)
 * @param WMMA_N WMMA fragment width (typically 16) 
 * @param WMMA_K WMMA fragment depth (typically 16)
 * @param WMMA_TILE_M Number of WMMA tiles per warp in M dimension
 * @param WMMA_TILE_N Number of WMMA tiles per warp in N dimension
 * @param WARP_TILE_M Number of warps per block in M dimension
 * @param WARP_TILE_N Number of warps per block in N dimension
 * 
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, FP16, device memory)
 * @param B Input matrix B (K×N, FP16, device memory)
 * @param C Output matrix C (M×N, FP16, device memory)
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16,
          int WMMA_TILE_M = 4, int WMMA_TILE_N = 2, int WARP_TILE_M = 2,
          int WARP_TILE_N = 4>
__global__ void __launch_bounds__(WMMA_TILE_M * WMMA_TILE_N * 32)
    gemm_wmma_tiled(int M, int N, int K, const fp16 *__restrict__ A,
                    const fp16 *__restrict__ B, fp16 *__restrict__ C) {
  // Block-level tile dimensions
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  // Block height: 16*4*2 = 128
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  // Block width: 16*2*4 = 128  
  constexpr int BK = WMMA_K;                              // Block depth: 16

  // Calculate block's position in the global matrix
  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;
  if (blockRow >= M || blockCol >= N) return;  // Early exit if block is out of bounds

  // Shared memory allocation for block-level tiling
  __shared__ fp16 sA[BM][BK];  // Shared memory for A tile
  __shared__ fp16 sB[BK][BN];  // Shared memory for B tile
  __shared__ fp16 sC[BM][BN];  // Shared memory for C tile

  // Thread and warp indexing
  const int tid = threadIdx.x;           // Thread ID within block
  const int warp_id = tid / 32;          // Warp ID within block
  const int lane_id = tid % 32;          // Lane ID within warp
  const int warp_m = warp_id / 2;        // Warp position in M dimension
  const int warp_n = warp_id % 2;        // Warp position in N dimension

  // Memory loading indices for coalesced access
  const int load_smem_a_m = tid / 2;       // Row index for A loading
  const int load_smem_a_k = (tid % 2) * 8; // Column index for A loading (8-element vector)
  const int load_smem_b_k = tid / 16;      // Row index for B loading
  const int load_smem_b_n = (tid % 16) * 8; // Column index for B loading (8-element vector)

  // Calculate number of K-dimension tiles to process
  const int numKTiles = CEIL_DIV(K, BK);
  const fp16 zero = __float2half(0.0f);  // Zero value in FP16

  // WMMA accumulator fragments for each warp tile
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp16>
      C_frag[WARP_TILE_M][WARP_TILE_N];

  // Initialize accumulator fragments to zero
  for (int i = 0; i < WARP_TILE_M; ++i) {
    for (int j = 0; j < WARP_TILE_N; ++j) {
      wmma::fill_fragment(C_frag[i][j], zero);
    }
  }

  for (int tile_k = 0; tile_k < numKTiles; ++tile_k) {
    const int global_a_m = blockRow + load_smem_a_m;
    const int global_a_k = tile_k * BK + load_smem_a_k;
    fp16 *smem_a_ptr = &sA[load_smem_a_m][load_smem_a_k];
    const fp16 *gmem_a_ptr = A + global_a_m * K + global_a_k;
    const bool a_in_bounds = (global_a_m < M);
    const bool a_vec_valid = a_in_bounds && (global_a_k + 8) <= K &&
                             (((reinterpret_cast<uintptr_t>(gmem_a_ptr)) & 0xF) == 0) &&
                             (((reinterpret_cast<uintptr_t>(smem_a_ptr)) & 0xF) == 0);

    if (a_vec_valid) {
      *reinterpret_cast<int4 *>(smem_a_ptr) =
          *reinterpret_cast<const int4 *>(gmem_a_ptr);
    } else {
      
      for (int i = 0; i < 8 && (load_smem_a_k + i) < BK; ++i) {
        int k_idx = global_a_k + i;
        smem_a_ptr[i] = (a_in_bounds && k_idx < K) ? gmem_a_ptr[i] : zero;
      }
    }

    const int global_b_k = tile_k * BK + load_smem_b_k;
    const int global_b_n = blockCol + load_smem_b_n;
    fp16 *smem_b_ptr = &sB[load_smem_b_k][load_smem_b_n];
    const fp16 *gmem_b_ptr = B + global_b_k * N + global_b_n;
    const bool b_in_bounds = (global_b_k < K);
    const bool b_vec_valid = b_in_bounds && (global_b_n + 8) <= N &&
                             (((reinterpret_cast<uintptr_t>(gmem_b_ptr)) & 0xF) == 0) &&
                             (((reinterpret_cast<uintptr_t>(smem_b_ptr)) & 0xF) == 0);

    if (b_vec_valid) {
      *reinterpret_cast<int4 *>(smem_b_ptr) =
          *reinterpret_cast<const int4 *>(gmem_b_ptr);
    } else {
      
      for (int i = 0; i < 8 && (load_smem_b_n + i) < BN; ++i) {
        int n_idx = global_b_n + i;
        smem_b_ptr[i] = (b_in_bounds && n_idx < N) ? gmem_b_ptr[i] : zero;
      }
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16, wmma::row_major>
        A_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16, wmma::row_major>
        B_frag[WARP_TILE_N];

    
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      wmma::load_matrix_sync(A_frag[i], &sA[warp_smem_a_m][0], BK);
    }

    
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::load_matrix_sync(B_frag[j], &sB[0][warp_smem_b_n], BN);
    }

    
    for (int i = 0; i < WARP_TILE_M; ++i) {
      
      for (int j = 0; j < WARP_TILE_N; ++j) {
        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }

    __syncthreads();
  }

  
  for (int i = 0; i < WARP_TILE_M; ++i) {
    
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_row = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
      int store_col = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
      wmma::store_matrix_sync(&sC[store_row][store_col], C_frag[i][j], BN,
                              wmma::mem_row_major);
    }
  }

  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += blockDim.x) {
    int row = idx / BN;
    int col = idx % BN;
    int globalRow = blockRow + row;
    int globalCol = blockCol + col;
    if (globalRow < M && globalCol < N) {
      C[globalRow * N + globalCol] = sC[row][col];
    }
  }
}

/**
 * Launcher function for WMMA Tensor Core GEMM kernel
 * Configures grid and block dimensions for optimal performance
 * 
 * @param M Number of rows in matrices A and C
 * @param N Number of columns in matrices B and C
 * @param K Number of columns in A and rows in B
 * @param A Input matrix A (M×K, FP16, device memory)
 * @param B Input matrix B (K×N, FP16, device memory)
 * @param C Output matrix C (M×N, FP16, device memory)
 * @param DB Debug buffer (unused, for compatibility)
 */
void runKernel8(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  // Block tile dimensions (must match kernel template parameters)
  constexpr int BM = 128;  // Block height
  constexpr int BN = 128;  // Block width
  
  // Configure grid and block dimensions
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));  // One block per tile
  dim3 blockDim(256);  // 8 warps per block (8 * 32 = 256 threads)
  
  // Launch WMMA Tensor Core kernel
  gemm_wmma_tiled<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
