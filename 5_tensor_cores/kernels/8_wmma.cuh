
typedef __half fp16;

using namespace nvcuda;

template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16,
          int WMMA_TILE_M = 4, int WMMA_TILE_N = 2, int WARP_TILE_M = 2,
          int WARP_TILE_N = 4>
__global__ void __launch_bounds__(WMMA_TILE_M * WMMA_TILE_N * 32)
    gemm_wmma_tiled(int M, int N, int K, const fp16 *__restrict__ A,
                    const fp16 *__restrict__ B, fp16 *__restrict__ C) {
  constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;  
  constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;  
  constexpr int BK = WMMA_K;                              

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;
  if (blockRow >= M || blockCol >= N) return;

  __shared__ fp16 sA[BM][BK];
  __shared__ fp16 sB[BK][BN];
  __shared__ fp16 sC[BM][BN];

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int warp_m = warp_id / 2;  
  const int warp_n = warp_id % 2;  

  const int load_smem_a_m = tid / 2;       
  const int load_smem_a_k = (tid % 2) * 8;  
  const int load_smem_b_k = tid / 16;       
  const int load_smem_b_n = (tid % 16) * 8; 

  const int numKTiles = CEIL_DIV(K, BK);
  const fp16 zero = __float2half(0.0f);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp16>
      C_frag[WARP_TILE_M][WARP_TILE_N];

  
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

void runKernel8(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C, int *DB = nullptr) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(256);
  gemm_wmma_tiled<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
