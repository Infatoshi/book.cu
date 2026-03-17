
/**
 * @file layout_utils.cuh
 * @brief Utility functions for matrix layout conversion
 * 
 * WGMMA operations require column-major layout for optimal performance, but
 * many applications use row-major layout. This file provides conversion utilities.
 * 
 * Layout Conversion:
 * - row_to_col: Converts row-major matrix to column-major
 * - col_to_row: Converts column-major matrix to row-major
 * - transpose: General matrix transpose operation
 * 
 * These kernels use simple 2D thread mapping for efficient conversion.
 */

namespace wgmma_layout {

inline dim3 make_grid(int rows, int cols, dim3 block) {
  return dim3((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
}

__global__ void row_to_col_kernel(const fp16 *src, fp16 *dst, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    dst[col * rows + row] = src[row * cols + col];
  }
}

__global__ void col_to_row_kernel(const fp16 *src, fp16 *dst, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    dst[row * cols + col] = src[col * rows + row];
  }
}

__global__ void transpose_kernel(const fp16 *src, fp16 *dst, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols) {
    dst[col * rows + row] = src[row * cols + col];
  }
}

/**
 * @brief Converts matrix from row-major to column-major layout
 * @param src Source matrix in row-major layout
 * @param dst Destination matrix in column-major layout
 * @param rows Number of rows
 * @param cols Number of columns
 * 
 * Launches a 2D grid kernel where each thread copies one element:
 * dst[col * rows + row] = src[row * cols + col]
 */
inline void row_to_col(const fp16 *src, fp16 *dst, int rows, int cols) {
  const dim3 block(32, 8);
  row_to_col_kernel<<<make_grid(rows, cols, block), block>>>(src, dst, rows, cols);
}

/**
 * @brief Converts matrix from column-major to row-major layout
 * @param src Source matrix in column-major layout
 * @param dst Destination matrix in row-major layout
 * @param rows Number of rows
 * @param cols Number of columns
 * 
 * Inverse operation of row_to_col:
 * dst[row * cols + col] = src[col * rows + row]
 */
inline void col_to_row(const fp16 *src, fp16 *dst, int rows, int cols) {
  const dim3 block(32, 8);
  col_to_row_kernel<<<make_grid(rows, cols, block), block>>>(src, dst, rows, cols);
}

inline void transpose(const fp16 *src, fp16 *dst, int rows, int cols) {
  const dim3 block(32, 8);
  transpose_kernel<<<make_grid(rows, cols, block), block>>>(src, dst, rows, cols);
}

inline void transpose_inplace(fp16 *src_dst, int rows, int cols, fp16 *workspace) {
  
  transpose_kernel<<<make_grid(rows, cols, dim3(32,8)), dim3(32,8)>>>(src_dst, workspace, rows, cols);
  cudaMemcpy(src_dst, workspace, sizeof(fp16) * rows * cols, cudaMemcpyDeviceToDevice);
}

}  
