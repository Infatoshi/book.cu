#include "common.h"

/**
 * @file conv2d.cu
 * @brief Naive CUDA implementation of 2D convolution
 * 
 * This file demonstrates 2D sliding window operations, the core operation in CNNs.
 * 2D convolution is fundamental to many operations including:
 * - Convolutional Neural Networks (CNNs): feature extraction from images
 * - Image processing: edge detection, blurring, sharpening
 * - Computer vision: object detection, segmentation
 * 
 * Convolution Operation: out[r,c] = sum(kr=0 to kernel_dim-1, kc=0 to kernel_dim-1) 
 *                         in[r+kr, c+kc] * kernel[kr, kc]
 * 
 * Padding: Valid padding (no padding) - output dimensions reduced by kernel_dim - 1
 * 
 * CPU Implementation: Nested loops for 2D sliding window
 * GPU Implementation: 2D thread grid, each thread computes one output element
 * 
 * Algorithm:
 * - For each output position (r, c):
 *   - Compute 2D dot product of kernel with input window starting at (r, c)
 *   - Window spans kernel_dim × kernel_dim elements
 * 
 * Memory Access Pattern:
 * - CPU: Sequential row-major access to input, nested loops over kernel
 * - GPU: 2D grid ensures coalesced access along columns within each warp
 * 
 * Performance Characteristics:
 * - CPU: O(H*W*K*K) time complexity where H=height, W=width, K=kernel_dim
 * - GPU: Highly parallelizable, each output element independent
 * - Memory bandwidth limited (K*K reads per output element)
 */

/**
 * CPU implementation of 2D convolution
 * 
 * Applies a square 2D convolution kernel to a 2D input image using valid padding.
 * Each output element is computed as the 2D dot product of the kernel with
 * a sliding window of the input image.
 * 
 * Algorithm:
 * - For each output position (r, c):
 *   1. Initialize accumulator sum = 0
 *   2. For kr from 0 to kernel_dim-1:
 *      For kc from 0 to kernel_dim-1:
 *        - sum += in[r+kr][c+kc] * kernel[kr][kc]
 *   3. out[r][c] = sum
 * 
 * Padding: Valid padding (no padding)
 * - Output height = height - kernel_dim + 1
 * - Output width = width - kernel_dim + 1
 * - Ensures kernel always fits within input boundaries
 * 
 * Memory Access Pattern:
 * - Input: Row-major access (cache-friendly, sequential within rows)
 * - Kernel: Sequential access (cache-friendly, small kernel)
 * - Output: Row-major write (cache-friendly, sequential within rows)
 * 
 * Time Complexity: O(H * W * K * K) where H=height, W=width, K=kernel_dim
 * Space Complexity: O(1) excluding input/output matrices
 * 
 * @param in Input image (host memory, row-major order, size height × width)
 * @param out Output feature map (host memory, row-major order, 
 *            size output_h × output_w = (height-kernel_dim+1) × (width-kernel_dim+1))
 *            Stores convolution result: out[r,c] = sum(kr,kc) in[r+kr][c+kc] * kernel[kr][kc]
 * @param kernel Convolution kernel (host memory, row-major order, size kernel_dim × kernel_dim)
 *               Contains filter weights to be applied
 * @param height Height of input image (must be >= kernel_dim)
 * @param width Width of input image (must be >= kernel_dim)
 * @param kernel_dim Dimension of square convolution kernel (must be >= 1)
 */
void conv2d_cpu(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) {
    // Output dimensions are reduced by kernel_dim - 1 (valid padding)
    // Formula: output_h = height - kernel_dim + 1, output_w = width - kernel_dim + 1
    // This ensures the kernel always fits within input boundaries
    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;
    
    // Iterate through each output position
    // Each output element corresponds to one position of the 2D sliding window
    for (int r = 0; r < output_h; ++r) {
        for (int c = 0; c < output_w; ++c) {
            float sum = 0.0f;  // Accumulator for 2D dot product
            
            // Compute 2D convolution: dot product of kernel with input window
            // Window starts at position (r, c) and spans kernel_dim × kernel_dim elements
            // Kernel is applied element-wise to the input window
            for (int kr = 0; kr < kernel_dim; ++kr) {
                for (int kc = 0; kc < kernel_dim; ++kc) {
                    // Calculate input position: (r+kr, c+kc)
                    int input_row = r + kr;
                    int input_col = c + kc;
                    // Access in[input_row][input_col]: row-major order (cache-friendly)
                    // Access kernel[kr][kc]: row-major order (cache-friendly, small kernel)
                    // Multiply and accumulate: sum += in[input_row][input_col] * kernel[kr][kc]
                    sum += in[input_row * width + input_col] * kernel[kr * kernel_dim + kc];
                }
            }
            // Store computed convolution result
            out[r * output_w + c] = sum;
        }
    }
}

/**
 * CUDA kernel for 2D convolution
 * 
 * Parallel GPU implementation using 2D thread indexing to map threads to output positions.
 * Each thread computes one output element independently.
 * 
 * Algorithm (same as CPU):
 * - Each thread computes one output element
 * - Thread at position (output_row, output_col) computes:
 *   out[output_row][output_col] = sum(kr,kc) in[output_row+kr][output_col+kc] * kernel[kr][kc]
 * 
 * Thread Indexing (2D Grid):
 * - Grid dimensions: (blocksPerGrid.x, blocksPerGrid.y) covering output feature map
 * - Block dimensions: (blockDim.x, blockDim.y)
 * - Output column index: output_col = blockIdx.x * blockDim.x + threadIdx.x
 * - Output row index: output_row = blockIdx.y * blockDim.y + threadIdx.y
 * 
 * Memory Access Pattern:
 * - Input: Each thread reads kernel_dim × kernel_dim elements (coalesced along columns)
 * - Kernel: All threads read same kernel (cached in constant memory or texture)
 * - Output: Each thread writes one element (coalesced)
 * 
 * Performance:
 * - Highly parallelizable: each output element independent
 * - Memory bandwidth limited: kernel_dim × kernel_dim reads per output element
 * - No shared memory: direct global memory access (acceptable for small kernels)
 * 
 * Optimization Opportunities:
 * - Shared memory: Cache input tiles to reduce global memory accesses
 * - Constant memory: Store kernel in constant memory for fast access
 * - Texture memory: Use texture cache for input image
 * - Tiling: Process output in tiles to improve data reuse
 * 
 * @param in Input image (device memory, row-major order, size height × width)
 * @param out Output feature map (device memory, row-major order, 
 *            size output_h × output_w = (height-kernel_dim+1) × (width-kernel_dim+1))
 *            Stores convolution result: out[r,c] = sum(kr,kc) in[r+kr][c+kc] * kernel[kr][kc]
 * @param kernel Convolution kernel (device memory, row-major order, size kernel_dim × kernel_dim)
 *               Contains filter weights to be applied
 * @param height Height of input image (must be >= kernel_dim)
 * @param width Width of input image (must be >= kernel_dim)
 * @param kernel_dim Dimension of square convolution kernel (must be >= 1)
 */
__global__ void conv2d_kernel(const float* in, float* out, const float* kernel, int height, int width, int kernel_dim) {
    // Calculate 2D output coordinates from thread indices
    // X-dimension (columns): maps to blockIdx.x and threadIdx.x
    // Formula: output_col = block_index_x * threads_per_block_x + thread_index_x
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in output feature map
    
    // Y-dimension (rows): maps to blockIdx.y and threadIdx.y
    // Formula: output_row = block_index_y * threads_per_block_y + thread_index_y
    int output_row = blockIdx.y * blockDim.y + threadIdx.y;   // Row index in output feature map

    // Output dimensions (valid padding)
    // Formula: output_h = height - kernel_dim + 1, output_w = width - kernel_dim + 1
    int output_h = height - kernel_dim + 1;
    int output_w = width - kernel_dim + 1;

    // Bounds check: critical for correctness when dimensions are not divisible by block size
    // Checks both row and column bounds independently
    // Without this check, threads beyond output boundaries would access invalid memory
    if (output_row < output_h && output_col < output_w) {
        float sum = 0.0f;  // Accumulator for 2D dot product
        
        // Compute 2D convolution: dot product of kernel with input window
        // Window starts at position (output_row, output_col) and spans kernel_dim × kernel_dim elements
        // Kernel is applied element-wise to the input window
        for (int kernel_row = 0; kernel_row < kernel_dim; ++kernel_row) {
            for (int kernel_col = 0; kernel_col < kernel_dim; ++kernel_col) {
                // Calculate input position: (output_row + kernel_row, output_col + kernel_col)
                int input_row = output_row + kernel_row;
                int input_col = output_col + kernel_col;
                // Access in[input_row][input_col]: row-major order
                // Memory access pattern: coalesced along columns within each warp
                // Access kernel[kernel_row][kernel_col]: row-major order (small kernel, may be cached)
                // Multiply and accumulate: sum += in[input_row][input_col] * kernel[kernel_row][kernel_col]
                sum += in[input_row * width + input_col] * kernel[kernel_row * kernel_dim + kernel_col];
            }
        }
        // Store computed convolution result
        // Memory access: coalesced (threads write consecutive elements)
        out[output_row * output_w + output_col] = sum;
    }
}

int main() {
    // Image and kernel dimensions
    const int height = 256, width = 256;  
    const int kernel_dim = 3;             
    const int output_h = height - kernel_dim + 1;
    const int output_w = width - kernel_dim + 1;

    const int input_size = height * width;
    const int kernel_size = kernel_dim * kernel_dim;
    const int output_size = output_h * output_w;

    std::cout << "2D Convolution: Input[" << height << "x" << width << "] * Kernel["
              << kernel_dim << "x" << kernel_dim << "] -> Output[" << output_h << "x" << output_w << "]" << std::endl;

    // Allocate host memory for input, kernel, and output
    float *h_in, *h_kernel, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, input_size);
    allocate_host(&h_kernel, kernel_size);
    allocate_host(&h_out_cpu, output_size);
    allocate_host(&h_out_gpu, output_size);

    // Initialize input image with test data (pattern based on position)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_in[i * width + j] = static_cast<float>((i + j) % 10);
        }
    }

    // Initialize kernel as edge detection filter (Laplacian-like)
    float kernel_data[9] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    for (int i = 0; i < kernel_size; ++i) {
        h_kernel[i] = kernel_data[i];
    }

    // Run CPU version with timing
    {
        Timer cpu_timer("CPU 2D Convolution");
        conv2d_cpu(h_in, h_out_cpu, h_kernel, height, width, kernel_dim);
    }

    // Allocate device memory
    float *d_in, *d_kernel, *d_out;
    allocate_device(&d_in, input_size);
    allocate_device(&d_kernel, kernel_size);
    allocate_device(&d_out, output_size);

    // Copy data from host to device
    copy_to_device(d_in, h_in, input_size);
    copy_to_device(d_kernel, h_kernel, kernel_size);

    // Configure 2D kernel launch parameters
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid(
        (output_w + threadsPerBlock.x - 1) / threadsPerBlock.x,  // Blocks in x-dimension
        (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y   // Blocks in y-dimension
    );

    std::cout << "GPU: Launching " << blocksPerGrid.x << "x" << blocksPerGrid.y
              << " blocks with " << threadsPerBlock.x << "x" << threadsPerBlock.y
              << " threads per block" << std::endl;

    // Run GPU version with timing
    {
        Timer gpu_timer("GPU 2D Convolution");
        conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_kernel, height, width, kernel_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result back from device to host
    copy_to_host(h_out_gpu, d_out, output_size);

    // Verify GPU results against CPU results
    if (verify_results(h_out_gpu, h_out_cpu, output_size)) {
        std::cout << "✓ 2D convolution results match!" << std::endl;

        // Display kernel and sample results
        std::cout << "\n3x3 Edge Detection Kernel:" << std::endl;
        for (int i = 0; i < kernel_dim; ++i) {
            for (int j = 0; j < kernel_dim; ++j) {
                std::cout << kernel_data[i * kernel_dim + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nInput image (top-left 5x5):" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                std::cout << h_in[i * width + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nOutput feature map (top-left 5x5):" << std::endl;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                std::cout << h_out_gpu[i * output_w + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nNote: Edge detection kernel highlights edges where values change." << std::endl;
        std::cout << "Center pixel (2,2) of output should be high due to edge detection." << std::endl;
    }

    // Clean up memory
    free_host(h_in);
    free_host(h_kernel);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_kernel);
    free_device(d_out);

    return 0;
}
