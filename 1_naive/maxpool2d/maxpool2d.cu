#include <iomanip>
#include "common.h"

/**
 * @file maxpool2d.cu
 * @brief Naive CUDA implementation of 2D max pooling
 * 
 * This file demonstrates spatial reduction operations, common in CNNs for downsampling.
 * Max pooling is fundamental to many operations including:
 * - Convolutional Neural Networks (CNNs): reduces spatial dimensions
 * - Feature extraction: preserves strongest activations while reducing size
 * - Translation invariance: makes network less sensitive to small translations
 * 
 * Max Pooling Operation: out[r,c] = max(in[r*pool_dim+pr, c*pool_dim+pc]) 
 *                        for pr in [0, pool_dim), pc in [0, pool_dim)
 * 
 * Downsampling: Output dimensions are reduced by pooling factor
 * - Output height = height / pool_dim
 * - Output width = width / pool_dim
 * 
 * CPU Implementation: Nested loops for 2D sliding window with max reduction
 * GPU Implementation: 2D thread grid, each thread computes one output element
 * 
 * Algorithm:
 * - For each output position (r, c):
 *   - Find maximum value in pool_dim × pool_dim window starting at (r*pool_dim, c*pool_dim)
 *   - Store maximum in output
 * 
 * Memory Access Pattern:
 * - CPU: Sequential row-major access to input, nested loops over pooling window
 * - GPU: 2D grid ensures coalesced access along columns within each warp
 * 
 * Performance Characteristics:
 * - CPU: O(H*W*P*P) time complexity where H=height, W=width, P=pool_dim
 * - GPU: Highly parallelizable, each output element independent
 * - Memory bandwidth limited (P*P reads per output element)
 */

/**
 * CPU implementation of 2D max pooling
 * 
 * Applies max pooling operation to reduce spatial dimensions of feature maps.
 * Each output element is the maximum value within a pool_dim × pool_dim window.
 * 
 * Algorithm:
 * - For each output position (r, c):
 *   1. Initialize max_val = very small value
 *   2. For pr from 0 to pool_dim-1:
 *      For pc from 0 to pool_dim-1:
 *        - val = in[r*pool_dim+pr][c*pool_dim+pc]
 *        - max_val = max(max_val, val)
 *   3. out[r][c] = max_val
 * 
 * Downsampling:
 * - Output height = height / pool_dim
 * - Output width = width / pool_dim
 * - Reduces spatial dimensions by pooling factor
 * 
 * Memory Access Pattern:
 * - Input: Row-major access (cache-friendly, sequential within rows)
 * - Output: Row-major write (cache-friendly, sequential within rows)
 * 
 * Time Complexity: O(H * W * P * P) where H=height, W=width, P=pool_dim
 * Space Complexity: O(1) excluding input/output matrices
 * 
 * @param in Input feature map (host memory, row-major order, size height × width)
 * @param out Output feature map (host memory, row-major order, 
 *            size output_h × output_w = (height/pool_dim) × (width/pool_dim))
 *            Stores max pooling result: out[r,c] = max(in[r*pool_dim+pr][c*pool_dim+pc])
 * @param height Height of input feature map (must be divisible by pool_dim)
 * @param width Width of input feature map (must be divisible by pool_dim)
 * @param pool_dim Dimension of square pooling window (must be >= 1)
 */
void maxpool2d_cpu(const float* in, float* out, int height, int width, int pool_dim) {
    // Output dimensions are reduced by pooling factor
    // Formula: output_h = height / pool_dim, output_w = width / pool_dim
    // Downsampling reduces spatial dimensions while preserving strongest activations
    int output_h = height / pool_dim;
    int output_w = width / pool_dim;
    
    // Iterate through each output position
    // Each output element corresponds to one pooling window
    for (int r = 0; r < output_h; ++r) {
        for (int c = 0; c < output_w; ++c) {
            float max_val = -1e20f;  // Initialize to very small value
            
            // Find maximum value in the pooling window
            // Window starts at position (r*pool_dim, c*pool_dim) and spans pool_dim × pool_dim elements
            for (int pr = 0; pr < pool_dim; ++pr) {
                for (int pc = 0; pc < pool_dim; ++pc) {
                    // Calculate input position: (r*pool_dim+pr, c*pool_dim+pc)
                    int input_row = r * pool_dim + pr;
                    int input_col = c * pool_dim + pc;
                    // Access input value
                    float val = in[input_row * width + input_col];
                    // Update maximum: max_val = max(max_val, val)
                    if (val > max_val) max_val = val;
                }
            }
            // Store maximum value in output
            out[r * output_w + c] = max_val;
        }
    }
}

/**
 * CUDA kernel for 2D max pooling
 * 
 * Parallel GPU implementation using 2D thread indexing to map threads to output positions.
 * Each thread computes one output element independently.
 * 
 * Algorithm (same as CPU):
 * - Each thread computes one output element
 * - Thread at position (output_row, output_col) computes:
 *   out[output_row][output_col] = max(in[output_row*pool_dim+pr][output_col*pool_dim+pc])
 *                                  for pr,pc in [0, pool_dim)
 * 
 * Thread Indexing (2D Grid):
 * - Grid dimensions: (blocksPerGrid.x, blocksPerGrid.y) covering output feature map
 * - Block dimensions: (blockDim.x, blockDim.y)
 * - Output column index: output_col = blockIdx.x * blockDim.x + threadIdx.x
 * - Output row index: output_row = blockIdx.y * blockDim.y + threadIdx.y
 * 
 * Memory Access Pattern:
 * - Input: Each thread reads pool_dim × pool_dim elements (coalesced along columns)
 * - Output: Each thread writes one element (coalesced)
 * 
 * Performance:
 * - Highly parallelizable: each output element independent
 * - Memory bandwidth limited: pool_dim × pool_dim reads per output element
 * - No shared memory: direct global memory access (acceptable for small pooling windows)
 * 
 * Optimization Opportunities:
 * - Shared memory: Cache input tiles to reduce global memory accesses
 * - Warp-level reductions: Use warp shuffles for efficient max computation
 * - Tiling: Process output in tiles to improve data reuse
 * 
 * @param in Input feature map (device memory, row-major order, size height × width)
 * @param out Output feature map (device memory, row-major order, 
 *            size output_h × output_w = (height/pool_dim) × (width/pool_dim))
 *            Stores max pooling result: out[r,c] = max(in[r*pool_dim+pr][c*pool_dim+pc])
 * @param height Height of input feature map (must be divisible by pool_dim)
 * @param width Width of input feature map (must be divisible by pool_dim)
 * @param pool_dim Dimension of square pooling window (must be >= 1)
 */
__global__ void maxpool2d_kernel(const float* in, float* out, int height, int width, int pool_dim) {
    // Calculate 2D output coordinates from thread indices
    // X-dimension (columns): maps to blockIdx.x and threadIdx.x
    // Formula: output_col = block_index_x * threads_per_block_x + thread_index_x
    int output_col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index in output feature map
    
    // Y-dimension (rows): maps to blockIdx.y and threadIdx.y
    // Formula: output_row = block_index_y * threads_per_block_y + thread_index_y
    int output_row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in output feature map

    // Output dimensions (downsampling)
    // Formula: output_h = height / pool_dim, output_w = width / pool_dim
    int output_h = height / pool_dim;
    int output_w = width / pool_dim;

    // Bounds check: critical for correctness when dimensions are not divisible by block size
    // Checks both row and column bounds independently
    // Without this check, threads beyond output boundaries would access invalid memory
    if (output_row < output_h && output_col < output_w) {
        float max_val = -1e20f;  // Initialize to very small value
        
        // Find maximum value in the pooling window
        // Window starts at position (output_row*pool_dim, output_col*pool_dim) 
        // and spans pool_dim × pool_dim elements
        for (int pool_row = 0; pool_row < pool_dim; ++pool_row) {
            for (int pool_col = 0; pool_col < pool_dim; ++pool_col) {
                // Calculate input position: (output_row*pool_dim + pool_row, output_col*pool_dim + pool_col)
                int input_row = output_row * pool_dim + pool_row;
                int input_col = output_col * pool_dim + pool_col;
                // Access input value
                float val = in[input_row * width + input_col];
                // Update maximum: max_val = max(max_val, val)
                // This is a reduction operation: finding max over pooling window
                if (val > max_val) max_val = val;
            }
        }
        // Store maximum value in output
        // Memory access: coalesced (threads write consecutive elements)
        out[output_row * output_w + output_col] = max_val;
    }
}

int main() {
    // Feature map and pooling dimensions
    const int height = 256, width = 256;  
    const int pool_dim = 2;               
    const int output_h = height / pool_dim;
    const int output_w = width / pool_dim;

    const int input_size = height * width;
    const int output_size = output_h * output_w;

    std::cout << "2D Max Pooling: Input[" << height << "x" << width << "] with "
              << pool_dim << "x" << pool_dim << " pooling -> Output[" << output_h << "x" << output_w << "]" << std::endl;
    std::cout << "Downsampling factor: " << pool_dim * pool_dim << "x" << std::endl;

    // Allocate host memory for input and output feature maps
    float *h_in, *h_out_cpu, *h_out_gpu;
    allocate_host(&h_in, input_size);
    allocate_host(&h_out_cpu, output_size);
    allocate_host(&h_out_gpu, output_size);

    // Initialize input feature map with test data (pattern with noise)
    srand(42); 
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float base_val = static_cast<float>((i / 16 + j / 16) % 10);
            float noise = static_cast<float>(rand() % 100) / 100.0f; 
            h_in[i * width + j] = base_val + noise;
        }
    }

    // Run CPU version with timing
    {
        Timer cpu_timer("CPU 2D Max Pooling");
        maxpool2d_cpu(h_in, h_out_cpu, height, width, pool_dim);
    }

    // Allocate device memory
    float *d_in, *d_out;
    allocate_device(&d_in, input_size);
    allocate_device(&d_out, output_size);

    // Copy data from host to device
    copy_to_device(d_in, h_in, input_size);

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
        Timer gpu_timer("GPU 2D Max Pooling");
        maxpool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, height, width, pool_dim);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy result back from device to host
    copy_to_host(h_out_gpu, d_out, output_size);

    // Verify GPU results against CPU results
    if (verify_results(h_out_gpu, h_out_cpu, output_size)) {
        std::cout << "✓ 2D max pooling results match!" << std::endl;

        // Display sample input and output
        std::cout << "\nInput feature map (top-left 8x8):" << std::endl;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                std::cout << std::fixed << std::setprecision(1) << h_in[i * width + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nOutput after 2x2 max pooling (top-left 4x4):" << std::endl;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(1) << h_out_gpu[i * output_w + j] << " ";
            }
            std::cout << std::endl;
        }

        // Verify that each output value is the maximum of its corresponding block
        bool pooling_correct = true;
        for (int r = 0; r < output_h && pooling_correct; ++r) {
            for (int c = 0; c < output_w && pooling_correct; ++c) {
                float max_in_block = -1e20f;
                for (int pr = 0; pr < pool_dim; ++pr) {
                    for (int pc = 0; pc < pool_dim; ++pc) {
                        int input_row = r * pool_dim + pr;
                        int input_col = c * pool_dim + pc;
                        max_in_block = std::max(max_in_block, h_in[input_row * width + input_col]);
                    }
                }
                if (std::abs(h_out_gpu[r * output_w + c] - max_in_block) > 1e-6f) {
                    pooling_correct = false;
                }
            }
        }

        if (pooling_correct) {
            std::cout << "\n✓ Verified: Each output value is the maximum of its corresponding " << pool_dim << "x" << pool_dim << " input block" << std::endl;
        }
    }

    // Clean up memory
    free_host(h_in);
    free_host(h_out_cpu);
    free_host(h_out_gpu);
    free_device(d_in);
    free_device(d_out);

    return 0;
}
