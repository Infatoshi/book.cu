#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

/**
 * Timer class for measuring execution time of operations
 * Automatically prints elapsed time when destroyed (RAII pattern)
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    /**
     * Constructor - starts timing immediately
     * @param operation_name Name of the operation being timed (optional)
     */
    Timer(const std::string& operation_name = "") : name(operation_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    /**
     * Destructor - automatically prints elapsed time if name was provided
     */
    ~Timer() {
        if (!name.empty()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << name << " took " << duration.count() << " ms" << std::endl;
        }
    }

    /**
     * Reset the timer to start timing from now
     */
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    /**
     * Get elapsed time in milliseconds without stopping the timer
     * @return Elapsed time in milliseconds
     */
    long long elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
};

/**
 * CUDA error checking macro
 * Wraps CUDA calls and exits with error message if operation fails
 * @param call CUDA function call to check
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * Allocate host (CPU) memory
 * @param ptr Pointer to pointer that will hold allocated memory
 * @param size Number of elements to allocate
 */
template<typename T>
void allocate_host(T** ptr, size_t size) {
    *ptr = (T*)malloc(size * sizeof(T));
    if (*ptr == nullptr) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        exit(1);
    }
}

/**
 * Allocate device (GPU) memory
 * @param d_ptr Pointer to pointer that will hold allocated device memory
 * @param size Number of elements to allocate
 */
template<typename T>
void allocate_device(T** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)d_ptr, size * sizeof(T)));
}

/**
 * Copy data from host to device
 * @param d_dst Destination device memory pointer
 * @param h_src Source host memory pointer
 * @param size Number of elements to copy
 */
template<typename T>
void copy_to_device(T* d_dst, const T* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size * sizeof(T), cudaMemcpyHostToDevice));
}

/**
 * Copy data from device to host
 * @param h_dst Destination host memory pointer
 * @param d_src Source device memory pointer
 * @param size Number of elements to copy
 */
template<typename T>
void copy_to_host(T* h_dst, const T* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

/**
 * Free host (CPU) memory
 * @param ptr Pointer to memory to free
 */
template<typename T>
void free_host(T* ptr) {
    free(ptr);
}

/**
 * Free device (GPU) memory
 * @param d_ptr Pointer to device memory to free
 */
template<typename T>
void free_device(T* d_ptr) {
    CUDA_CHECK(cudaFree(d_ptr));
}

/**
 * Verify that two arrays of floats match within tolerance
 * @param result Array to verify
 * @param reference Reference array for comparison
 * @param size Number of elements to compare
 * @param tolerance Maximum allowed difference between elements
 * @return true if arrays match within tolerance, false otherwise
 */
bool verify_results(const float* result, const float* reference, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(result[i] - reference[i]) > tolerance) {
            std::cout << "Verification failed at index " << i
                      << ": got " << result[i] << ", expected " << reference[i] << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * Print a matrix in readable format
 * @param matrix Pointer to matrix data (row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param name Optional name/title for the matrix
 */
void print_matrix(const float* matrix, int rows, int cols, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << ":" << std::endl;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
