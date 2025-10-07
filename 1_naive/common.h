


class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    Timer(const std::string& operation_name = "") : name(operation_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        if (!name.empty()) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << name << " took " << duration.count() << " ms" << std::endl;
        }
    }

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    long long elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
};

    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

template<typename T>
void allocate_host(T** ptr, size_t size) {
    *ptr = (T*)malloc(size * sizeof(T));
    if (*ptr == nullptr) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        exit(1);
    }
}

template<typename T>
void allocate_device(T** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)d_ptr, size * sizeof(T)));
}

template<typename T>
void copy_to_device(T* d_dst, const T* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copy_to_host(T* h_dst, const T* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void free_host(T* ptr) {
    free(ptr);
}

template<typename T>
void free_device(T* d_ptr) {
    CUDA_CHECK(cudaFree(d_ptr));
}

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

