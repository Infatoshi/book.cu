

struct ValueIndex {
    float value;
    int index;
    
    __device__ __forceinline__ bool operator<(const ValueIndex& other) const {
        return value < other.value;
    }
    
    __device__ __forceinline__ bool operator>(const ValueIndex& other) const {
        return value > other.value;
    }
};

