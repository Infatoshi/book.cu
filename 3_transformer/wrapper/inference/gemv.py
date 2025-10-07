import custom_inference_extension

def gemv(a, b):
    """Matrix-vector multiplication wrapper for custom CUDA kernel
    Handles batch-aware GEMV for attention computation in inference"""
    return custom_inference_extension.gemv_forward(a, b)
