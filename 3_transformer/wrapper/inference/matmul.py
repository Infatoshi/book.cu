import custom_inference_extension


def matmul(a, b):
    """Matrix multiplication wrapper for custom CUDA kernel"""
    return custom_inference_extension.matmul_forward(a, b)
