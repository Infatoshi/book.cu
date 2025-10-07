import custom_inference_extension

def mul(a, b):
    """Elementwise multiplication wrapper for custom CUDA kernel"""
    return custom_inference_extension.mul_forward(a, b)
