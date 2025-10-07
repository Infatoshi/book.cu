import custom_inference_extension

def add(a, b):
    """Elementwise addition wrapper for custom CUDA kernel"""
    return custom_inference_extension.add_forward(a, b)
