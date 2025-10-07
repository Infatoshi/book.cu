import custom_inference_extension

def gelu(x):
    """GELU activation wrapper for custom CUDA kernel"""
    return custom_inference_extension.gelu_forward(x)
