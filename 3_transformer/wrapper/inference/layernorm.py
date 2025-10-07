import custom_inference_extension

def layernorm(x, weight, bias):
    """Layer normalization wrapper for custom CUDA kernel"""
    return custom_inference_extension.layernorm_forward(x, weight, bias)
