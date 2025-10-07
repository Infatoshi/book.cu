import custom_inference_extension

def topk(input, k):
    """Top-K selection wrapper for custom CUDA kernel"""
    return custom_inference_extension.topk_forward(input, k)
