import custom_inference_extension

def softmax(x):
    """Softmax wrapper for custom CUDA kernel"""
    return custom_inference_extension.softmax_forward(x)
