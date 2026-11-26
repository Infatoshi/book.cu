"""
Element-wise Addition Wrapper for Inference

This module provides a Python interface to the custom CUDA element-wise addition kernel.
Used for residual connections in transformer blocks during inference.

Usage:
    from wrapper.inference.add import add
    result = add(a, b)  # element-wise addition

Architecture:
    - Simple wrapper around custom_inference_extension.add_forward()
    - Used for residual connections: x = x + attention(x)
    - Forward-only operation (no gradients needed in inference)
"""

import custom_inference_extension


def add(a, b):
    """
    Element-wise addition wrapper for custom CUDA kernel
    
    Computes element-wise addition: c = a + b
    
    Args:
        a: First input tensor, CUDA tensor (must match shape of b)
        b: Second input tensor, CUDA tensor (must match shape of a)
    
    Returns:
        Output tensor c = a + b, same shape as inputs, CUDA tensor
    
    Note:
        This is primarily used for residual connections in transformer blocks:
        - Post-attention: x = x + attention_output
        - Post-feedforward: x = x + ff_output
        No gradient computation is performed since this is inference-only.
    """
    return custom_inference_extension.add_forward(a, b)
