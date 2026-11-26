"""
Element-wise Multiplication Wrapper for Inference

This module provides a Python interface to the custom CUDA element-wise multiplication kernel.
Used for scaling operations during transformer inference.

Usage:
    from wrapper.inference.mul import mul
    result = mul(a, b)  # element-wise multiplication

Architecture:
    - Simple wrapper around custom_inference_extension.mul_forward()
    - Used for scaling attention weights, expert routing probabilities, etc.
    - Forward-only operation (no gradients needed in inference)
"""

import custom_inference_extension


def mul(a, b):
    """
    Element-wise multiplication wrapper for custom CUDA kernel
    
    Computes element-wise multiplication: c = a * b
    
    Args:
        a: First input tensor, CUDA tensor (must match shape of b)
        b: Second input tensor, CUDA tensor (must match shape of a)
    
    Returns:
        Output tensor c = a * b, same shape as inputs, CUDA tensor
    
    Note:
        Common uses in transformer inference:
        - Scaling attention weights by expert probabilities (MoE)
        - Re-normalizing top-k probabilities after selection
        - Element-wise scaling operations
    """
    return custom_inference_extension.mul_forward(a, b)
