"""
GELU Activation Wrapper for Inference

This module provides a Python interface to the custom CUDA GELU (Gaussian Error Linear Unit)
activation kernel. GELU is the activation function used in transformer feed-forward networks.

Usage:
    from wrapper.inference.activation import gelu
    result = gelu(x)  # applies GELU activation

Architecture:
    - Wraps custom_inference_extension.gelu_forward()
    - Implements GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    - Used in feed-forward network activation layers
    - Forward-only operation for inference
"""

import custom_inference_extension


def gelu(x):
    """
    GELU activation wrapper for custom CUDA kernel
    
    Applies Gaussian Error Linear Unit activation: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    
    Args:
        x: Input tensor of any shape, CUDA tensor
    
    Returns:
        Output tensor with GELU activation applied, same shape as input, CUDA tensor
    
    Note:
        GELU is used in transformer feed-forward networks:
        - Feed-forward: FFN(x) = Linear(GELU(Linear(x)))
        - Provides smooth activation that outperforms ReLU in transformers
        - The implementation uses an approximation based on erf (error function)
        
        Common in models like GPT, BERT, and other transformer architectures.
    """
    return custom_inference_extension.gelu_forward(x)
