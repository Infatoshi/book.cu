"""
Layer Normalization Wrapper for Inference

This module provides a Python interface to the custom CUDA layer normalization kernel.
Layer normalization is applied before attention and after feed-forward layers in transformers.

Usage:
    from wrapper.inference.layernorm import layernorm
    result = layernorm(x, weight, bias)

Architecture:
    - Wraps custom_inference_extension.layernorm_forward()
    - Normalizes input across the last dimension (hidden dimension)
    - Applies learned scale (gamma/weight) and shift (beta/bias) parameters
    - Forward-only operation optimized for inference
"""

import custom_inference_extension


def layernorm(x, weight, bias):
    """
    Layer normalization wrapper for custom CUDA kernel
    
    Applies layer normalization: normalized = (x - mean) / sqrt(var + eps)
                                 output = normalized * weight + bias
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_size), CUDA tensor
        weight: Scale parameter (gamma) of shape (hidden_size,), CUDA tensor
        bias: Shift parameter (beta) of shape (hidden_size,), CUDA tensor
    
    Returns:
        Normalized output tensor, same shape as input x, CUDA tensor
    
    Note:
        Layer normalization is applied:
        1. Before attention: x = layernorm(x) -> attention(x)
        2. After feed-forward: x = layernorm(x + ff_output)
        
        This helps stabilize training and improve model performance by normalizing
        activations across the hidden dimension for each (batch, sequence) position.
    """
    return custom_inference_extension.layernorm_forward(x, weight, bias)
