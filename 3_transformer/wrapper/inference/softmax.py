"""
Softmax Activation Wrapper for Inference

This module provides a Python interface to the custom CUDA softmax kernel.
Softmax is used to convert attention logits into attention probabilities.

Usage:
    from wrapper.inference.softmax import softmax
    result = softmax(logits)  # converts logits to probabilities

Architecture:
    - Wraps custom_inference_extension.softmax_forward()
    - Applies softmax along the last dimension (vocabulary/attention dimension)
    - Uses numerically stable implementation (subtracts max before exp)
    - Forward-only operation for inference
"""

import custom_inference_extension


def softmax(x):
    """
    Softmax wrapper for custom CUDA kernel
    
    Applies softmax activation: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Args:
        x: Input logits tensor of shape (batch_size, seq_len, vocab_size), CUDA tensor
           For attention: (batch_size, seq_len, seq_len) attention scores
    
    Returns:
        Output probabilities tensor, same shape as input, CUDA tensor
        All values are in [0, 1] and sum to 1 along the last dimension
    
    Note:
        In transformer inference:
        1. Attention weights: softmax(Q @ K^T / sqrt(d_k))
        2. MoE routing: softmax(gate_logits) to get expert probabilities
        
        The implementation uses max subtraction for numerical stability:
        - Prevents overflow when computing exp() of large values
        - Ensures numerical precision is maintained
    """
    return custom_inference_extension.softmax_forward(x)
