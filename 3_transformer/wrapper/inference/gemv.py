"""
General Matrix-Vector Multiplication (GEMV) Wrapper for Inference

This module provides a Python interface to the custom CUDA GEMV kernel optimized
for transformer inference. GEMV is more efficient than full matrix multiplication
when processing single tokens during autoregressive generation.

Usage:
    from wrapper.inference.gemv import gemv
    result = gemv(matrix, vector)  # matrix is (M, N), vector is (N,)

Architecture:
    - Optimized for single-token generation in autoregressive models
    - Supports both single and batched operations
    - Used in attention computation when processing one token at a time
    - Key optimization: avoids computing full attention matrix for incremental generation
"""

import custom_inference_extension


def gemv(a, b):
    """
    Matrix-vector multiplication wrapper for custom CUDA kernel
    
    Computes General Matrix-Vector multiplication: y = A @ x
    Supports both single and batched operations:
    - Single: A is (M, N), x is (N,) -> y is (M,)
    - Batched: A is (batch, M, N), x is (batch, N) -> y is (batch, M)
    
    Args:
        a: Input matrix A, CUDA tensor
           - Single: shape (M, N)
           - Batched: shape (batch, M, N)
        b: Input vector x, CUDA tensor
           - Single: shape (N,)
           - Batched: shape (batch, N)
    
    Returns:
        Output vector y, CUDA tensor
        - Single: shape (M,)
        - Batched: shape (batch, M)
    
    Note:
        This operation is optimized for inference where we process one token at a time.
        During autoregressive generation, we only need to compute attention for the
        new token, making GEMV more efficient than full matrix multiplication.
    """
    return custom_inference_extension.gemv_forward(a, b)
