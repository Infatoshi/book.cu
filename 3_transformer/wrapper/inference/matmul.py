"""
Matrix Multiplication Wrapper for Inference

This module provides a Python interface to the custom CUDA matrix multiplication kernel
used during transformer inference. It wraps the C++ extension function to provide
a clean API for Python code.

Usage:
    from wrapper.inference.matmul import matmul
    result = matmul(a, b)  # where a is (M, K) and b is (K, N)

Architecture:
    - Simple wrapper around custom_inference_extension.matmul_forward()
    - No backward pass needed (inference only)
    - Used for attention score computation and feed-forward layers
    - Optimized for inference workloads (no gradient tracking overhead)
"""

import custom_inference_extension


def matmul(a, b):
    """
    Matrix multiplication wrapper for custom CUDA kernel
    
    Computes matrix multiplication: C = A @ B
    where A is (M, K) and B is (K, N), resulting in C of shape (M, N)
    
    Args:
        a: Input tensor A of shape (M, K), CUDA tensor
        b: Input tensor B of shape (K, N), CUDA tensor
    
    Returns:
        Output tensor C of shape (M, N), CUDA tensor
    
    Note:
        This is a forward-only operation optimized for inference.
        No gradient computation is performed, making it faster than training operations.
    """
    return custom_inference_extension.matmul_forward(a, b)
