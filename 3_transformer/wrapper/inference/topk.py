"""
Top-K Selection Wrapper for Inference

This module provides a Python interface to the custom CUDA Top-K selection kernel.
Used for Mixture of Experts (MoE) routing to select the top K experts for each token.

Usage:
    from wrapper.inference.topk import topk
    values, indices = topk(input, k=2)  # get top 2 experts

Architecture:
    - Wraps custom_inference_extension.topk_forward()
    - Returns both values and indices of top K elements
    - Used in MoE architectures for sparse expert routing
    - Critical for reducing computation in MoE models
"""

import custom_inference_extension


def topk(input, k):
    """
    Top-K selection wrapper for custom CUDA kernel
    
    Finds the K largest values and their indices in each row of the input tensor.
    
    Args:
        input: Input tensor of shape (batch_size, n), CUDA tensor
               Typically expert gate logits or probabilities
        k: Number of top elements to select (must be <= n)
    
    Returns:
        Tuple of (values, indices), both CUDA tensors:
        - values: Top-K values of shape (batch_size, k), sorted in descending order
        - indices: Indices of top-K values of shape (batch_size, k), dtype=int32
    
    Note:
        This is primarily used for MoE (Mixture of Experts) routing:
        1. Compute expert gate logits: gate_logits = gate(x)
        2. Apply softmax: gate_probs = softmax(gate_logits)
        3. Select top-K experts: topk_probs, topk_indices = topk(gate_probs, k=2)
        4. Route tokens to selected experts only
        
        By selecting only top-K experts (typically k=2), we sparsify computation
        and reduce the number of experts that need to process each token.
        
        Known limitation: The current implementation uses naive insertion sort,
        which may have numerical precision issues when combined with custom softmax.
        See README.md for details on MoE routing challenges.
    """
    return custom_inference_extension.topk_forward(input, k)
