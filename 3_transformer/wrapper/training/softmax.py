"""
Softmax Activation Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA softmax
kernels used during transformer training. Softmax converts attention logits to probabilities.

Usage:
    from wrapper.training.softmax import Softmax
    softmax_op = Softmax()
    output = softmax_op(logits)  # converts logits to probabilities

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: Softmax with numerical stability (subtracts max before exp)
    - Backward: Computes gradient using softmax gradient formula
    - Note: Backward pass requires recomputing softmax output (stored, not saved)
"""

import torch
from torch.autograd import Function


class SoftmaxFunction(Function):
    """
    Custom autograd Function for softmax activation
    
    This class implements the forward and backward passes needed for automatic
    differentiation. The backward pass requires the softmax output, which is
    recomputed rather than saved to save memory.
    """
    
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Apply softmax activation
        
        Softmax formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        
        Args:
            ctx: Context object to save tensors for backward pass
            x: Input logits tensor of shape (batch_size, seq_len, n_embd), CUDA tensor
        
        Returns:
            Output probabilities tensor, same shape as input, CUDA tensor
            All values are in [0, 1] and sum to 1 along the last dimension
        """
        # Save input for backward pass (we'll recompute softmax in backward)
        ctx.save_for_backward(x)

        # Allocate output tensor
        out = torch.empty_like(x)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.softmax_fwd(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: Compute gradient with respect to input
        
        Gradient formula for softmax:
        - Let s = softmax(x), then grad_x = s * (grad_out - sum(grad_out * s))
        
        The backward pass requires the softmax output, which is recomputed here
        rather than saved to save memory. This is acceptable since softmax forward
        is relatively cheap compared to other operations.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_out: Gradient with respect to output probabilities
        
        Returns:
            Gradient with respect to input logits x, same shape as input
        """
        # Retrieve saved tensor from forward pass
        x, = ctx.saved_tensors

        # Recompute softmax output (needed for backward pass)
        # This is done to save memory (don't need to store softmax output)
        out = torch.empty_like(x)
        import custom_training_extension as cte
        cte.softmax_fwd(x, out)

        # Allocate gradient tensor
        grad_x = torch.empty_like(x)
        
        # Call CUDA kernel for backward pass
        cte.softmax_bwd(grad_out, out, grad_x)

        return grad_x


class Softmax(torch.nn.Module):
    """
    PyTorch Module wrapper for softmax activation
    
    This module provides a convenient interface that can be used like any other
    PyTorch activation function. It wraps SoftmaxFunction to integrate with the autograd system.
    """
    
    def __init__(self, dim=-1):
        """
        Initialize the Softmax module
        
        Args:
            dim: Dimension along which to apply softmax (currently not used, always last dim)
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass through the module
        
        Args:
            x: Input logits tensor of shape (batch_size, seq_len, n_embd), CUDA tensor
        
        Returns:
            Output probabilities tensor, same shape as input, CUDA tensor
        """
        return SoftmaxFunction.apply(x)
