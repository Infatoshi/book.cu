"""
Element-wise Multiplication Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA element-wise
multiplication kernels used during transformer training.

Usage:
    from wrapper.training.mul import Mul
    mul_op = Mul()
    output = mul_op(a, b)  # element-wise multiplication

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: c = a * b
    - Backward: grad_a = grad_out * b, grad_b = grad_out * a (product rule)
"""

import torch
from torch.autograd import Function


class MulFunction(Function):
    """
    Custom autograd Function for element-wise multiplication
    
    This class implements the forward and backward passes needed for automatic
    differentiation. The backward pass uses the product rule from calculus.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass: Compute element-wise multiplication
        
        Args:
            ctx: Context object to save tensors for backward pass
            a: First input tensor, CUDA tensor (must match shape of b)
            b: Second input tensor, CUDA tensor (must match shape of a)
        
        Returns:
            Output tensor c = a * b, same shape as inputs, CUDA tensor
        """
        # Save input tensors for backward pass
        ctx.save_for_backward(a, b)

        # Allocate output tensor
        out = torch.empty_like(a)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.mul_fwd(a, b, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: Compute gradients with respect to inputs
        
        Gradient formulas (product rule):
        - grad_a = grad_out * b
        - grad_b = grad_out * a
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_out: Gradient with respect to output
        
        Returns:
            Tuple of (grad_a, grad_b):
            - grad_a: Gradient w.r.t. input a
            - grad_b: Gradient w.r.t. input b
        """
        # Retrieve saved tensors from forward pass
        a, b = ctx.saved_tensors

        # Allocate gradient tensors
        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(b)

        # Call CUDA kernel for backward pass
        import custom_training_extension as cte
        cte.mul_bwd(grad_out, a, b, grad_a, grad_b)

        return grad_a, grad_b


class Mul(torch.nn.Module):
    """
    PyTorch Module wrapper for element-wise multiplication
    
    This module provides a convenient interface that can be used like any other
    PyTorch operation. It wraps MulFunction to integrate with the autograd system.
    """
    
    def __init__(self):
        """Initialize the Mul module"""
        super().__init__()

    def forward(self, a, b):
        """
        Forward pass through the module
        
        Args:
            a: First input tensor, CUDA tensor
            b: Second input tensor, CUDA tensor
        
        Returns:
            Output tensor c = a * b, same shape as inputs, CUDA tensor
        """
        return MulFunction.apply(a, b)
