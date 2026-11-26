"""
GELU Activation Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA GELU
activation kernels used during transformer training. GELU is used in feed-forward networks.

Usage:
    from wrapper.training.activation import GELU
    gelu_op = GELU()
    output = gelu_op(x)  # applies GELU activation

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    - Backward: Computes gradient using chain rule
"""

import torch
from torch.autograd import Function


class GELUFunction(Function):
    """
    Custom autograd Function for GELU activation
    
    This class implements the forward and backward passes needed for automatic
    differentiation. The backward pass computes the derivative of GELU.
    """
    
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass: Apply GELU activation
        
        GELU formula: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        
        Args:
            ctx: Context object to save tensors for backward pass
            x: Input tensor of any shape, CUDA tensor
        
        Returns:
            Output tensor with GELU activation applied, same shape as input, CUDA tensor
        """
        # Save input tensor for backward pass
        ctx.save_for_backward(x)

        # Allocate output tensor
        out = torch.empty_like(x)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.gelu_fwd(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: Compute gradient with respect to input
        
        The gradient is computed using the derivative of GELU:
        dGELU/dx = 0.5 * (1 + erf(x/√2)) + 0.5 * x * d(erf(x/√2))/dx
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_out: Gradient with respect to output
        
        Returns:
            Gradient with respect to input x, same shape as input
        """
        # Retrieve saved tensor from forward pass
        x, = ctx.saved_tensors

        # Allocate gradient tensor
        grad_x = torch.empty_like(x)

        # Call CUDA kernel for backward pass
        import custom_training_extension as cte
        cte.gelu_bwd(grad_out, x, grad_x)

        return grad_x


class GELU(torch.nn.Module):
    """
    PyTorch Module wrapper for GELU activation
    
    This module provides a convenient interface that can be used like any other
    PyTorch activation function. It wraps GELUFunction to integrate with the autograd system.
    """
    
    def __init__(self):
        """Initialize the GELU module"""
        super().__init__()

    def forward(self, x):
        """
        Forward pass through the module
        
        Args:
            x: Input tensor of any shape, CUDA tensor
        
        Returns:
            Output tensor with GELU activation applied, same shape as input, CUDA tensor
        """
        return GELUFunction.apply(x)
