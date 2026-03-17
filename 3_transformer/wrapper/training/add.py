"""
Element-wise Addition Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA element-wise
addition kernels used during transformer training. Used for residual connections.

Usage:
    from wrapper.training.add import Add
    add_op = Add()
    output = add_op(a, b)  # element-wise addition

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: c = a + b
    - Backward: grad_a = grad_out, grad_b = grad_out (both receive same gradient)
    - Handles tensor broadcasting if shapes don't match
"""

import torch
from torch.autograd import Function


class AddFunction(Function):
    """
    Custom autograd Function for element-wise addition
    
    This class implements the forward and backward passes needed for automatic
    differentiation. Handles tensor broadcasting for inputs with different shapes.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass: Compute element-wise addition with broadcasting
        
        Args:
            ctx: Context object to save tensors for backward pass
            a: First input tensor, CUDA tensor
            b: Second input tensor, CUDA tensor
        
        Returns:
            Output tensor c = a + b (after broadcasting), CUDA tensor
        """
        # Handle tensor broadcasting (PyTorch's automatic broadcasting)
        a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)

        # Save original tensors and broadcast shapes for backward pass
        ctx.save_for_backward(a, b)
        ctx.a_broadcast_shape = a_broadcast.shape
        ctx.b_broadcast_shape = b_broadcast.shape

        # Allocate output tensor
        out = torch.empty_like(a_broadcast)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.add_fwd(a_broadcast, b_broadcast, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: Compute gradients with respect to inputs
        
        Gradient formulas:
        - grad_a = grad_out (summed over broadcast dimensions if needed)
        - grad_b = grad_out (summed over broadcast dimensions if needed)
        
        If broadcasting occurred, we need to sum gradients over the broadcast dimensions
        to match the original input shapes.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_out: Gradient with respect to output
        
        Returns:
            Tuple of (grad_a, grad_b):
            - grad_a: Gradient w.r.t. input a (summed if broadcasting occurred)
            - grad_b: Gradient w.r.t. input b (summed if broadcasting occurred)
        """
        # Retrieve saved tensors from forward pass
        a, b = ctx.saved_tensors
        a_broadcast_shape = ctx.a_broadcast_shape
        b_broadcast_shape = ctx.b_broadcast_shape

        # Allocate gradient tensors with broadcast shapes
        grad_a_broadcast = torch.empty(a_broadcast_shape, dtype=a.dtype, device=a.device)
        grad_b_broadcast = torch.empty(b_broadcast_shape, dtype=b.dtype, device=b.device)

        # Call CUDA kernel for backward pass
        import custom_training_extension as cte
        cte.add_bwd(grad_out, grad_a_broadcast, grad_b_broadcast)

        # If broadcasting occurred, sum gradients over broadcast dimensions
        # to match original input shapes
        if a_broadcast_shape != a.shape:
            grad_a = grad_a_broadcast.sum_to_size(a.shape)
        else:
            grad_a = grad_a_broadcast

        if b_broadcast_shape != b.shape:
            grad_b = grad_b_broadcast.sum_to_size(b.shape)
        else:
            grad_b = grad_b_broadcast

        return grad_a, grad_b


class Add(torch.nn.Module):
    """
    PyTorch Module wrapper for element-wise addition
    
    This module provides a convenient interface that can be used like any other
    PyTorch operation. It wraps AddFunction to integrate with the autograd system.
    """
    
    def __init__(self):
        """Initialize the Add module"""
        super().__init__()

    def forward(self, a, b):
        """
        Forward pass through the module
        
        Args:
            a: First input tensor, CUDA tensor
            b: Second input tensor, CUDA tensor
        
        Returns:
            Output tensor c = a + b (after broadcasting), CUDA tensor
        """
        return AddFunction.apply(a, b)
