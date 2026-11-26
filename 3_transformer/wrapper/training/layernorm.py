"""
Layer Normalization Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA layer
normalization kernels used during transformer training. Layer normalization is
applied before attention and after feed-forward layers.

Usage:
    from wrapper.training.layernorm import LayerNorm
    ln = LayerNorm(normalized_shape=128)
    output = ln(x)  # x is (batch, seq, 128)

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: Normalize and apply affine transformation
    - Backward: Compute gradients w.r.t. input, gamma, and beta
    - Saves mean and variance from forward pass for efficient backward computation
"""

import torch
from torch.autograd import Function


class LayerNormFunction(Function):
    """
    Custom autograd Function for layer normalization
    
    This class implements the forward and backward passes needed for automatic
    differentiation. The forward pass saves mean and variance statistics that
    are reused in the backward pass for efficiency.
    """
    
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-5):
        """
        Forward pass: Normalize input and apply affine transformation
        
        Algorithm:
        1. Compute mean and variance across hidden dimension
        2. Normalize: normalized = (x - mean) / sqrt(var + eps)
        3. Apply affine: output = normalized * gamma + beta
        
        Args:
            ctx: Context object to save tensors for backward pass
            x: Input tensor of shape (batch_size, seq_len, n_embd), CUDA tensor
            gamma: Scale parameter (weight) of shape (n_embd,), CUDA tensor
            beta: Shift parameter (bias) of shape (n_embd,), CUDA tensor
            eps: Small epsilon to prevent division by zero (default: 1e-5)
        
        Returns:
            Normalized output tensor, same shape as input x, CUDA tensor
        """
        # Allocate output and statistics tensors
        out = torch.empty_like(x)
        mean = torch.empty(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        var = torch.empty(x.size(0), x.size(1), dtype=x.dtype, device=x.device)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.layernorm_fwd(x, gamma, beta, out, mean, var, eps)

        # Save tensors and metadata for backward pass
        ctx.save_for_backward(x, gamma, beta, mean, var)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: Compute gradients with respect to inputs and parameters
        
        Gradient formulas:
        - grad_gamma = sum(grad_out * normalized) across (batch, seq) dimensions
        - grad_beta = sum(grad_out) across (batch, seq) dimensions
        - grad_x = inv_std * (grad_out * gamma - mean(grad_out * gamma) - normalized * mean(grad_out * gamma * normalized))
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_out: Gradient with respect to output, shape (batch, seq, n_embd)
        
        Returns:
            Tuple of (grad_x, grad_gamma, grad_beta, None):
            - grad_x: Gradient w.r.t. input x, shape (batch, seq, n_embd)
            - grad_gamma: Gradient w.r.t. gamma, shape (n_embd,)
            - grad_beta: Gradient w.r.t. beta, shape (n_embd,)
            - None: No gradient for eps parameter
        """
        # Retrieve saved tensors from forward pass
        x, gamma, beta, mean, var = ctx.saved_tensors
        eps = ctx.eps

        # Allocate gradient tensors
        grad_x = torch.empty_like(x)
        grad_gamma = torch.zeros_like(gamma)  # Must be zero-initialized for accumulation
        grad_beta = torch.zeros_like(beta)     # Must be zero-initialized for accumulation

        # Call CUDA kernel for backward pass
        import custom_training_extension as cte
        cte.layernorm_bwd(grad_out, x, gamma, mean, var, grad_x, grad_gamma, grad_beta, eps)

        return grad_x, grad_gamma, grad_beta, None


class LayerNorm(torch.nn.Module):
    """
    PyTorch Module wrapper for layer normalization
    
    This module provides a convenient interface similar to torch.nn.LayerNorm.
    It wraps LayerNormFunction and manages learnable parameters (gamma and beta).
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initialize the LayerNorm module
        
        Args:
            normalized_shape: Shape of the normalized dimension (hidden size)
            eps: Small epsilon to prevent division by zero (default: 1e-5)
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Initialize learnable parameters
        # gamma (scale) starts at 1.0, beta (shift) starts at 0.0
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        Forward pass through the module
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, normalized_shape), CUDA tensor
        
        Returns:
            Normalized output tensor, same shape as input, CUDA tensor
        """
        return LayerNormFunction.apply(x, self.gamma, self.beta, self.eps)
