"""
Matrix Multiplication Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA matrix
multiplication kernels used during transformer training. It integrates with PyTorch's
automatic differentiation system to enable gradient computation.

Usage:
    from wrapper.training.matmul import MatMul
    matmul_op = MatMul()
    output = matmul_op(A, B)  # A is (M, K), B is (K, N) -> output is (M, N)

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: C = A @ B
    - Backward: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
    - Used in attention mechanisms and feed-forward networks during training
"""

import torch
from torch.autograd import Function


class MatMulFunction(Function):
    """
    Custom autograd Function for matrix multiplication
    
    This class implements the forward and backward passes needed for automatic
    differentiation. PyTorch's autograd system calls forward() during the forward
    pass and backward() during backpropagation.
    """
    
    @staticmethod
    def forward(ctx, A, B):
        """
        Forward pass: Compute C = A @ B
        
        Args:
            ctx: Context object to save tensors for backward pass
            A: Input matrix A of shape (M, K), CUDA tensor
            B: Input matrix B of shape (K, N), CUDA tensor
        
        Returns:
            Output matrix C of shape (M, N), CUDA tensor
        """
        # Save input tensors for backward pass
        ctx.save_for_backward(A, B)

        # Extract dimensions
        M, K = A.shape
        N = B.shape[1]
        
        # Allocate output tensor
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.matmul_fwd(A, B, C)

        return C

    @staticmethod
    def backward(ctx, grad_C):
        """
        Backward pass: Compute gradients with respect to inputs
        
        Gradient formulas:
        - grad_A = grad_C @ B^T  (gradient w.r.t. A)
        - grad_B = A^T @ grad_C  (gradient w.r.t. B)
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_C: Gradient with respect to output C, shape (M, N)
        
        Returns:
            Tuple of (grad_A, grad_B):
            - grad_A: Gradient w.r.t. A, shape (M, K)
            - grad_B: Gradient w.r.t. B, shape (K, N)
        """
        # Retrieve saved tensors from forward pass
        A, B = ctx.saved_tensors

        # Allocate gradient tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)

        # Call CUDA kernels for backward pass
        import custom_training_extension as cte
        cte.matmul_bwd(A, B, grad_C, grad_A, grad_B)

        return grad_A, grad_B


class MatMul(torch.nn.Module):
    """
    PyTorch Module wrapper for matrix multiplication
    
    This module provides a convenient interface that can be used like any other
    PyTorch operation. It wraps MatMulFunction to integrate with the autograd system.
    """
    
    def __init__(self):
        """Initialize the MatMul module"""
        super().__init__()

    def forward(self, A, B):
        """
        Forward pass through the module
        
        Args:
            A: Input matrix A of shape (M, K), CUDA tensor
            B: Input matrix B of shape (K, N), CUDA tensor
        
        Returns:
            Output matrix C = A @ B of shape (M, N), CUDA tensor
        """
        return MatMulFunction.apply(A, B)
