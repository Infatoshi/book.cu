"""
Batched Matrix Multiplication Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA batched
matrix multiplication kernels used during transformer training. Used for processing
multiple sequences in parallel.

Usage:
    from wrapper.training.batched_matmul import BatchedMatMul
    batched_matmul_op = BatchedMatMul()
    output = batched_matmul_op(A, B)  # A is (batch, M, K), B is (batch, K, N)

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: C[batch] = A[batch] @ B[batch] for all batches
    - Backward: Computes gradients for each batch separately
    - Used in attention mechanisms for batched sequence processing
"""

import torch
from torch.autograd import Function


class BatchedMatMulFunction(Function):
    """
    Custom autograd Function for batched matrix multiplication
    
    This class implements the forward and backward passes needed for automatic
    differentiation. Processes multiple matrix multiplications in parallel.
    """
    
    @staticmethod
    def forward(ctx, A, B):
        """
        Forward pass: Compute batched matrix multiplication
        
        Computes: C[batch] = A[batch] @ B[batch] for all batches in parallel
        
        Args:
            ctx: Context object to save tensors for backward pass
            A: Input matrix A of shape (batch_size, M, K), CUDA tensor
            B: Input matrix B of shape (batch_size, K, N), CUDA tensor
        
        Returns:
            Output matrix C of shape (batch_size, M, N), CUDA tensor
        """
        # Save input tensors for backward pass
        ctx.save_for_backward(A, B)

        # Extract dimensions
        batch_size, M, K = A.shape
        _, _, N = B.shape
        
        # Allocate output tensor
        C = torch.empty(batch_size, M, N, dtype=A.dtype, device=A.device)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.batched_matmul_fwd(A, B, C)

        return C

    @staticmethod
    def backward(ctx, grad_C):
        """
        Backward pass: Compute gradients with respect to inputs
        
        Gradient formulas (applied per batch):
        - grad_A[batch] = grad_C[batch] @ B[batch]^T
        - grad_B[batch] = A[batch]^T @ grad_C[batch]
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_C: Gradient with respect to output C, shape (batch_size, M, N)
        
        Returns:
            Tuple of (grad_A, grad_B):
            - grad_A: Gradient w.r.t. A, shape (batch_size, M, K)
            - grad_B: Gradient w.r.t. B, shape (batch_size, K, N)
        """
        # Retrieve saved tensors from forward pass
        A, B = ctx.saved_tensors

        # Allocate gradient tensors
        grad_A = torch.empty_like(A)
        grad_B = torch.empty_like(B)

        # Call CUDA kernel for backward pass
        import custom_training_extension as cte
        cte.batched_matmul_bwd(A, B, grad_C, grad_A, grad_B)

        return grad_A, grad_B


class BatchedMatMul(torch.nn.Module):
    """
    PyTorch Module wrapper for batched matrix multiplication
    
    This module provides a convenient interface that can be used like any other
    PyTorch operation. It wraps BatchedMatMulFunction to integrate with the autograd system.
    """
    
    def __init__(self):
        """Initialize the BatchedMatMul module"""
        super().__init__()

    def forward(self, A, B):
        """
        Forward pass through the module
        
        Args:
            A: Input matrix A of shape (batch_size, M, K), CUDA tensor
            B: Input matrix B of shape (batch_size, K, N), CUDA tensor
        
        Returns:
            Output matrix C = A @ B of shape (batch_size, M, N), CUDA tensor
        """
        return BatchedMatMulFunction.apply(A, B)
