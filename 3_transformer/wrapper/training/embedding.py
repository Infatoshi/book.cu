"""
Embedding Layer Wrapper for Training

This module provides a PyTorch autograd-compatible wrapper for custom CUDA embedding
kernels used during transformer training. Embeddings convert token indices to dense
vector representations.

Usage:
    from wrapper.training.embedding import Embedding
    emb = Embedding(num_embeddings=100, embedding_dim=128)
    output = emb(indices)  # indices is (batch, seq) -> output is (batch, seq, 128)

Architecture:
    - Uses torch.autograd.Function to integrate with PyTorch's autograd
    - Implements both forward and backward passes
    - Forward: Lookup embedding vectors for given token indices
    - Backward: Accumulate gradients to embedding weight matrix using atomic operations
    - Handles sparse gradient updates (multiple tokens may share the same embedding)
"""

import torch
from torch.autograd import Function


class EmbeddingFunction(Function):
    """
    Custom autograd Function for embedding layer
    
    This class implements the forward and backward passes needed for automatic
    differentiation. The backward pass uses atomic operations to accumulate gradients
    since multiple tokens may reference the same embedding vector.
    """
    
    @staticmethod
    def forward(ctx, weight, indices):
        """
        Forward pass: Lookup embedding vectors for token indices
        
        Performs embedding lookup: output[i] = weight[indices[i]]
        
        Args:
            ctx: Context object to save tensors for backward pass
            weight: Embedding weight matrix of shape (vocab_size, n_embd), CUDA tensor
            indices: Token indices of shape (batch_size, seq_len), CUDA tensor (int32)
        
        Returns:
            Output embeddings of shape (batch_size, seq_len, n_embd), CUDA tensor
        """
        # Save input tensors for backward pass
        ctx.save_for_backward(weight, indices)

        # Extract dimensions
        batch_size, seq_len = indices.shape
        n_embd = weight.shape[1]
        
        # Allocate output tensor
        out = torch.empty(batch_size, seq_len, n_embd, dtype=weight.dtype, device=weight.device)

        # Call CUDA kernel for forward pass
        import custom_training_extension as cte
        cte.embedding_fwd(weight, indices, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass: Accumulate gradients to embedding weight matrix
        
        The gradient for embedding weights is accumulated using atomic operations
        because multiple tokens may reference the same embedding vector. This means
        we need to sum gradients from all positions that use each embedding.
        
        Args:
            ctx: Context object containing saved tensors from forward pass
            grad_out: Gradient with respect to output, shape (batch, seq, n_embd)
        
        Returns:
            Tuple of (grad_weight, None):
            - grad_weight: Gradient w.r.t. embedding weights, shape (vocab_size, n_embd)
            - None: No gradient for indices (they are discrete)
        """
        # Retrieve saved tensors from forward pass
        weight, indices = ctx.saved_tensors

        # Allocate gradient tensor (must be zero-initialized for accumulation)
        grad_weight = torch.zeros_like(weight)

        # Convert indices to int32 (required by CUDA kernel)
        indices_int32 = indices.to(torch.int32)

        # Call CUDA kernel for backward pass
        # Uses atomic operations to accumulate gradients
        import custom_training_extension as cte
        cte.embedding_bwd(grad_out, indices_int32, grad_weight)

        return grad_weight, None


class Embedding(torch.nn.Module):
    """
    PyTorch Module wrapper for embedding layer
    
    This module provides a convenient interface similar to torch.nn.Embedding.
    It wraps EmbeddingFunction and manages the learnable embedding weight matrix.
    """
    
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initialize the Embedding module
        
        Args:
            num_embeddings: Size of vocabulary (number of unique tokens)
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embedding weight matrix
        # Typically initialized with small random values
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, indices):
        """
        Forward pass through the module
        
        Args:
            indices: Token indices of shape (batch_size, seq_len), CUDA tensor (int32 or int64)
        
        Returns:
            Output embeddings of shape (batch_size, seq_len, embedding_dim), CUDA tensor
        """
        # Convert indices to int32 (required by CUDA kernel)
        indices_int32 = indices.to(torch.int32)
        return EmbeddingFunction.apply(self.weight, indices_int32)
