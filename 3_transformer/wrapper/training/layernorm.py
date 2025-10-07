import torch
from torch.autograd import Function

class LayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps=1e-5):
        out = torch.empty_like(x)
        mean = torch.empty(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        var = torch.empty(x.size(0), x.size(1), dtype=x.dtype, device=x.device)

        import custom_training_extension as cte
        cte.layernorm_fwd(x, gamma, beta, out, mean, var, eps)

        ctx.save_for_backward(x, gamma, beta, mean, var)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, gamma, beta, mean, var = ctx.saved_tensors
        eps = ctx.eps

        grad_x = torch.empty_like(x)
        grad_gamma = torch.zeros_like(gamma)
        grad_beta = torch.zeros_like(beta)

        import custom_training_extension as cte
        cte.layernorm_bwd(grad_out, x, gamma, mean, var, grad_x, grad_gamma, grad_beta, eps)

        return grad_x, grad_gamma, grad_beta, None

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return LayerNormFunction.apply(x, self.gamma, self.beta, self.eps)
