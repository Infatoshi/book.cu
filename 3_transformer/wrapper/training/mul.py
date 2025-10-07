import torch
from torch.autograd import Function

class MulFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)

        out = torch.empty_like(a)

        import custom_training_extension as cte
        cte.mul_fwd(a, b, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors

        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(b)

        import custom_training_extension as cte
        cte.mul_bwd(grad_out, a, b, grad_a, grad_b)

        return grad_a, grad_b

class Mul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return MulFunction.apply(a, b)
