import torch
from torch.autograd import Function

class GELUFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        out = torch.empty_like(x)

        import custom_training_extension as cte
        cte.gelu_fwd(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors

        grad_x = torch.empty_like(x)

        import custom_training_extension as cte
        cte.gelu_bwd(grad_out, x, grad_x)

        return grad_x

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GELUFunction.apply(x)
