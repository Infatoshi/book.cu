import torch
from torch.autograd import Function

class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        out = torch.empty_like(x)

        import custom_training_extension as cte
        cte.softmax_fwd(x, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors

        out = torch.empty_like(x)
        import custom_training_extension as cte
        cte.softmax_fwd(x, out)

        grad_x = torch.empty_like(x)
        cte.softmax_bwd(grad_out, out, grad_x)

        return grad_x

class Softmax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return SoftmaxFunction.apply(x)
