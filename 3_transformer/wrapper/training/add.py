import torch
from torch.autograd import Function

class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        a_broadcast, b_broadcast = torch.broadcast_tensors(a, b)

        ctx.save_for_backward(a, b)
        ctx.a_broadcast_shape = a_broadcast.shape
        ctx.b_broadcast_shape = b_broadcast.shape

        out = torch.empty_like(a_broadcast)

        import custom_training_extension as cte
        cte.add_fwd(a_broadcast, b_broadcast, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        a_broadcast_shape = ctx.a_broadcast_shape
        b_broadcast_shape = ctx.b_broadcast_shape

        grad_a_broadcast = torch.empty(a_broadcast_shape, dtype=a.dtype, device=a.device)
        grad_b_broadcast = torch.empty(b_broadcast_shape, dtype=b.dtype, device=b.device)

        import custom_training_extension as cte
        cte.add_bwd(grad_out, grad_a_broadcast, grad_b_broadcast)

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
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return AddFunction.apply(a, b)
