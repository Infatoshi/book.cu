"""
Compare each custom training op in wrapper/training against PyTorch (forward output and gradients).

Run after building the extensions:
    python setup.py build_ext --inplace
    python check_training_ops.py

Each op is checked with three kinds of inputs, because all of them occur in train.py:
  - contiguous tensors
  - a non-contiguous input or upstream gradient (CustomLinear passes self.weight.t(), and
    torch.cat's backward hands each attention head a strided slice of the gradient)
  - broadcasting (Add with a bias vector)
Exits non-zero if any check fails.
"""
import sys
import torch
import torch.nn.functional as F
from wrapper.training import MatMul, BatchedMatMul, Add, Mul, GELU, Softmax, LayerNorm, Embedding

torch.manual_seed(0)
dev = "cuda"
failures = []


def rel(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def check(name, f_custom, f_ref, inputs, strided_grad=False, tol=1e-4):
    xs_c = [x.detach().clone().requires_grad_() for x in inputs]
    xs_r = [x.detach().clone().requires_grad_() for x in inputs]
    y_c, y_r = f_custom(*xs_c), f_ref(*xs_r)
    if strided_grad:
        # what torch.cat(..., dim=-1) does in backward: each piece gets a strided slice
        G = torch.randn(*y_r.shape[:-1], 2 * y_r.shape[-1], device=dev)
        torch.cat([y_c, y_c.detach()], -1).backward(G)
        torch.cat([y_r, y_r.detach()], -1).backward(G)
    else:
        g = torch.randn_like(y_r)
        (y_c * g).sum().backward()
        (y_r * g).sum().backward()
    errs = [rel(y_c, y_r)] + [rel(a.grad, b.grad) for a, b in zip(xs_c, xs_r)]
    ok = all(e < tol for e in errs)
    if not ok:
        failures.append(name)
    print(f"{'OK  ' if ok else 'FAIL'} {name:<46} fwd {errs[0]:.1e}  grads " + " ".join(f"{e:.1e}" for e in errs[1:]))


mm, bmm, add, mul, gelu, sm = MatMul(), BatchedMatMul(), Add(), Mul(), GELU(), Softmax()
X = torch.randn(1024, 128, device=dev)
W = torch.randn(512, 128, device=dev)
H = torch.randn(16, 64, 128, device=dev)
S = torch.randn(16, 64, 64, device=dev).masked_fill(torch.tril(torch.ones(64, 64, device=dev)) == 0, float("-inf"))
P, V = torch.softmax(torch.randn(16, 64, 64, device=dev), -1), torch.randn(16, 64, 128, device=dev)
ln = LayerNorm(128).to(dev)

for strided in (False, True):
    tag = " [strided grad]" if strided else ""
    check("MatMul x @ W.t() (CustomLinear)" + tag, lambda x, w: mm(x, w.t()), lambda x, w: x @ w.t(), [X, W], strided)
    check("BatchedMatMul wei @ v" + tag, bmm, torch.matmul, [P, V], strided)
    check("Softmax, causal mask" + tag, sm, lambda s: torch.softmax(s, -1), [S], strided)
    check("GELU" + tag, gelu, F.gelu, [H], strided)
    check("Mul" + tag, mul, torch.mul, [H, torch.randn_like(H)], strided)
    check("Add" + tag, add, torch.add, [H, torch.randn_like(H)], strided)
    check("LayerNorm (input grad)" + tag, ln,
          lambda h: F.layer_norm(h, (128,), ln.gamma.detach(), ln.beta.detach(), 1e-5), [H], strided)
check("Add with broadcast bias", add, torch.add, [H, torch.randn(1, 1, 128, device=dev)])

# parameter gradients: LayerNorm gamma/beta, Embedding weight
for C in (128, 768):
    lnc, lnr = LayerNorm(C).to(dev), torch.nn.LayerNorm(C).to(dev)
    with torch.no_grad():
        torch.nn.init.normal_(lnc.gamma)
        lnr.weight.copy_(lnc.gamma)
        lnr.bias.copy_(lnc.beta)
    x, g = torch.randn(16, 64, C, device=dev), torch.randn(16, 64, C, device=dev)
    (lnc(x) * g).sum().backward()
    (lnr(x) * g).sum().backward()
    errs = (rel(lnc.gamma.grad, lnr.weight.grad), rel(lnc.beta.grad, lnr.bias.grad))
    ok = all(e < 1e-4 for e in errs)
    if not ok:
        failures.append(f"LayerNorm gamma/beta grads, n_embd={C}")
    print(f"{'OK  ' if ok else 'FAIL'} {f'LayerNorm gamma/beta grads, n_embd={C}':<46} gamma {errs[0]:.1e}  beta {errs[1]:.1e}")

emb = Embedding(81, 128).to(dev)
ref = torch.nn.Embedding(81, 128).to(dev)
with torch.no_grad():
    torch.nn.init.normal_(emb.weight)
    ref.weight.copy_(emb.weight)
idx = torch.randint(0, 81, (16, 64), device=dev)
g = torch.randn(16, 64, 128, device=dev)
(emb(idx) * g).sum().backward()
(ref(idx) * g).sum().backward()
e = rel(emb.weight.grad, ref.weight.grad)
if e >= 1e-4:
    failures.append("Embedding weight grad")
print(f"{'OK  ' if e < 1e-4 else 'FAIL'} {'Embedding weight grad':<46} {e:.1e}")

print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
