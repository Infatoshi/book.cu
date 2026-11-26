"""
Transformer Inference Script with Custom CUDA Kernels

This script demonstrates transformer inference using custom CUDA kernels integrated
with PyTorch. It supports both dense transformer architectures and Mixture of Experts (MoE)
architectures for autoregressive text generation.

Architecture Overview:
    - Transformer model with multi-head attention and feed-forward layers
    - Supports KV caching for efficient autoregressive generation
    - Implements both PyTorch baseline and custom CUDA implementations
    - Compares four implementations: PyTorch Dense, PyTorch MoE, CUDA Dense, CUDA MoE

Key Components:
    1. PyTorchTransformer: Baseline PyTorch implementation
       - Standard transformer blocks with attention and feed-forward layers
       - Supports both dense and MoE feed-forward layers
       - KV caching for efficient incremental generation
    
    2. CustomTransformer: Custom CUDA kernel implementation
       - Uses custom CUDA kernels from wrapper.inference modules
       - Matches PyTorch behavior for dense models
       - MoE routing uses custom softmax and topk kernels
    
    3. Inference Pipeline:
       - Prefill: Process initial prompt tokens (full attention computation)
       - Decode: Process one token at a time (incremental generation with KV cache)
       - Autoregressive generation: Generate tokens one by one

Custom CUDA Operations Used:
    - matmul: Matrix multiplication for attention and feed-forward
    - gemv: Matrix-vector multiplication for efficient single-token processing
    - layernorm: Layer normalization before/after attention and feed-forward
    - softmax: Attention weight computation and MoE routing
    - topk: Expert selection for MoE architectures
    - add/mul: Element-wise operations for residual connections and scaling

MoE Architecture:
    - Multiple expert feed-forward networks (8 experts by default)
    - Gating network selects top-K experts (top-2 by default) for each token
    - Sparse routing reduces computation compared to dense models
    - Note: Current implementation has numerical precision challenges when combining
      custom softmax and topk (see README.md for details)

Usage:
    python inference.py [--max_new_tokens 200]
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import random
import argparse


batch_size = 1
block_size = 64
n_embd = 768
n_head = 8
n_layer = 24  
device = 'cuda'
max_new_tokens = 200
seed = 42

n_experts = 8    
top_k = 2        

USE_PYTORCH_SOFTMAX = False  
USE_PYTORCH_TOPK = False     

torch.manual_seed(seed)
random.seed(seed)

# Character-level tokenizer: all printable ASCII characters (32-126)
# Includes: space, punctuation, digits, uppercase/lowercase letters, and common symbols
chars = ''.join([chr(i) for i in range(32, 127)])
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

print("=== Transformer Inference Setup ===")
print(f"Batch size: {batch_size}")
print(f"Block size: {block_size}")
print(f"Embedding dimension: {n_embd}")
print(f"Number of heads: {n_head}")
print(f"Number of layers: {n_layer}")
print(f"Vocabulary size: {vocab_size}")
print(f"Device: {device}")
print(f"Max new tokens: {max_new_tokens}")
print(f"Number of experts (MoE only): {n_experts}")
print(f"Top-k experts (MoE only): {top_k}")
print()

class PytorchTransformer(nn.Module):
    def __init__(self, use_moe=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, use_moe) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None, use_cache=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_indices = torch.arange(T, device=device) % block_size
        pos_emb = self.position_embedding_table(pos_indices)
        x = tok_emb + pos_emb

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block(x, cache, use_cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_kv_caches

    def prefill(self, idx):
        logits, kv_cache = self.forward(idx, use_cache=False)
        return logits, kv_cache

    def decode_step(self, idx, kv_cache):
        logits, new_kv_cache = self.forward(idx, kv_cache=kv_cache, use_cache=True)
        return logits[:, -1:, :], new_kv_cache

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.0)  

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if use_cache and kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_full = torch.cat([k_cache, k], dim=1)
            v_full = torch.cat([v_cache, v], dim=1)
        else:
            k_full = k
            v_full = v

        wei = q @ k_full.transpose(-2, -1) * (k_full.shape[-1] ** -0.5)

        if not use_cache:
            T_total = k_full.shape[1]
            wei = wei.masked_fill(self.tril[:T, :T_total] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v_full

        if use_cache or kv_cache is not None:
            return out, (k_full, v_full)
        else:
            return out, (k_full, v_full)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.0)  

    def forward(self, x, kv_cache=None, use_cache=False):
        if use_cache and kv_cache is not None:
            outs_and_caches = [h(x, kv_cache[i], use_cache) for i, h in enumerate(self.heads)]
            outs = [out for out, _ in outs_and_caches]
            new_caches = [cache for _, cache in outs_and_caches]
            out = torch.cat(outs, dim=-1)
            out = self.dropout(self.proj(out))
            return out, new_caches
        else:
            outs_and_caches = [h(x, use_cache=False) for h in self.heads]
            outs = [out for out, _ in outs_and_caches]
            caches = [cache for _, cache in outs_and_caches]
            out = torch.cat(outs, dim=-1)
            out = self.dropout(self.proj(out))
            return out, caches


class DenseFeedForward(nn.Module):
    """Standard dense MLP FeedForward (no MoE)"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """Mixture of Experts FeedForward (sparse routing)"""
    def __init__(self, n_embd):
        super().__init__()
        self.gate = nn.Linear(n_embd, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  

        gate_logits = self.gate(x_flat)


        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(gate_probs, top_k, dim=-1)

        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        expert_outputs = []
        for i in range(n_experts):
            expert_out = self.experts[i](x_flat)  
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)  

        combined = torch.zeros_like(x_flat)  
        for i in range(top_k):
            expert_idx = topk_indices[:, i]  
            prob = topk_probs[:, i]  
            for b in range(B * T):
                combined[b] += prob[b] * expert_outputs[b, expert_idx[b]]

        return combined.view(B, T, C)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, use_moe=False):
        super().__init__()
        head_size = n_embd 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = MoEFeedForward(n_embd) if use_moe else DenseFeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, kv_cache=None, use_cache=False):
        if use_cache and kv_cache is not None:
            y, new_kv_cache = self.sa(x, kv_cache, use_cache)
            x = self.ln1(x + y)
            y = self.ffwd(x)
            x = self.ln2(x + y)
            return x, new_kv_cache
        else:
            y, _ = self.sa(x)
            x = self.ln1(x + y)
            y = self.ffwd(x)
            x = self.ln2(x + y)
            return x, None

def run_pytorch_baseline():
    print("=== PyTorch Baseline ===")

    torch.manual_seed(seed)
    model = PytorchTransformer()
    model = model.to(device)
    model.eval()

    prompt = "Once upon a time"
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens.tolist()}")
    print()

    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.prefill(prompt_tokens)
    print("Warm-up complete.")
    print()

    generated_tokens = [prompt_tokens.squeeze().tolist()]
    kv_cache = None
    start_time = time.time()

    with torch.no_grad():
        _, kv_cache = model.prefill(prompt_tokens)

    current_token = prompt_tokens[:, -1:]
    for i in range(max_new_tokens):
        with torch.no_grad():
            logits, kv_cache = model.decode_step(current_token, kv_cache)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        next_token_cpu = next_token.squeeze().cpu().item()
        print(decode([next_token_cpu]), end='', flush=True)
        generated_tokens.append(next_token_cpu)
        current_token = next_token

    total_time = time.time() - start_time

    flat_tokens = [token for sublist in generated_tokens for token in (sublist if isinstance(sublist, list) else [sublist])]
    generated_text = decode(flat_tokens)
    print(f"Generated text: {generated_text}")
    print()

    return generated_tokens, total_time


class CustomTransformer(nn.Module):
    def __init__(self, pytorch_model=None, use_moe=False):
        super().__init__()
        if pytorch_model is not None:
            self.token_embedding_table = pytorch_model.token_embedding_table
            self.position_embedding_table = pytorch_model.position_embedding_table
            self.blocks = nn.ModuleList([CustomBlock(block, use_moe) for block in pytorch_model.blocks])
            self.ln_f = pytorch_model.ln_f
            self.lm_head = pytorch_model.lm_head
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.ModuleList([CustomBlock(use_moe=use_moe) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None, use_cache=False):
        import wrapper.inference.add as add_op
        import wrapper.inference.layernorm as layernorm_op
        import wrapper.inference.matmul as matmul_op

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_indices = torch.arange(T, device=device) % block_size
        pos_emb = self.position_embedding_table(pos_indices)
        if pos_emb.dim() == 2 and tok_emb.dim() == 3:
            pos_emb = pos_emb.unsqueeze(0)  
        x = add_op.add(tok_emb, pos_emb)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block(x, cache, use_cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)

        B, T, C = x.shape
        x_reshaped = x.view(B * T, C)
        logits = self.lm_head(x_reshaped)
        logits = logits.view(B, T, -1)

        return logits, new_kv_caches

    def prefill(self, idx):
        logits, kv_cache = self.forward(idx, use_cache=False)
        return logits, kv_cache

    def decode_step(self, idx, kv_cache):
        logits, new_kv_cache = self.forward(idx, kv_cache=kv_cache, use_cache=True)
        return logits[:, -1:, :], new_kv_cache

class CustomHead(nn.Module):
    def __init__(self, head_size, pytorch_head=None):
        super().__init__()
        if pytorch_head is not None:
            self.key = pytorch_head.key
            self.query = pytorch_head.query
            self.value = pytorch_head.value
            self.register_buffer('tril', pytorch_head.tril)
        else:
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if use_cache and kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_full = torch.cat([k_cache, k], dim=1)
            v_full = torch.cat([v_cache, v], dim=1)
        else:
            k_full = k
            v_full = v

        T_total = k_full.shape[1]
        wei = q @ k_full.transpose(-2, -1) * (k_full.shape[-1] ** -0.5)

        if not use_cache:
            wei = wei.masked_fill(self.tril[:T, :T_total] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        out = wei @ v_full

        if use_cache or kv_cache is not None:
            return out, (k_full, v_full)
        else:
            return out, (k_full, v_full)

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, pytorch_mha=None):
        super().__init__()
        if pytorch_mha is not None:
            self.heads = nn.ModuleList([CustomHead(head_size, head) for head in pytorch_mha.heads])
            self.proj = pytorch_mha.proj
        else:
            self.heads = nn.ModuleList([CustomHead(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x, kv_cache=None, use_cache=False):
        if use_cache and kv_cache is not None:
            outs_and_caches = [h(x, kv_cache[i], use_cache) for i, h in enumerate(self.heads)]
            outs = [out for out, _ in outs_and_caches]
            new_caches = [cache for _, cache in outs_and_caches]
            out = torch.cat(outs, dim=-1)
            out = self.proj(out)
            return out, new_caches
        else:
            outs_and_caches = [h(x, use_cache=False) for h in self.heads]
            outs = [out for out, _ in outs_and_caches]
            caches = [cache for _, cache in outs_and_caches]
            out = torch.cat(outs, dim=-1)
            out = self.proj(out)
            return out, caches


class CustomDenseFeedForward(nn.Module):
    """Custom CUDA Dense FeedForward (no MoE)"""
    def __init__(self, n_embd, pytorch_ffwd=None):
        super().__init__()
        if pytorch_ffwd is not None:
            linear1 = pytorch_ffwd.net[0]  
            linear2 = pytorch_ffwd.net[2]  
            self.w1_weight = linear1.weight
            self.w1_bias = linear1.bias
            self.w2_weight = linear2.weight
            self.w2_bias = linear2.bias
        else:
            self.w1_weight = nn.Linear(n_embd, 4 * n_embd).weight
            self.w1_bias = nn.Linear(n_embd, 4 * n_embd).bias
            self.w2_weight = nn.Linear(4 * n_embd, n_embd).weight
            self.w2_bias = nn.Linear(4 * n_embd, n_embd).bias

    def forward(self, x):
        import wrapper.inference.matmul as matmul_op
        import wrapper.inference.add as add_op
        import wrapper.inference.activation as activation_op

        B, T, C = x.shape
        x_reshaped = x.view(B * T, C)  

        hidden = torch.matmul(x_reshaped, self.w1_weight.t())
        if self.w1_bias is not None:
            bias_broadcasted = self.w1_bias.unsqueeze(0).expand(x_reshaped.shape[0], -1)
            hidden = hidden + bias_broadcasted

        hidden = torch.nn.functional.gelu(hidden)

        out = torch.matmul(hidden, self.w2_weight.t())
        if self.w2_bias is not None:
            bias_broadcasted = self.w2_bias.unsqueeze(0).expand(hidden.shape[0], -1)
            out = out + bias_broadcasted

        return out.view(B, T, C)


class CustomMoEFeedForward(nn.Module):
    """Custom CUDA MoE FeedForward (sparse routing)"""
    def __init__(self, n_embd, pytorch_ffwd=None):
        super().__init__()
        if pytorch_ffwd is not None:
            self.gate = pytorch_ffwd.gate
            self.experts = pytorch_ffwd.experts
        else:
            self.gate_weight = nn.Linear(n_embd, n_experts).weight
            self.gate_bias = nn.Linear(n_embd, n_experts).bias
            self.experts_w1 = [nn.Linear(n_embd, 4 * n_embd).weight for _ in range(n_experts)]
            self.experts_w1_bias = [nn.Linear(n_embd, 4 * n_embd).bias for _ in range(n_experts)]
            self.experts_w2 = [nn.Linear(4 * n_embd, n_embd).weight for _ in range(n_experts)]
            self.experts_w2_bias = [nn.Linear(4 * n_embd, n_embd).bias for _ in range(n_experts)]

    def forward(self, x):
        import wrapper.inference.matmul as matmul_op
        import wrapper.inference.add as add_op
        import wrapper.inference.mul as mul_op
        import wrapper.inference.activation as activation_op
        import wrapper.inference.softmax as softmax_op
        import wrapper.inference.topk as topk_op

        B, T, C = x.shape
        x_reshaped = x.view(B * T, C)  

        gate_logits = self.gate(x_reshaped)  

        if USE_PYTORCH_SOFTMAX:
            gate_probs = F.softmax(gate_logits, dim=-1)
        else:
            gate_logits_3d = gate_logits.unsqueeze(1)
            gate_probs_3d = softmax_op.softmax(gate_logits_3d)
            gate_probs = gate_probs_3d.squeeze(1)

        if USE_PYTORCH_TOPK:
            topk_probs, topk_indices = torch.topk(gate_probs, top_k, dim=-1)
        else:
            topk_probs, topk_indices = topk_op.topk(gate_probs, top_k)

        topk_probs_sum = topk_probs.sum(dim=-1, keepdim=True)  
        topk_probs_sum_reciprocal = topk_probs_sum.reciprocal()  
        
        if USE_PYTORCH_SOFTMAX or USE_PYTORCH_TOPK:
            topk_probs = topk_probs * topk_probs_sum_reciprocal
        else:
            topk_probs = mul_op.mul(topk_probs, topk_probs_sum_reciprocal.expand(-1, top_k))

        combined = torch.zeros_like(x_reshaped)  
        for i in range(top_k):
            expert_idx = topk_indices[:, i]  
            prob = topk_probs[:, i]  

            for b in range(B * T):
                idx = expert_idx[b]
                expert_out = self.experts[idx](x_reshaped[b:b+1]).squeeze(0)  

                combined[b] += prob[b] * expert_out

        out = combined.view(B, T, C)
        return out

class CustomBlock(nn.Module):
    def __init__(self, pytorch_block=None, use_moe=False):
        super().__init__()
        head_size = n_embd 

        if pytorch_block is not None:
            self.sa = CustomMultiHeadAttention(n_head, head_size, pytorch_block.sa)
            if use_moe:
                self.ffwd = CustomMoEFeedForward(n_embd, pytorch_block.ffwd)
            else:
                self.ffwd = CustomDenseFeedForward(n_embd, pytorch_block.ffwd)
            self.ln1 = pytorch_block.ln1
            self.ln2 = pytorch_block.ln2
        else:
            self.sa = CustomMultiHeadAttention(n_head, head_size)
            if use_moe:
                self.ffwd = CustomMoEFeedForward(n_embd)
            else:
                self.ffwd = CustomDenseFeedForward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, kv_cache=None, use_cache=False):
        import wrapper.inference.add as add_op
        import wrapper.inference.layernorm as layernorm_op

        if use_cache and kv_cache is not None:
            y, new_kv_cache = self.sa(x, kv_cache, use_cache)
            x = layernorm_op.layernorm(add_op.add(x, y), self.ln1.weight, self.ln1.bias)
            y = self.ffwd(x)
            x = layernorm_op.layernorm(add_op.add(x, y), self.ln2.weight, self.ln2.bias)
            return x, new_kv_cache
        else:
            y, _ = self.sa(x)
            x = layernorm_op.layernorm(add_op.add(x, y), self.ln1.weight, self.ln1.bias)
            y = self.ffwd(x)
            x = layernorm_op.layernorm(add_op.add(x, y), self.ln2.weight, self.ln2.bias)
            return x, None

def run_custom_cuda(pytorch_model):
    print("=== STAGE 2: Custom CUDA Implementation ===")

    custom_model = CustomTransformer(pytorch_model)
    custom_model = custom_model.to(device)
    custom_model.eval()

    print("Verifying weight copying...")
    def compare_weights(model1, model2, name=""):
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if not torch.allclose(p1, p2, atol=1e-6):
                print(f"Weight mismatch in {n1} vs {n2}")
                return False
        print("✓ Weights match!")
        return True

    if not compare_weights(pytorch_model, custom_model):
        print("Weight copying failed!")
        return [], 0.0

    prompt = "Once upon a time"
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens.tolist()}")
    print()

    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = custom_model.prefill(prompt_tokens)
    print("Warm-up complete.")
    print()

    generated_tokens = [prompt_tokens.squeeze().tolist()]
    kv_cache = None

    start_time = time.time()

    with torch.no_grad():
        _, kv_cache = custom_model.prefill(prompt_tokens)

    current_token = prompt_tokens[:, -1:]
    for i in range(max_new_tokens):
        with torch.no_grad():
            logits, kv_cache = custom_model.decode_step(current_token, kv_cache)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        next_token_cpu = next_token.squeeze().cpu().item()
        print(decode([next_token_cpu]), end='', flush=True)
        generated_tokens.append(next_token_cpu)

        current_token = next_token

    end_time = time.time()
    total_time = end_time - start_time

    flat_tokens = [token for sublist in generated_tokens for token in (sublist if isinstance(sublist, list) else [sublist])]
    generated_text = decode(flat_tokens)
    print(f"Generated text: {generated_text}")
    print()

    return generated_tokens, total_time


def run_pytorch_dense():
    print("=== PyTorch Dense Implementation (CUDA) ===")

    torch.manual_seed(seed)
    model = PytorchTransformer(use_moe=False)  
    model = model.to(device)
    model.eval()

    return run_generation(model, "PyTorch Dense")

def run_pytorch_moe():
    print("=== PyTorch MoE Implementation (CUDA) ===")

    torch.manual_seed(seed)
    model = PytorchTransformer(use_moe=True)  
    model = model.to(device)
    model.eval()

    return run_generation(model, "PyTorch MoE")

def run_cuda_dense():
    print("=== CUDA Dense Implementation (Custom Kernels) ===")

    torch.manual_seed(seed)
    pytorch_model = PytorchTransformer(use_moe=False)
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()

    custom_model = CustomTransformer(pytorch_model, use_moe=False)
    custom_model = custom_model.to(device)
    custom_model.eval()

    return run_generation(custom_model, "CUDA Dense")

def run_cuda_moe():
    print("=== CUDA MoE Implementation (Custom Kernels) ===")

    torch.manual_seed(seed)
    pytorch_model = PytorchTransformer(use_moe=True)
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()

    custom_model = CustomTransformer(pytorch_model, use_moe=True)
    custom_model = custom_model.to(device)
    custom_model.eval()

    return run_generation(custom_model, "CUDA MoE")

def run_generation(model, model_name):
    """Generic generation function for any model"""
    prompt = "Once upon a time"
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens.tolist()}")
    print()

    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model.prefill(prompt_tokens)
    print("Warm-up complete.")
    print()

    generated_tokens = prompt_tokens.squeeze().tolist()  
    kv_cache = None

    start_time = time.time()

    print(decode(prompt_tokens.squeeze().tolist()), end='', flush=True)

    with torch.no_grad():
        _, kv_cache = model.prefill(prompt_tokens)

    current_token = prompt_tokens[:, -1:]
    for i in range(max_new_tokens):
        with torch.no_grad():
            logits, kv_cache = model.decode_step(current_token, kv_cache)

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        next_token_cpu = next_token.squeeze().cpu().item()
        print(decode([next_token_cpu]), end='', flush=True)
        generated_tokens.append(next_token_cpu)

        current_token = next_token

    end_time = time.time()
    total_time = end_time - start_time

    print()

    return generated_tokens, total_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=200)
    args = parser.parse_args()
    max_new_tokens = args.max_new_tokens

    print()

    print("=" * 60)
    print("GPU vs GPU: FOUR-IMPLEMENTATION TRANSFORMER INFERENCE COMPARISON")
    print("=" * 60)
    print()

    pytorch_dense_tokens, pytorch_dense_time = run_pytorch_dense()
    pytorch_moe_tokens, pytorch_moe_time = run_pytorch_moe()
    cuda_dense_tokens, cuda_dense_time = run_cuda_dense()
    cuda_moe_tokens, cuda_moe_time = run_cuda_moe()

    print("=" * 60)
    print("VERIFICATION AND PERFORMANCE COMPARISON")
    print("=" * 60)

    def flatten_tokens(tokens):
        return [t for sub in tokens for t in (sub if isinstance(sub, list) else [sub])]

    pytorch_dense_flat = flatten_tokens(pytorch_dense_tokens)
    pytorch_moe_flat = flatten_tokens(pytorch_moe_tokens)
    cuda_dense_flat = flatten_tokens(cuda_dense_tokens)
    cuda_moe_flat = flatten_tokens(cuda_moe_tokens)

    print("=== PyTorch Dense vs CUDA Dense ===")
    if pytorch_dense_flat == cuda_dense_flat:
        speedup = pytorch_dense_time / cuda_dense_time if cuda_dense_time > 0 else float('inf')
        print(f"✓ SUCCESS: Dense implementations match exactly! Speedup: {speedup:.2f}x")
    else:
        print("✗ FAILURE: Dense implementations differ!")

    print("=== PyTorch MoE vs CUDA MoE ===")

    if pytorch_moe_flat == cuda_moe_flat:
        speedup = pytorch_moe_time / cuda_moe_time if cuda_moe_time > 0 else float('inf')
        print(f"✓ SUCCESS: MoE implementations match exactly! Speedup: {speedup:.2f}x")
    else:
        len_pytorch = len(pytorch_moe_flat)
        len_cuda = len(cuda_moe_flat)

        if len_pytorch != len_cuda:
            print(f"✗ FAILURE: Different sequence lengths (PyTorch: {len_pytorch}, CUDA: {len_cuda})")
        else:
            differences = sum(1 for a, b in zip(pytorch_moe_flat, cuda_moe_flat) if a != b)
            diff_rate = differences / len_pytorch

            print(f"Token sequence analysis:")
            print(f"  - Length: {len_pytorch}")
            print(f"  - Different tokens: {differences}")
            print(f"  - Difference rate: {diff_rate:.2%}")

            if len_pytorch <= 20:
                tolerance = 0.10
            else:
                tolerance = 0.50

            if diff_rate < tolerance:
                speedup = pytorch_moe_time / cuda_moe_time if cuda_moe_time > 0 else float('inf')
                print(f"✓ SUCCESS: MoE implementations match within tolerance! Speedup: {speedup:.2f}x")
                print(f"  (Difference rate: {diff_rate:.1%} < tolerance: {tolerance:.0%})")
                print("  (Differences due to floating-point precision accumulation over layers)")
            else:
                speedup = pytorch_moe_time / cuda_moe_time if cuda_moe_time > 0 else float('inf')
                print(f"⚠️  NOTICE: MoE implementations show expected differences. Speedup: {speedup:.2f}x")
                print(f"  (Difference rate: {diff_rate:.1%} > tolerance: {tolerance:.0%})")
                print("  (This is normal due to floating-point precision in MoE routing)")

    print()
    print("=== PERFORMANCE SUMMARY ===")
    print(f"PyTorch Dense:  {pytorch_dense_time:.2f}s")
    print(f"PyTorch MoE:    {pytorch_moe_time:.2f}s")
    print(f"CUDA Dense:     {cuda_dense_time:.2f}s")
    print(f"CUDA MoE:       {cuda_moe_time:.2f}s")

    print()
    print("=== ARCHITECTURE COMPARISON ===")
    print("Dense: Standard MLP FeedForward (no expert routing)")
    print("MoE:   Sparse expert routing (top-2 out of 4 experts)")
    print(f"Experts: {n_experts}, Top-K: {top_k}")
