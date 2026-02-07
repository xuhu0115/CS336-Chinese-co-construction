import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from transformers import PreTrainedTokenizerFast

# 基础组件
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        sigma = math.sqrt(2.0 / (self.weight.shape[0] + self.weight.shape[1]))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x):
        return x @ self.weight.t()

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x_fp32 * rms * self.weight.to(torch.float32)
        return x_normed.to(orig_dtype)

class SiLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(8/3*d_model)
        d_ff = (d_ff+63)//64*64
        self.f_c1 = Linear(d_model, d_ff)
        self.f_c2 = Linear(d_ff, d_model)
    def forward(self, x):
        return self.f_c2(F.silu(self.f_c1(x)))
# RoPE
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int):
        super().__init__()
        assert d_k % 2 == 0, "d_k必须为偶数"
        self.d_k = d_k
        self.theta = theta
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    def _build_cache(self, seq_len, device, dtype):
        if self.cos is not None and seq_len <= self.cos.shape[0]:
            return
        i = torch.arange(0, self.d_k, 2, device=device)
        inv_freq = self.theta ** (-i / self.d_k)
        t = torch.arange(seq_len, device=device)
        freqs = torch.outer(t, inv_freq).to(dtype)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,H,T,D = x.shape
        self._build_cache(T, x.device, x.dtype)
        cos = self.cos[:T, :].view(1,1,T,D//2)
        sin = self.sin[:T, :].view(1,1,T,D//2)
        x_even = x[...,0::2]
        x_odd  = x[...,1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        x_rot = torch.zeros_like(x)
        x_rot[...,0::2] = x_rot_even
        x_rot[...,1::2] = x_rot_odd
        return x_rot

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

        # flash attention可以对GPU加速
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.rope = RotaryPositionalEmbedding(10000.0, self.d_k)

    def forward(self, x, mask):
        B,T,_ = x.shape
        q = self.w_q(x).view(B,T,self.num_heads,self.d_k).transpose(1,2)
        k = self.w_k(x).view(B,T,self.num_heads,self.d_k).transpose(1,2)
        v = self.w_v(x).view(B,T,self.num_heads,self.d_k).transpose(1,2)

        q = self.rope(q)
        k = self.rope(k)

        if self.flash:
            # flash attention输出结果直接就是Q、K点积结果和V的乘积
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0,
                is_causal=True   # 表示自回归自动生成下三角全为True的因果掩码（mask）矩阵
            )
        else:
            scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)
            if mask is not None:
                mask = mask.expand(B, self.num_heads, T, T)
                scores = scores.masked_fill(~mask, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,T,-1)
        return self.w_o(out)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SiLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

# TransformerLM
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads) for _ in range(num_layers)])
        self.norm_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

        self.config = {
            "vocab_size": vocab_size,
            "context_length": max_seq_len,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_model": d_model,
        }

    def forward(self, idx):
        B,T = idx.shape
        x = self.token_embedding(idx)
        mask = torch.tril(torch.ones((B,1,T,T), device=idx.device, dtype=torch.bool))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:,-self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:,-1,:] / temperature
            if top_k is not None:
                v,_ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
