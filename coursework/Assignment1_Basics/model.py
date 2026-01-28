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

class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = int(8/3*d_model)
        d_ff = (d_ff+63)//64*64
        self.w_gate = Linear(d_model, d_ff)
        self.w_up   = Linear(d_model, d_ff)
        self.w_down = Linear(d_ff, d_model)

    def forward(self, x):
        gate = self.w_gate(x)
        swish = gate * torch.sigmoid(gate)
        return self.w_down(swish * self.w_up(x))

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
        self.ffn = SwiGLU(d_model)
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



class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        self.id_to_token = {i:t for t,i in self.token_to_id.items()}
        self.unk_id = self.token_to_id.get("<unk>", 0)

    def encode(self, text):
        return [self.token_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        return "".join([self.id_to_token.get(i, "<unk>") for i in ids])


@torch.no_grad()
def generate_with_sampling(
    model,
    idx,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    model.eval()

    # 根据./bpe_tokenizer/tokenizer.josn可以得到特殊字符的id为0、1、2、3、4，生成文本中需要阻止其生成
    special_ids = [0, 1, 2, 3,4]
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]

        logits = model(idx_cond)[:, -1, :]  # (B, V)

        # temperature防御
        temperature = max(temperature, 1e-5)
        logits = logits / temperature

        # top-k
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        # top-p
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)

            # 关键：数值稳定
            sorted_logits = sorted_logits - sorted_logits.max(dim=-1, keepdim=True).values

            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            mask = cum_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            sorted_logits[mask] = -float("inf")
            logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

        for i in special_ids:
            logits[:, i] = -float('inf')

        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            # print("[WARN] 检测到无效的概率，退回到均匀分布")
            probs = torch.ones_like(probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def decode_generated_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None, device='cpu'):
    model.eval()
    model.to(device)
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated_idx = generate_with_sampling(model, idx, max_new_tokens=max_new_tokens,
                                           temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(generated_idx[0].tolist())

class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None: continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Correct bias correction logic
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                denom = exp_avg_sq.sqrt().add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)
                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)
        return loss

# 使用示例
if __name__ == "__main__":

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="bpe_tokenizer/tokenizer.json"
    )

    vocab_size = tokenizer.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型的参数必须和训练模型的参数一致
    model = TransformerLM(vocab_size=vocab_size, d_model=256, num_heads=8, num_layers=6, max_seq_len=128)

    if os.path.exists("ckpt/epoch_5.pt"):
        #model.load_state_dict(torch.load("ckpt/epoch_5.pt", map_location=device))
        ckpt = torch.load("ckpt/epoch_5.pt", map_location=device)

        # 1. 模型
        model.load_state_dict(ckpt["model"])

        # 2. 优化器
        optimizer = CustomAdamW(model.parameters(), lr=2e-4, weight_decay=0.1)
        optimizer.load_state_dict(ckpt["optimizer"])

        # 3. 训练状态
        global_step = ckpt["iteration"]
        start_epoch = ckpt["epoch"]
        print(ckpt["config"])

        # 4. 配置一致性检查（非常重要）
        # assert ckpt["config"]["vocab_size"] == model.config.vocab_size
        # assert ckpt["config"]["context_length"] == model.config.context_length

        print(f"[INFO] 成功加载权重文件，使用设备: {device}")
    else:
        print("[WARNING] 权重文件不存在，使用随机初始化模型")

    model.to(device)
    model.eval()

    prompt = "Lily like play computer games"
    generated_text = decode_generated_text(model, tokenizer, prompt,
                                           max_new_tokens=60,
                                           temperature=0.8,
                                           top_k=None,
                                           top_p=0.9,
                                           device=device)
    print("Generated:", generated_text)
