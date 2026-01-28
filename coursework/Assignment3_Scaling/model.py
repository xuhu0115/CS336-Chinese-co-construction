from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        # 存储模型配置以便于序列化/反序列化
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.residual_pdrop = residual_pdrop
        # 报告参数数量
        logger.info(
            "非嵌入参数数量: %.2fM" % (self.get_num_params() / 1e6,)
        )

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
            n_params -= self.token_embeddings.weight.numel()
        return n_params

    def forward(self, x: torch.LongTensor):
        _, sequence_length = x.size()
        # (batch size, sequence_length, d_model)
        # 注意：论文提到“在嵌入层中，我们将这些权重乘以 sqrt(d_model)”，但我们这里没有这样做。
        embedded_tokens = self.token_embeddings(x)

        # 形状: (1, sequence_length)
        positions = torch.arange(
            0, sequence_length, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        # (1, sequence_length, d_model)
        embedded_positions = self.position_embeddings(positions)
        # (batch size, sequence_length, d_model)
        x = embedded_tokens + embedded_positions
        if self.residual_pdrop:
            # (batch size, sequence_length, d_model)
            x = F.dropout(x, self.residual_pdrop)
        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)
        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)
        # (batch size, sequence_length, vocab_size)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # 如果输入超过模型的上下文长度，取最后`context_length`个标记
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # 获取模型的logits
            logits = self.forward(x)
            # 获取下一个标记的logits
            next_token_logits = logits[:, -1]
            # 应用温度缩放
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # 如果提供了top-k，则只保留得分最高的标记
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # 获取保留的第 k 个项的分数——分数低于此阈值的项应被遮蔽（mask）。
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(
                    topk_mask, float("-inf")
                )
            next_token_probabilities = F.softmax(
                temperature_scaled_next_token_logits, dim=-1
            )
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # 如果看到EOS标记ID，则结束生成
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attn_pdrop if attn_pdrop else 0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.residual_pdrop = residual_pdrop

    def forward(self, x: torch.FloatTensor):
        # 应用多头自注意力子层
        x_ln = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x_attn = self.attn(
            x_ln, x_ln, x_ln, need_weights=False, attn_mask=causal_mask, is_causal=True
        )[0]
        if self.residual_pdrop is not None:
            x_attn = F.dropout(x_attn, self.residual_pdrop)
        attn_sublayer_output = x + x_attn

        # 应用前馈网络子层
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        if self.residual_pdrop is not None:
            x_ffn = F.dropout(x_ffn, self.residual_pdrop)
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class FFN(nn.Module):
    """前馈网络（Feed-Forward Network）"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x

def compute_params_from_config(config: dict, non_embedding: bool = True):
    model = BasicsTransformerLM(**config)
    n_params = model.get_num_params(non_embedding=non_embedding)
    total = sum(p.numel() for p in model.parameters())
    n_embd = total - n_params
    return n_params, n_embd

if __name__ == "__main__":

    config = {
        "vocab_size": 32000,
        "context_length": 2048,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
        "attn_pdrop": 0.1,
        "residual_pdrop": 0.1,
    }

    params, n_embd = compute_params_from_config(config)
    print(f"Non-embedding params: {params / 1e6:.2f}M")
    print(f"Embedding params: {n_embd / 1e6:.2f}M")
