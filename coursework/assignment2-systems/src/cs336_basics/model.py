from __future__ import annotations

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int


from .nn_utils import softmax

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """使用截断正态分布fan-in fan-out初始化的线性层。

        Args:
            d_in: int
                输入特征的数量。
            d_out: int
                输出特征的数量。
        """
        
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        std = 1.0
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(vocab_size, d_model), std=std, a=-3 * std, b=3 * std),
            requires_grad=True
        )
    
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]
    
    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


class RMSNorm(nn.Module):
    """
    此模块实现根均方层归一化，如
    https://arxiv.org/abs/1910.07467 中方程4所述

    Args:
        hidden_size: int
            要归一化的输入维度。
        eps: float, 默认值为1e-5
            为数值稳定性添加到分母中的值。

    Returns:
        与输入相同形状的FloatTensor。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                要应用根均方层归一化的输入。

        Returns:
            与输入相同形状的FloatTensor
        """
        # 注意：在实际中，许多实现会
        # 在此处手动将输入转换为fp32，以防止在
        # 平方输入时发生溢出。
        # https://github.com/pytorch/pytorch/issues/66707
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        return (self.weight * x).to(in_dtype)
    
    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 应用于x中对的2D旋转矩阵
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"


class BasicsTransformerLM(nn.Module):
    """一个Transformer语言模型。

    Args:
        vocab_size: int
            输出词汇表中要预测的唯一项目数量。
        context_length: int,
            一次处理的最大token数量。
        d_model: int
            模型嵌入和子层输出的维度。
        num_layers: int
            要使用的Transformer层数。
        num_heads: int
            多头注意力中使用的头数。`d_model`必须
            能被`num_heads`整除。
        d_ff: int
            前馈内层（3.3节）的维度。
        rope_theta: float
            RoPE位置编码的theta值。

    Returns:
        形状为（batch size, sequence_length, vocab_size）的FloatTensor，包含每个token的
        预测非标准化下一个词分布。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        # 存储模型配置以用于序列化/反序列化
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length,
            dim=d_head,
            theta=rope_theta
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        # 报告参数数量
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        返回模型中参数的数量。
        对于非嵌入计数（默认），会减去lm_head参数。
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()

        return n_params

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        """
        Args:
            x: 用于语言建模的输入ID。

        Returns: 形状为
            (batch size, sequence_length, vocab_size)的FloatTensor，包含每个token的预测非标准化下一个词
            分布。
        """
        _, sequence_length = x.size()

        # (batch size, sequence_length, d_model)
        x = self.token_embeddings(x)

        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)

        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)

        # (batch size, sequence_length, vocab_size)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                生成时用于条件化的输入ID。
            max_new_tokens: int
                要生成的最大token数量。
            temperature: float
                生成期间使用的温度。
            top_k: int
                如果提供，仅从`top_k`词汇项目（按概率）中采样。
            eos_token_id: int
                如果提供，在生成此ID时停止生成。

        Returns: 形状为(max_new_tokens,)的LongTensor，包含生成的模型输出。
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # 如果输入超过模型的上下文长度，
            # 取最后的`context_length`个token
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # 从模型获取logits
            logits = self.forward(x)
            # 获取下一个token的logits
            next_token_logits = logits[:, -1]
            # 应用温度缩放
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # 如果提供了top-k，选择得分最高的token
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # 获取我们保留的第k个项目的分数——得分较低的项应该被遮蔽。
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # 如果我们看到EOS token ID，结束生成
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

        # 移除来自序列化编译模型的_orig_mod.前缀
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model


class TransformerBlock(nn.Module):
    """单个Transformer层。

    此实现Transformer的单个层，如论文3.1节
    所述。

    Args:
        d_model: int
            模型嵌入和子层输出的维度。
        num_heads: int
            多头注意力中使用的头数。`d_model`必须
            能被`num_heads`整除。
        d_ff: int
            前馈内层（3.3节）的维度。
        positional_encoder: RotaryEmbedding
            要使用的RoPE模块。

    Returns:
        形状为`(batch_size, sequence_length, d_model)`的FloatTensor。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                用Transformer块处理的输入。

        Returns:
            形状为`(batch_size, sequence_length, d_model)`的FloatTensor。
        """
        # 注意：这是一个pre-norm Transformer，与论文中的原始
        # 描述不同。
        # 应用多头自注意力子层
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        # 应用前馈子层
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """缩放点积注意力。

    此函数实现了Transformer论文中的公式1。

    Args:
        Q: 查询张量，可以有任意数量的前导维度。
        K: 键张量，与Q共享前导维度。
        V: 值张量，与Q和K共享前导维度。
        mask: 一个（可选的）形状为(..., seq_len, seq_len)的掩码。
            对于掩码值为`False`的位置，注意力分数应该
            被屏蔽，即不影响softmax后的注意力概率。

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        使用提供的键、查询和值张量运行缩放点积注意力
        实现的输出。
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # 在键维度上应用softmax

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    """多头自注意力

    此函数实现了Transformer论文的3.2.2节。特别是，
    给定形状为`(batch_size, sequence_length, d_model)`的输入张量，我们投影
    它以创建查询、键和值，然后对这些查询、键和值执行因果多头注意力。

    Args:
        d_model: int
            模型嵌入和子层输出的维度。
        num_heads: int
            多头注意力中使用的头数。`d_model`必须
            能被`num_heads`整除。
        positional_encoder: RotaryEmbedding
            要使用的RoPE模块。

    Returns:
        形状为`(batch_size, sequence_length, d_model)`的张量。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE位置编码

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: 执行多头自注意力的输入。
            token_positions: 输入嵌入沿序列维度的位置索引。

        Returns:
            自注意力输出。
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 从Q、K、V的嵌入维度中分解每个头，形状为(..., num_heads, seq_len, d_k)。
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )  # fmt: skip

        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

        # 为每个头复制token位置
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # 构造因果掩码
        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)

        # 形状: (..., num_heads, sequence_length, d_k)
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        # 连接所有头的注意力输出。
        # (..., sequence_length, num_heads * d_v)。
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # 应用输出投影
        output = self.output_proj(attn_output)
        return output

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
