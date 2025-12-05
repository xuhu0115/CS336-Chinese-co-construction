import torch
import torch.nn as nn
import torch.nn.functional as F

# 按字节切分输入
def byte_tokenize(text):
    return list(text.encode("utf-8"))

# 可学习的Byte Embedding
class ByteEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.embed = nn.Embedding(256, dim)
    def forward(self, byte_ids):
        return self.embed(byte_ids)

# Expert FFN
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
    def forward(self, x):
        return self.ffn(x)

# EC MoE
class EC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(self, x, tokens=None, verbose=False):
        B_total, D = x.shape
        gate_scores = F.softmax(self.router(x), dim=-1)  # [B*L, E]

        # EC：每个专家选择top-ktoken
        scores_T = gate_scores.transpose(0, 1)  # [E, B*L]
        topk_scores, topk_idx = scores_T.topk(min(self.k, B_total), dim=-1)

        # dispatch table (token→expert权重)
        dispatch_weights = x.new_zeros((B_total, self.num_experts))
        for e in range(self.num_experts):
            for t_idx, s in zip(topk_idx[e].tolist(), topk_scores[e].tolist()):
                dispatch_weights[t_idx, e] = s

        # 可视化路由
        if verbose and tokens is not None:
            self.visualize(tokens, topk_idx, topk_scores)

        # 专家负载（每个专家处理多少token）
        expert_load = (dispatch_weights > 0).sum(dim=0)  # [E]

        print("\n========== 专家负载统计（Expert → Tokens） ==========")
        for e in range(self.num_experts):
            print(f"Expert {e}: {int(expert_load[e])} tokens")

        # 统计未被任何专家处理过的token（同字符只输出一次）
        used_tokens = (dispatch_weights.sum(dim=1) > 0)  # [B]
        unused_indices = (~used_tokens).nonzero(as_tuple=False).squeeze().tolist()

        print("\n========== 未被任何专家处理的 Token（对应字符不同） ==========")

        if isinstance(unused_indices, int):
            unused_indices = [unused_indices]

        if len(unused_indices) == 0:
            print("所有token都被至少1个专家处理。")
        else:
            printed_chars = set()
            for idx in unused_indices:
                ch = tokens[idx]
                if ch not in printed_chars:
                    printed_chars.add(ch)
                    print(f"Token {idx}: '{ch}'")

        # 计算专家输出
        out = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = (dispatch_weights[:, e_id] > 0).float().unsqueeze(1)
            if mask.sum() == 0:
                continue
            expert_out = expert(x * mask)
            out += expert_out * dispatch_weights[:, e_id].unsqueeze(1)

        return out

    # 路由可视化（专家选token）
    def visualize(self, tokens, topk_idx, topk_scores):
        E = topk_idx.shape[0]
        print("\n========== EC MoE 路由可视化 ==========\n")
        for e in range(E):
            print(f"Expert {e} 选择的 token:")
            for t_idx, s in zip(topk_idx[e].tolist(), topk_scores[e].tolist()):
                print(f"   Token {t_idx:3d}: '{tokens[t_idx]}'  score={s:.4f}")
            print("-"*40)


# 运行示例（Batch > 1）
if __name__ == "__main__":
    sentences = ["MoE是很强大的机制！", "专家混合模型非常高效。"]

    # byte-tokenize + padding
    byte_lists = [byte_tokenize(s) for s in sentences]
    max_len = max(len(b) for b in byte_lists)
    padded_bytes = [bl + [0]*(max_len - len(bl)) for bl in byte_lists]
    byte_ids = torch.tensor(padded_bytes)  # [B, L]

    B, L = byte_ids.shape

    embed = ByteEmbedding(dim=32)
    x = embed(byte_ids)                 # [B, L, D]
    x_flat = x.reshape(B * L, -1)       # [B*L, D]

    # 展平的token字符，用于打印
    tokens_flat = []
    for sen in sentences:
        for ch in sen:
            byte_len = len(ch.encode("utf-8"))  # 中文会是3
            tokens_flat.extend([ch] * byte_len)
        # padding：补足字节长度
        used_bytes = sum(len(ch.encode("utf-8")) for ch in sen)
        tokens_flat.extend(["<pad>"] * (max_len - used_bytes))

    ec_moe = EC_MoE(dim=32, num_experts=10, k=2)
    out_flat = ec_moe(x_flat, tokens=tokens_flat, verbose=True)

    out = out_flat.reshape(B, L, -1)
    print("\nEC MoE输出shape:", out.shape)
