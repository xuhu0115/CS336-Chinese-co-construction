import torch
import torch.nn as nn
import torch.nn.functional as F

# 按字节切分输入
def byte_tokenize(text):
    # 将文本编码为UTF-8字节序列并转换为列表
    return list(text.encode("utf-8"))


# 可学习的 Byte Embedding
class ByteEmbedding(nn.Module):
    def __init__(self, dim=32):
        # 初始化父类
        super().__init__()
        # 创建词嵌入层，词汇表大小为256（字节值范围0-255），嵌入维度为dim
        self.embed = nn.Embedding(256, dim)

    def forward(self, byte_ids):
        # 将字节ID转换为嵌入向量
        return self.embed(byte_ids)


# Expert FFN
class Expert(nn.Module):
    def __init__(self, dim):
        # 初始化父类
        super().__init__()
        # 定义专家前馈网络：线性层 -> ReLU激活 -> 线性层
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 将维度扩展4倍
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(dim * 4, dim)  # 将维度恢复为原始维度
        )

    def forward(self, x):
        # 前向传播：通过前馈网络
        return self.ffn(x)

# TC MoE
class TC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        # 初始化父类
        super().__init__()
        # 设置专家数量
        self.num_experts = num_experts
        # 设置每个token选用的专家数量（top-k）
        self.k = k
        # 路由器：将输入映射到专家分数
        self.router = nn.Linear(dim, num_experts)
        # 创建专家模块列表
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

    def forward(self, x, tokens=None, verbose=False):
        # 获取批量大小和特征维度
        B, D = x.shape
        # 计算每个专家对每个token的分数，使用softmax归一化
        gate_scores = F.softmax(self.router(x), dim=-1)  # [B, E]
        # token选取分数最高的k个专家及其分数
        # 返回的两个张量是token选中的Top-K专家信息，从左到右形状分别为：[batch,TopE_scores]、[batch,TopE_index]
        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)

        # 如果verbose为True且提供了tokens，则显示路由可视化
        if verbose and tokens is not None:
            self.visualize(tokens, gate_scores, topk_idx, topk_scores)

        # topk_idx: [B, k]展平后统计每个专家被选中的次数
        expert_load = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts)
        # 打印负载统计标题
        print("\n========== 专家负载统计（Token → Expert） ==========")
        # 打印每个专家处理的token数量
        for e in range(self.num_experts):
            print(f"Expert {e}: {expert_load[e].item()} tokens")

        # 打印直方图标题
        print("\n---------- 直方图（越长表示处理 token 越多） ----------")
        # 获取最大负载用于归一化显示
        max_load = expert_load.max().item()
        # 为每个专家绘制ASCII直方图
        for e in range(self.num_experts):
            # 根据负载比例绘制条形图
            bar = "█" * int((expert_load[e].item() / max_load) * 30) if max_load > 0 else ""
            print(f"E{e}: {bar}")
        print("-----------------------------------------------------\n")

        # 初始化输出张量
        out = torch.zeros_like(x)
        # 遍历每个top-k专家
        for i in range(self.k):
            # 获取当前第i个专家ID
            expert_ids = topk_idx[:, i]
            # 获取当前第i个专家权重
            expert_weight = topk_scores[:, i]
            # 初始化专家输出
            expert_output = torch.zeros_like(x)
            # 遍历所有专家模块
            for e_id, expert in enumerate(self.experts):
                # 创建掩码：标记哪些token应该被当前专家处理
                mask = (expert_ids == e_id).float().unsqueeze(1)
                # 如果没有token被分配给当前专家，跳过
                if mask.sum() == 0:
                    continue
                # 应用专家前馈网络，只处理掩码标记的token
                expert_output += expert(x * mask)
            # 将专家输出按权重加权后累加到总输出
            out += expert_output * expert_weight.unsqueeze(1)

        # out本质是各个token在对应Top-K专家的特征空间的语义表征它融合了最适合处理该token的k个专家的知识，比单一FFN能够表达更丰富、更专业的语义信息。
        return out



    # ----------------------------
    # 路由可视化
    # ----------------------------
    def visualize(self, tokens, gate_scores, topk_idx, topk_scores):
        # 获取批量大小和专家数量
        B, E = gate_scores.shape
        # 打印可视化标题
        print("\n========== TC MoE路由可视化 ==========\n")
        # 将张量转换为CPU并脱离计算图
        gate_scores = gate_scores.detach().cpu()
        topk_idx = topk_idx.detach().cpu()
        topk_scores = topk_scores.detach().cpu()

        # 遍历每个token
        for i in range(B):
            # 显示当前token对应的字符
            print(f"Token {i}: '{tokens[i]}'")
            # 显示所有专家得分
            print("  全部专家得分：")
            for e in range(E):
                print(f"    Expert {e:2d}: {gate_scores[i, e]:.4f}")
            # 显示top-k专家信息
            print("  Top-k 专家：")
            for k in range(topk_idx.size(1)):
                print(f"    Expert {topk_idx[i, k].item()} score={topk_scores[i, k]:.4f}")
            print("-" * 40)



# 运行示例（Batch >1）
if __name__ == "__main__":
    # 定义两个示例句子
    sentences = ["MoE 是很强大的机制！", "专家混合模型非常高效。"]

    # byte-tokenize + padding
    # 将每个句子转换为字节列表
    byte_lists = [byte_tokenize(s) for s in sentences]
    # 找到最大序列长度
    max_len = max(len(b) for b in byte_lists)
    # 对每个字节列表进行填充，使其长度一致
    padded_bytes = [bl + [0] * (max_len - len(bl)) for bl in byte_lists]
    # 将填充后的列表转换为PyTorch张量
    byte_ids = torch.tensor(padded_bytes)

    # 获取批量大小和序列长度
    B, L = byte_ids.shape

    # token embedding
    # 创建字节嵌入层
    embed = ByteEmbedding(dim=32)
    # 将字节ID转换为嵌入向量 [B, L, D]
    x = embed(byte_ids)
    # 展平为二维张量 [B*L, D]
    x_flat = x.reshape(B * L, -1)

    # 展平后的 token 字符
    # 为每个字节创建对应的字符表示
    tokens_flat = []
    for s in sentences:
        # 遍历句子中的每个字符
        for ch in s:
            # 对于每个字符的每个字节，添加字符表示
            for _ in ch.encode("utf-8"):
                tokens_flat.append(ch)
        # 添加填充标记
        tokens_flat += ["<pad>"] * (max_len - len(s.encode("utf-8")))

    # TC MoE
    # 创建TC MoE层
    tc_moe = TC_MoE(dim=32, num_experts=15, k=2)
    # 前向传播，verbose=True显示路由信息
    out_flat = tc_moe(x_flat, tokens=tokens_flat, verbose=True)
    # 将输出重新整形为三维张量 [B, L, D],第一个B表示处理文本数量即批次大小
    out = out_flat.reshape(B, L, -1)
    # 打印输出形状
    print("\nTC MoE输出shape:", out.shape)
