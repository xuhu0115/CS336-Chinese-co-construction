# 第五章 专家混合模型
专家混合模型（MoE）是当前LLM领域中一项至关重要的技术，它有效地解决了模型规模与计算成本之间的矛盾。这种机制允许模型在不显著增加训练和推理计算量的前提下，大幅扩展其总参数规模和表达能力，从而实现了模型容量（参数数量）与计算效率之间的动态平衡。正是凭借MoE这一机制，Switch Transformer、DeepSeek等模型得以问世并展现出卓越性能。然而，MoE在实际应用中会有负载均衡、跨设备并行、训练不稳定和路由机制设计等工程挑战。接下来，我们将剖析MoE的核心概念、工作原理以及实际应用，并提供解决这些工程挑战的实用思路，旨在以更高的计算效率，正确应用这一机制来扩展LLM的能力。

## 5.1 分析MoE
混合专家模型通过将原本的单一前馈网络（如MLP、FFN）替换为由多个并行子网络组成的专家集合，并通过路由机制在每次计算中仅激活少数专家，从而在保持单次前向计算量（FLOPs）基本不变的前提下显著提升模型的参数容量与表达能力。其核心思想是：模型总体包含大规模参数，但每个输入只使用其中一小部分专家，使得**容量大但计算稀疏**。**值得注意的是多数研究表明，MoE架构在参数规模较大、数据和计算资源充足时优势最为明显；在小规模或资源受限的环境下，其表现可能不如对应的稠密模型，具体效果还取决于任务类型、数据量以及实现细节。**

**概念直观理解：**
`MoE模型`就像一个拥有大量书籍的图书馆（专家集合）,当一个读者（输入数据）来访时，他并不需要翻遍所有书架，而是根据自己感兴趣的主题由“图书管理员”（路由机制）指引，只去相关书架寻找书籍区域（稀疏激活）。这样一来，读者仍然可以访问图书馆中所有书籍的知识（模型的巨大参数容量），同时查找过程却快速高效（FLOPs）。


### 5.1.1 路由机制与负载均衡

路由机制在一次前向传播中，从所有专家中选择出少量最适合当前输入的专家共同参与计算，通常也被称为门控机制，接下来介绍的是基于 $Top-K$ 实现的`词元选择模式（TC）`和`专家选择模式（EC）`的两种路由机制。

假设一共有 $N$ 个专家，输入为 $x$ ，门控函数为 $G(\cdot)$ ，用于决定每个专家的权重， $E_i(\cdot)$ 表示第 $i$ 个专家的输出，则TC和EC的通用门控机制核心计算公式为：

$$
y = \sum_{i \in \mathcal{T}} G_i(x) E_i(x)
$$

**关键点在于集合 $\mathcal{T}$：**
实现稀疏化的关键步骤，这里的 $\mathcal{T}$ 是通过 **$Top\text{-}k$** 机制选出的索引集合。无论是TC还是EC $G(x)$ 都有的计算过程包含两个步骤：

   - **打分：** 计算路由分数 $h(x) = x \cdot W_g$ 。
   - **稀疏化：** 仅保留分数最高的 $k$ 个专家（激活 $k$ 个专家），并对这些分值进行Softmax归一化。未被选中的 $N-k$ 个专家权重被强制置零，这意味着这部分专家在本次前向传播中*完全不参与计算*，从而在模型参数巨大的情况下，保证了实际FLOPs的高效。

> $W_g$ 是路由器中的可学习线性投影层，它将每个token的特征映射为与专家数量相同的得分向量（logits），表示token与各个专家的匹配程度，模型会对这些得分应用 $top-k$ 策略：在TC模式下，每个token选择最适合处理的专家；在EC模式下，每个专家主动选择最适合处理的token。

<div align="center">
<img width="1000" height="520" alt="c5b3cdd83a238e8a6484d51975277f8a" src="https://github.com/user-attachments/assets/648d1892-b01e-4d40-9c2b-50478d2eeccf" />
   <p>图5.1 词元选择模式</p>
 </div>

- `TC`中 $W_g$ ：在打分步骤中，它可以理解为一个 “专家特长档案”。它将token的隐藏特征映射到专家集合的能力空间，并告诉token不同专家分别擅长什么语义，每个token会根据与各个专家“特长档案”的匹配程度，主动挑选最适合处理自己的Top-K专家。

**简易MoE的 $Top-k$ 词元选择模式实现步骤：**

第一步：定义专家网络
```python
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            # 维度升高
            nn.Linear(dim, dim * 4),
            # 非线性激活，提高表达能力
            nn.ReLU(),
            # 还原到初始维度
            nn.Linear(dim * 4, dim)  
        )

    def forward(self, x):
        return self.ffn(x)  # 前向传播
```
每个专家网络由`线性层 → ReLU → 线性层`组成，用于在该专家独有的特征子空间中处理被路由到的 token，从而提供可区分的语义变换，使Top-K路由后的组合输出更具专家特性。

第二步：定义TC MoE网络
```python
class TC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        # 设置专家数量
        self.num_experts = num_experts
        # 设置每个token选用的专家数量
        self.k = k
        # 路由器：将输入映射到专家特征空间
        self.router = nn.Linear(dim, num_experts)
        # 创建专家模块列表（每个专家是独立的）
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
    def forward(self, x, tokens=None, verbose=False):
        # 获取批量大小和特征维度
        B, D = x.shape
        # 计算每个专家对每个token的分数，使用softmax得到概率分布
        gate_scores = F.softmax(self.router(x), dim=-1)  # gate_scores: [B, E]

        # token选取分数最高的k个专家及其分数
        # topk_scores: [B, k]（对应被选中的专家概率值）
        # topk_idx:    [B, k]（对应被选中的专家索引）
        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)
 
        # 初始化输出张量与输入同形状
        out = torch.zeros_like(x)

        # 每个token对应的每一个top-k位置单独处理（同一个token可能被不同专家处理）
        for i in range(self.k):
            # B表示处理的token总数
            # expert_ids表示每个token在第i个Top-K选择的专家编号，形状：[B]
            expert_ids = topk_idx[:, i]
            # expert_weight表示第每个token在第i个Top-K选择上专家所占的权重，形状：[B]
            expert_weight = topk_scores[:, i]

            # 用于累加当前第i个选择位置上所有专家的输出
            expert_output = torch.zeros_like(x)

            # 遍历所有专家，让对应专家处理被分配给它的token
            # e_id表示对应Top-K专家处理的token索引值
            for e_id, expert in enumerate(self.experts):
                # 创建掩码当token在第i个选择位置的专家索引等于当前专家e_id时为1，否则为0
                # mask形状为[B, 1]，用于在计算专家网络前把不属于该专家的token置0
                mask = (expert_ids == e_id).float().unsqueeze(1)

                # mask.sum()表示属于该专家的token数量；若为0表示该专家在本轮没有任务
                if mask.sum() == 0:
                    continue

                # 只把属于该专家的token送入该专家的前馈网络
                # 注意：这里采用x * mask的方式，能保持张量形状一致并保留反向传播路径
                expert_output += expert(x * mask)

            # 将第i个选择位置上专家的输出按对应权重加权并累加到最终out
            # expert_weight.unsqueeze(1)变为[B, 1]以便广播乘到[B, D]
            out += expert_output * expert_weight.unsqueeze(1)

        # out：每个token在Top-K专家上的加权聚合的向量表示
        return out
```
**以上展示的是Top-K TC混合专家的关键板块，运行的代码在[Top-K TC](https://github.com/1iyouzhen/CS336-Chinese-co-construction/blob/main/docs/chapter5/Top-K%20TC.py)。**

TC_MoE(dim=32, num_experts=10, k=2)，输入文本：
>"MoE是很强大的机制！", "专家混合模型非常高效。"

输出：
>按照字节级切分文本，得到33个token，专家负载统计从0到9号专家处理token总数统计依次为[13, 13, 16, 14, 9, 6, 20, 19, 18, 4]

<div align="center">
<img width="1000" height="773" alt="1980610e1c138e069cd6fdcdc3b196a3" src="https://github.com/user-attachments/assets/d665c6bd-88be-4b35-9199-71dbfe74b9ba" />
   <p>图5.2 专家选择模式</p>
 </div>  

- `EC`中 $W_g$ ：在打分步骤中，它可以理解为一个 “语义导航器”，将token的隐藏特征映射到每个专家的语义空间，并把这份导航信号提供给所有专家。每个专家会根据这份“导航信息”，主动挑选最符合自己能力范围的Top-K token进行处理。
  
**简易MoE的 $Top-k$ 专家选择模式实现步骤：**
第一步：定义专家网络
```python
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
```

第二步：定义EC MoE网络
```python
class EC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        # 专家的总数量
        self.num_experts = num_experts
        # 每个专家最多选多少个token  
        self.k = k
        # 用于给每个token输出E个专家得分的路由器                
        self.router = nn.Linear(dim, num_experts)  
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])  
    def forward(self, x, tokens=None, verbose=False):
        # 获取输入token的数量B_total和维度D
        # B_total表示所有token的总数（批次 × token数）
        B_total, D = x.shape

        # 路由器计算每个token、每个专家的匹配得分输出维度：[B_total, num_experts]
        # softmax确保所有专家得分加起来为1
        gate_scores = F.softmax(self.router(x), dim=-1)

        # EC模式： “专家挑token”
        # 转置后变成[num_experts, B_total]
        # scores_T[e][t] = 第e个专家对第t个token的评分
        scores_T = gate_scores.transpose(0, 1)

        # 每个专家从所有token中挑选top-k个最相关的token
        # topk_idx: 每个专家选中Top-K token的索引
        # topk_scores: 对应的路由得分
        # 维度：[num_experts, k]
        topk_scores, topk_idx = scores_T.topk(min(self.k, B_total), dim=-1)

        # dispatch_weights大小：[B_total, num_experts]
        # 初始化dispatch_weights
        dispatch_weights = x.new_zeros((B_total, self.num_experts))

        # 对每个专家e，把top-k token的得分写入对应位置
        for e in range(self.num_experts):
            # topk_idx[e]是一个Top-K token索引列表
            # topk_scores[e]是Top-K token的得分
            # 填写dispatch_weights：每个专家对各个token评分归一化处理
            for t_idx, s in zip(topk_idx[e].tolist(), topk_scores[e].tolist()):
                dispatch_weights[t_idx, e] = s

        # 初始化输出out，与输入x大小相同
        out = torch.zeros_like(x)

        # 对每个专家进行前向计算
        for e_id, expert in enumerate(self.experts):

            # mask: 这个专家是否选择了该token
            # mask[t] == 1 → token t被这个专家选中
            # 维度：[B_total, 1]
            mask = (dispatch_weights[:, e_id] > 0).float().unsqueeze(1)

            # 如果专家没有选择任何token，则跳过计算
            if mask.sum() == 0:
                continue

            # 确保每个专家只处理其选中的Top-K token
            # mask会把不属于该专家的token设置为0，不同专家可能会处理相同的token
            expert_out = expert(x * mask)

            # 将专家输出按其权重加回到最终输出中
            # dispatch_weights[:, e_id]是每一组Top-K token对这个专家的权重
            out += expert_out * dispatch_weights[:, e_id].unsqueeze(1)
        return out
```
**以上展示的是Top-K EC混合专家的关键板块，运行的代码在[Top-K EC](https://github.com/1iyouzhen/CS336-Chinese-co-construction/blob/main/docs/chapter5/Top-K%20EC.py)。**

EC_MoE(dim=32, num_experts=10, k=2)，输入文本：
>"MoE是很强大的机制！", "专家混合模型非常高效。"

输出：
>按照字节级切分文本，得到33个token，专家负载统计从0到9号专家处理token总数均为2，但是有部分token一次都没有被处理过比如：['混'，'合'， '模'， '型'...]。


因此，在每一次前向传播中，模型只会对Top-K路由机制挑选出的专家子集 $T$ 进行计算，从而实现稀疏化推理。路由机制的核心作用可以概括为：**为每个输入选择最合适的少数专家+对这些激活专家的输出按路由权重进行加权融合**。

值得注意的是，路由机制的选择依据是输入的隐藏状态。具体来说，输入词元在经过嵌入、位置编码及前置处理后生成隐藏状态，然后作为路由器`通常是线性层或小型MLP`的输入计算专家分数，这种行列维度的区别决定了稀疏化的粒度。

   - **TC模式**：每个 token 独立选择最适合自己的Top-K专家，因此Top-K操作在<ins>列维度（专家维度）</ins>上进行，每列对应一个专家。
   - **EC模式**：每个专家独立选择自己最适合处理的Top-K token，因此Top-K操作在<ins>行维度（token维度）</ins>上进行，每行对应一个 token。

**TC vs EC** ：

通过运行两段代码[Top-K EC](https://github.com/1iyouzhen/CS336-Chinese-co-construction/blob/main/docs/chapter5/Top-K%20EC.py)和[Top-K TC](https://github.com/1iyouzhen/CS336-Chinese-co-construction/blob/main/docs/chapter5/Top-K%20TC.py)。
1. 在TC模式下，每个token主动选择自己最合适的Top-K专家像“学生找导师”。优势是每个token都会被至少尝试分配到专家上，因此语义完整性较高，减少了信息丢失的风险。缺点问题是专家`负载不均衡`，少数“热门”专家会处理绝大部分token，得到充分训练并显著优于其他专家；而大量“冷门”专家长期闲置能力停滞。这种差距导致模型出现“偏科”现象在即高频领域表现良好，而在低频领域能力不足。
2. 在EC模式下，每个专家主动从所有token中挑选自己最想处理的Top-K像“导师挑学生”。这种机制天然约束了每个专家的处理量（招生名额），从而显著缓解或消除专家`负载不均衡`，有利于专家能力的均衡提升。但代价是部分token可能完全未被任何专家处理（被丢弃），导致语义信息缺失或上下文片段被跳过，从而增加模型在理解、推理时出现断章取义或错误的风险从而降低LLM最后的表现能力。

**结论** TC在语义完整性上占优但易遭遇专家能力两极化；EC在负载均衡上占优但要承受潜在的语义丢失。两种模式代表了稀疏专家系统在信息完整性 vs 负载均衡上的典型权衡抉择。根据近期的研究也有解决这个权衡问题的思路：
| 策略名 | 核心思路 | 说明 |
| ------ | -------- | ----------- |
| [辅助负载均衡](https://yangyutu.github.io/llm_book.github.io/docs/chapter_LLM_arch/LLM_moe_sparse_architectures.html) | 在训练 loss中加入一个正则项，鼓励专家之间接收的token量、路由概率趋于均匀分布，避免少数专家“吃”掉绝大多数token | 这是最早也是最经典的方法，在`Switch Transformer`以及后续很多MoE实现中被采用。 |
| [容量控制 + expert capacity + overflow机制](https://mljourney.com/mixture-of-experts-moe-routing-algorithms-for-sparse-llms) | 给每个专家设定一个容量上限，超过后不再接token、转为fallback路径(或dropout、备用expert)；避免单个专家过载，也避免忽略“冷门” expert | 多MoE系统建议通过capacity factor+expert capacity控制单专家负载以及治理overflow情况。 |
| [动态、无辅助损失的负载均衡](https://www.emergentmind.com/papers/2408.15664) | 避免引入额外训练梯度，通过对每个专家加 bias（基于过去负载统计）动态调整路由分数，从而平衡专家负载，无需aux‑loss也可稳定路由分布 | 最近研究Loss‑Free Balancing for MoE提出该方式，显示比传统aux‑loss更稳定，不破坏原模型优化目标。|
| [改善路由器、相似度保持路由](https://arxiv.org/abs/2506.14038) | 设计路由器，使相似语义的token → 相似分配专家、在专家间分布均匀；减少重复路由和专家负载偏移 | 提升收敛速度和负载均衡效果。|
| [改进专家结构、路由机制](https://arxiv.org/abs/2511.10971) | 通过改变专家参数化例如用正交基、basis或用更稳定、可解释的路由评分而非简单linear logits，提升路由稳定性以及专家利用率 | 最新工作不仅减轻传统路由不稳定和专家闲置，还自然实现更均匀专家负载。 |
| [混合共享 + 路由专家池](https://arxiv.org/pdf/2401.06066) | 将部分专家设为共享专家 ，所有token都激活其余为路由专家，共享专家保证即使路由阶段极不均衡也能覆盖所有token，减少token dropout与语义丢失 | 工程实践中如DeepSeekMoE使用该办法以折中保持语义覆盖+专家特长训练。|

  ### 5.1.2 训练稳定性
  ### 5.1.3 混合专家与稠密模型
## 5.2 MoE的应用

  ### 5.2.1 MoE与LLM
  ### 5.2.2 推理与部署
  ### 5.2.3 MoE变形
## 5.3 DeepSeek创新与实战复现
  ### 5.3.1 DeepSeek的创新关键点
  ### 5.3.2 小规模复现实验
## 5.4 近期MoE研究
和其他领域的联系。
## 思考
## 参考文献

