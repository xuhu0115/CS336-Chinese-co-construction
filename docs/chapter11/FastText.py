# FastText简易代码实现（其分类效果容易收到数据量大小以及参数设置影响）
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- 超参数设置 ---
H = 16  # Embedding（嵌入）维度即n-gram被表示为一个16维的向量
K = 2  # 分类数：二分类（正面1\负面0）
num_epochs = 1000  # 训练轮数
lr = 3e-2  # 学习率
num_buckets = 5  # 哈希桶数量：模拟真实fastText中的千万级哈希空间，这里为了演示设为5
ngram = 2  # n-gram 词袋大小：同时考虑单个词和相邻的两个词对
batch_size = 1  # 批处理大小

# 数据准备
texts = [
    "I love this product",
    "This is terrible",
    "Amazing experience",
    "I hate it",
    "Pretty good",
    "Worst ever"
]
labels = [1, 0, 1, 0, 1, 0]  # 1代表正面，0代表负面


# 词袋n-gram获取函数
def get_ngrams(tokens, n):
    """
    生成n-gram词组。
    """
    ngrams = []
    for i in range(len(tokens)):
        for j in range(1, n + 1):
            if i + j <= len(tokens):
                ngrams.append(" ".join(tokens[i:i + j]))
    return ngrams


def hash_ngrams(tokens, num_buckets, ngram):
    ngrams = get_ngrams(tokens, ngram)
    # 使用内置hash并取模，转化为Tensor格式
    return torch.tensor([hash(g) % num_buckets for g in ngrams], dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 预处理：转小写、分词、哈希化
        tokens = self.texts[idx].lower().split()
        hashed_ids = hash_ngrams(tokens, num_buckets, ngram)
        label = self.labels[idx]
        return hashed_ids, label


def collate_fn(batch):
    """
    整理函数：处理同一个Batch中长度不一的句子，进行填充。
    """
    max_len = max(len(x[0]) for x in batch)
    padded = []
    labels = []

    for hashed_ids, label in batch:
        pad_len = max_len - len(hashed_ids)
        # 在右侧填充0，处理文本数据长度对齐
        padded_ids = F.pad(hashed_ids, (0, pad_len), value=0)
        padded.append(padded_ids)
        labels.append(label)

    return torch.stack(padded), torch.tensor(labels)


# 实例化数据加载器
dataset = TextDataset(texts, labels)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)


class FastTextClassifier(nn.Module):
    def __init__(self, num_buckets, embed_dim, num_classes):
        super().__init__()
        # 嵌入层：将哈希索引转为连续向量
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        # 全连接层：从嵌入特征映射到分类空间
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 获得嵌入向量: [Batch_Size, Seq_Len, Embed_Dim]
        embedded = self.embedding(x)

        # 平均池化 ：fastText的核心逻辑，忽略词序只关注特征平均值
        avg_embedded = embedded.mean(dim=1)  # 结果形状: [Batch_Size, Embed_Dim]

        # 输出分类逻辑值: [Batch_Size, Num_Classes]
        logits = self.fc(avg_embedded)
        return logits


# 初始化模型、优化器和损失函数
model = FastTextClassifier(num_buckets, H, K)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 使用随机梯度下降
criterion = nn.CrossEntropyLoss()  # 交叉熵损失适用于分类
print("开始训练...")
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()  # 梯度清零
        logits = model(x)  # 前向传播
        loss = criterion(logits, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()

    # 每 100 轮打印一次进度
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss={total_loss:.4f}")

test_text = "I hate this product"
print("\n--- 测试 ---")
tokens = test_text.lower().split()
# 将测试文本转化为模型可接受的哈希Tensor并增加Batch维度
hashed_ids = hash_ngrams(tokens, num_buckets, ngram).unsqueeze(0)

model.eval()  # 切换到评估模式
with torch.no_grad():
    logits = model(hashed_ids)
    probs = F.softmax(logits, dim=1)  # 计算概率分布
print("输入文本:", test_text)
print(f"预测概率: 正面={probs[0][1]:.4f}, 负面={probs[0][0]:.4f}")
print("预测类别:", "正面" if torch.argmax(probs).item() == 1 else "负面")
