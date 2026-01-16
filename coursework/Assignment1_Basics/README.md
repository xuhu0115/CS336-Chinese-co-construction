# Assignment1

基于`PyTorch`实现的mini Transformer语言模型，支持从零训练和文本生成。

## 环境要求

```bash
pip install torch numpy transformers tokenizers tqdm psutil matplotlib
```

## 快速开始

### 1. 准备训练数据

确保你有以下文件：
- TinyStoriesV2-GPT4-train.txt - 训练文本数据（这里因为文件大小限制没有需要自己下载点击👉[数据集](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main)）
- bpe_tokenizer/tokenizer.json - BPE分词器（通过运行CS336_Assignment1_BPE得到）

生成训练数据的二进制文件：

```bash
python "get  train data.py"
```

这会生成`data.bin`文件（从训练文本中随机抽取80万个样本）。

### 3. 训练模型

使用默认参数训练：

```bash
python train.py
```

自定义参数训练：

```bash
python train.py --epochs 10 --batch_size 16 --d_model 256 --num_heads 8 --num_layers 6 --lr 3e-4
```

### 4. 文本生成

训练完成后，直接运行model.py进行文本生成：

```bash
python model.py
```

默认会加载`ckpt/epoch_5.pt`权重文件，并根据`prompt`生成文本。

## 主要参数说明

### 训练参数(train.py 可以自己根据需求进行修改)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 5 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--context_length` | 128 | 上下文长度 |
| `--d_model` | 256 | 模型维度 |
| `--num_heads` | 8 | 注意力头数 |
| `--num_layers` | 6 | Transformer层数 |
| `--vocab_size` | 50257 | 词表大小 |
| `--lr` | 3e-4 | 学习率 |
| `--min_lr` | 3e-5 | 最小学习率 |
| `--checkpoint_dir` | ./ckpt | 检查点保存目录 |
| `--data_path` | data.bin | 训练数据路径 |

### 生成参数(model.py)

在model.py的`__main__`部分修改：

```txt
# 建议不要同时使用top_k和top_p
generated_text = decode_generated_text(
    model, tokenizer, prompt,
    max_new_tokens=60,      # 生成的 token 数量
    temperature=0.8,        # 温度参数（0.1-2.0）
    top_k=None,            # Top-K 采样
    top_p=0.9,             # Top-P 采样
    device=device
)
```

## 代码结构

```
├── train.py                    # 训练脚本
├── model.py                    # 模型定义、生成
├── get train data.py          # 数据处理
├── bpe_tokenizer/              # BPE分词器目录
│   └── tokenizer.json
├── ckpt/                       # checkpoint保存
│   ├── epoch_{epoch}.pt
|   |——...
├── TinyStoriesV2-GPT4-train.txt
├── CS336_Assignment1_Ablations/       # 3个消融实验
│   ├── data.in                 # 训练数据
│   ├── not_RMSNorm_model.py    # 完全移除归一化层
│   ├── not_RMSNorm_train.py    # 实验1训练函数
│   ├── post_Norm_model.py      # 后归一化
│   ├── post_Norm_train.py      # 实验2训练函数
│   ├── SiLU_model.py           # SwiGLU换成SiLU
│   ├── SiLU_train.py           # 实验3训练函数
│   └── tokenizer.json
└── data.bin                    # 处理后的训练数据
```

## 模型特性

- **RoPE位置编码**：旋转位置嵌入；
- **SwiGLU激活函数**：改进的FFN层；
- **RMSNorm**：高效的归一化层；
- **Flash Attention**：自动使用PyTorch的优化注意力实现（注意力机制计算时分块计算最后结果精确度基本不变，提高对GPU的利用）；
- **自定义AdamW优化器**：带权重衰减的Adam；
- **余弦学习率调度**：带预热的余弦退火。

## 训练监控

训练过程中会：
- 每100步打印一次训练状态
- 显示实时内存、显存占用
- 每个epoch结束后进行验证（训练数据量:验证数据量 = 4:1）
- 保存训练和验证的困惑度曲线图
- 自动保存检查点到`./ckpt/`目录

## 注意事项

1. 运行train.py时，如果没有`data.bin`文件，训练脚本会自动生成模拟数据。
2. 模型生成时需要确保加载的检查点参数与模型初始化参数一致。
3. Windows系统已配置OpenMP环境变量防止冲突。
4. 可以支持CPU和GPU训练，自动检测可用设备。
5. 加载模型参数以及相关状态时，在model.py中`__main__`设置的相关参数比如num_heads之类的参数必须和训练模型的参数一致否则会报错无法生成文本。

## 常见问题

**Q1: 如何修改生成的提示词？**

A1: 编辑model.py文件中__main__里面的`prompt`变量：
```txt
prompt = "填写你的提示词..."
```
---
**Q2: 训练时内存不足怎么办？**

A2: 减小batch_size或context_length参数。

---
**Q3: 如何使用自己的数据集？**

A3: 编辑get train data.py文件中__main__里面的build_random_data_bin()函数中的`input_txt=`：
```txt
    build_random_data_bin(
        input_txt="填写自己准备的 .txt类型的训练文本",
        tokenizer_json="bpe_tokenizer/tokenizer.json",
        output_bin="data.bin",
        target_samples=填写一个整数
    )
建议：得到文本生成尽可能好的模型建议考虑缩放定律，抽取样本数量时考虑`模型总参数x(大概100~200倍)=训练总token数量`）.
```
