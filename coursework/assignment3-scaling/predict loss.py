import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scaling import *
from model import *


data = np.load("ckpt/scaling_records_70M.npy", allow_pickle=True)

# 假设data中包含了不同模型在不同步数下的记录
# data结构示例: [{"params": 70e6, "tokens": 1e9, "loss": 3.2}, ...]

N_data = np.array([d["flops"] for d in data])  # 需要你有params记录
D_data = np.array([d["tokens"] for d in data])  # 需要你有tokens记录
losses = np.array([d["loss"] for d in data])

popt, pcov = fit_chinchilla_scaling(N_data, D_data, losses)
E_fit, A_fit, B_fit, alpha_fit, beta_fit = popt

# 模型配置
config = {
    "vocab_size": 50720,
    "context_length": 128,
    "d_model": 512,
    "num_layers": 36,
    "num_heads": 8,
    "d_ff": 512 * 4,
    "attn_pdrop": 0.1,
    "residual_pdrop": 0.1,
}

# 可以选择读取数据或则是自己固定数据总量
def count_total_tokens(
    path,
    dtype=np.uint16,
    pad_id=0,
    special_ids=None,
    chunk_size=10_000_000,  # 每次读10M tokens
):
    if special_ids is None:
        special_ids = []

    token_bytes = np.dtype(dtype).itemsize
    file_size = os.path.getsize(path)
    assert file_size % token_bytes == 0, "文件大小与dtype不匹配"

    total_tokens = 0

    mm = np.memmap(path, dtype=dtype, mode="r")
    num_tokens = mm.shape[0]

    for i in range(0, num_tokens, chunk_size):
        chunk = mm[i : i + chunk_size]

        mask = chunk != pad_id
        for sid in special_ids:
            mask &= (chunk != sid)

        total_tokens += np.sum(mask)

    return int(total_tokens)

def get_params(config: dict):
    model = BasicsTransformerLM(**config)
    no_embeddings = model.get_num_params(non_embedding=True)
    return no_embeddings

non_embed = get_params(config)

# C=FLOPs=6*N_{non_embed}*D_{token}
num_tokens = count_total_tokens(
    "data.bin",
    dtype=np.uint16,
    pad_id=0,
    special_ids=[0, 1, 2, 3, 4],  # 例如<BOS> id=1, <EOS> id=2
)

#tokens = num_tokens

tokens = 2e11

# FLOPs = 6 * non_embed * tokens
# 方法A: 使用自己训练出来的参数拟合关系
pred_loss = chinchilla_scaling_law(
    (non_embed, tokens),
    *popt
)

# 方式 B: 直接使用Chinchilla论文中的参数 (如果你数据不够，想看理论值)
# 注意：论文参数是基于MassiveWeb数据集的，可能与你的数据集分布不同，仅供参考，参数: E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28
"""pred_loss = chinchilla_scaling_law(
    (non_embed, tokens),
    E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28
)"""

print(f"非嵌入参数: {non_embed}")
print(f"pred_loss: {pred_loss: .6f}")