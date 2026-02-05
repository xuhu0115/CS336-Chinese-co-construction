from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from typing import List
from tokenizers import Tokenizer

# 加载tokenizer
def load_tokenizer(tokenizer_json_path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(tokenizer_json_path)
    return tokenizer

# 文本转化为toekn id
def text_to_token_ids(
    text: str,
    tokenizer: Tokenizer,
    add_eos: bool = True
    ) -> List[int]:
    encoding = tokenizer.encode(text)
    ids = encoding.ids

    if add_eos:
        eos_id = tokenizer.token_to_id("<|eos|>")
        if eos_id is not None:
            ids.append(eos_id)

    return ids



def build_random_data_bin(
        input_txt: str,
        tokenizer_json: str,
        output_bin: str,
        target_samples: int = 10000,
        dtype=np.int32
):
    tokenizer = load_tokenizer(tokenizer_json)

    print(f"正在扫描文件行数...")
    # 第一遍扫描：获取总行数（用于随机索引）
    with open(input_txt, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"总计 {total_lines} 行，正在抽取 {target_samples} 个样本...")

    # 随机生成要抽取的行号集合（排好序以便流式读取）
    target_indices = set(random.sample(range(total_lines), min(target_samples, total_lines)))

    all_ids = []

    # 第二遍扫描：只处理选中的行
    with open(input_txt, "r", encoding="utf-8") as f:
        sampled_count = 0  # 记录真正取到的样本数
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing")):
            if i in target_indices:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                # 修改点：不再使用break，而是正常处理
                # 如果这一行仅仅是<|endoftext|>，我们也可以把它转成ID
                ids = text_to_token_ids(line_stripped, tokenizer)
                all_ids.extend(ids)

                sampled_count += 1
                # 只有当真正取够了n条随机样本时才停止
                if sampled_count >= target_samples:
                    break

    # 保存文件
    arr = np.array(all_ids, dtype=dtype)
    arr.tofile(output_bin)
    print(f"成功随机抽取并保存{len(arr):,}tokens到{output_bin}")


if __name__ == "__main__":
    build_random_data_bin(
        input_txt="TinyStoriesV2-GPT4-train.txt",
        tokenizer_json="bpe_tokenizer/tokenizer.json",
        output_bin="data.bin",
        target_samples=800000
    )