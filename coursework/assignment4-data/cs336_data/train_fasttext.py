#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 FastText 质量分类器（二分类：hq vs lq）
步骤：合并 → 打乱 → 划分 → 训练 → 保存模型
"""
import fasttext
import random
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm

# ---------- 1. 合并 + 打乱 ---------- #
def merge_and_shuffle_samples(
    positive_file: str,
    negative_file: str,
    output_file: str,
    shuffle: bool = True,
    random_seed: int = 42
) -> int:
    """
    合并正负样本并打乱顺序，返回总样本数
    注意：必须混排，否则 batch 内类别单一，训练失效
    """
    random.seed(random_seed)
    all_samples: List[str] = []

    # 读正样本
    for line in tqdm(open(positive_file, 'r', encoding='utf-8'), desc="读取正样本"):
        line = line.strip()
        if line.startswith('__label__hq'):
            all_samples.append(line)

    # 读负样本
    for line in tqdm(open(negative_file, 'r', encoding='utf-8'), desc="读取负样本"):
        line = line.strip()
        if line.startswith('__label__lq'):
            all_samples.append(line)

    if shuffle:
        random.shuffle(all_samples)

    # 写合并文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(all_samples, desc="写入合并文件"):
            f.write(line + '\n')

    return len(all_samples)

# ---------- 2. 训练 / 验证划分 ---------- #
def split_train_val(
    merged_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[int, int]:
    """按 val_ratio 划分训练集与验证集，返回 (训练条数, 验证条数)"""
    random.seed(random_seed)
    lines = [ln.strip() for ln in open(merged_file, 'r', encoding='utf-8') if ln.strip()]
    random.shuffle(lines)
    val_size = int(len(lines) * val_ratio)
    with open(val_file, 'w', encoding='utf-8') as f:
        for ln in tqdm(lines[:val_size], desc="写验证集"):
            f.write(ln + '\n')
    with open(train_file, 'w', encoding='utf-8') as f:
        for ln in tqdm(lines[val_size:], desc="写训练集"):
            f.write(ln + '\n')
    return len(lines) - val_size, val_size

# ---------- 3. 训练 + 评估 ---------- #
def train_quality_classifier(
    train_path: str,
    model_path: str,
    val_path: Optional[str] = None,
    **kwargs
) -> fasttext.FastText:
    """
    训练 fastText 分类器并保存
    默认参数已针对中文/英文维基质量过滤调优，可外部覆盖
    """
    defaults = dict(
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss='softmax',
        minCount=5,
        bucket=2_000_000,
        thread=8,          # 多线程加速
    )
    defaults.update(kwargs)
    model = fasttext.train_supervised(input=train_path, **defaults)
    model.save_model(model_path)
    if val_path and Path(val_path).exists():
        n, p, r = model.test(val_path)
        print(f"验证集  样本数:{n}  precision:{p:.4f}  recall:{r:.4f}  F1:{2*p*r/(p+r):.4f}")
    return model

# ---------- 4. 主流程 ---------- #
def main() -> None:
    script_dir = Path(__file__).parent
    pos_file   = script_dir / "wiki_positive_samples.txt"
    neg_file   = script_dir / "wiki_negative_samples.txt"
    merged     = script_dir / "quality_merged.txt"
    train_f    = script_dir / "quality_train.txt"
    val_f      = script_dir / "quality_val.txt"
    model_bin  = script_dir / "quality_classifier.bin"

    # 1. 合并 & 打乱
    total = merge_and_shuffle_samples(str(pos_file), str(neg_file), str(merged))
    if total == 0:
        print("没有有效样本，程序结束")
        return

    # 2. 划分
    train_n, val_n = split_train_val(str(merged), str(train_f), str(val_f))

    # 3. 训练
    print(f"开始训练：训练集 {train_n} 条，验证集 {val_n} 条")
    train_quality_classifier(str(train_f), str(model_bin), str(val_f))

    print(f"模型已保存至 {model_bin}")


if __name__ == "__main__":
    main()
