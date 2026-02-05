
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Common Crawl WET 文件中提取训练数据并使用 FastText 质量分类器
支持：文件级 + 记录级 双进度条，实时跳过统计，支持 Ctrl-C 中断
"""

import json
import gzip
import signal
import sys
import random
from pathlib import Path
from typing import Optional, List, Tuple,  Dict, Any
from tqdm import tqdm

# 如果脚本在子目录，把项目根加入 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
from filter import (
    run_mask_emails,
    run_mask_ips,
    run_mask_phone_numbers,
    run_gopher_quality_filter,
    run_extract_text_from_html_bytes,
)

import fasttext


# -------------------- 配置 -------------------- #
FASTTEXT_MODEL_PATH = "quality_classifier.bin"  # 你的 FastText 模型路径


# -------------------- 工具函数 -------------------- #
def desensitize_text(text: str) -> str:
    """邮箱 / 电话 / IP 脱敏"""
    text, _ = run_mask_emails(text)
    text, _ = run_mask_phone_numbers(text)
    text, _ = run_mask_ips(text)
    return text


def process_text(text: str) -> Optional[str]:
    """脱敏 + 单行化 + 基础过滤"""
    if not text or not text.strip():
        return None

    text = text.strip()

    # 基础长度过滤（与 Gopher 类似）
    if len(text) < 200:  # 太短不要
        return None
    if len(text) > 100_000:  # 太长截断
        text = text[:100_000]

    # 脱敏处理
    text = desensitize_text(text)

    # 单行化（FastText 格式要求）
    return ' '.join(text.split())


def classify_quality(model, text: str) -> Tuple[str, float]:
    """
    使用 FastText 模型进行质量分类
    返回: (标签, 置信度分数)
    """
    # FastText 要求输入不能包含换行符
    clean_text = text.replace('\n', ' ').strip()
    if not clean_text:
        return ("unknown", 0.0)

    try:
        prediction = model.predict(clean_text)
        label = prediction[0][0].replace('__label__', '')
        score = prediction[1][0]
        return (label, score)
    except Exception as e:
        return ("error", 0.0)


def extract_text_from_wet_record(record_lines: List[str]) -> Optional[str]:
    """
    从 WET 记录的行列表中提取文本内容

    WET 格式：
    - 以 URL 行开始
    - 然后是 WARC-Header 元数据
    - 空行后是实际内容
    """
    if not record_lines:
        return None

    # 找到第一个空行后的内容
    content_start = 0
    for i, line in enumerate(record_lines):
        if line.strip() == '':
            content_start = i + 1
            break

    # 提取内容部分
    content = '\n'.join(record_lines[content_start:]).strip()

    # 基础过滤
    if len(content) < 50:  # 太短
        return None
    if len(content.split()) < 10:  # 词数太少
        return None

    return content


def parse_wet_file(file_obj) -> List[List[str]]:
    """
    解析 WET 文件，返回记录列表（每个记录是一个行列表）

    WET 文件格式：记录之间用 "WARC/1.0" 分隔
    """
    records = []
    current_record = []

    for line in file_obj:
        line = line.decode('utf-8', errors='ignore')

        # 新记录开始
        if line.startswith('WARC/1.0'):
            if current_record:
                records.append(current_record)
            current_record = [line]
        else:
            if current_record is not None:
                current_record.append(line)

    # 添加最后一个记录
    if current_record:
        records.append(current_record)

    return records


# -------------------- 核心提取 -------------------- #
def extract_samples_from_wet(
    wet_paths: List[str],
    target_count: int,
    model_path: str = FASTTEXT_MODEL_PATH,
    quality_threshold: float = 0.5,  # 质量分数阈值
    random_seed: int = 42,
) -> Tuple[List[str], List[str], dict]:
    """
    多文件 WET → 提取训练样本并使用 FastText 分类

    返回: (高质量样本列表, 低质量样本列表, 统计信息字典)
    """
    random.seed(random_seed)

    # 加载 FastText 模型
    print(f"加载 FastText 模型: {model_path}")
    try:
        model = fasttext.load_model(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    hq_samples: List[str] = []  # 高质量
    lq_samples: List[str] = []  # 低质量
    stats = {
        "total_records": 0,
        "processed": 0,
        "skipped_empty": 0,
        "skipped_short": 0,
        "hq_count": 0,
        "lq_count": 0,
        "error": 0,
    }

    # 文件级进度条
    pbar_files = tqdm(wet_paths, desc="WET文件", unit="file", position=0)

    for wet_path in pbar_files:
        # 检查是否已达到目标
        total_collected = len(hq_samples) + len(lq_samples)
        if total_collected >= target_count * 2:  # 两种质量都收集够
            pbar_files.write("已达到目标数量，提前结束")
            break

        if not Path(wet_path).exists():
            pbar_files.write(f"文件不存在: {wet_path}")
            continue

        # 解析 WET 文件
        try:
            with gzip.open(wet_path, 'rb') as f:
                records = parse_wet_file(f)
        except Exception as e:
            pbar_files.write(f"解析失败 {wet_path}: {e}")
            continue

        total_records = len(records)
        stats["total_records"] += total_records
        pbar_files.set_postfix(
            file=Path(wet_path).name, 
            total_records=total_records,
            hq=len(hq_samples),
            lq=len(lq_samples)
        )

        # 记录级进度条
        for record_lines in tqdm(
            records,
            total=total_records,
            desc="记录",
            unit="rec",
            position=1,
            leave=False,
        ):
            # 检查是否已收集足够
            if len(hq_samples) >= target_count and len(lq_samples) >= target_count:
                break

            # 提取文本
            text = extract_text_from_wet_record(record_lines)
            if text is None:
                stats["skipped_empty"] += 1
                continue

            # 处理文本
            processed = process_text(text)
            if processed is None:
                stats["skipped_short"] += 1
                continue

            stats["processed"] += 1

            # FastText 质量分类
            label, score = classify_quality(model, processed)

            # 根据分类结果和阈值决定保留
            if label == "hq" and score >= quality_threshold:
                if len(hq_samples) < target_count:
                    hq_samples.append(processed)
                    stats["hq_count"] += 1
            elif label == "lq" and score >= quality_threshold:
                if len(lq_samples) < target_count:
                    lq_samples.append(processed)
                    stats["lq_count"] += 1
            else:
                stats["error"] += 1

        # 更新文件级进度条信息
        pbar_files.set_postfix(
            file=Path(wet_path).name,
            hq=len(hq_samples),
            lq=len(lq_samples),
            processed=stats["processed"]
        )

    return hq_samples, lq_samples, stats


# -------------------- 辅助函数 -------------------- #
def find_wet_files(datasets_dir: Path) -> List[str]:
    """扫描目录下所有 .wet.gz 文件"""
    patterns = ["*.warc.wet.gz"]
    files = []
    for pattern in patterns:
        files.extend(datasets_dir.glob(pattern))
    return sorted(str(f) for f in files if f.exists())



def save_samples(hq_samples: List[str], lq_samples: List[str], output_dir: Path):
    """保存样本为 JSON 格式"""
    output_dir.mkdir(parents=True, exist_ok=True)

    hq_path = output_dir / "train_hq.jsonl"
    lq_path = output_dir / "train_lq.jsonl"
    combined_path = output_dir / "train_combined.jsonl"
    metadata_path = output_dir / "metadata.json"

    # 保存高质量样本 (JSON Lines 格式)
    with open(hq_path, 'w', encoding='utf-8') as f:
        for text in hq_samples:
            record = {
                "text": text,
                "label": "hq",
                "quality_score": 1.0
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 保存低质量样本 (JSON Lines 格式)
    with open(lq_path, 'w', encoding='utf-8') as f:
        for text in lq_samples:
            record = {
                "text": text,
                "label": "lq",
                "quality_score": 0.0
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 保存合并文件 (JSON Lines 格式)
    with open(combined_path, 'w', encoding='utf-8') as f:
        for text in hq_samples:
            record = {
                "text": text,
                "label": "hq",
                "quality_score": 1.0
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        for text in lq_samples:
            record = {
                "text": text,
                "label": "lq",
                "quality_score": 0.0
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # 保存元数据
    metadata = {
        "total_samples": len(hq_samples) + len(lq_samples),
        "hq_samples": len(hq_samples),
        "lq_samples": len(lq_samples),
        "format": "jsonl",
        "fields": ["text", "label", "quality_score"]
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return hq_path, lq_path, combined_path, metadata_path

# -------------------- 主入口 -------------------- #
def main():
    script_dir = Path(__file__).parent
    datasets_dir = script_dir  # 或改为你的数据目录
    output_dir = script_dir / "cc_output"

    # 配置
    target_count = 10_0000  # 每种质量的目标数量
    quality_threshold = 0.8  # FastText 置信度阈值

    # 查找 WET 文件
    wet_paths = find_wet_files(datasets_dir)
    if not wet_paths:
        print(f"未找到任何 .wet.gz 文件 in {datasets_dir}")
        print("支持的格式: *.wet.gz, *.warc.wet.gz")
        return

    print(f"找到 {len(wet_paths)} 个 WET 文件")
    for p in wet_paths[:5]:  # 只显示前5个
        print(f"  - {Path(p).name}")
    if len(wet_paths) > 20:
        print(f"  ... 等共 {len(wet_paths)} 个文件")

    # 检查模型
    if not Path(FASTTEXT_MODEL_PATH).exists():
        print(f"错误: FastText 模型不存在: {FASTTEXT_MODEL_PATH}")
        return

    # 提取样本
    print(f"\n开始提取样本...")
    print(f"目标: {target_count} 高质量 + {target_count} 低质量")
    print(f"质量阈值: {quality_threshold}")

    hq_samples, lq_samples, stats = extract_samples_from_wet(
        wet_paths=wet_paths,
        target_count=target_count,
        quality_threshold=quality_threshold,
    )

    # 保存结果
    if hq_samples or lq_samples:
        hq_path, lq_path, combined_path,_ = save_samples(hq_samples, lq_samples, output_dir)

        print(f"\n{'='*50}")
        print("处理完成!")
        print(f"{'='*50}")
        print(f"总记录数: {stats['total_records']}")
        print(f"成功处理: {stats['processed']}")
        print(f"  - 跳过(空/解析失败): {stats['skipped_empty']}")
        print(f"  - 跳过(太短): {stats['skipped_short']}")
        print(f"  - 分类错误: {stats['error']}")
        print(f"\n收集样本:")
        print(f"  - 高质量 (hq): {len(hq_samples)} / {target_count}")
        print(f"  - 低质量 (lq): {len(lq_samples)} / {target_count}")
        print(f"\n保存位置:")
        print(f"  - 高质量: {hq_path}")
        print(f"  - 低质量: {lq_path}")
        print(f"  - 合并: {combined_path}")
    else:
        print("\n没有提取到任何样本")


if __name__ == "__main__":
    # 支持 Ctrl-C 中断
    signal.signal(signal.SIGINT, lambda *_: (print("\n用户中断，退出"), sys.exit(0)))
    main()
