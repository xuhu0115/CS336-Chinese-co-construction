
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Common Crawl WARC 中提取负样本（lq）并保存为 FastText 格式
新增：文件级 + 记录级 双进度条，实时跳过统计，支持 Ctrl-C 中断
"""
import gzip
import signal
import sys
import random
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

# 如果脚本在子目录，把项目根加入 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
from filter import (
    run_mask_emails,
    run_mask_ips,
    run_mask_phone_numbers,
    run_gopher_quality_filter,
    run_extract_text_from_html_bytes,
)

# -------------------- 工具函数 -------------------- #
def desensitize_text(text: str) -> str:
    """邮箱 / 电话 / IP 脱敏"""
    text, _ = run_mask_emails(text)
    text, _ = run_mask_phone_numbers(text)
    text, _ = run_mask_ips(text)
    return text


def extract_text_from_warc_record(record) -> Optional[str]:
    """
    从单个 WARC 记录中提取纯文本
    只处理 HTTP 200 且 Content-Type 为 text/html 的响应
    """
    if record.rec_type != 'response':
        return None
    content_type = record.http_headers.get_header('Content-Type', '')
    if 'text/html' not in content_type.lower():
        return None

    try:
        html_bytes = record.content_stream().read()
        text = run_extract_text_from_html_bytes(html_bytes)
        if text and len(text.strip()) > 100:          # 太短不要
            return text
    except Exception:
        return None
    return None


def process_text(text: str) -> Optional[str]:
    """脱敏 + 单行化"""
    if not text or not text.strip():
        return None
    text = desensitize_text(text)
    return ' '.join(text.split())                     # 去多余空白/换行


# -------------------- 核心提取 -------------------- #
def extract_negative_samples_from_warc(
    warc_paths: List[str],
    target_count: int,
    random_seed: int = 42,
) -> List[str]:
    """
    多文件 WARC → 提取负样本
    文件级 + 记录级 双进度条，实时显示跳过原因
    """
    random.seed(random_seed)
    results: List[str] = []

    # 文件级进度条
    pbar_files = tqdm(warc_paths, desc="WARC文件", unit="file", position=0)
    skipped  = {"empty": 0, "quality": 0}

    for warc_path in pbar_files:
        if len(results) >= target_count:
            pbar_files.write("已达到目标数量，提前结束")
            break
        if not Path(warc_path).exists():
            skipped["file_not_found"] = skipped.get("file_not_found", 0) + 1
            continue

        # 先扫一遍总记录数，用于记录级进度条
        with gzip.open(warc_path, 'rb') as f:
            total_records = sum(1 for _ in ArchiveIterator(f))
        pbar_files.set_postfix(file=Path(warc_path).name, total_records=total_records)

        # 第二遍真正提取
        with gzip.open(warc_path, 'rb') as f:
            for record in tqdm(
                ArchiveIterator(f),
                total=total_records,
                desc="记录",
                unit="rec",
                position=1,
                leave=False,
            ):
                if len(results) >= target_count:
                    break
                text = extract_text_from_warc_record(record)
                if text is None:
                    skipped["empty"] += 1
                    continue
                processed = process_text(text)
                if processed is None:
                    skipped["quality"] += 1
                    continue
                results.append(processed)

        pbar_files.set_postfix(
            valid=len(results), skipped_empty=skipped["empty"], skipped_quality=skipped["quality"]
        )

    return results


# -------------------- 辅助函数 -------------------- #
def count_positive_samples(positive_file: str) -> int:
    """统计正样本行数（__label__hq 开头）"""
    try:
        with open(positive_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip().startswith('__label__hq'))
    except FileNotFoundError:
        return 0


def find_warc_files(datasets_dir: Path) -> List[str]:
    """扫描 datasets 目录下 *.warc.gz，排除 .wet.gz"""
    return sorted(str(f) for f in datasets_dir.glob("*.warc.gz") if not f.name.endswith(".wet.gz"))


# -------------------- 主入口 -------------------- #
def main():
    script_dir    = Path(__file__).parent
    datasets_dir  = Path(__file__).parent
    positive_file = str(script_dir / "wiki_positive_samples.txt")
    output_path   = str(script_dir / "wiki_negative_samples.txt")

    # 1. 确定目标数量
    target_count  = count_positive_samples(positive_file) or 10_000
    print("正样本数量为",target_count)
    warc_paths    = find_warc_files(datasets_dir)
    if not warc_paths:
        print("未找到任何 WARC 文件，程序结束")
        return

    # 2. 提取负样本
    print(f"目标负样本数量：{target_count}")
    negative_texts = extract_negative_samples_from_warc(warc_paths, target_count)

    # 3. 保存
    if negative_texts:
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in negative_texts:
                f.write(f"__label__lq {text}\n")
        print(f"已保存 {len(negative_texts)} 条负样本 → {output_path}")
    else:
        print("没有提取到任何负样本")


if __name__ == "__main__":
    # 支持 Ctrl-C 中断
    signal.signal(signal.SIGINT, lambda *_: (print("\n用户中断，退出"), sys.exit(0)))
    main()
