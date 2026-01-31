from __future__ import annotations
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from langdetect import detect_langs
import os
from typing import Any
import re


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    从包含 HTML 的字节字符串中提取纯文本。

    参数:
        html_bytes (bytes): 原始 HTML 字节串

    返回:
        str: 提取出的纯文本
    """
    # 1. 尝试直接用 UTF-8 解码（最快路径）
    try:
        html_text = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 2. UTF-8 失败时，使用 resiliparse 的编码检测
        encoding = detect_encoding(html_bytes)
        # 使用检测到的编码进行解码，忽略无法解码的字符
        html_text = html_bytes.decode(encoding, errors="ignore")

    # 3. 使用 Resiliparse 从 HTML 字符串中提取纯文本
    text = extract_plain_text(html_text)

    return 1



def run_identify_language(text: str) -> tuple[Any, float]:
    """
    对输入的 Unicode 文本进行语言识别，返回最可能的语言及其置信度。

    参数:
        text (str): Unicode 字符串（已解码的自然语言文本）

    返回:
        (language_code, confidence_score)
        - language_code: 语言标识符（如 "en", "zh"）
        - confidence_score: 介于 0 和 1 之间的概率值
    """

    # -------------------------------
    # 1. 防御性检查（edge case 处理）
    # -------------------------------
    # 如果文本为空或只包含空白字符：
    # - 语言识别模型无法做出可靠判断
    # - 直接返回一个兜底结果，避免抛异常
    if not text or len(text.strip()) == 0:
        return ("unknown", 0.0)

    # ---------------------------------------
    # 2. 调用 langdetect 进行语言概率预测
    # ---------------------------------------
    # detect_langs 返回一个列表，形式如：
    # [LangProbability(lang='en', prob=0.98),
    #  LangProbability(lang='de', prob=0.02)]
    #
    # 这些结果已经按概率从高到低排序
    langs = detect_langs(text)

    # 取概率最高的语言作为“主要语言”
    top_lang = langs[0]

    # 语言代码（ISO 639-1 / 内部约定）
    lang_code = top_lang.lang

    # 该语言的置信度（0~1）
    confidence = top_lang.prob

    # ------------------------------------------------
    # 3. 语言代码规范化（为通过测试 & 下游统一）
    # ------------------------------------------------

    if lang_code.startswith("zh"):
        lang_code = "zh"

    # 英文在 langdetect 中本身就是 "en"
    # 这里显式保留，强调与测试约定一致
    elif lang_code == "en":
        lang_code = "en"

    # ---------------------------------------
    # 4. 返回最终预测结果
    # ---------------------------------------
    return (lang_code, confidence)



def run_mask_emails(text: str) -> tuple[str, int]:
    """
    屏蔽字符串中的电子邮件地址。

    参数:
        text (str): 输入文本

    返回:
        (masked_text, count)
    """
    print("ENTER run_mask_emails")
    # 常见电子邮件地址的正则表达式
    email_pattern = re.compile(
        r'\b[a-zA-Z0-9._%+-]+'
        r'@'
        r'[a-zA-Z0-9.-]+'
        r'\.[a-zA-Z]{2,}\b'
    )

    # findall 用于统计匹配数量
    matches = email_pattern.findall(text)

    # sub 用于统一替换
    masked_text = email_pattern.sub("|||EMAIL_ADDRESS|||", text)

    return masked_text, len(matches)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:

    phone_pattern = re.compile(
        r'''
        \b
        (?:\+?1[\s.-]?)?          # 可选国家码
        (?:\(?\d{3}\)?[\s.-]?)    # 区号
        \d{3}[\s.-]?\d{4}         # 主号码
        \b
        ''',
        re.VERBOSE
    )

    matches = phone_pattern.findall(text)
    masked_text = phone_pattern.sub("|||PHONE_NUMBER|||", text)

    return masked_text, len(matches)



def run_mask_ips(text: str) -> tuple[str, int]:
    """
    屏蔽 IPv4 地址
    """

    ip_pattern = re.compile(
        r'\b'
        r'(?:25[0-5]|2[0-4]\d|1?\d{1,2})'
        r'(?:\.'
        r'(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}'
        r'\b'
    )

    matches = ip_pattern.findall(text)
    masked_text = ip_pattern.sub("|||IP_ADDRESS|||", text)

    return masked_text, len(matches)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    raise NotImplementedError


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
