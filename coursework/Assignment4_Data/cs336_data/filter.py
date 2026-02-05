import re
from typing import Tuple
import re
import nltk
from nltk.tokenize import word_tokenize
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str:
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

    return text

def run_mask_emails(text: str) -> Tuple[str, int]:
    """
    屏蔽字符串中的电子邮件地址。

    参数:
        text (str): 输入文本

    返回:
        (masked_text, count)
    """

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


def run_mask_phone_numbers(text: str) -> Tuple[str, int]:


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

def run_mask_ips(text: str) -> Tuple[str, int]:
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


def run_gopher_quality_filter(text: str) -> bool:
    """
    Gopher 质量过滤器子集实现
    
    接受一个字符串 text，返回 True 如果文本通过过滤器，False 如果文本被过滤掉。
    
    实现的过滤规则：
    1. 字数少于50字或超过10万字 -> 不通过
    2. 单词平均长度超出 3 到 10 个字符 -> 不通过
    3. 超过30%的行以省略号结尾 -> 不通过
    4. 少于80%的单词至少含有一个字母 -> 不通过
    
    Args:
        text (str): 输入文本
    
    Returns:
        bool: True 表示通过质量过滤，False 表示不通过
    """
    
    # 规则1：字数过滤
    num_chars = len(text)
    if num_chars < 50 or num_chars > 100000:
        return False  # 字数过少或过多
    
    # 将文本按空白字符分词
    words = word_tokenize(text)
    
    # 规则2：单词平均长度过滤
    if len(words) > 0:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3 or avg_word_len > 10:
            return False  # 单词平均长度不符合要求
    
    # 规则3：省略号行比例过滤
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False  # 省略号行过多
    
    # 规则4：至少含有一个字母的单词比例过滤
    if words:
        letter_word_count = sum(1 for w in words if re.search(r"[A-Za-z]", w))
        if letter_word_count / len(words) < 0.8:
            return False  # 含字母单词比例过低
    
    # 所有规则都通过
    return True

