import fasttext
from functools import lru_cache

# fastText 语言识别模型路径
MODEL_PATH = "/mnt/d/code/项目/cs336/CS336-Chinese-co-construction/coursework/Assignment4_Data/cs336_data/lid.176.bin"


@lru_cache(maxsize=1)
def _load_model():
    """
    加载 fastText 语言识别模型（单例模式）

    使用 lru_cache 的原因：
    - fastText 模型体积较大（~126MB）
    - 避免在每次函数调用时重复加载模型
    - 提高整体运行效率，尤其在批量文本处理时
    """
    return fasttext.load_model(MODEL_PATH)


def run_identify_language(text: str):
    """
    识别输入 Unicode 字符串中的主要语言

    参数：
        text (str): 任意 Unicode 文本字符串

    返回：
        (language_id, confidence_score)
        - language_id: 语言标识符（如 "en", "zh"）
        - confidence_score: 介于 [0, 1] 的置信度分数

    注意：
    - 该函数是测试要求的“适配器函数”
    - 必须保证英文返回 "en"，中文返回 "zh"
    """

    # 对空字符串或只包含空白符的情况进行兜底处理
    # fastText 在这种情况下预测结果不可靠
    if not text or not text.strip():
        return ("unknown", 0.0)

    # 加载（或从缓存中获取）语言识别模型
    model = _load_model()

    # fastText 的 predict 接口：
    # - 返回 labels 和 scores 两个列表
    # - labels 形如 "__label__en"
    # - scores 为预测置信度
    labels, scores = model.predict(
        text.replace("\n", " "),  # 去除换行，避免影响模型判断
        k=1                        # 只取概率最高的一个语言
    )

    # 取出预测结果
    raw_label = labels[0]                 # "__label__en"
    confidence = float(scores[0])         # 置信度（0~1）

    # 去掉 fastText 特有的 "__label__" 前缀
    language = raw_label.replace("__label__", "")

    # 根据作业测试要求，对语言标签进行统一映射
    # fastText 可能返回 zh, zh-cn, zh-tw 等
    if language.startswith("zh"):
        language = "zh"
    elif language == "en":
        language = "en"
    # 其他语言保持原样（如 "fr", "de", "ja" 等）

    return (language, confidence)

if __name__ == "__main__":
    examples = [
        "This is a simple English sentence.",
        "这是一个用于测试的中文句子。",
        "Bonjour, comment allez-vous ?",
        "こんにちは、元気ですか？",
        "",
        "    "
    ]

    for text in examples:
        lang, score = run_identify_language(text)
        print(f"文本: {repr(text)}")
        print(f"预测语言: {lang}, 置信度: {score:.4f}")
        print("-" * 50)

