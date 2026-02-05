from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# eval_model_path = "/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B"
eval_model_path = "openai-community/gpt2"
zero_shot_tasks = ["arc_easy", "piqa", "lambada", "triviaqa"]
few_shot_tasks = ["humaneval", "mbpp", "gsm8k", "minerva_math"]
all_results = {}

# 🔥 关键：只创建一次模型实例
lm = HFLM(
    pretrained=eval_model_path,      # 可以是路径，会自动加载
    tokenizer=eval_model_path,
    batch_size=32,
    device="cpu",
    max_length=1024,
)

# ===== 1. Zero-shot evaluation =====
print(f"Evaluating zero-shot tasks: {zero_shot_tasks}")
results_zero = evaluator.simple_evaluate(
    model=lm,  # ← 复用同一个 lm 实例
    tasks=zero_shot_tasks,
    num_fewshot=0,
    limit=1,
    batch_size=32,
    gen_kwargs={"max_gen_toks": 512}, 
    # device='cuda',
    confirm_run_unsafe_code=True
)
all_results.update(results_zero['results'])

# ===== 2. Few-shot evaluation =====
print(f"Evaluating 3-shot tasks: {few_shot_tasks}")
results_few = evaluator.simple_evaluate(
    model=lm,  # ← 同一个 lm 实例
    tasks=few_shot_tasks,
    num_fewshot=3,
    limit=1,
    batch_size=32,
    gen_kwargs={"max_gen_toks": 512}, 
    # device='cuda',
    confirm_run_unsafe_code=True
)
all_results.update(results_few['results'])

print("Combined results keys:", list(all_results.keys()))
print("all_results:",all_results)