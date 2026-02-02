from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch

def log_generations(
        prompts: List[str],
        responses: List[str],
        ground_truths: List[str],
        reward_infos: Optional[List[Dict[str, Any]]] = None,
        token_entropies: Optional[List[float]] = None,   # 每条 response 的平均 token entropy（外面算好）
        response_lengths: Optional[List[int]] = None,    # 每条 response 的长度（外面算好，token数或字数都行，但要一致）
        *,
        step: Optional[int] = None,
        max_examples_to_print: int = 3,
) -> Dict[str, Any]:
    """
    Log generations info (no generation inside).
    Records per-example:
      1) prompt
      2) response
      3) ground-truth
      4) reward (format/answer/total etc. if provided)
      5) avg token entropy (if provided)
      6) length stats overall/correct/incorrect (if provided)

    Returns a dict you can feed to wandb/tensorboard or print.
    """
    n = len(prompts)
    assert len(responses) == n and len(ground_truths) == n, "length mismatch"
    if reward_infos is None:
        reward_infos = [{} for _ in range(n)]
    else:
        assert len(reward_infos) == n, "reward_infos length mismatch"

    if token_entropies is None:
        token_entropies = [float("nan")] * n
    else:
        assert len(token_entropies) == n, "token_entropies length mismatch"

    if response_lengths is None:
        # fallback：用字符长度（你也可以改成 split() 的词数）
        response_lengths = [len(r) for r in responses]
    else:
        assert len(response_lengths) == n, "response_lengths length mismatch"

    examples: List[Dict[str, Any]] = []
    lengths_all: List[float] = []
    lengths_correct: List[float] = []
    lengths_incorrect: List[float] = []
    ent_all: List[float] = []

    for i in range(n):
        rinfo = reward_infos[i] or {}

        # 判对错：优先用 reward 里给的；否则退化成字符串完全匹配
        if "is_correct" in rinfo:
            is_correct = bool(rinfo["is_correct"])
        elif "answer_reward" in rinfo:
            try:
                is_correct = float(rinfo["answer_reward"]) > 0
            except Exception:
                is_correct = responses[i].strip() == ground_truths[i].strip()
        else:
            is_correct = responses[i].strip() == ground_truths[i].strip()

        L = float(response_lengths[i])
        lengths_all.append(L)
        (lengths_correct if is_correct else lengths_incorrect).append(L)

        ent = float(token_entropies[i])
        if ent == ent:  # not NaN
            ent_all.append(ent)

        examples.append({
            "prompt": prompts[i],
            "response": responses[i],
            "ground_truth": ground_truths[i],
            "reward": {
                "format_reward": rinfo.get("format_reward", None),
                "answer_reward": rinfo.get("answer_reward", None),
                "total_reward": rinfo.get("total_reward", None),
                **{k: v for k, v in rinfo.items()
                   if k not in {"format_reward", "answer_reward", "total_reward"}},
            },
            "avg_token_entropy": ent,
            "response_length": int(response_lengths[i]),
            "is_correct": is_correct,
        })

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    stats = {
        "step": step,
        "num_examples": n,
        "num_correct": len(lengths_correct),
        "num_incorrect": len(lengths_incorrect),
        "avg_response_length": _mean(lengths_all),
        "avg_response_length_correct": _mean(lengths_correct),
        "avg_response_length_incorrect": _mean(lengths_incorrect),
        "avg_token_entropy": _mean(ent_all),
    }

    log_dict = {"stats": stats, "examples": examples}

    # 可选：打印几条（纯 log，不做任何生成）
    if max_examples_to_print > 0:
        header = f"\n=== LOG GENERATIONS (step={step}) ===" if step is not None else "\n=== LOG GENERATIONS ==="
        print(header)
        print("Stats:", stats)
        for j in range(min(n, max_examples_to_print)):
            ex = examples[j]
            print(f"\n--- Example {j} ---")
            print("Prompt:", ex["prompt"])
            print("Response:", ex["response"])
            print("GT:", ex["ground_truth"])
            print("Reward:", ex["reward"])
            print("Avg token entropy:", ex["avg_token_entropy"])
            print("Resp length:", ex["response_length"], "Correct:", ex["is_correct"])

    return log_dict