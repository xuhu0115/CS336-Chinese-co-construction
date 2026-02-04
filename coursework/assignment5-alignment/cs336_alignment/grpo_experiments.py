# --------------------------- grpo_experiments.py ---------------------------
"""
GRPO 实验脚本 - 实现作业5的所有GRPO实验

包括：
1. 学习率调整实验 (grpo_learning_rate)
2. 基线影响实验 (grpo_baselines)
3. 长度归一化实验 (grpo_length_normalization)
4. 标准差归一化实验 (grpo_group_standard_deviation)
5. 离策略 GRPO 实现 (grpo_off_policy)
6. 离策略超参数扫描 (grpo_off_policy_sweep)
7. 裁剪消融实验 (grpo_off_policy_clip_ablation)
8. 提示消融实验 (grpo_prompt_ablation)
9. 排行榜挑战 (leaderboard)
"""

import os
os.environ["VLLM_USE_V1"] = "0"  # Use vLLM v0 for compatibility
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import wandb
import bitsandbytes as bnb

# --- Import your components ---
from sft_helper import tokenize_prompt_and_output, get_response_log_probs
from gpro_helper import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from evaluate_math import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
import gc

# --------------------------- Config ---------------------------

@dataclass
class GRPOConfig:
    # Model & Data
    base_model: str = "/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B"
    train_data_path: str = "data/sft/sft_gpt-oss-120b_filtered.jsonl"
    val_data_path: str = "data/gsm8k/test.jsonl"
    output_dir: str = "results/grpo"
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"

    # GRPO Hyperparams
    n_grpo_steps: int = 200
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    epochs_per_rollout_batch: int = 1
    advantage_eps: float = 1e-6
    cliprange: float = 0.2
    use_std_normalization: bool = False
    length_norm: str = "masked_mean"  # "masked_mean" or "masked_normalize"
    reward_fn: str=r1_zero_reward_fn     # r1_zero_reward_fn, question_only_reward_fn

    # Loss Type: "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    loss_type: str = "reinforce_with_baseline"

    # Training Hyperparams
    lr: float = 1e-5
    grad_accum_steps: int = 64
    max_grad_norm: float = 1.0
    seed: int = 2026

    # Evaluation
    eval_every_steps: int = 10
    eval_max_examples: int = 256  # max=1319

    # Devices
    device_policy: str = "cuda:1"
    device_eval: str = "cuda:0"

    # vLLM memory behavior
    vllm_gpu_memory_utilization: float = 0.7

    # Generation params
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    gen_max_tokens: int = 1024
    gen_min_tokens: int = 4
    gen_stop: tuple = ("</answer>",)
    gen_include_stop_str_in_output: bool = True

    # Eval generation params
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_max_tokens: int = 1024
    eval_stop: tuple = ("</answer>",)
    eval_include_stop_str_in_output: bool = True

    # Optional knobs
    wandb_project: str = "cs336-a5-grpo"
    run_name: Optional[str] = None  # set to a string if you want


# --------------------------- Helper Functions ---------------------------

def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        return "{question}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template: str, question: str) -> str:
    return template.format(question=question.strip())

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.5) -> LLM:
    # Lower memory usage for vLLM to leave room for the policy model if on same node
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=True,  # Sometimes helps stability in loops
        )

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    print("Syncing Policy weights to vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        start_char = f.read(1)
        f.seek(0)
        if start_char == "[":
            data = json.load(f)
        else:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def extract_question(ex: Dict[str, Any]) -> str:
    return ex.get("problem", ex.get("question", ex.get("query", "")))

def extract_ground_truth(ex: Dict[str, Any]) -> str:
    return ex.get("expected_answer", ex.get("answer", ex.get("solution", "")))


# --------------------------- Core GRPO Training ---------------------------

def run_grpo_experiment(cfg: GRPOConfig) -> Dict[str, float]:
    """
    运行单个 GRPO 实验并返回最终指标
    """
    # ---- Sanity checks ----
    assert cfg.train_batch_size % cfg.grad_accum_steps == 0, "train_batch_size must be divisible by grad_accum_steps"
    micro_batch_size = cfg.train_batch_size // cfg.grad_accum_steps
    assert cfg.rollout_batch_size % cfg.group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout = cfg.rollout_batch_size // cfg.group_size
    assert cfg.train_batch_size >= cfg.group_size

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Reproducibility ----
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- W&B ----
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        mode="offline",
        config=asdict(cfg),
        id=cfg.run_name, # <--- 关键修改：强制指定 run ID (如果想覆盖旧记录)
        resume="allow",  # <--- 配合 id 使用，允许覆盖或继续
    )

    # ---- Prompt template ----
    template = load_prompt_template(cfg.prompt_template_path)

    # ---- Init policy model (training) ----
    print(f"Init Policy on {cfg.device_policy}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    policy = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(cfg.device_policy)

    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    # ---- Init vLLM (rollouts + eval) ----
    print(f"Init vLLM on {cfg.device_eval}...")
    llm = init_vllm(model_id=cfg.base_model, device=cfg.device_eval, seed=cfg.seed, gpu_memory_utilization=cfg.vllm_gpu_memory_utilization)

    # ---- Load data ----
    print(f"Loading train questions from {cfg.train_data_path}...")
    all_questions = load_json_or_jsonl(cfg.train_data_path)

    print(f"Loading validation from {cfg.val_data_path}...")
    val_data = load_json_or_jsonl(cfg.val_data_path)

    val_prompts, val_gts = [], []
    for ex in val_data:
        val_prompts.append(format_prompt(template, extract_question(ex)))
        val_gts.append(extract_ground_truth(ex))
        if len(val_prompts) >= cfg.eval_max_examples:
            break

    # ---- vLLM sampling params ----
    rollout_params = SamplingParams(
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
        max_tokens=cfg.gen_max_tokens,
        min_tokens=cfg.gen_min_tokens,
        stop=list(cfg.gen_stop),
        include_stop_str_in_output=cfg.gen_include_stop_str_in_output,
        n=cfg.group_size,
    )
    eval_params = SamplingParams(
        temperature=cfg.eval_temperature,
        top_p=cfg.eval_top_p,
        max_tokens=cfg.eval_max_tokens,
        stop=list(cfg.eval_stop),
        include_stop_str_in_output=cfg.eval_include_stop_str_in_output,
    )

    # ---- GRPO loop ----
    print(f"Starting GRPO for {cfg.n_grpo_steps} steps...")

    final_metrics = {}

    for step in range(1, cfg.n_grpo_steps + 1):
        print(f"\n=== GRPO Step {step}/{cfg.n_grpo_steps} ===")

        # A) Sample prompts
        batch_questions = random.sample(all_questions, n_prompts_per_rollout)
        prompts = [format_prompt(template, extract_question(q)) for q in batch_questions]
        ground_truths = [extract_ground_truth(q) for q in batch_questions]

        # B) Rollouts (vLLM)
        load_policy_into_vllm_instance(policy, llm)
        print(f"Generating {len(prompts) * cfg.group_size} rollouts...")
        outputs = llm.generate(prompts, rollout_params)

        # C) Flatten rollouts
        rollout_responses: List[str] = []
        repeated_prompts: List[str] = []
        repeated_gts: List[str] = []

        for i, req_output in enumerate(outputs):
            for completion in req_output.outputs:
                rollout_responses.append(completion.text)
                repeated_prompts.append(prompts[i])
                repeated_gts.append(ground_truths[i])

        assert len(rollout_responses) == cfg.rollout_batch_size

        # D) Rewards & advantages (CPU tensors)
        advantages, raw_rewards, meta = compute_group_normalized_rewards(
            reward_fn=cfg.reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.use_std_normalization,
        )

        wandb.log({
            "grpo/step": step,
            "grpo/avg_reward": meta.get("raw_reward_mean", 0.0),
            "grpo/avg_format": meta.get("raw_format_reward_mean", 0.0),
            "grpo/avg_answer": meta.get("raw_answer_reward_mean", 0.0),
            "grpo/adv_std": meta.get("adv_std", 0.0),
        })

        advantages = advantages.unsqueeze(-1)   # (B,1) on CPU for now
        raw_rewards = raw_rewards.unsqueeze(-1) # (B,1) on CPU for now

        # E) Compute old logprobs (for GRPO-Clip / GRPO-No-Clip)
        old_log_probs: Optional[torch.Tensor] = None
        if cfg.loss_type in ("grpo_clip", "grpo_no_clip"):
            print("Computing old logprobs...")
            policy.eval()
            old_log_probs_list: List[torch.Tensor] = []

            with torch.no_grad():
                bs = micro_batch_size
                for i in range(0, len(rollout_responses), bs):
                    p_batch = repeated_prompts[i: i + bs]
                    r_batch = rollout_responses[i: i + bs]

                    tokenized = tokenize_prompt_and_output(p_batch, r_batch, tokenizer)
                    input_ids = tokenized["input_ids"].to(cfg.device_policy)
                    labels = tokenized["labels"].to(cfg.device_policy)

                    out = get_response_log_probs(policy, input_ids, labels)
                    old_log_probs_list.append(out["log_probs"].cpu())

            # Pad to same T to allow concatenation (masked later by response_mask)
            max_T = max(t.size(1) for t in old_log_probs_list)
            padded = []
            for t in old_log_probs_list:
                if t.size(1) < max_T:
                    t = F.pad(t, (0, max_T - t.size(1)), value=0.0)
                padded.append(t)
            old_log_probs = torch.cat(padded, dim=0)  # (B, max_T) on CPU

        # F) Training inner loop
        policy.train()
        dataset_indices = list(range(len(rollout_responses)))

        for epoch in range(cfg.epochs_per_rollout_batch):
            random.shuffle(dataset_indices)

            acc_loss = 0.0
            acc_entropy = 0.0
            acc_clip_frac = 0.0
            acc_ratio = 0.0
            micro_steps_done = 0

            for start in range(0, len(dataset_indices), micro_batch_size):
                micro_steps_done += 1
                indices = dataset_indices[start: start + micro_batch_size]

                batch_prompts = [repeated_prompts[j] for j in indices]
                batch_responses = [rollout_responses[j] for j in indices]
                batch_advantages = advantages[indices].to(cfg.device_policy)
                batch_raw_rewards = raw_rewards[indices].to(cfg.device_policy)

                tokenized = tokenize_prompt_and_output(batch_prompts, batch_responses, tokenizer)
                input_ids = tokenized["input_ids"].to(cfg.device_policy)
                labels = tokenized["labels"].to(cfg.device_policy)
                response_mask = tokenized["response_mask"].to(cfg.device_policy)

                out = get_response_log_probs(policy, input_ids, labels, return_token_entropy=True)
                policy_log_probs = out["log_probs"]          # (B_micro, T)
                token_entropy = out["token_entropy"]         # (B_micro, T)

                batch_old_log_probs = None
                if old_log_probs is not None:
                    # Align to current T (since sequences may differ by batch)
                    T = policy_log_probs.size(1)
                    batch_old_log_probs = old_log_probs[indices][:, :T].to(cfg.device_policy)

                loss, loss_meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=cfg.grad_accum_steps,
                    loss_type=cfg.loss_type,
                    raw_rewards=batch_raw_rewards,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cfg.cliprange,
                    length_norm=cfg.length_norm,
                )

                acc_loss += loss.item()

                mask_f = response_mask.float()
                num_tokens = mask_f.sum().item()

                if num_tokens > 0:
                    acc_entropy += (token_entropy * mask_f).sum().item() / num_tokens

                    if "is_clipped" in loss_meta:
                        acc_clip_frac += (loss_meta["is_clipped"].float().to(mask_f.device) * mask_f).sum().item() / num_tokens

                    if "ratio" in loss_meta:
                        acc_ratio += (loss_meta["ratio"].to(mask_f.device) * mask_f).sum().item() / num_tokens

                # Update step when enough microbatches accumulated OR at end
                is_update_step = (micro_steps_done == cfg.grad_accum_steps) or (start + micro_batch_size >= len(dataset_indices))

                if is_update_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    denom = micro_steps_done

                    wandb.log({
                        "grpo/step": step,
                        "train/loss": acc_loss * (cfg.grad_accum_steps / denom),
                        "train/grad_norm": float(grad_norm.item()),
                        "train/entropy": acc_entropy / denom if denom > 0 else 0.0,
                        "train/clip_fraction": acc_clip_frac / denom if denom > 0 else 0.0,
                        "train/mean_ratio": acc_ratio / denom if denom > 0 else 0.0,
                        "train/epoch_in_rollout_batch": epoch,
                    })

                    acc_loss = 0.0
                    acc_entropy = 0.0
                    acc_clip_frac = 0.0
                    acc_ratio = 0.0
                    micro_steps_done = 0

                # Minor cleanup (optional)
                del input_ids, labels, response_mask, out, policy_log_probs, token_entropy

        # G) Validation
        if step % cfg.eval_every_steps == 0:
            print(f"Evaluating at Step {step}...")
            load_policy_into_vllm_instance(policy, llm)

            eval_dir = os.path.join(cfg.output_dir, f"step_{step}")
            os.makedirs(eval_dir, exist_ok=True)

            metrics = evaluate_vllm(
                vllm_model=llm,
                reward_fn=cfg.reward_fn,
                dataset_path=cfg.val_data_path,
                prompt_template=template,
                eval_sampling_params=eval_params,
                output_filepath=os.path.join(eval_dir, "results.jsonl"),
                fast=True,
            )

            wandb.log({
                "grpo/step": step,
                "val/acc": metrics.get("answer_accuracy", 0.0),
            })
            print(f"Val Acc: {metrics.get('answer_accuracy', 0.0):.4f}")

            # Save latest
            # latest_dir = os.path.join(cfg.output_dir, "latest")
            # os.makedirs(latest_dir, exist_ok=True)
            # policy.save_pretrained(latest_dir)
            # tokenizer.save_pretrained(latest_dir)

            # Store final metrics
            if step == cfg.n_grpo_steps:
                final_metrics = metrics
    print(f"Finishing wandb run: {cfg.run_name}")
    wandb.finish()

    print("GRPO Training Complete.")
    return final_metrics


# --------------------------- Individual Experiments ---------------------------

# def run_lr_sweep_experiment():   # 快速测试
#     """学习率调整实验 (grpo_learning_rate) - 快速测试版"""
#     print("=== Running Learning Rate Sweep Experiment (DEBUG MODE) ===")

#     # 1. 减少测试的学习率数量，只测 2 个，验证循环逻辑即可
#     # 原来是 [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
#     LEARNING_RATES = [1e-5, 2e-5] 
#     results = {}

#     # 2. 修改基础配置为“极小资源消耗”模式
#     base_config = GRPOConfig(
#         # --- 关键流程控制 ---
#         n_grpo_steps=1,          # 只跑 2 步，足够验证 step 1 -> step 2 的转换
#         eval_every_steps=1,      # 每步都评估，确保评估代码也能跑通
        
#         # --- 显存与速度优化 ---
#         rollout_batch_size=4,    # 极小的 rollout batch (必须能被 group_size 整除)
#         group_size=4,            # 最小分组
#         train_batch_size=4,      # 极小的训练 batch
#         grad_accum_steps=2,      # 4/2 = 2，确保微批次逻辑能跑通 (必须能整除 train_batch_size)
        
#         # --- 生成速度优化 ---
#         gen_max_tokens=20,       # 生成极短的回复，大幅减少 vLLM 等待时间
#         gen_min_tokens=2,
#         eval_max_tokens=20,      # 评估时也生成极短回复
#         eval_max_examples=1,     # 评估集只取 2 条数据，瞬间完成
        
#         # --- 其他 ---
#         run_name="lr_sweep_debug",
#         wandb_project="grpo-debug", # 建议换个 project 名，避免污染正式实验的 wandb 面板
#     )

#     for lr in LEARNING_RATES:
#         config = GRPOConfig(**asdict(base_config))
#         config.lr = lr
#         config.run_name = f"debug_grpo_lr_{lr}"
#         config.output_dir = f"results/debug_grpo_lr_{lr}"

#         print(f"\n--- Testing LR: {lr} ---")
#         final_metrics = run_grpo_experiment(config)
#         results[lr] = final_metrics.get("answer_accuracy", 0.0)

#     print("\nLearning Rate Sweep Results (DEBUG):")
#     for lr, acc in results.items():
#         print(f"LR {lr}: {acc:.4f}") # 修复了原代码 print(".2e") 的格式错误

#     best_lr = max(results.keys(), key=lambda x: results[x])
#     print(f"\nBest LR: {best_lr} with accuracy: {results[best_lr]:.4f}")

#     return results

def run_lr_sweep_experiment():
    """学习率调整实验 (grpo_learning_rate)"""
    print("=== Running Learning Rate Sweep Experiment ===")

    LEARNING_RATES = [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]
    results = {}

    base_config = GRPOConfig(
        n_grpo_steps=50,  # Shorter for sweep
        run_name="lr_sweep",
    )

    for lr in LEARNING_RATES:
        config = GRPOConfig(**asdict(base_config))
        config.lr = lr
        config.run_name = f"grpo_lr_{lr}"
        config.output_dir = f"results/grpo_lr_{lr}"

        print(f"\n--- Testing LR: {lr} ---")
        final_metrics = run_grpo_experiment(config)
        results[lr] = final_metrics.get("answer_accuracy", 0.0)

    print("\nLearning Rate Sweep Results:")
    for lr, acc in results.items():
        print(f"LR {lr}: {acc:.4f}") # 修复了原代码 print(".2e") 的格式错误

    best_lr = max(results.keys(), key=lambda x: results[x])
    print(f"\nBest LR: {best_lr} with accuracy: {results[best_lr]:.4f}")

    return results




def run_baseline_experiment():# 未完成，卡在reinforce_with_baseline
    """基线影响实验 (grpo_baselines)"""
    print("=== Running Baseline Experiment ===")

    # BASELINE_TYPES = ["no_baseline", "reinforce_with_baseline"]
    BASELINE_TYPES = ["reinforce_with_baseline", "no_baseline"]

    results = {}

    for loss_type in BASELINE_TYPES:
        # 显存清理：虽然分卡了，但保持好习惯

        torch.cuda.empty_cache()
        gc.collect()

        config = GRPOConfig(
            # n_grpo_steps=2, 
            # eval_every_steps=1,
            # # --- 测试 ---
            # rollout_batch_size=4,    # 极小的 rollout batch (必须能被 group_size 整除)
            # group_size=4,            # 最小分组
            # train_batch_size=4,      # 极小的训练 batch
            # grad_accum_steps=2,      # 4/2 = 2，确保微批次逻辑能跑通 (必须能整除 train_batch_size)
            # gen_max_tokens=20,       # 生成极短的回复，大幅减少 vLLM 等待时间
            # gen_min_tokens=2,
            # eval_max_tokens=20,      # 评估时也生成极短回复
            # eval_max_examples=2,     # 评估集只取 2 条数据，瞬间完成
            # 
            device_policy="cuda:1",
            device_eval="cuda:0",
            n_grpo_steps=50,  # Shorter for sweep
            # rollout_batch_size=64, 
            loss_type=loss_type,
            lr=2e-5,  # Best from LR sweep
            use_std_normalization=True,
            run_name=f"grpo_baseline_{loss_type}",
            output_dir=f"results/grpo_baseline_{loss_type}",
        )

        print(f"\n--- Testing {loss_type} ---")
        final_metrics = run_grpo_experiment(config)
        print("final_metrics:",final_metrics)
        results[loss_type] = final_metrics.get("answer_accuracy", 0.0)

    print("\nBaseline Experiment Results:")
    for baseline, acc in results.items():
        print(f"baseline {baseline}: {acc:.4f}") 

    return results


def run_normalization_experiment():
    """长度归一化实验 (grpo_length_normalization)"""
    print("=== Running Length Normalization Experiment ===")

    NORMALIZATION_TYPES = ["masked_mean", "masked_normalize"]
    results = {}

    for norm_type in NORMALIZATION_TYPES:
        config = GRPOConfig(
            n_grpo_steps=50,  # Shorter for sweep
            # rollout_batch_size=64, 
            length_norm=norm_type,
            loss_type="reinforce_with_baseline",
            lr=2e-5,
            run_name=f"grpo_norm_{norm_type}",
            output_dir=f"results/grpo_norm_{norm_type}",
        )

        print(f"\n--- Testing {norm_type} ---")
        final_metrics = run_grpo_experiment(config)
        results[norm_type] = final_metrics.get("answer_accuracy", 0.0)

    print("\nLength Normalization Results:")
    for norm, acc in results.items():
        print(f"norm {norm}: {acc:.4f}") 

    return results


def run_std_norm_experiment():
    """标准差归一化实验 (grpo_group_standard_deviation)"""
    print("=== Running Standard Deviation Normalization Experiment ===")

    STD_SETTINGS = [True, False]
    results = {}

    for use_std in STD_SETTINGS:
        config = GRPOConfig(
            use_std_normalization=use_std,
            length_norm="masked_normalize",
            loss_type="reinforce_with_baseline",
            n_grpo_steps=50,  # Shorter for sweep
            lr=2e-5,
            run_name=f"grpo_std_{use_std}",
            output_dir=f"results/grpo_std_{use_std}",
        )

        print(f"\n--- Testing std_norm={use_std} ---")
        final_metrics = run_grpo_experiment(config)
        results[use_std] = final_metrics.get("answer_accuracy", 0.0)

    print("\nStandard Deviation Normalization Results:")
    for std, acc in results.items():
        print(f"std_norm={std}: {acc:.4f}")

    return results


def run_off_policy_sweep_experiment():
    """离策略超参数扫描 (grpo_off_policy_sweep)"""
    print("=== Running Off-Policy Hyperparameter Sweep ===")

    OFF_POLICY_CONFIGS = [
        {"epochs_per_rollout_batch": 1, "train_batch_size": 256, "grad_accum_steps": 128, "name": "on_policy"},
        {"epochs_per_rollout_batch": 2, "train_batch_size": 128, "grad_accum_steps": 128, "name": "off_policy_2x"},
        {"epochs_per_rollout_batch": 4, "train_batch_size": 128, "grad_accum_steps": 64, "name": "off_policy_4x"},
        {"epochs_per_rollout_batch": 8, "train_batch_size": 128, "grad_accum_steps": 32, "name": "off_policy_8x"},
    ]

    results = {}

    for config_dict in OFF_POLICY_CONFIGS:
        name = config_dict["name"]
        config = GRPOConfig(
            epochs_per_rollout_batch=config_dict["epochs_per_rollout_batch"],
            train_batch_size=config_dict["train_batch_size"],
            grad_accum_steps=config_dict["grad_accum_steps"],
            loss_type="grpo_clip",  # Use clip for off-policy
            use_std_normalization=True,
            length_norm="masked_normalize",
            lr=2e-5,
            n_grpo_steps=50,  # Shorter for sweep
            run_name=f"grpo_off_policy_{name}",
            output_dir=f"results/grpo_off_policy_{name}",
        )

        print(f"\n--- Testing {name} ---")
        final_metrics = run_grpo_experiment(config)
        results[name] = final_metrics.get("answer_accuracy", 0.0)

    print("\nOff-Policy Sweep Results:")
    for name, acc in results.items():
        print(f"name={name}: {acc:.4f}")

    return results


def run_clip_ablation_experiment():
    """裁剪消融实验 (grpo_off_policy_clip_ablation)"""
    print("=== Running Clip Ablation Experiment ===")

    CLIP_SETTINGS = ["grpo_clip", "grpo_no_clip"]
    results = {}

    # Use best off-policy config
    for loss_type in CLIP_SETTINGS:
        config = GRPOConfig(
            loss_type=loss_type,
            epochs_per_rollout_batch=2,
            train_batch_size=128,  # Fixed: 128 % 128 = 0
            grad_accum_steps=128,  # Ensure compatibility
            n_grpo_steps=50,  # Shorter for sweep
            use_std_normalization=False,
            length_norm="masked_normalize",
            lr=2e-5,
            eval_max_examples = 256,  # max=1319
            run_name=f"grpo_clip_ablation_{loss_type}",
            output_dir=f"results/grpo_clip_ablation_{loss_type}",
        )

        print(f"\n--- Testing {loss_type} ---")
        final_metrics = run_grpo_experiment(config)
        results[loss_type] = final_metrics.get("answer_accuracy", 0.0)

    print("\nClip Ablation Results:")
    for clip, acc in results.items():
        print(f"clip={clip}: {acc:.4f}")

    return results


def run_prompt_ablation_experiment():
    """提示消融实验 (grpo_prompt_ablation)"""
    print("=== Running Prompt Ablation Experiment ===")

    PROMPT_TYPES = ["r1_zero", "question_only"]
    results = {}

    for prompt_type in PROMPT_TYPES:
        if prompt_type == "r1_zero":
            template_path = "cs336_alignment/prompts/r1_zero.prompt"
            reward_fn = r1_zero_reward_fn
        else:
            template_path = "cs336_alignment/prompts/question_only.prompt"
            reward_fn = question_only_reward_fn

        config = GRPOConfig(
            prompt_template_path=template_path,
            reward_fn=reward_fn,
            use_std_normalization=False,
            length_norm="masked_normalize",
            loss_type="grpo_clip",
            n_grpo_steps=50,  # Shorter for sweep
            lr=2e-5,
            run_name=f"grpo_prompt_{prompt_type}",
            output_dir=f"results/grpo_prompt_{prompt_type}",
        )

        print(f"\n--- Testing {prompt_type} ---")
        final_metrics = run_grpo_experiment(config)
        results[prompt_type] = final_metrics.get("answer_accuracy", 0.0)

    print("\nPrompt Ablation Results:")
    for prompt, acc in results.items():
        print(f"prompt={prompt}: {acc:.4f}")

    return results


def run_leaderboard_experiment():
    """排行榜挑战 (leaderboard)"""
    print("=== Running Leaderboard Challenge ===")

    # Use all best practices from previous experiments
    config = GRPOConfig(
        # Best hyperparameters from experiments
        lr=1e-5,
        loss_type="grpo_clip",
        epochs_per_rollout_batch=4,
        train_batch_size=256,  # Fixed: 256 % 256 = 0
        use_std_normalization=False,
        length_norm="masked_normalize",
        advantage_eps=1e-6,
        cliprange=0.2,

        # Extended training
        n_grpo_steps=400,  # More steps for leaderboard

        # Memory optimizations
        grad_accum_steps=256,  # Larger batch
        vllm_gpu_memory_utilization=0.4,  # Less memory for vLLM

        run_name="grpo_leaderboard_final",
        output_dir="results/grpo_leaderboard",
    )

    print("Starting leaderboard challenge with optimized config...")
    final_metrics = run_grpo_experiment(config)

    print("Leaderboard Final Result:")
    print(f"Validation Accuracy: {final_metrics.get('answer_accuracy', 0.0):.4f}")
    print(f"Format Rate: {final_metrics.get('format_rate', 0.0):.4f}")

    return final_metrics


# --------------------------- Main ---------------------------

if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def lr_sweep():
        """运行学习率扫描实验"""
        run_lr_sweep_experiment()

    @app.command()
    def baselines():
        """运行基线实验"""
        run_baseline_experiment()

    @app.command()
    def length_norm():
        """运行长度归一化实验"""
        run_normalization_experiment()

    @app.command()
    def std_norm():
        """运行标准差归一化实验"""
        run_std_norm_experiment()

    @app.command()
    def off_policy_sweep():
        """运行离策略超参数扫描"""
        run_off_policy_sweep_experiment()

    @app.command()
    def clip_ablation():
        """运行裁剪消融实验"""
        run_clip_ablation_experiment()

    @app.command()
    def prompt_ablation():
        """运行提示消融实验"""
        run_prompt_ablation_experiment()

    @app.command()
    def leaderboard():
        """运行排行榜挑战"""
        run_leaderboard_experiment()

    @app.command()
    def all_experiments():
        """运行所有实验（按顺序）"""
        print("Running all GRPO experiments in sequence...")

        print("\n1. Learning Rate Sweep")
        lr_results = run_lr_sweep_experiment()

        print("\n2. Baseline Experiment")
        baseline_results = run_baseline_experiment()

        print("\n3. Length Normalization Experiment")
        norm_results = run_normalization_experiment()

        print("\n4. Standard Deviation Normalization Experiment")
        std_results = run_std_norm_experiment()

        print("\n5. Off-Policy Hyperparameter Sweep")
        off_policy_results = run_off_policy_sweep_experiment()

        print("\n6. Clip Ablation Experiment")
        clip_results = run_clip_ablation_experiment()

        print("\n7. Prompt Ablation Experiment")
        prompt_results = run_prompt_ablation_experiment()

        print("\n8. Leaderboard Challenge")
        leaderboard_results = run_leaderboard_experiment()

        # Summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Best LR: {max(lr_results.keys(), key=lambda x: lr_results[x])}")
        print(f"Best Baseline: {max(baseline_results.keys(), key=lambda x: baseline_results[x])}")
        print(f"Best Length Norm: {max(norm_results.keys(), key=lambda x: norm_results[x])}")
        print(f"Best Std Norm: {max(std_results.keys(), key=lambda x: std_results[x])}")
        print(f"Best Off-Policy: {max(off_policy_results.keys(), key=lambda x: off_policy_results[x])}")
        print(f"Best Clip Setting: {max(clip_results.keys(), key=lambda x: clip_results[x])}")
        print(f"Best Prompt: {max(prompt_results.keys(), key=lambda x: prompt_results[x])}")
        print(f"Leaderboard Score: {leaderboard_results.get('answer_accuracy', 0.0):.4f}")

    app()