# --------------------------- grpo.py (config-based, no argparse) ---------------------------
import os
os.environ["VLLM_USE_V1"] = "0"  # Use vLLM v0 for compatibility

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
from drgrpo_grader import r1_zero_reward_fn


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

    # Loss Type: "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    loss_type: str = "grpo_clip"

    # Training Hyperparams
    lr: float = 1e-5
    grad_accum_steps: int = 128
    max_grad_norm: float = 1.0
    seed: int = 2026

    # Evaluation
    eval_every_steps: int = 10
    eval_max_examples: int = 256 # max=1319

    # Devices
    device_policy: str = "cuda:0"
    device_eval: str = "cuda:1"

    # vLLM memory behavior
    vllm_gpu_memory_utilization: float = 0.5

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


# --------------------------- Prompt helpers ---------------------------

def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        return "{question}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template: str, question: str) -> str:
    return template.format(question=question.strip())


# --------------------------- vLLM helpers ---------------------------

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


# --------------------------- Data loading ---------------------------

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


# --------------------------- Main training ---------------------------

def main(cfg: GRPOConfig):
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
    llm = init_vllm(cfg.base_model, cfg.device_eval, cfg.seed, gpu_memory_utilization=cfg.vllm_gpu_memory_utilization)

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
            reward_fn=r1_zero_reward_fn,
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

            # metrics = evaluate_vllm(
            #     llm,
            #     r1_zero_reward_fn,
            #     val_prompts,
            #     val_gts,
            #     eval_params,
            #     os.path.join(eval_dir, "results.jsonl"),
            #     fast=True,
            # )

            metrics = evaluate_vllm(
                vllm_model=llm,
                reward_fn=r1_zero_reward_fn,
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
            latest_dir = os.path.join(cfg.output_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)
            policy.save_pretrained(latest_dir)
            tokenizer.save_pretrained(latest_dir)

    print("GRPO Training Complete.")


if __name__ == "__main__":
    cfg = GRPOConfig(
        # You can override any fields here, e.g.:
        # n_grpo_steps=50,
        # rollout_batch_size=128,
        # group_size=8,
        # loss_type="grpo_clip",
        # run_name="grpo-qwen-math-1.5b",
    )
    main(cfg)