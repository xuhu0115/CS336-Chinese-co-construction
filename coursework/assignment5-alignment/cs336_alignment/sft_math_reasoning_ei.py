"""
SFT Math Reasoning EI (Expert Iteration)
Implements Expert Iteration for mathematical reasoning on MATH dataset.

Based on sft_ei.py and run_sft_sweep_ei.sh, adapted for the specific requirements.
"""

import os
os.environ["VLLM_USE_V1"] = "0"  # Use vLLM v0 for compatibility

import json
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
import torch.nn.functional as F

import wandb

# Import your components
from sft_helper import tokenize_prompt_and_output, sft_microbatch_train_step
from evaluate_math import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn
from datetime import datetime

# Configuration constants (as specified by user)
BASE_MODEL = "/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B"
TRAIN_DATA = "data/sft/sft_gpt-oss-120b_filtered.jsonl"
VAL_DATA = "data/gsm8k/test.jsonl"
PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"

# Training hyperparameters (following run_sft_sweep_ei.sh)
BATCH_SIZE = 4
GRAD_ACCUM = 8
LR = 5e-5
MAX_GRAD_NORM = 1.0
N_EI_STEPS = 5

# Dataset sizes to sweep (expert_batch_size Db)
EXPERT_BATCH_SIZES = [512, 1024, 2048]

# Rollouts per question (G)
ROLLOUTS_PER_QUESTION = [1, 4, 8, 16]

# SFT epochs per step
SFT_EPOCHS_PER_STEP = [1, 4, 8, 16]

SEED = 2026

# --- Helper: Prompt Formatting ---
def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        # Fallback if file missing
        return (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
            "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
            "User: {question}\n"
            "Assistant: <think>"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def format_prompt(template: str, question: str) -> str:
    """Applies the R1-Zero template to the raw math question."""
    return template.format(question=question.strip())

# --- Helper: Entropy Calculation ---
def compute_mean_entropy(logits, mask):
    """
    Computes mean entropy of the predicted tokens (masked).
    logits: (B, Seq, V)
    mask: (B, Seq)
    """
    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropy = - sum(p * log p)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, Seq)

    # Apply mask
    masked_entropy = entropy * mask

    # Avoid division by zero
    sum_mask = mask.sum()
    if sum_mask == 0:
        return torch.tensor(0.0, device=logits.device)

    return masked_entropy.sum() / sum_mask

# --- vLLM Setup ---
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.50):
    """Initialize vLLM for evaluation."""
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
        )

def load_policy_into_vllm(policy, llm):
    """Load policy weights into vLLM instance for evaluation."""
    print("Syncing Policy weights to vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

def log_generations_to_wandb(results_path: str, step: int, run_name: str):
    """Log generation results to wandb table."""
    if not os.path.exists(results_path):
        return

    table_data = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if len(table_data) < 20:  # Limit to 20 examples
                table_data.append([
                    step,
                    rec['prompt'][:100] + "..." if len(rec['prompt']) > 100 else rec['prompt'],
                    rec['model_response'][:200] + "..." if len(rec['model_response']) > 200 else rec['model_response'],
                    rec['ground_truth'],
                    rec['rewards']['format_reward'],
                    rec['rewards']['answer_reward']
                ])

    if table_data:
        wandb.log({
            "generations": wandb.Table(
                columns=["step", "prompt", "generation", "gold", "format_rew", "ans_rew"],
                data=table_data
            )
        })

# --- Main Expert Iteration Logic ---

def run_expert_iteration_experiment(
    expert_batch_size: int = 512,
    rollouts_per_question: int = 4,
    sft_epochs_per_step: int = 1,
    experiment_tag: str = "default"
):
    """Run a single Expert Iteration experiment."""

    print(f"\n{'='*60}")
    print(f"EXPERT ITERATION EXPERIMENT: {experiment_tag}")
    print(f"Db={expert_batch_size}, G={rollouts_per_question}, Epochs={sft_epochs_per_step}")
    print(f"{'='*60}")

    # Setup wandb
    run_name = f"ei_{experiment_tag}_Db{expert_batch_size}_G{rollouts_per_question}_Ep{sft_epochs_per_step}"
    wandb.init(project="cs336-a5-sft-ei", name=run_name, mode="offline", config={
        "expert_batch_size": expert_batch_size,
        "rollouts_per_question": rollouts_per_question,
        "sft_epochs_per_step": sft_epochs_per_step,
        "n_ei_steps": N_EI_STEPS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR
    })

    # Set seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Load Prompt Template
    template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    print("Loaded Prompt Template.")

    # 2. Load Training Questions (Robustly)
    all_questions = []
    print(f"Loading questions from {TRAIN_DATA}...")
    with open(TRAIN_DATA, 'r', encoding='utf-8') as f:
        # Check if file starts with '[' (JSON List) or '{' (JSONL)
        start_char = f.read(1)
        f.seek(0)
        if start_char == '[':
            print("Detected JSON List format.")
            all_questions = json.load(f)
        else:
            print("Detected JSONL format.")
            for line in f:
                if line.strip():
                    all_questions.append(json.loads(line))

    print(f"Loaded {len(all_questions)} training examples.")

    # 3. Initialize Policy Model (GPU 0)
    print("Initializing Policy Model on cuda:0...")
    device_policy = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    policy = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)

    # 4. Initialize vLLM (GPU 1)
    print("Initializing vLLM on cuda:1...")
    device_eval = "cuda:1"
    llm = init_vllm(BASE_MODEL, device_eval, SEED)

    # 5. Prepare Dataset (Sample questions ONCE for all EI steps)
    print(f"Sampling {expert_batch_size} questions for Expert Iteration...")
    current_db_size = min(len(all_questions), expert_batch_size)
    batch_questions = random.sample(all_questions, current_db_size)

    # Pre-format prompts
    prompts = [format_prompt(template, q.get('problem', q.get('question', ''))) for q in batch_questions]
    ground_truths = [q.get('answer', q.get('expected_answer', '')) for q in batch_questions]

    print(f"Prepared {len(prompts)} prompts for Expert Iteration.")

    # 6. Sampling Parameters
    rollout_params = SamplingParams(
        temperature=1.0,  # High temp for diversity in Expert Iteration
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,  # Prevent empty responses
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=rollouts_per_question  # Generate G outputs per prompt
    )

    eval_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # --- EXPERT ITERATION LOOP ---
    for step in range(1, N_EI_STEPS + 1):
        print(f"\n=== Expert Iteration Step {step}/{N_EI_STEPS} ===")

        # A. Update vLLM with current policy weights
        load_policy_into_vllm(policy, llm)

        # B. Generate Rollouts
        print(f"Generating {len(prompts) * rollouts_per_question} rollouts...")
        outputs = llm.generate(prompts, rollout_params)

        # C. Filter for Correctness
        new_sft_data = []
        correct_count = 0
        total_gen = 0

        for i, req_output in enumerate(outputs):
            gt = ground_truths[i]
            prompt = prompts[i]

            for completion in req_output.outputs:
                total_gen += 1
                generated_text = completion.text

                # Check correctness using reward function
                score = r1_zero_reward_fn(generated_text, gt)

                if score['answer_reward'] == 1.0:
                    correct_count += 1
                    new_sft_data.append({
                        "prompt": prompt,
                        "response": generated_text
                    })

        correct_rate = correct_count / total_gen if total_gen > 0 else 0.0
        print(f"Step {step}: Generated {total_gen}, Kept {correct_count} ({correct_rate:.2%}) correct traces.")

        wandb.log({
            "ei/step": step,
            "ei/correct_rate": correct_rate,
            "ei/dataset_size": len(new_sft_data)
        })

        if not new_sft_data:
            print("No correct samples found! Skipping training step.")
            continue

        # D. SFT Training Step
        print(f"Training on {len(new_sft_data)} examples for {sft_epochs_per_step} epochs...")
        policy.train()

        batch_loss = 0
        batch_entropy = 0
        optimizer.zero_grad()

        micro_step = 0
        global_step = (step - 1) * sft_epochs_per_step  # For wandb logging

        for epoch in range(sft_epochs_per_step):
            random.shuffle(new_sft_data)

            for i in range(0, len(new_sft_data), BATCH_SIZE):
                micro_step += 1

                batch = new_sft_data[i : i + BATCH_SIZE]
                p_list = [x['prompt'] for x in batch]
                r_list = [x['response'] for x in batch]

                # Tokenize
                tokenized = tokenize_prompt_and_output(p_list, r_list, tokenizer)
                input_ids = tokenized["input_ids"].to(device_policy)
                labels = tokenized["labels"].to(device_policy)
                mask = tokenized["response_mask"].to(device_policy)

                # Forward
                logits = policy(input_ids).logits

                # Entropy Calculation (on response tokens only)
                with torch.no_grad():
                    ent = compute_mean_entropy(logits, mask)
                    batch_entropy += ent.item()

                # Loss
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)

                loss, _ = sft_microbatch_train_step(token_log_probs, mask, GRAD_ACCUM)
                batch_loss += loss.item()

                # Optimizer step
                if micro_step % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log training metrics
                    wandb.log({
                        "train/loss": batch_loss,
                        "train/entropy": batch_entropy / GRAD_ACCUM,
                        "global_step": global_step + epoch
                    })
                    batch_loss = 0
                    batch_entropy = 0

        # E. Evaluation after this EI step
        print(f"Evaluating after EI step {step}...")
        load_policy_into_vllm(policy, llm)

        eval_dir = f"results/ei_experiments_{experiment_tag}_Db{expert_batch_size}_G{rollouts_per_question}_Ep{sft_epochs_per_step}/step_{step}"
        os.makedirs(eval_dir, exist_ok=True)

        metrics = evaluate_vllm(
            vllm_model=llm,
            reward_fn=r1_zero_reward_fn,
            dataset_path=VAL_DATA,
            prompt_template=template,
            eval_sampling_params=eval_params,
            output_filepath=os.path.join(eval_dir, "results.jsonl")
        )

        wandb.log({
            "eval/acc": metrics["answer_accuracy"],
            "eval/format_rate": metrics["format_rate"],
            "ei_step": step
        })

        print(f"Validation Accuracy: {metrics['answer_accuracy']:.4f}")

        # Log generations to wandb
        log_generations_to_wandb(
            results_path=os.path.join(eval_dir, "results.jsonl"),
            step=step,
            run_name=f"ei_{experiment_tag}"
        )

        # Save checkpoint (only keep latest)
        checkpoint_dir = f"results/ei_experiments_{experiment_tag}_Db{expert_batch_size}_G{rollouts_per_question}_Ep{sft_epochs_per_step}/latest"
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    print("Expert Iteration Complete.")
    wandb.finish()

def main():
    """Main function to run all Expert Iteration experiments."""

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("Starting Expert Iteration experiments on MATH dataset...")

    # 1. Batch Size Sweep (Fix G=4, Ep=4, Vary Db)
    print("\n=== PART 1: Expert Batch Size Sweep ===")
    for db in EXPERT_BATCH_SIZES:
        run_expert_iteration_experiment(
            expert_batch_size=db,
            rollouts_per_question=4,
            sft_epochs_per_step=4,
            experiment_tag="batch_sweep"
        )

    # 2. Rollout & Epoch Sweep (Fix Db=512)
    print("\n=== PART 2: Rollout and Epoch Sweep ===")

    # Different rollout numbers
    for rollouts in ROLLOUTS_PER_QUESTION:
        if rollouts != 4:  # Skip G=4 as it was already run above
            run_expert_iteration_experiment(
                expert_batch_size=512,
                rollouts_per_question=rollouts,
                sft_epochs_per_step=4,
                experiment_tag="rollout_sweep"
            )

    # Different epoch numbers
    for i, epochs in enumerate(SFT_EPOCHS_PER_STEP):
        if epochs not in [4]:  # Skip already run combinations
            tag = "epoch_sweep" if i == 1 else "combo_sweep"  # Ep=8 -> epoch_sweep, Ep=16 -> combo_sweep
            run_expert_iteration_experiment(
                expert_batch_size=512,
                rollouts_per_question=4,
                sft_epochs_per_step=epochs,
                experiment_tag=tag
            )

    print("\nAll Expert Iteration experiments completed!")

if __name__ == "__main__":
    main()