"""
SFT Math Reasoning Script
Based on sft.py and run_sft.sh logic, adapted for the specific requirements.
"""

import os
os.environ["VLLM_USE_V1"] = "0" # LLM v0.9.x 虽然引入了 V1 引擎，但仍然保留了完整的 V0 引擎代码。设置 VLLM_USE_V1=0 会告诉 vLLM 初始化旧版的 LLMEngine
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

import wandb

# Import your components
from sft_helper import tokenize_prompt_and_output, sft_microbatch_train_step
from log import log_generations
from evaluate_math import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn
# from datetime import datetime

# Configuration constants (as specified)
BASE_MODEL = "/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B"
VAL_PATH = "data/gsm8k/test.jsonl"
RAW_TRAIN = "data/sft/sft_gpt-oss-120b.jsonl"
FILTERED_TRAIN = "data/sft/sft_gpt-oss-120b_filtered.jsonl"
PROMPT_TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"

# Training hyperparameters (following run_sft.sh)
BATCH_SIZE = 4
GRAD_ACCUM = 8
LR = 5e-5
EPOCHS = 1
EVAL_EVERY_STEPS = 5
MAX_GRAD_NORM = 1.0
SEED = 2026

# Dataset sizes to sweep
DATASET_SIZES = [128, 256, 512, 1024, 2048]

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

# --- vLLM Helper Functions ---
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.50):
    
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

def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    print("Loading policy weights into vLLM...")
    state_dict = policy.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
    print("Weights loaded.")

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

# --- Main Training Logic ---

def run_sft_experiment(train_data_path: str, max_examples: int = -1,
                      dataset_tag: str = "raw", size_tag: str = "full"):
    """Run a single SFT experiment."""

    print(f"\n{'='*50}")
    print(f"RUNNING SFT EXPERIMENT: {dataset_tag} | size={size_tag}")
    print(f"Train data: {train_data_path}")
    print(f"Val data: {VAL_PATH}")
    print(f"Max examples: {max_examples}")
    print(f"{'='*50}")

    # Setup wandb
    run_name = f"sft_{dataset_tag}_{size_tag}_{max_examples if max_examples > 0 else 'full'}"
    wandb.init(project="cs336-a5-sft-v2", name=run_name, mode="offline", config={
        "dataset_tag": dataset_tag,
        "size_tag": size_tag,
        "max_examples": max_examples,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "epochs": EPOCHS
    })

    # Set seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Load Prompt Template
    template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    print("Loaded Prompt Template.")

    # 2. Load Training Dataset (following sft.py logic)
    print(f"Loading training data from {train_data_path}...")
    sft_data = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        # Check first char to see if it's a list '[' or a dict '{'
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # Case A: It is a JSON List (Your format)
            print("Detected JSON List format.")
            sft_data = json.load(f)
        else:
            # Case B: It is JSONL (Line-by-line)
            print("Detected JSONL format.")
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    sft_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Subsample if needed
    if max_examples > 0 and len(sft_data) > max_examples:
        sft_data = random.sample(sft_data, max_examples)
        print(f"Subsampled training data to {len(sft_data)} examples.")

    # 3. Validation data will be loaded directly by evaluate_math.py

    # 4. Initialize Policy Model (GPU 0)
    print("Initializing Policy Model on cuda:0...")
    device_policy = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device_policy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 5. Initialize vLLM (GPU 1)
    print("Initializing vLLM on cuda:1...")
    device_eval = "cuda:1"
    llm = init_vllm(BASE_MODEL, device_eval, SEED)

    # Evaluation sampling parameters
    eval_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # 6. Training Loop
    global_step = 0
    model.train()

    # Calculate eval frequency (roughly every half epoch)
    steps_per_epoch = max(1, len(sft_data) // BATCH_SIZE)
    eval_every_steps = EVAL_EVERY_STEPS
    print(f"Training on {len(sft_data)} examples, {steps_per_epoch} steps per epoch, eval every {eval_every_steps} steps")

    eval_count = 0
    batch_loss = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        random.shuffle(sft_data)

        for i in tqdm(range(0, len(sft_data), BATCH_SIZE), desc=f"Epoch {epoch + 1}"):
            # 1. 准备批次数据
            batch_data = sft_data[i : i + BATCH_SIZE]
            if not batch_data:
                continue

            # 2. 格式化提示和响应
            prompt_strs = []
            output_strs = []

            for x in batch_data:
                question = x.get('problem', x.get('question', ''))
                formatted_prompt = format_prompt(template, question)

                # Handle different response keys
                if 'reasoning_trace' in x:
                    response = x['reasoning_trace']
                elif 'response' in x:
                    response = x['response']
                else:
                    continue

                prompt_strs.append(formatted_prompt)
                output_strs.append(response)

            if not prompt_strs:
                continue

            # 3. 分词
            tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)
            
            # 4. 前向传播
            outputs = model(input_ids)
            logits = outputs.logits

            log_probs = torch.log_softmax(logits, dim=-1)
            per_token_log_probs = torch.gather(
                log_probs,
                dim=2,
                index=labels.unsqueeze(2)
            ).squeeze(2)

            # 5. 计算损失并反向传播
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=per_token_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=GRAD_ACCUM
            )
            batch_loss += loss.item()

            # 6. 梯度累积和优化器步骤
            if (global_step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()

                # Log training metrics
                wandb.log({
                    "train/loss": batch_loss,
                    "train_step": global_step
                })
                batch_loss = 0.0

            # 7. 定期评估
            if (global_step + 1) % eval_every_steps == 0:
                print(f"Step {global_step}: Evaluating...")
                model.eval()
                load_policy_into_vllm_instance(model, llm)

                eval_dir = f"results/sft_experiments_{dataset_tag}_{size_tag}/step_{global_step}"
                os.makedirs(eval_dir, exist_ok=True)

                # Use GSM8K test set for evaluation
                metrics = evaluate_vllm(
                    vllm_model=llm,
                    reward_fn=r1_zero_reward_fn,
                    dataset_path=VAL_PATH,
                    prompt_template=template,
                    eval_sampling_params=eval_sampling_params,
                    output_filepath=os.path.join(eval_dir, "results.jsonl")
                )

                wandb.log({
                    "eval/acc": metrics["answer_accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval_step": eval_count,
                    "global_step": global_step
                })
                eval_count += 1

                print(f"Eval Acc: {metrics['answer_accuracy']:.4f}")

                # Log generations to wandb
                log_generations_to_wandb(
                    results_path=os.path.join(eval_dir, "results.jsonl"),
                    step=global_step,
                    run_name=f"sft_{dataset_tag}_{size_tag}"
                )

                model.train()

            global_step += 1

    # Save final model
    # print("Saving final model...")
    # final_save_dir = os.path.join(f"results/sft_experiments_{dataset_tag}_{size_tag}", "latest")
    # model.save_pretrained(final_save_dir)
    # tokenizer.save_pretrained(final_save_dir)
    # print(f"Final model saved to {final_save_dir}")

    print("Training Complete.")
    wandb.finish()

def main():
    """Main function to run all SFT experiments."""

    # Create results directory
    os.makedirs("results", exist_ok=True)

    print("Starting SFT experiments on MATH dataset...")

    # Part 1: Raw dataset sweep with different sizes
    print("\n=== PART 1: Raw Dataset Sweep ===")
    for size in DATASET_SIZES:
        run_sft_experiment(
            train_data_path=RAW_TRAIN,
            max_examples=size,
            dataset_tag="raw",
            size_tag=str(size)
        )

    # Raw dataset full
    print("\n=== PART 1b: Raw Dataset Full ===")
    run_sft_experiment(
        train_data_path=RAW_TRAIN,
        max_examples=-1,  # Use full dataset
        dataset_tag="raw",
        size_tag="full"
    )

    # Part 2: Filtered dataset experiments — now loaded directly from file
    print("\n=== PART 2: Filtered Dataset Experiments (loaded from pre-saved file) ===")

    # Load the number of examples in filtered dataset to support subsampling
    with open(FILTERED_TRAIN, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            filtered_data = json.load(f)
        else:
            filtered_data = [json.loads(line) for line in f if line.strip()]
    filtered_size = len(filtered_data)

    # Run filtered experiments with different sizes
    for size in DATASET_SIZES:
        if size <= filtered_size:
            run_sft_experiment(
                train_data_path=FILTERED_TRAIN,
                max_examples=size,
                dataset_tag="filtered",
                size_tag=str(size)
            )

    # Filtered full
    run_sft_experiment(
        train_data_path=FILTERED_TRAIN,
        max_examples=-1,
        dataset_tag="filtered",
        size_tag="full"
    )

    print("\nAll SFT experiments completed!")

if __name__ == "__main__":
    main()