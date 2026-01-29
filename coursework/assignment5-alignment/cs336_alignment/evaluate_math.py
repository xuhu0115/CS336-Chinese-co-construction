import json
import os
from typing import Callable, List, Dict, Any
from collections import Counter

from vllm import LLM, SamplingParams
import torch

from drgrpo_grader import r1_zero_reward_fn

# Define the model path and prompt path
MODEL_PATH = "/home/magnus-share/xuhu/model/Qwen2___5-Math-1___5B"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
MATH_VALIDATION_PATH = "data/gsm8k/test.jsonl"
OUTPUT_DIR = "results/base"

def load_r1_zero_prompt(prompt_file_path: str) -> str:
    """Loads the r1_zero prompt template from a file."""
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_prompt(question: str, prompt_template: str) -> str:
    """Formats the question into the r1_zero prompt template."""
    return prompt_template.format(question=question)

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str, bool], Dict[str, float]], # Added bool for 'fast'
    dataset_path: str,
    prompt_template: str,
    eval_sampling_params: SamplingParams,
    output_filepath: str,
    fast: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))

    questions = []
    ground_truths = []
    prompts = []
    
    # Handle both JSONL and JSON array formats
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            # JSON array format (MATH data)
            data = json.load(f)
        else:
            # JSONL format (GSM8K data)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    for example in data:
        # Handle different field names
        question = example.get('question', example.get('problem', ''))
        if 'expected_answer' in example:
            # MATH format: direct answer
            answer = example['expected_answer']
        elif 'answer' in example:
            # GSM8K format: extract final answer after ####
            answer_text = example['answer']
            import re
            m = re.search(r"####\s*(.+)\s*$", answer_text.strip())
            answer = m.group(1).strip() if m else answer_text.strip()
        else:
            answer = ''

        questions.append(question)
        ground_truths.append(answer)
        prompts.append(format_prompt(question, prompt_template))

    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_answer_reward = 0.0
    total_format_reward = 0.0
    total_reward = 0.0
    combo_counts = Counter()

    for i, output in enumerate(outputs):
        # Fix for potential empty outputs
        generated_text = output.outputs[0].text if output.outputs else ""
        
        # Use r1_zero_reward_fn which expects `response` and `ground_truth`
        # The `response` for r1_zero_reward_fn is the raw model generated text
        # and it handles the extraction of the answer part itself.
        rewards = reward_fn(generated_text, ground_truths[i], fast=fast)

        fr = int(rewards.get("format_reward", 0.0) >= 0.5)
        ar = int(rewards.get("answer_reward", 0.0) >= 0.5)
        combo_counts[(fr, ar)] += 1

        results.append({
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "prompt": output.prompt, # Use output.prompt for exact prompt sent to vLLM
            "model_response": generated_text,
            "rewards": rewards
        })
        total_answer_reward += rewards["answer_reward"]
        total_format_reward += rewards["format_reward"]
        total_reward += rewards["reward"]

    num_examples = len(results)
    avg_answer_reward = (total_answer_reward / num_examples) if num_examples else 0.0
    avg_format_reward = (total_format_reward / num_examples) if num_examples else 0.0
    avg_reward = (total_reward / num_examples) if num_examples else 0.0

    print(f"Evaluation Results:")
    print(f"  Average Answer Reward: {avg_answer_reward:.4f}")
    print(f"  Average Format Reward: {avg_format_reward:.4f}")
    print(f"  Average Total Reward: {avg_reward:.4f}")
    print(f"  Combo Counts: {combo_counts}")

    with open(output_filepath, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_filepath}")

    combo_table = {
        "format=1 answer=1": combo_counts[(1, 1)],
        "format=1 answer=0": combo_counts[(1, 0)],
        "format=0 answer=0": combo_counts[(0, 0)],
        "format=0 answer=1": combo_counts[(0, 1)], # Should ideally be 0
    }

    metrics = {
        "n": num_examples,
        "format_rate": avg_format_reward,
        "answer_accuracy": avg_answer_reward,
        "reward_mean": avg_reward,
        "counts": combo_table,
    }
    return metrics

if __name__ == "__main__":
    # Load prompt template
    r1_zero_template = load_r1_zero_prompt(PROMPT_PATH)

    # Initialize vLLM
    print(f"Initializing vLLM model from {MODEL_PATH}...")
    # Removed `torch.bfloat16` as `LLM` init doesn't take it directly, `dtype` parameter handles it.
    llm = LLM(model=MODEL_PATH, dtype=torch.bfloat16) 

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Define output file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "zero_shot_math_evaluation.jsonl")

    # Run evaluation
    evaluation_metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        dataset_path=MATH_VALIDATION_PATH,
        prompt_template=r1_zero_template,
        eval_sampling_params=sampling_params,
        output_filepath=output_file,
        fast=True,
    )
    
    print("\nFinal Evaluation Metrics:")
    print(json.dumps(evaluation_metrics, indent=4, ensure_ascii=False))
