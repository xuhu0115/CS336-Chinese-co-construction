from __future__ import annotations

import logging

import typer
import torch
from transformers import AutoTokenizer

from cs336_basics.model import BasicsTransformerLM

logger = logging.getLogger(__name__)


def generate(
    model_path: str,
    prompt: str = "Linda was on a walk in the park",
    device: str = "cuda:0",
    num_samples: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    eos_token_id = tokenizer.eos_token_id
    model = BasicsTransformerLM.from_pretrained(model_path)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for _ in range(num_samples):
            output = model.generate(
                prompt_ids,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
            )
            print("=" * 100)
            print("Prefix: ", prompt)
            print("-" * 100)
            print("Generated: ", tokenizer.decode(output[0].tolist()))
            print("=" * 100)


if __name__ == "__main__":
    """
    Script used to debug that our training script produces a model that generates reasonable text (the validation losses also look good).

    Usage: uv run scripts/generate_with_gpt2_tok.py /path/to/model
    """
    typer.run(generate)
