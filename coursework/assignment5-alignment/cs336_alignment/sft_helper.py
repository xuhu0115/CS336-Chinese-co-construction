
from typing import Any, Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F

def tokenize_prompt_and_output(
        prompt_strs: List[str],
        output_strs: List[str],
        tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    bs = len(prompt_strs)

    prompt_tok = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    output_tok = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    prompt_ids_list = prompt_tok["input_ids"]
    output_ids_list = output_tok["input_ids"]

    # 拼接 full_ids = prompt + output
    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    prompt_lens = [len(p) for p in prompt_ids_list]
    output_lens = [len(o) for o in output_ids_list]

    # pad_id 选择：Qwen2 通常 pad_token_id=None，所以用 eos 作为 pad（常见做法）
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # ========= 关键：先 pad full_ids，再 shift =========
    max_full_len = max(len(x) for x in full_ids_list) if bs > 0 else 0
    full_padded = torch.full((bs, max_full_len), pad_id, dtype=torch.long)

    for i, ids in enumerate(full_ids_list):
        if len(ids) > 0:
            full_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # shift
    input_ids = full_padded[:, :-1].contiguous()
    labels = full_padded[:, 1:].contiguous()

    # response_mask 对齐 labels（只覆盖 output 部分，不包括 padding）
    response_mask = torch.zeros_like(labels, dtype=torch.long)
    for i in range(bs):
        p_len = prompt_lens[i]
        o_len = output_lens[i]
        if o_len == 0:
            continue
        start = max(p_len - 1, 0)
        end = p_len + o_len - 1  # exclusive in labels coords
        end = min(end, labels.size(1))
        if end > start:
            response_mask[i, start:end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy of next-token predictions.

    Args:
        logits: Tensor of shape (batch_size, sequence_length, vocab_size)

    Returns:
        Tensor of shape (batch_size, sequence_length)
    """
    # log Z = logsumexp over vocab dimension
    log_z = torch.logsumexp(logits, dim=-1)  # (B, T)

    # p = softmax(logits)
    probs = torch.softmax(logits, dim=-1)   # (B, T, V)

    # sum_v p_v * l_v
    expected_logit = (probs * logits).sum(dim=-1)  # (B, T)

    # H = log Z - E_p[l]
    entropy = log_z - expected_logit

    return entropy




def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities log p_theta(x_t | x_<t)
    for a causal LM, and optionally per-token entropy.

    Args:
        model: HF causal LM (on correct device; set eval/no_grad outside if desired)
        input_ids: (B, T)
        labels: (B, T)
        return_token_entropy: if True, also return token_entropy (B, T)

    Returns:
        dict with:
          - "log_probs": (B, T)
          - "token_entropy": (B, T) if requested
    """
    # Forward: logits (B, T, V)
    logits = model(input_ids).logits

    # log-probs over vocab: (B, T, V)
    log_probs_vocab = F.log_softmax(logits, dim=-1)

    # Select log-prob of the label token at each position.
    # labels: (B, T) -> (B, T, 1) for gather
    log_probs = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    out: Dict[str, torch.Tensor] = {"log_probs": log_probs}

    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)

    return out



def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Sum tensor elements where mask == 1 and normalize by a constant.
    """
    # 只保留 mask == 1 的位置，其余置 0
    masked_tensor = tensor * mask

    # 求和
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    # 归一化
    return summed / normalize_constant


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    One SFT microbatch step: masked NLL, batch-mean, normalize, grad-acc scaling, backward.
    """
    # mask -> same dtype as log probs
    mask = response_mask.to(dtype=policy_log_probs.dtype)

    # per-token NLL
    per_token_nll = -policy_log_probs  # (B, T)

    # sum over sequence per example (mask out prompt/pad)
    per_example_nll = (per_token_nll * mask).sum(dim=1)  # (B,)

    # normalize by constant (as assignment says)
    per_example_nll = per_example_nll / float(normalize_constant)

    # batch mean
    microbatch_loss = per_example_nll.mean()  # scalar

    # scale for gradient accumulation
    loss = microbatch_loss / float(gradient_accumulation_steps)

    # backward
    loss.backward()

    metadata = {
        "microbatch_loss": microbatch_loss.detach(),
    }
    return loss, metadata


    return log_dict




