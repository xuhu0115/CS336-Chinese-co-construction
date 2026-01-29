from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Any,Literal
import torch


def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], Dict[str, float]],
        rollout_responses: List[str],
        repeated_ground_truths: List[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute per-response rewards and group-normalized advantages.

    Args:
        reward_fn: Callable[[response, ground_truth], dict] returning keys:
                  "reward", "format_reward", "answer_reward"
        rollout_responses: list[str], length = rollout_batch_size
        repeated_ground_truths: list[str], length = rollout_batch_size
        group_size: int, number of responses per question
        advantage_eps: float, small constant to avoid division by zero
        normalize_by_std: bool, if True use (r-mean)/(std+eps) else (r-mean)

    Returns:
        advantages: torch.Tensor, shape (rollout_batch_size,)
        raw_rewards: torch.Tensor, shape (rollout_batch_size,)
        metadata: dict[str, float] with useful stats
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            f"rollout_responses and repeated_ground_truths must have same length, "
            f"got {len(rollout_responses)} vs {len(repeated_ground_truths)}"
        )
    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size == 0:
        # Return empty tensors consistently
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty, {"rollout_batch_size": 0.0}

    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if rollout_batch_size % group_size != 0:
        raise ValueError(
            f"rollout_batch_size must be divisible by group_size, got "
            f"{rollout_batch_size} % {group_size} != 0"
        )

    # ---- 1) Compute raw rewards for each rollout response ----
    raw_rewards_list: List[float] = []
    raw_format_list: List[float] = []
    raw_answer_list: List[float] = []

    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(resp, gt)  # expected keys: reward/format_reward/answer_reward
        # Be strict so bugs surface early
        if "reward" not in scores:
            raise KeyError(f"reward_fn output missing key 'reward': {scores.keys()}")
        raw_rewards_list.append(float(scores["reward"]))
        raw_format_list.append(float(scores.get("format_reward", 0.0)))
        raw_answer_list.append(float(scores.get("answer_reward", 0.0)))

    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    raw_format = torch.tensor(raw_format_list, dtype=torch.float32)
    raw_answer = torch.tensor(raw_answer_list, dtype=torch.float32)

    # ---- 2) Group-normalize ----
    n_groups = rollout_batch_size // group_size
    rewards_g = raw_rewards.view(n_groups, group_size)

    group_mean = rewards_g.mean(dim=1, keepdim=True)  # (n_groups, 1)
    centered = rewards_g - group_mean  # (n_groups, group_size)

    if normalize_by_std:
        # std per group; use unbiased=False for stability (population std)
        group_std = rewards_g.std(dim=1, keepdim=True, unbiased=True)  # (n_groups, 1)
        denom = group_std + float(advantage_eps)
        advantages_g = centered / denom
        zero_std_groups = (group_std.squeeze(1) == 0).sum().item()
    else:
        advantages_g = centered
        group_std = rewards_g.std(dim=1, keepdim=True, unbiased=True)
        zero_std_groups = (group_std.squeeze(1) == 0).sum().item()

    advantages = advantages_g.reshape(-1)  # (rollout_batch_size,)

    # ---- 3) Metadata (all floats) ----
    # Group-level stats (over groups)
    group_means = rewards_g.mean(dim=1)  # (n_groups,)
    group_stds = rewards_g.std(dim=1, unbiased=False)  # (n_groups,)

    metadata: Dict[str, float] = {
        "rollout_batch_size": float(rollout_batch_size),
        "n_groups": float(n_groups),
        "group_size": float(group_size),
        "normalize_by_std": float(1.0 if normalize_by_std else 0.0),
        "advantage_eps": float(advantage_eps),
        # Raw reward stats (overall)
        "raw_reward_mean": float(raw_rewards.mean().item()),
        "raw_reward_std": float(raw_rewards.std(unbiased=False).item()),
        "raw_reward_min": float(raw_rewards.min().item()),
        "raw_reward_max": float(raw_rewards.max().item()),
        # Optional sub-reward stats (overall)
        "raw_format_reward_mean": float(raw_format.mean().item()),
        "raw_answer_reward_mean": float(raw_answer.mean().item()),
        # Group stats
        "group_reward_mean_mean": float(group_means.mean().item()),
        "group_reward_mean_std": float(group_means.std(unbiased=False).item()),
        "group_reward_std_mean": float(group_stds.mean().item()),
        "zero_std_groups": float(zero_std_groups),
        # Advantage stats (overall)
        "adv_mean": float(advantages.mean().item()),
        "adv_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "adv_min": float(advantages.min().item()),
        "adv_max": float(advantages.max().item()),
    }

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token naive policy gradient loss:
        loss_{b,t} = -A_b * log_prob_{b,t}

    Args:
        raw_rewards_or_advantages: Tensor of shape (batch_size, 1)
        policy_log_probs: Tensor of shape (batch_size, sequence_length)

    Returns:
        Tensor of shape (batch_size, sequence_length)
    """
    if raw_rewards_or_advantages.dim() != 2 or raw_rewards_or_advantages.shape[1] != 1:
        raise ValueError(
            f"raw_rewards_or_advantages must have shape (B, 1), got {tuple(raw_rewards_or_advantages.shape)}"
        )
    if policy_log_probs.dim() != 2:
        raise ValueError(
            f"policy_log_probs must have shape (B, T), got {tuple(policy_log_probs.shape)}"
        )
    if raw_rewards_or_advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            f"Batch size mismatch: {raw_rewards_or_advantages.shape[0]} vs {policy_log_probs.shape[0]}"
        )

    # Broadcast (B,1) -> (B,T) and compute per-token loss
    advantages = raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    loss = -(advantages * policy_log_probs)  # broadcasting along T

    return loss

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
        loss_type: Literal["grpo_clip", "grpo_no_clip"] = "grpo_clip",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Per-token GRPO-Clip loss:
        loss_{b,t} = - min( r_{b,t} * A_b, clip(r_{b,t}, 1-eps, 1+eps) * A_b )

    where r_{b,t} = exp(policy_log_probs - old_log_probs).

    Args:
        advantages: (B, 1)
        policy_log_probs: (B, T)
        old_log_probs: (B, T)
        cliprange: eps

    Returns:
        loss: (B, T)
        metadata: dict of tensors (e.g., is_clipped mask)
    """
    if advantages.dim() != 2 or advantages.shape[1] != 1:
        raise ValueError(f"advantages must have shape (B, 1), got {tuple(advantages.shape)}")
    if policy_log_probs.dim() != 2 or old_log_probs.dim() != 2:
        raise ValueError(
            f"policy_log_probs and old_log_probs must have shape (B, T), got "
            f"{tuple(policy_log_probs.shape)} and {tuple(old_log_probs.shape)}"
        )
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError(
            f"policy_log_probs and old_log_probs must have same shape, got "
            f"{tuple(policy_log_probs.shape)} vs {tuple(old_log_probs.shape)}"
        )
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            f"Batch size mismatch: advantages {advantages.shape[0]} vs log_probs {policy_log_probs.shape[0]}"
        )

    # Broadcast A: (B,1) -> (B,T)
    A = advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)

    # ratio r = pi_theta / pi_old = exp(log_pi - log_pi_old)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    if loss_type == "grpo_no_clip":
        # Eq.(34): per-token loss = - r_{b,t} * A_b
        loss = -(ratio * A)  # (B,T)
        metadata: Dict[str, torch.Tensor] = {
            "ratio": ratio,
            "log_ratio": log_ratio,
            "is_clipped": torch.zeros_like(ratio, dtype=torch.bool),
            "unclipped_obj": ratio * A,
        }
        return loss, metadata

    elif loss_type == "grpo_clip":
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        unclipped_obj = ratio * A
        clipped_obj = clipped_ratio * A

        min_obj = torch.minimum(unclipped_obj, clipped_obj)
        loss = -min_obj

        is_clipped = clipped_obj < unclipped_obj
        metadata = {
            "ratio": ratio,
            "log_ratio": log_ratio,
            "clipped_ratio": clipped_ratio,
            "is_clipped": is_clipped,
            "unclipped_obj": unclipped_obj,
            "clipped_obj": clipped_obj,
        }
        return loss, metadata


def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convenience wrapper to compute different policy-gradient losses.

    Args:
        policy_log_probs: (B, T) log-probabilities from current policy.
        loss_type: "no_baseline" | "reinforce_with_baseline" | "grpo_clip"
        raw_rewards: required if loss_type == "no_baseline", shape (B, 1)
        advantages: required if loss_type in {"reinforce_with_baseline", "grpo_clip"}, shape (B, 1)
        old_log_probs: required if loss_type == "grpo_clip", shape (B, T)
        cliprange: required if loss_type == "grpo_clip", scalar eps

    Returns:
        loss: (B, T) per-token loss
        metadata: dict[str, Tensor] auxiliary stats
    """
    if policy_log_probs.dim() != 2:
        raise ValueError(f"policy_log_probs must have shape (B, T), got {tuple(policy_log_probs.shape)}")

    metadata: Dict[str, torch.Tensor] = {}

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required when loss_type == 'no_baseline'")
        # raw_rewards expected shape (B,1); compute_naive_policy_gradient_loss checks it
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata["used_advantages"] = raw_rewards.detach()

    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required when loss_type == 'reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata["used_advantages"] = advantages.detach()

    elif loss_type == "grpo_clip" or loss_type == "grpo_no_clip":
        if advantages is None:
            raise ValueError("advantages is required when loss_type == 'grpo_clip'")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required when loss_type == 'grpo_clip'")
        if cliprange is None:
            raise ValueError("cliprange is required when loss_type == 'grpo_clip'")

        loss, clip_meta = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=float(cliprange),
            loss_type=loss_type,
        )
        metadata.update(clip_meta)

        # 常用额外统计：clip fraction（被 clip 的 token 比例）
        if "is_clipped" in clip_meta:
            metadata["clip_fraction"] = clip_meta["is_clipped"].to(torch.float32).mean()

        metadata["used_advantages"] = advantages.detach()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Compute mean of `tensor` considering only elements where mask == 1.

    Args:
        tensor: torch.Tensor, data to average
        mask: torch.Tensor, same shape as tensor; 1/True positions included
        dim: int or None. If None, mean over all masked elements.

    Returns:
        torch.Tensor, masked mean with semantics like tensor.mean(dim)
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"tensor and mask must have same shape, got {tensor.shape} vs {mask.shape}")

    # Make mask numeric and on same device/dtype for multiplication
    m = mask.to(device=tensor.device, dtype=tensor.dtype)

    masked_sum = (tensor * m).sum() if dim is None else (tensor * m).sum(dim=dim)
    denom = m.sum() if dim is None else m.sum(dim=dim)

    return masked_sum / denom

def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
        constant_normalizer: float = 1.0,
) -> torch.Tensor:
    """
    Sum `tensor` over masked elements, then divide by a constant scalar (not by masked count).

    This is useful for length normalization where you want each token to have the same weight
    regardless of sequence length.

    Args:
        tensor: torch.Tensor, same shape as mask (e.g., (B,T))
        mask: torch.Tensor, same shape as tensor; 1/True positions included
        dim: int or None. If None, normalize over all elements (single scalar).
        constant_normalizer: scalar C to divide by (e.g., max_gen_len)

    Returns:
        torch.Tensor with semantics like tensor.sum(dim) / C
    """
    if tensor.shape != mask.shape:
        raise ValueError(f"tensor and mask must have same shape, got {tensor.shape} vs {mask.shape}")
    if constant_normalizer <= 0:
        raise ValueError(f"constant_normalizer must be > 0, got {constant_normalizer}")

    m = mask.to(device=tensor.device, dtype=tensor.dtype)

    masked_sum = (tensor * m).sum() if dim is None else (tensor * m).sum(dim=dim)

    return masked_sum / float(constant_normalizer)


def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip","grpo_no_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        length_norm: Literal["masked_mean", "masked_normalize"] = "masked_mean",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute one GRPO microbatch forward+backward step.

    Steps:
      1) compute per-token PG loss (B,T)
      2) masked_mean over response tokens -> per-example loss (B,)
      3) batch mean -> microbatch_loss (scalar)
      4) scale by gradient_accumulation_steps
      5) backward
    """
    if policy_log_probs.dim() != 2:
        raise ValueError(f"policy_log_probs must be (B,T), got {tuple(policy_log_probs.shape)}")
    if response_mask.shape != policy_log_probs.shape:
        raise ValueError(
            f"response_mask must match policy_log_probs shape, got "
            f"{tuple(response_mask.shape)} vs {tuple(policy_log_probs.shape)}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be > 0, got {gradient_accumulation_steps}")

    # 1) per-token loss
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )  # (B,T)

    # 2) masked mean over response tokens => (B,)
    mask = response_mask.to(dtype=per_token_loss.dtype, device=per_token_loss.device)
    if length_norm == "masked_mean":
        per_example_loss = masked_mean(per_token_loss, mask, dim=1)  # (B,)
    elif length_norm == "masked_normalize":
        per_example_loss = masked_normalize(per_token_loss, mask, dim=1, constant_normalizer=1024)  # (B,)
    else:
        raise ValueError(f"Unknown length_norm: {length_norm}")

    # 3) batch mean => scalar
    microbatch_loss = per_example_loss.mean()

    # 4) grad-acc scaling
    loss = microbatch_loss / float(gradient_accumulation_steps)

    # 5) backward
    loss.backward()

    # ---- metadata to log ----
    out_meta: Dict[str, torch.Tensor] = dict(meta)  # copy underlying metadata
    out_meta["microbatch_loss"] = microbatch_loss.detach()
    out_meta["per_example_loss_mean"] = per_example_loss.detach().mean()
    out_meta["per_example_loss_std"] = per_example_loss.detach().std(unbiased=False) if per_example_loss.numel() > 1 else torch.zeros((), device=per_example_loss.device)

    # Useful: average entropy / clip fraction could be handled elsewhere, but if present, keep it.
    return loss, out_meta




