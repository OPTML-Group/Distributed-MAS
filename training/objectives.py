"""Role-local objectives f_i (Sec. 4, Eq. (8)).

Everything reduces to one primitive: the teacher-forced log-likelihood of an
agent's message given its conditioning context c_i (its role prompt, the query
and the upstream messages), with the graph kept so `torch.autograd.grad` reaches
the LoRA factors.

    dpo_loss       Eq. (8): -log sigma(beta [log pi(m+|c)/pi_ref(m+|c) - log pi(m-|c)/pi_ref(m-|c)])
    kl_forward     KL(pi_ref || pi_theta | c), the forward-KL regulariser added
                   (weighted by beta_KL) to every scored sample
    advantage_ce   advantage * NLL(m | c); with advantage 1 this is plain SFT on
                   the message (used by `--fit_mode sft`)
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _target_logits(model, context_ids: torch.Tensor, target_ids: torch.Tensor):
    """Logits at the positions that predict `target_ids`, in FP32. [1, T, V]."""
    ids = torch.cat([context_ids, target_ids], dim=1)
    ctx_len = context_ids.shape[1]
    # Only materialise logits for the target span (the context can be several
    # thousand tokens). Fall back to the full forward on transformers versions
    # without `logits_to_keep`.
    keep = target_ids.shape[1] + 1
    out = None
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            out = model.model(input_ids=ids, use_cache=False, return_dict=True,
                              **{kw: keep})
            break
        except TypeError:
            continue
    if out is None:
        out = model.model(input_ids=ids, use_cache=False, return_dict=True)
        return out.logits[:, ctx_len - 1: -1, :].float()
    logits = out.logits[:, :-1, :].float()
    if logits.shape[1] != target_ids.shape[1]:      # kwarg silently ignored
        logits = out.logits[:, ctx_len - 1: -1, :].float()
    return logits


def seq_nll(model, context_ids: torch.Tensor, target_ids: torch.Tensor, *,
            reduction: str = "mean") -> torch.Tensor:
    """Teacher-forced NLL of `target_ids` given `context_ids` ([1, T] each). Differentiable."""
    if context_ids.dim() == 1:
        context_ids = context_ids.unsqueeze(0)
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)
    if target_ids.shape[1] == 0:
        raise ValueError("seq_nll got an empty target span")
    logp = F.log_softmax(_target_logits(model, context_ids, target_ids), dim=-1)
    tok_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if reduction == "sum":
        return -tok_logp.sum()
    return -tok_logp.mean()


def seq_logp(model, context_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Total log-probability of `target_ids` given `context_ids`. Differentiable."""
    return -seq_nll(model, context_ids, target_ids, reduction="sum")


def dpo_loss(model, prompt_ids: torch.Tensor,
             chosen_ids: torch.Tensor, rejected_ids: torch.Tensor,
             ref_logp_chosen: float, ref_logp_rejected: float,
             beta: float = 0.1) -> torch.Tensor:
    """Eq. (8). pi_ref is the backbone with the adapter disabled; its two
    log-probabilities are constants and are passed in precomputed. Both messages
    are scored under the same context (the preferred rollout's)."""
    lc = seq_logp(model, prompt_ids, chosen_ids)
    lr = seq_logp(model, prompt_ids, rejected_ids)
    logits = beta * ((lc - ref_logp_chosen) - (lr - ref_logp_rejected))
    return -torch.nn.functional.logsigmoid(logits)


def span_logprobs(model, context_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Next-token log-distribution at each target position. [1, T, V]."""
    if context_ids.dim() == 1:
        context_ids = context_ids.unsqueeze(0)
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)
    if target_ids.shape[1] == 0:
        raise ValueError("span_logprobs got an empty target span")
    return torch.log_softmax(_target_logits(model, context_ids, target_ids), dim=-1)


def kl_forward(ref_logp: torch.Tensor, cur_logp: torch.Tensor) -> torch.Tensor:
    """KL(reference || current), averaged over target positions."""
    return (ref_logp.exp() * (ref_logp - cur_logp)).sum(-1).mean()


def advantage_ce(model, prompt_ids: torch.Tensor, output_ids: torch.Tensor,
                 advantage: float) -> Optional[torch.Tensor]:
    """`advantage * NLL(message | context)`; None when the advantage is zero."""
    if advantage == 0.0:
        return None
    return advantage * seq_nll(model, prompt_ids, output_ids)


def group_advantages(rewards: Sequence[float]) -> Tuple[List[float], bool]:
    """Group-relative advantages (r - mean) / (std + eps), and whether the group
    is degenerate (all rewards equal). Used to rank rollouts for DPO pairing."""
    if not rewards:
        return [], True
    r = torch.tensor(list(rewards), dtype=torch.float32)
    std = float(r.std(unbiased=False))
    if std < 1e-8:
        return [0.0] * len(rewards), True
    adv = (r - r.mean()) / (std + 1e-6)
    return [float(a) for a in adv], False

