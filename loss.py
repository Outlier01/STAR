from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import Experience
from transformers import LlamaForCausalLM

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cbs: int, # caculate batch size
) -> torch.Tensor:
    log_probs = []
    for i in range((sequence_ids.shape[0] + cbs - 1) // cbs):
        start_i = i * cbs
        end_i = min(start_i + cbs, sequence_ids.shape[0])

        output = model(
            input_ids=sequence_ids[start_i: end_i],
            attention_mask=attention_mask[start_i: end_i],
            use_cache=False,
        )
        logits = output["logits"]
        bs_log_probs = sequence_log_probs_from_logits(
            logits=logits[:, :-1],
            output_ids=sequence_ids[start_i: end_i, 1:],
        )
        log_probs.append(bs_log_probs)

    log_probs = torch.cat(log_probs, dim=0)
    return log_probs

def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, low_clip_eps: float, high_clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.low_clip_eps = low_clip_eps
        self.high_clip_eps = high_clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.low_clip_eps, 1 + self.high_clip_eps) * advantages

        if self.kl_weight > 0.0:
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            )
            loss = -torch.min(surr1, surr2) + self.kl_weight * kl
            loss = masked_mean(loss, action_mask, dim=-1).mean()
            return loss, kl.mean()
        
        else:
            loss = -torch.min(surr1, surr2)
            loss = masked_mean(loss, action_mask, dim=-1).mean()
            return loss
