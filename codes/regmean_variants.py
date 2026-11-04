#!/usr/bin/env python3
"""Closed-form RegMean variants. Does not train and does not launch a merge.

The existing RegMean result is left untouched. New outputs, when a run is
later requested, go under results/regmean_variants/.

lambda is the penalty already used by RegMean:
    lambda = ridge * mean(diag(sum_i G_i)), ridge = 1e-4.
Task weights alpha scale the expert Grams and are normalized to sum to 4,
so equal weights are exactly [1, 1, 1, 1]. They do not rescale lambda.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

DATASETS = ("dbpedia", "amazon", "yahoo", "agnews")
RIDGE = 1e-4
GAMMAS = (0.5, 1.0, 2.0)
NORM_EPS = 1e-12
N_TASKS = 4

T5_SPECS = {
    42: {
        "experts_root": Path("results/independent"),
        "grams": Path("results/alternative_merges/t5-large/seed_42/grams"),
        "learning_rate": 1e-4,
    },
    123: {
        "experts_root": Path("results/seeds/seed_123/independent"),
        "grams": Path("results/alternative_merges/t5-large/seed_123/grams"),
        "learning_rate": 1e-4,
    },
    2026: {
        "experts_root": Path("results/seeds/seed_2026/independent"),
        "grams": Path("results/alternative_merges/t5-large/seed_2026/grams"),
        "learning_rate": 1e-4,
    },
}
LLAMA_SPECS = {
    seed: {
        "experts_root": Path(f"results/llama-3.2-1b/lr_5e-6/seed_{seed}/independent"),
        "grams": Path(f"results/llama-3.2-1b/lr_5e-6/seed_{seed}/regmean_grams"),
        "learning_rate": 5e-6,
    }
    for seed in (42, 123, 2026)
}


def regmean_penalty(grams: Sequence[torch.Tensor], ridge: float = RIDGE) -> torch.Tensor:
    """Scalar lambda from the unweighted Gram sum. Same definition as RegMean."""
    total = torch.stack([gram.float() for gram in grams], dim=0).sum(dim=0)
    mean_diag = torch.trace(total) / total.shape[0]
    return float(ridge) * torch.clamp(mean_diag, min=NORM_EPS)


def weighted_regmean(
    experts: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    alphas: Sequence[float],
    ridge: float = RIDGE,
) -> torch.Tensor:
    """W = (sum alpha_i W_i G_i) (sum alpha_i G_i + lambda I)^{-1}."""
    if len(experts) != len(grams) or len(experts) != len(alphas):
        raise ValueError("experts, grams, and alphas must have the same length")
    scaled = [float(alpha) * gram.float() for alpha, gram in zip(alphas, grams)]
    total_gram = torch.stack(scaled, dim=0).sum(dim=0)
    total_mixed = torch.stack(
        [torch.matmul(expert.float(), gram) for expert, gram in zip(experts, scaled)],
        dim=0,
    ).sum(dim=0)
    penalty = regmean_penalty(grams, ridge)
    eye = torch.eye(total_gram.shape[0], device=total_gram.device, dtype=total_gram.dtype)
    solved = torch.linalg.solve(total_gram + penalty * eye, total_mixed.transpose(0, 1))
    return solved.transpose(0, 1).contiguous()


def anchor_regmean(
    w0: torch.Tensor,
    experts: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    ridge: float = RIDGE,
    penalty_scale: float = 1.0,
) -> torch.Tensor:
    """W = W0 + (sum (W_i - W0) G_i) (sum G_i + lambda_l I)^{-1}.

    penalty_scale is 1 for AnchorRegMean. TopoRegMean-Anchor uses
    1 + gamma * drift_l, where drift_l still needs an agreed normalization.
    """
    deltas = [expert.float() - w0.float() for expert in experts]
    total_gram = torch.stack([gram.float() for gram in grams], dim=0).sum(dim=0)
    total_mixed = torch.stack(
        [torch.matmul(delta, gram.float()) for delta, gram in zip(deltas, grams)],
        dim=0,
    ).sum(dim=0)
    penalty = regmean_penalty(grams, ridge) * float(penalty_scale)
    eye = torch.eye(total_gram.shape[0], device=total_gram.device, dtype=total_gram.dtype)
    solved = torch.linalg.solve(total_gram + penalty * eye, total_mixed.transpose(0, 1))
    return (w0.float() + solved.transpose(0, 1)).contiguous()


def normalize_alpha(raw: Sequence[float]) -> List[float]:
    """Scale raw scores so their mean is 1. Equal scores return all ones."""
    total = sum(float(value) for value in raw)
    if total <= 0.0:
        raise ValueError(f"alpha weights must sum to a positive number, got {total}")
    count = len(raw)
    return [count * float(value) / total for value in raw]


def layer_epsilon(distances: Sequence[float]) -> float:
    """Per-layer scale: 0.01 times the mean task distance for that layer."""
    if not distances:
        raise ValueError("epsilon needs at least one task distance")
    return 0.01 * (sum(distances) / len(distances)) + 1e-12


def percentile_ranks(scores: Dict[str, float]) -> Dict[str, float]:
    """Map scores onto [0, 1]. The smallest is 0, the largest is 1, ties share the average rank."""
    names = list(scores)
    count = len(names)
    if count == 0:
        return {}
    if count == 1:
        return {names[0]: 0.0}
    order = sorted(names, key=lambda name: scores[name])
    ranks: Dict[str, float] = {}
    start = 0
    while start < count:
        end = start
        while end + 1 < count and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average = 0.5 * (start + end)
        for index in range(start, end + 1):
            ranks[order[index]] = average / (count - 1)
        start = end + 1
    return ranks


def agreement_alphas(deltas: Sequence[torch.Tensor]) -> List[float]:
    """alpha_i = N * softmax(c_i), with c_i the mean pairwise cosine agreement."""
    agreement = pairwise_cosine_agreement(deltas)
    return normalize_alpha([math.exp(value) for value in agreement])


def topo_raw_high(distances: Sequence[float], eps: float) -> List[float]:
    return [distance + eps for distance in distances]


def topo_raw_low(distances: Sequence[float], eps: float) -> List[float]:
    return [1.0 / (distance + eps) for distance in distances]


def pairwise_cosine_agreement(deltas: Sequence[torch.Tensor]) -> List[float]:
    """Mean cosine of each task delta with the other task deltas. Can be negative."""
    flat = [delta.float().reshape(-1) for delta in deltas]
    scores = []
    for index, left in enumerate(flat):
        others = []
        for other_index, right in enumerate(flat):
            if other_index == index:
                continue
            denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            if float(denom) <= NORM_EPS:
                cosine = 0.0
            else:
                cosine = float(torch.dot(left, right) / denom)
            others.append(cosine)
        scores.append(sum(others) / len(others))
    return scores


def t5_attention_key(name: str) -> Optional[Tuple[str, int, str]]:
    """Map a T5 Linear name onto the TDA freeze key, or None if it is not QKVO."""
    if ".SelfAttention." in name and name.startswith("encoder."):
        component = "enc_self"
    elif ".SelfAttention." in name and name.startswith("decoder."):
        component = "dec_self"
    elif ".EncDecAttention." in name:
        component = "dec_cross"
    else:
        return None
    projection = name.rsplit(".", 2)[-2]
    if projection not in {"q", "k", "v", "o"}:
        return None
    marker = ".block."
    start = name.index(marker) + len(marker)
    layer_idx = int(name[start:].split(".", 1)[0])
    return component, layer_idx, projection


def cpu_checks() -> None:
    """Tiny identity checks. No checkpoints."""
    torch.manual_seed(0)
    w0 = torch.randn(3, 2)
    experts = [torch.randn(3, 2) for _ in range(4)]
    grams = []
    for _ in range(4):
        raw = torch.randn(2, 2)
        grams.append(raw.t() @ raw + torch.eye(2))
    equal = normalize_alpha([2.0, 2.0, 2.0, 2.0])
    if any(abs(value - 1.0) > 1e-12 for value in equal):
        raise SystemExit(f"equal raw scores must return [1, 1, 1, 1], got {equal}")
    distances = [1.0, 3.0, 1.0, 3.0]
    high = normalize_alpha(topo_raw_high(distances, eps=layer_epsilon(distances)))
    if abs(sum(high) - 4.0) > 1e-12 or high[1] <= high[0]:
        raise SystemExit("TopoRegMean-High weights are not mean-1 in drift order")
    low = normalize_alpha(topo_raw_low(distances, eps=layer_epsilon(distances)))
    if abs(sum(low) - 4.0) > 1e-12 or low[0] <= low[1]:
        raise SystemExit("TopoRegMean-Low did not favor the smaller drift")
    ranks = percentile_ranks({"a": 1.0, "b": 2.0, "c": 4.0})
    if ranks["a"] != 0.0 or ranks["c"] != 1.0 or abs(ranks["b"] - 0.5) > 1e-12:
        raise SystemExit(f"percentile ranks are wrong: {ranks}")
    agreed = agreement_alphas([torch.ones(2, 2) for _ in range(4)])
    if any(abs(value - 1.0) > 1e-6 for value in agreed):
        raise SystemExit(f"equal agreement must return [1, 1, 1, 1], got {agreed}")
    anchored = anchor_regmean(w0, [w0, w0], grams[:2])
    if not torch.allclose(anchored, w0, atol=1e-4):
        raise SystemExit("AnchorRegMean did not stay at W0 when every delta is zero")
    print("REGMEAN_VARIANT_CHECKS_OK")


if __name__ == "__main__":
    cpu_checks()
