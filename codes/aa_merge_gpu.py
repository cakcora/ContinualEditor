#!/usr/bin/env python3
"""Standalone GPU runner: merge the four full-finetune experts and evaluate every merge rule.

Tasks (O-LoRA order 1, the same files and prompts as run_alternative_merges.py):
    dbpedia (TC, 14 classes), amazon (SC, 5 classes), yahoo (TC, 10 classes), agnews (TC, 4 classes)
Data: the CL_Benchmark folder of O-LoRA (https://github.com/cmnfriend/O-LoRA), laid out as
    CL_Benchmark/TC/dbpedia/{train,test,labels}.json, CL_Benchmark/SC/amazon/..., TC/yahoo, TC/agnews
Prompt (zero-shot, task and dataset name on, as DataCollatorForUIE with add_task_name=True,
add_dataset_name=True):
    Task:TC\nDataset:dbpedia\nWhat is the topic of the following paragraph? Choose one from the option.\n
    Option: Company, ..., Written Work \n<sentence>\nAnswer:
Score: greedy generation (50 new tokens), exact match after lower-casing and removing punctuation,
as in O-LoRA's compute_metrics.

Nothing here needs the project's run_merge.py. The script does not train; it needs
    --base     the base model (T5-large or Llama-3.2-1B)
    --experts-root DIR  with DIR/<task>/<W_DBPedia|W_Amazon|W_Yahoo|W_AGNews>   (project layout)
               or --expert dbpedia=PATH --expert amazon=PATH ...
    --data-dir the CL_Benchmark folder

Steps, all on one GPU:
    1. Input Gram matrices G_i = X_i^T X_i of every Linear layer, for every expert on its own
       training set (stored on disk, one file per layer; reused on the next run).
    2. Diagonal output Fisher (only for AA_Fisher).
    3. For each method: merge layer by layer on the GPU; the sequential methods then refit
       block by block on the merged model's own inputs.
    4. Evaluate the merged model on the four test sets; one row per method in results.csv.
    The base model and the experts (each on its own task) are evaluated as reference rows.

Methods (--methods):
    EqualQ (OrthoMerge recomputed), PlainQ, WeightedQ, RegMean, ActivationAware (the original
    output-side rule), and the input-side activation-aware family:
    ActivationAwareInput, ActivationAwareInputRes, AA_Anchor, AA_TIES, AA_LowRank, AA_Balanced,
    AA_Fisher, AA_Sequential, AA_Fast, AA_Consistent, ActivationAwarePlus,
    AAPlus-TSVD-16, AAPlus-TSVD-32, AAPlus-TSVD-64, AAPlus-TSVD-128, AAPlus-TSVD-256.
    The TSVD variants are ActivationAwarePlus with sequential refit off. Their expert
    rotation uses a rank-k randomized SVD, completed to a rotation that is the identity
    outside the span of those singular vectors.
Formulas: see activation_aware_methods.pdf. The merge code is the same as in
run_alternative_merges_fixed.py.

Examples
    python aa_merge_gpu.py --arch t5 --experts-root results/independent --data-dir CL_Benchmark
    python aa_merge_gpu.py --arch llama --experts-root results/llama-3.2-1b/lr_2e-5/seed_42/independent \
        --data-dir CL_Benchmark --methods EqualQ RegMean ActivationAwareInput ActivationAwarePlus
    python aa_merge_gpu.py --self-check          # formula checks only, no models, no GPU
A run can be stopped and restarted: finished methods are read back from results.json
(use --force to recompute them).

Practical notes (step-by-step instructions, run times and multi-GPU use: HOW_TO_RUN.md)
    * EqualQ is OrthoMerge's own rule recomputed here; its row is the OrthoMerge baseline.
    * --fast: 30 accelerated rotation-search steps instead of 100 safe ones, bf16 evaluation.
    * Disk: the Gram matrices take about 6 GB per task for Llama-3.2-1B (about 4.5 GB for T5-large);
      the rotation cache (expert Procrustes rotations, shared by all methods) about as much again.
      --no-cache turns the rotation cache off (slower: the SVDs are recomputed for every method).
    * GPU memory: up to three fp32 copies of the model at once (statistics, merged model for the
      sequential methods, evaluation) plus the Gram matrices of one expert while they are collected;
      for Llama-3.2-1B a 40 GB GPU is comfortable. --eval-dtype bf16 halves the evaluation copy.
    * Time: the rotation search on the widest inputs (Llama down_proj, 8192; T5 wo, 4096) dominates.
      AA_Fast and ActivationAwarePlus use the faster search; --methods selects a subset.
    * T5 decoder layers see only the answer tokens (a few per example); for the sequential methods
      --seq-max-samples 2048 gives better-conditioned decoder statistics than the default 512.
    * Wide layers (out < in): the expert rotation is not unique there; --procrustes minimal (default)
      takes the one closest to I, --procrustes svd the plain SVD choice. W0 R_i is the same.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================================
# Settings
# ============================================================================

DATASETS = ("dbpedia", "amazon", "yahoo", "agnews")
TASK_FOLDERS = {"dbpedia": ("TC", "dbpedia"), "amazon": ("SC", "amazon"), "yahoo": ("TC", "yahoo"), "agnews": ("TC", "agnews")}
INSTRUCTIONS = {
    "TC": "What is the topic of the following paragraph? Choose one from the option.\n",
    "SC": "What is the sentiment of the following paragraph? Choose one from the option.\n",
}
ANSWER_PREFIX = "Answer:"
# Training-set sizes; WeightedQ weights the experts by them (as in run_alternative_merges.py).
TASK_WEIGHTS = {"dbpedia": 14000.0, "amazon": 5000.0, "yahoo": 10000.0, "agnews": 4000.0}
EXPERT_DIR = {"dbpedia": "W_DBPedia", "amazon": "W_Amazon", "yahoo": "W_Yahoo", "agnews": "W_AGNews"}
ARCHITECTURES = {
    "t5": {"base": "/data/cuneyt-topo/numpy_weights/t5-continual-learning/external_baselines/GORP/initial_model/t5-large",
           "model_type": "seq2seq", "eval_batch_size": 64},
    "llama": {"base": "meta-llama/Llama-3.2-1B", "model_type": "causal", "eval_batch_size": 32},
}
# Activation-aware (input-side) family. Values given as strings are read from OPTIONS.
AA_CONFIG = {
    "ActivationAwareInput": {"residual": "mean"},
    "ActivationAwareInputRes": {"residual": "corr", "alpha": "aa_alpha"},
    "AA_Anchor": {"residual": "mean", "anchor": "aa_anchor"},
    "AA_TIES": {"residual": "ties"},
    "AA_LowRank": {"residual": "corr", "alpha": "aa_alpha", "rank": "aa_rank"},
    "AA_Balanced": {"residual": "mean", "balance": True},
    "AA_Fisher": {"residual": "mean", "fisher": True},
    "AA_Sequential": {"residual": "mean", "seq": True},
    "AA_Fast": {"residual": "mean", "solver": "accel"},
    "AA_Consistent": {"residual": "mean", "consistent": True},
    "ActivationAwarePlus": {"residual": "corr", "alpha": "aap_alpha", "balance": True, "seq": True,
                            "solver": "accel", "iters": "aap_iters"},
}
# Same merge as ActivationAwarePlus, without the sequential refit. procrustes_rank truncates only
# the expert rotation; the activation fit and the 0.3 correction are unchanged.
for _tsvd_rank in (16, 32, 64, 128, 256):
    AA_CONFIG[f"AAPlus-TSVD-{_tsvd_rank}"] = {
        "residual": "corr", "alpha": "aap_alpha", "balance": True, "seq": False,
        "solver": "accel", "iters": "aap_iters", "procrustes_rank": _tsvd_rank,
    }
del _tsvd_rank
# Rank comparison: same as AAPlus-TSVD-k, but layers whose input dimension is at least
# 4096 (T5 wo, Llama down_proj) use the Cayley start as R_fit and skip rotation_fit.
# Smaller layers still run rotation_fit. The 0.3 correction is unchanged.
for _tsvd_rank in (16, 32, 64, 128):
    AA_CONFIG[f"AAPlus-TSVD-{_tsvd_rank}-SkipWideFit"] = {
        **AA_CONFIG[f"AAPlus-TSVD-{_tsvd_rank}"],
        "skip_fit_inputs": 4096,
    }
del _tsvd_rank
# AAPlus-Full formula with the hour-long pieces removed: no sequential refit, and no
# rotation_fit on input dimension >= 4096. Expert rotations and the remaining fits use
# SVD when the run passes --procrustes svd --aa-polar svd.
AA_CONFIG["AAPlus-Full-Minutes"] = {
    "residual": "corr", "alpha": "aap_alpha", "balance": True, "seq": False,
    "solver": "accel", "iters": "aap_iters", "skip_fit_inputs": 4096,
}
SKIPPED_FIT = {"count": 0}
METHODS = ("EqualQ", "PlainQ", "WeightedQ", "RegMean", "ActivationAware") + tuple(AA_CONFIG)
AA_METHODS = set(AA_CONFIG)
GRAM_METHODS = {"RegMean", "ActivationAware"} | AA_METHODS
FISHER_METHODS = {name for name, cfg in AA_CONFIG.items() if cfg.get("fisher")}
SEQ_METHODS = {name for name, cfg in AA_CONFIG.items() if cfg.get("seq")}
NORM_EPS = 1e-8
# The original ActivationAware rule rotates outputs, (out x out); vocab-size outputs are averaged instead.
MAX_ACTIVATION_OUTPUT = 16384
OPTIONS = {
    "weighted_residual": False,
    "regmean_alpha": 1.0,
    "aa_iters": 100,
    "aa_tol": 1e-6,
    "aa_alpha": 0.5,
    "aa_solver": "safe",
    "aa_polar": "ns",
    "aa_anchor": 0.1,
    "aa_rank": 64,
    "aa_ties_keep": 0.2,
    "aap_alpha": 0.3,
    "aap_iters": 50,
    "seq_max_samples": 512,
    "fisher_max_samples": 512,
    "batch_size": 8,
}


class MergeProfile:
    """Wall-time buckets for one merge. Disabled unless --profile. The math is unchanged."""

    def __init__(self) -> None:
        self.enabled = False
        self.seconds: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self.shape = ""
        self.dump_path: Optional[Path] = None

    def add_time(self, key: str, dt: float) -> None:
        self.seconds[key] += dt
        if self.shape:
            self.seconds[f"{key}|{self.shape}"] += dt

    def add_count(self, key: str, n: int = 1) -> None:
        self.counts[key] += int(n)
        if self.shape:
            self.counts[f"{key}|{self.shape}"] += int(n)

    def dump(self) -> None:
        if self.dump_path is None:
            return
        payload = {"seconds": self.seconds, "counts": self.counts}
        tmp = self.dump_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.dump_path)


PROFILE = MergeProfile()


def prof_begin():
    if not PROFILE.enabled:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def prof_end(token, key: str) -> None:
    if token is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    PROFILE.add_time(key, time.perf_counter() - token)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def aa_settings(method: str) -> dict:
    """Resolve a method's configuration; strings refer to OPTIONS entries."""
    cfg = {"residual": "mean", "alpha": 0.0, "rank": None, "anchor": 0.0, "balance": False, "fisher": False,
           "seq": False, "consistent": False, "solver": OPTIONS["aa_solver"], "iters": OPTIONS["aa_iters"]}
    for key, value in AA_CONFIG[method].items():
        cfg[key] = OPTIONS[value] if isinstance(value, str) and value in OPTIONS else value
    return cfg


# ============================================================================
# Activation-aware rotation merging: building blocks (as in run_alternative_merges_fixed.py)
# ============================================================================
# Weights are stored as (out, in), as in PyTorch. Every expert i gives
#     M_i  : its fine-tuned weight (out x in)
#     G_i  : X_i^T X_i, Gram matrix of the layer inputs on task i (in x in)
# For the sequential variant the inputs of the merged model, Xm_i, differ from the
# expert's own inputs, Xe_i, and two statistics are used instead:
#     gram_in_i = Xm_i^T Xm_i     cross_i = Xe_i^T Xm_i
# (for the non-sequential variants both are G_i).
#
# The rotation fit minimizes
#     f(R) = sum_i || S_i^{1/2} (W0 R Xm_i^T - T_i Xe_i^T) ||_F^2  [+ mu ||R - R_anchor||_F^2]
#          = sum_i tr(R gram_in_i R^T A_i) - 2 <R, K> + const,
#     A_i = W0^T S_i W0,   K = sum_i W0^T S_i T_i cross_i  [+ mu R_anchor],
# over rotations R (R^T R = I, det R = 1). T_i is the target (M_i, or M_i minus the
# merged leftover), S_i an optional diagonal output weight (Fisher), I by default.

# ----------------------------------------------------------------------------
# Closest rotation (special polar factor)
# ----------------------------------------------------------------------------

def closest_rotation_svd(matrix: torch.Tensor) -> torch.Tensor:
    """Closest rotation to `matrix` (det +1), from one SVD.

    The fit steps produce matrices close to a multiple of a rotation (clustered singular values),
    on which float32 SVD can fail to converge; float64 is used then."""
    try:
        u_factor, _sigma, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    except RuntimeError:  # torch.linalg.LinAlgError is a RuntimeError
        u_factor, _sigma, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
        u_factor, vh = u_factor.float(), vh.float()
    sign = torch.sign(torch.linalg.det(torch.matmul(u_factor, vh)))
    if float(sign) < 0:
        u_factor = u_factor.clone()
        u_factor[:, -1] = -u_factor[:, -1]
    return torch.matmul(u_factor, vh)


def _sigma_max(matrix: torch.Tensor, steps: int = 12) -> float:
    vector = torch.randn(matrix.shape[1], 1, device=matrix.device, dtype=matrix.dtype,
                         generator=torch.Generator(device=matrix.device).manual_seed(0))
    for _ in range(steps):
        vector = torch.matmul(matrix.t(), torch.matmul(matrix, vector))
        vector = vector / torch.linalg.norm(vector).clamp_min(1e-30)
    return float(torch.linalg.norm(torch.matmul(matrix, vector)))


def polar_newton_schulz(matrix: torch.Tensor, iters: int = 40, tol: float = 1e-6) -> Optional[torch.Tensor]:
    """Orthogonal polar factor by Newton-Schulz (matmuls only). None if it did not converge."""
    size = matrix.shape[0]
    eye = torch.eye(size, device=matrix.device, dtype=torch.float32)
    x = matrix.float() / (1.05 * max(_sigma_max(matrix.float()), 1e-30))
    for step in range(iters):
        x = 1.5 * x - 0.5 * torch.matmul(x, torch.matmul(x.t(), x))
        if step % 4 == 3:
            err = float(torch.linalg.norm(torch.matmul(x.t(), x) - eye)) / math.sqrt(size)
            if err < tol:
                return x
    return None


def closest_rotation(matrix: torch.Tensor, method: str = "auto", svd_max: int = 1024) -> torch.Tensor:
    """Closest rotation. 'svd' is exact; 'ns' uses Newton-Schulz (fast on GPU for large matrices).

    Newton-Schulz returns the orthogonal polar factor; if that is a reflection
    (determinant -1) the exact SVD answer is used instead.
    """
    if method == "svd" or (method == "auto" and matrix.shape[0] <= svd_max):
        return closest_rotation_svd(matrix)
    result = polar_newton_schulz(matrix)
    if result is None or float(torch.linalg.slogdet(result)[0]) < 0:  # not converged, or a reflection
        return closest_rotation_svd(matrix)
    return result


# ----------------------------------------------------------------------------
# Rotation fit
# ----------------------------------------------------------------------------

def _lambda_max_sym(matrix: torch.Tensor, steps: int = 50) -> float:
    """Largest eigenvalue of a symmetric PSD matrix: exact up to 4096, else power iteration
    with a 10% safety margin (power iteration approaches the value from below)."""
    if matrix.shape[0] <= 4096:
        return float(torch.linalg.eigvalsh(matrix.float())[-1])
    vector = torch.randn(matrix.shape[0], 1, device=matrix.device, dtype=matrix.dtype,
                         generator=torch.Generator(device=matrix.device).manual_seed(1))
    value = 0.0
    for _ in range(steps):
        vector = torch.matmul(matrix, vector)
        value = float(torch.linalg.norm(vector))
        vector = vector / max(value, 1e-30)
    return 1.1 * value


def rotation_fit(
    w0: torch.Tensor,
    targets: Sequence[torch.Tensor],
    gram_in: Sequence[torch.Tensor],
    cross: Sequence[torch.Tensor],
    starts: Sequence[torch.Tensor],
    s_diag: Optional[Sequence[torch.Tensor]] = None,
    anchor: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.0,
    iters: int = 100,
    tol: float = 1e-6,
    solver: str = "safe",
    polar: str = "auto",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Best single input-side rotation (see the module docstring for the objective).

    solver='safe'  : majorization step R <- rot(L R - grad + K), L = sum ||A_i|| ||gram_in_i||.
    solver='accel' : the same step with an adaptive L (backtracking) and Nesterov momentum.
                     A step that would increase the objective is replaced by a safe step,
                     so the objective never increases in either mode.
    Every rotation in `starts` is used as a starting point; the best result is returned,
    so the fit is never worse than any starting rotation (e.g. OrthoMerge's rotation).
    anchor_weight is relative: mu = anchor_weight * L.
    """
    base = w0.float()
    device = base.device
    weighted = s_diag is not None
    if weighted:
        a_list = [torch.matmul(base.t(), s.float().view(-1, 1) * base) for s in s_diag]
        g_list = [g.float() for g in gram_in]
        lipschitz = sum(_lambda_max_sym(a) * _lambda_max_sym(g) for a, g in zip(a_list, g_list))
        k_mat = sum(torch.matmul(torch.matmul(base.t(), s.float().view(-1, 1) * t.float()), c.float())
                    for s, t, c in zip(s_diag, targets, cross))
    else:
        c0 = torch.matmul(base.t(), base)
        g_sum = torch.stack([g.float() for g in gram_in]).sum(dim=0)
        lipschitz = _lambda_max_sym(c0) * _lambda_max_sym(g_sum)
        k_mat = sum(torch.matmul(torch.matmul(base.t(), t.float()), c.float()) for t, c in zip(targets, cross))
    lipschitz = max(lipschitz, 1e-12)
    if anchor is not None and anchor_weight > 0:
        k_mat = k_mat + anchor_weight * lipschitz * anchor.float().to(device)

    def quad_and_grad(rot: torch.Tensor) -> Tuple[float, torch.Tensor]:
        if weighted:
            grad = torch.zeros_like(rot)
            quad = 0.0
            for a, g in zip(a_list, g_list):
                rg = torch.matmul(rot, g)
                ar = torch.matmul(a, rot)
                quad += float(torch.sum(rg * ar))
                grad = grad + torch.matmul(ar, g)
            return quad, grad
        rg = torch.matmul(rot, g_sum)
        cr = torch.matmul(c0, rot)
        return float(torch.sum(rg * cr)), torch.matmul(cr, g_sum)

    def objective(rot: torch.Tensor) -> float:
        return quad_and_grad(rot)[0] - 2.0 * float(torch.sum(rot * k_mat))

    def step(point: torch.Tensor, grad: torch.Tensor, lip: float) -> torch.Tensor:
        return closest_rotation(lip * point - grad + k_mat, method=polar)

    accel = solver == "accel"
    best, best_value, total_steps = None, None, 0
    candidates = [s.float().to(device) for s in starts] + [closest_rotation(k_mat, method="svd" if k_mat.shape[0] <= 4096 else polar)]
    for start in candidates:
        rot, previous = start, start
        value = objective(rot)
        lip = lipschitz / 16.0 if accel else lipschitz
        momentum = 1.0
        for _ in range(int(iters)):
            if accel:
                momentum_next = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
                point = rot + ((momentum - 1.0) / momentum_next) * (rot - previous)
            else:
                momentum_next, point = 1.0, rot
            quad_p, grad_p = quad_and_grad(point)
            value_p = quad_p - 2.0 * float(torch.sum(point * k_mat))
            while True:
                candidate = step(point, grad_p, lip)
                total_steps += 1
                cand_value = objective(candidate)
                diff = candidate - point
                bound = value_p + 2.0 * float(torch.sum((grad_p - k_mat) * diff)) + lip * float(torch.sum(diff * diff))
                if not accel or cand_value <= bound + 1e-9 * abs(value_p) or lip >= lipschitz:
                    break
                lip = min(2.0 * lip, lipschitz)
            if cand_value > value:
                _q, grad_r = quad_and_grad(rot)
                candidate = step(rot, grad_r, lipschitz)
                cand_value = objective(candidate)
                total_steps += 1
                momentum_next = 1.0
                if cand_value > value:  # no improvement possible: keep the current rotation
                    break
            done = abs(value - cand_value) <= tol * max(abs(value), 1.0)
            previous, rot, value, momentum = rot, candidate, cand_value, momentum_next
            if accel:
                lip = max(lip / 1.5, lipschitz / 1024.0)
            if done:
                break
        if best_value is None or value < best_value:
            best, best_value = rot, value
    if float(torch.linalg.slogdet(best)[0]) < 0:  # Newton-Schulz may return a reflection
        best = closest_rotation_svd(best)
        best_value = objective(best)
    return best, {"objective": best_value, "steps": total_steps, "lipschitz": lipschitz}


def fit_batch_width(inputs: int) -> int:
    """How many independent same-shaped layers to search at once. The update is unchanged."""
    if inputs >= 8192:
        return 8
    if inputs >= 4096:
        return 16
    return 32


def _sigma_max_batch(matrices: torch.Tensor) -> torch.Tensor:
    """Per-matrix scale used by Newton-Schulz. Each slice matches ``_sigma_max``."""
    return torch.tensor([_sigma_max(matrices[index]) for index in range(matrices.shape[0])],
                        device=matrices.device, dtype=matrices.dtype)


def polar_newton_schulz_batch(matrices: torch.Tensor, iters: int = 40, tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched Newton-Schulz. A converged slice matches the single-matrix iteration, including its early stop."""
    batch, size, _ = matrices.shape
    eye = torch.eye(size, device=matrices.device, dtype=torch.float32)
    scale = (1.05 * _sigma_max_batch(matrices.float()).clamp_min(1e-30)).view(batch, 1, 1)
    current = matrices.float() / scale
    frozen = torch.zeros(batch, dtype=torch.bool, device=matrices.device)
    steps_run = iters
    for step in range(iters):
        updated = 1.5 * current - 0.5 * torch.matmul(current, torch.matmul(current.transpose(-1, -2), current))
        current = torch.where(frozen[:, None, None], current, updated)
        if step % 4 == 3:
            error = torch.linalg.matrix_norm(torch.matmul(current.transpose(-1, -2), current) - eye, ord="fro")
            error = error / math.sqrt(size)
            frozen = frozen | (error < tol)
            if bool(frozen.all()):
                steps_run = step + 1
                break
    if PROFILE.enabled:
        PROFILE.add_count("ns_inner_steps", int(steps_run) * int(batch))
    return current, frozen


def closest_rotation_batch(matrices: torch.Tensor, method: str = "ns") -> torch.Tensor:
    """Each slice matches ``closest_rotation``."""
    if PROFILE.enabled:
        PROFILE.add_count("projections", int(matrices.shape[0]))
    if method == "svd" or (method == "auto" and matrices.shape[-1] <= 1024):
        token = prof_begin()
        result = torch.stack([closest_rotation_svd(matrices[index]) for index in range(matrices.shape[0])])
        prof_end(token, "fit_svd")
        if PROFILE.enabled:
            PROFILE.add_count("svd_projections", int(matrices.shape[0]))
        return result
    token = prof_begin()
    polar, converged = polar_newton_schulz_batch(matrices)
    sign = torch.linalg.slogdet(polar)[0]
    use_polar = converged & (sign > 0)
    prof_end(token, "fit_ns")
    if PROFILE.enabled:
        PROFILE.add_count("ns_projections", int(matrices.shape[0]))
    if bool(use_polar.all()):
        return polar
    output = polar.clone()
    fallback = torch.where(~use_polar)[0]
    if PROFILE.enabled:
        PROFILE.add_count("svd_fallbacks", int(fallback.numel()))
    token = prof_begin()
    for index in fallback.tolist():
        output[index] = closest_rotation_svd(matrices[index])
    prof_end(token, "fit_svd")
    return output


def _lambda_max_batch(matrices: torch.Tensor) -> torch.Tensor:
    return torch.tensor([_lambda_max_sym(matrices[index]) for index in range(matrices.shape[0])],
                        device=matrices.device, dtype=torch.float32)


def rotation_fit_batch(
    w0: torch.Tensor,
    targets: torch.Tensor,
    gram_in: torch.Tensor,
    cross: torch.Tensor,
    starts: torch.Tensor,
    iters: int,
    tol: float,
    solver: str,
    polar: str,
    anchor: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.0,
) -> torch.Tensor:
    """Same search as ``rotation_fit`` for every layer in the batch. No Fisher weight and no anchor.

    w0 is (B, out, in), targets/gram_in/cross are (B, tasks, ...), starts is (B, in, in).
    """
    batch, out, inputs = w0.shape
    device = w0.device
    if PROFILE.enabled:
        PROFILE.shape = f"R{inputs}x{inputs}|W{out}x{inputs}"
        PROFILE.add_count("fitted_layers", batch)
    wall = prof_begin()
    base = w0.float()
    token = prof_begin()
    c0 = torch.stack([torch.matmul(base[index].t(), base[index]) for index in range(batch)])
    g_sum = torch.stack([gram_in[index].float().sum(dim=0) for index in range(batch)])
    lipschitz = _lambda_max_batch(c0) * _lambda_max_batch(g_sum)
    lipschitz = torch.clamp(lipschitz, min=1e-12)
    prof_end(token, "fit_eig")
    token = prof_begin()
    k_mat = torch.zeros(batch, inputs, inputs, device=device, dtype=torch.float32)
    # Same per-layer products as rotation_fit. A batched product is a different rounding and can flip the SVD.
    for index in range(batch):
        for task in range(targets.shape[1]):
            k_mat[index] = k_mat[index] + torch.matmul(
                torch.matmul(base[index].t(), targets[index, task].float()), cross[index, task].float())
    if anchor is not None and anchor_weight > 0:
        k_mat = k_mat + float(anchor_weight) * lipschitz[:, None, None] * anchor.float()
    prof_end(token, "fit_k")

    def quad_and_grad(rot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_q = prof_begin()
        right = torch.matmul(rot, g_sum)
        left = torch.matmul(c0, rot)
        quad = (right * left).sum(dim=(-1, -2))
        grad = torch.matmul(left, g_sum)
        prof_end(token_q, "fit_objective")
        return quad, grad

    def objective(rot: torch.Tensor) -> torch.Tensor:
        quad, _grad = quad_and_grad(rot)
        return quad - 2.0 * (rot * k_mat).sum(dim=(-1, -2))

    accel = solver == "accel"
    best = starts.float().clone()
    best_value = torch.full((batch,), float("inf"), device=device)
    start_k = closest_rotation_batch(k_mat, method="svd" if inputs <= 4096 else polar)
    for start in (starts.float(), start_k):
        rot = start.clone()
        previous = start.clone()
        value = objective(rot)
        lip = lipschitz / 16.0 if accel else lipschitz.clone()
        momentum = torch.ones(batch, device=device)
        active = torch.ones(batch, dtype=torch.bool, device=device)
        for _ in range(int(iters)):
            if not bool(active.any()):
                break
            if PROFILE.enabled:
                PROFILE.add_count("active_iters", int(active.sum().item()))
            if accel:
                momentum_next = (1.0 + torch.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
                point = rot + ((momentum - 1.0) / momentum_next)[:, None, None] * (rot - previous)
            else:
                momentum_next, point = torch.ones_like(momentum), rot
            _quad_p, grad_p = quad_and_grad(point)
            value_p = _quad_p - 2.0 * (point * k_mat).sum(dim=(-1, -2))
            step_lip = lip.clone()
            while True:
                candidate = closest_rotation_batch(step_lip[:, None, None] * point - grad_p + k_mat, method=polar)
                cand_value = objective(candidate)
                diff = candidate - point
                bound = value_p + 2.0 * ((grad_p - k_mat) * diff).sum(dim=(-1, -2)) + step_lip * (diff * diff).sum(dim=(-1, -2))
                need_more = active & (cand_value > bound + 1e-9 * value_p.abs()) & (step_lip < lipschitz) if accel else active.new_zeros(batch)
                # ``active`` is bool; new_zeros would be bool only if we use zeros_like
                if not accel:
                    need_more = torch.zeros(batch, dtype=torch.bool, device=device)
                if not bool(need_more.any()):
                    break
                step_lip = torch.where(need_more, torch.minimum(2.0 * step_lip, lipschitz), step_lip)
            worse = active & (cand_value > value)
            if bool(worse.any()):
                _quad_r, grad_r = quad_and_grad(rot)
                safe = closest_rotation_batch(lipschitz[:, None, None] * rot - grad_r + k_mat, method=polar)
                safe_value = objective(safe)
                use_safe = worse & (safe_value <= value)
                candidate = torch.where(use_safe[:, None, None], safe, candidate)
                cand_value = torch.where(use_safe, safe_value, cand_value)
                active = active & ~(worse & ~use_safe)
                momentum_next = torch.where(worse, torch.ones_like(momentum_next), momentum_next)
            done = (value - cand_value).abs() <= tol * torch.maximum(value.abs(), torch.ones_like(value))
            previous = torch.where(active[:, None, None], rot, previous)
            rot = torch.where(active[:, None, None], candidate, rot)
            value = torch.where(active, cand_value, value)
            momentum = torch.where(active, momentum_next, momentum)
            if accel:
                lip = torch.where(active, torch.maximum(step_lip / 1.5, lipschitz / 1024.0), lip)
            else:
                lip = step_lip
            active = active & ~done
        better = value < best_value
        best = torch.where(better[:, None, None], rot, best)
        best_value = torch.where(better, value, best_value)
    sign = torch.linalg.slogdet(best)[0]
    reflect = torch.where(sign < 0)[0]
    if reflect.numel():
        token = prof_begin()
        for index in reflect.tolist():
            best[index] = closest_rotation_svd(best[index])
        prof_end(token, "fit_svd")
        if PROFILE.enabled:
            PROFILE.add_count("svd_fallbacks", int(reflect.numel()))
    prof_end(wall, "fit_wall")
    return best


# ----------------------------------------------------------------------------
# Merging the leftovers (residuals rho_i = M_i - W0 R_i)
# ----------------------------------------------------------------------------

def ties_merge(tensors: Sequence[torch.Tensor], keep: float = 0.2, scale: float = 1.0) -> torch.Tensor:
    """TIES-merging (Yadav et al., 2023): trim to the top `keep` fraction by magnitude per
    tensor, elect a sign per entry from the sum, average the entries that agree with it."""
    trimmed = []
    for tensor in tensors:
        flat = tensor.float().abs().flatten()
        count = max(1, int(round(keep * flat.numel())))
        threshold = torch.topk(flat, count, largest=True).values.min()
        trimmed.append(torch.where(tensor.float().abs() >= threshold, tensor.float(), torch.zeros_like(tensor.float())))
    stacked = torch.stack(trimmed)
    positive = stacked.clamp(min=0).sum(dim=0)
    negative = stacked.clamp(max=0).sum(dim=0)
    n_pos = (stacked > 0).sum(dim=0).clamp(min=1)
    n_neg = (stacked < 0).sum(dim=0).clamp(min=1)
    elected = torch.sign(positive + negative)
    merged = torch.where(elected > 0, positive / n_pos, torch.where(elected < 0, negative / n_neg, torch.zeros_like(positive)))
    return scale * merged


def activation_correction(
    merged: torch.Tensor,
    targets: Sequence[torch.Tensor],
    gram_in: Sequence[torch.Tensor],
    cross: Sequence[torch.Tensor],
    ridge: float,
    rank: Optional[int] = None,
) -> torch.Tensor:
    """Least-squares correction E with  sum_i ||Xm_i (W + E)^T - Xe_i T_i^T||^2  minimal:
        E = (sum_i T_i cross_i - W sum_i gram_in_i)(sum_i gram_in_i + lambda I)^{-1},
    lambda = ridge * mean diagonal. With rank r, only the r strongest directions are kept."""
    g_sum = torch.stack([g.float() for g in gram_in]).sum(dim=0)
    # sum_i T_i cross_i - W sum_i gram_in_i, written so that experts equal to W give exactly zero
    # (the subtraction of two large products would leave rounding noise, which the inverse of a
    # rank-deficient Gram matrix amplifies).
    rhs = sum(torch.matmul(t.float() - merged.float(), c.float()) for t, c in zip(targets, cross))
    if any(c is not g for c, g in zip(cross, gram_in)):
        rhs = rhs + torch.matmul(merged.float(), sum(c.float() - g.float() for c, g in zip(cross, gram_in)))
    penalty = float(ridge) * float(torch.clamp(torch.trace(g_sum) / g_sum.shape[0], min=1e-12))
    system = g_sum + penalty * torch.eye(g_sum.shape[0], device=g_sum.device, dtype=g_sum.dtype)
    correction = torch.linalg.solve(system, rhs.t()).t()
    if rank is not None and 0 < rank < min(correction.shape):
        u_factor, sigma, vh = torch.linalg.svd(correction, full_matrices=False)
        correction = torch.matmul(u_factor[:, :rank] * sigma[:rank], vh[:rank])
    return correction


# ----------------------------------------------------------------------------
# Masks (same rules as GramCollector in run_alternative_merges.py)
# ----------------------------------------------------------------------------

def module_mask(model_type: str, module_name: str, encoder_mask, decoder_mask, length: int):
    if model_type == "causal":
        mask = decoder_mask
    elif module_name.startswith("encoder.") or ".EncDecAttention.k" in module_name or ".EncDecAttention.v" in module_name:
        mask = encoder_mask
    else:
        mask = decoder_mask
    if mask is None or mask.shape[-1] != length:
        return None
    return mask


def _masked_rows(hidden: torch.Tensor, mask) -> Tuple[torch.Tensor, int]:
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(1)
    values = hidden.detach().float()
    if mask is not None:
        values = values * mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        count = int(mask.sum().item())
    else:
        count = int(values.shape[0] * values.shape[1])
    return values.reshape(-1, values.shape[-1]), count


# ----------------------------------------------------------------------------
# Fisher (output sensitivity) statistics: diagonal of E[g g^T], g = dLoss/d(layer output)
# ----------------------------------------------------------------------------

class FisherCollector:
    """Accumulates sum over tokens of g^2 for every hooked Linear output.

    Labels are sampled from the model's own prediction (true Fisher), so the statistic
    measures which output directions the model's predictions are sensitive to.
    """

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.sums: Dict[str, torch.Tensor] = {}
        self.tokens: Dict[str, int] = {}
        self.encoder_mask = None
        self.decoder_mask = None

    def forward_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                return
            def grad_hook(grad):
                mask = module_mask(self.model_type, name, self.encoder_mask, self.decoder_mask,
                                   grad.shape[1] if grad.ndim == 3 else 1)
                rows, count = _masked_rows(grad, mask)
                total = (rows * rows).sum(dim=0)
                if name in self.sums:
                    self.sums[name] += total
                    self.tokens[name] += count
                else:
                    self.sums[name] = total
                    self.tokens[name] = count
            output.register_hook(grad_hook)
        return hook


def sampled_fisher_loss(model_type: str, logits: torch.Tensor, labels: Optional[torch.Tensor],
                        attention_mask: Optional[torch.Tensor], generator: torch.Generator) -> torch.Tensor:
    """Cross-entropy against labels sampled from softmax(logits) at the supervised positions."""
    if model_type == "causal":
        logits = logits[:, :-1]
        if labels is not None:
            valid = labels[:, 1:].ne(-100)
        else:
            valid = attention_mask[:, 1:].bool()
    else:
        valid = labels.ne(-100)
    chosen = logits[valid].float()
    probs = torch.softmax(chosen.detach(), dim=-1)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.nn.functional.cross_entropy(chosen, sampled, reduction="sum")


class _HandleGroup:
    def __init__(self, handles):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()


def enable_input_grad(model: nn.Module):
    """Let gradients flow to activations although all parameters are frozen: the outputs of
    every embedding module (encoder and decoder may use separate modules) require grad."""
    handles = [module.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))
               for module in model.modules() if isinstance(module, nn.Embedding)]
    return _HandleGroup(handles)


# ----------------------------------------------------------------------------
# Sequential statistics: inputs of the merged model next to the expert's own inputs
# ----------------------------------------------------------------------------

class StopForward(Exception):
    """Raised by a hook to stop a forward pass once a group of layers has been seen."""


_BLOCK = re.compile(r"^(.*?\.(?:block|layers|layer|h)\.\d+)\.")


def linear_call_order(model: nn.Module, module_names: Sequence[str], run_forward: Callable[[], None]) -> List[str]:
    """Names of the given Linear modules in the order in which a forward pass first calls them."""
    order: List[str] = []
    wanted = set(module_names)
    handles = []
    for name, module in model.named_modules():
        if name in wanted:
            handles.append(module.register_forward_hook(
                lambda _m, _i, _o, name=name: order.append(name) if name not in order else None))
    try:
        with torch.no_grad():
            run_forward()
    finally:
        for handle in handles:
            handle.remove()
    return order


def group_by_block(ordered_names: Sequence[str]) -> List[List[str]]:
    """Group module names by transformer block (encoder.block.3, model.layers.5, ...), in call order."""
    groups: Dict[str, List[str]] = {}
    for name in ordered_names:
        match = _BLOCK.match(name)
        key = match.group(1) if match else name
        groups.setdefault(key, []).append(name)
    return list(groups.values())


class PairedGroupCollector:
    """For one group of Linear modules, accumulates
        cross[name]   += Xe^T Xm    (expert inputs vs merged-model inputs, same tokens)
        gram_in[name] += Xm^T Xm
    Use: phase 'expert' on the expert model, then phase 'merged' on the merged model,
    on the same batch. The forward pass stops after the group's last module."""

    def __init__(self, model_type: str, names: Sequence[str], last_name: str):
        self.model_type = model_type
        self.names = list(names)
        self.last_name = last_name
        self.phase = "expert"
        self.stored: Dict[str, torch.Tensor] = {}
        self.cross: Dict[str, torch.Tensor] = {}
        self.gram_in: Dict[str, torch.Tensor] = {}
        self.tokens: Dict[str, int] = {}
        self.encoder_mask = None
        self.decoder_mask = None
        self._cache: Dict[Tuple[int, int], tuple] = {}
        self._expert_rows: Dict[int, tuple] = {}
        self.mismatches = 0

    def hook(self, name: str):
        def fn(_module, inputs, _output):
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or not torch.is_floating_point(hidden):
                return
            length = hidden.shape[1] if hidden.ndim == 3 else 1
            mask = module_mask(self.model_type, name, self.encoder_mask, self.decoder_mask, length)
            if mask is None:
                self.mismatches += 1
            if self.phase == "expert":
                # q, k, v (or gate, up) share one input tensor: compute its rows once. The input
                # is kept referenced so that its memory cannot be reused by another tensor.
                entry = self._expert_rows.get(id(hidden))
                if entry is None or entry[0] is not hidden:
                    entry = (hidden, _masked_rows(hidden, mask)[0])
                    self._expert_rows[id(hidden)] = entry
                self.stored[name] = entry[1]
            else:
                rows, count = _masked_rows(hidden, mask)
                expert_rows = self.stored[name]
                key = (id(expert_rows), id(hidden))
                entry = self._cache.get(key)
                if entry is None or entry[0] is not expert_rows or entry[1] is not hidden:
                    entry = (expert_rows, hidden, torch.matmul(expert_rows.t(), rows), torch.matmul(rows.t(), rows))
                    self._cache[key] = entry
                cross, gram = entry[2], entry[3]
                if name in self.cross:
                    self.cross[name] += cross
                    self.gram_in[name] += gram
                    self.tokens[name] += count
                else:
                    self.cross[name] = cross.clone()
                    self.gram_in[name] = gram.clone()
                    self.tokens[name] = count
            if name == self.last_name:
                raise StopForward()
        return fn

    def end_batch(self):
        self.stored = {}
        self._cache = {}
        self._expert_rows = {}


def run_until_stop(model: nn.Module, inputs: dict) -> None:
    try:
        with torch.no_grad():
            model(**inputs)
    except StopForward:
        pass

# ============================================================================
# OrthoMerge pieces (replace run_merge.py's helpers)
# ============================================================================

PROCRUSTES = {"mode": "minimal", "rank": None}


def _svd(matrix: torch.Tensor):
    try:
        return torch.linalg.svd(matrix.float(), full_matrices=False)
    except RuntimeError:
        u_factor, sigma, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
        return u_factor.float(), sigma.float(), vh.float()


def minimal_procrustes(cross: torch.Tensor, rank: int) -> torch.Tensor:
    """Among the rotations R that maximise <R, cross>, the one closest to the identity.

    For a wide layer (out < in) cross = W0^T M has rank at most out, so the Procrustes rotation is
    fixed only on that subspace; plain SVD fills the rest with an arbitrary map, which then enters
    the Cayley average. Here R = U_r V_r^T + U_n Z V_n^T with Z the closest orthogonal matrix to
    U_n^T V_n (det chosen so that det R = +1). W0 R is the same as with plain SVD."""
    u_factor, _sigma, vh = _svd(cross)
    v_factor = vh.t()
    u_r, v_r, u_n, v_n = u_factor[:, :rank], v_factor[:, :rank], u_factor[:, rank:], v_factor[:, rank:]
    zu, _zs, zvh = _svd(torch.matmul(u_n.t(), v_n))
    want = float(torch.sign(torch.linalg.det(u_factor)) * torch.sign(torch.linalg.det(v_factor)))
    if float(torch.sign(torch.linalg.det(torch.matmul(zu, zvh)))) != want:
        zu = zu.clone()
        zu[:, -1] = -zu[:, -1]
    return torch.matmul(u_r, v_r.t()) + torch.matmul(torch.matmul(u_n, torch.matmul(zu, zvh)), v_n.t())


def _rotation_on_span(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Orthogonal r x r map sending the frame alpha to beta, identity-like on the rest of the span.

    alpha and beta are r x k with orthonormal columns, k <= r. The free (r-k) block is the
    orthogonal matrix closest to the corresponding alpha/beta bases, with one sign chosen so
    the determinant is +1. A k x k SVD is the largest factorization here.
    """
    rank_r, k = alpha.shape
    if rank_r == k:
        rotation = torch.matmul(beta, alpha.t())
        if float(torch.linalg.slogdet(rotation)[0]) < 0:
            beta = beta.clone()
            beta[:, -1] = -beta[:, -1]
            rotation = torch.matmul(beta, alpha.t())
        return rotation
    a_perp = torch.linalg.qr(alpha, mode="complete")[0][:, k:]
    b_perp = torch.linalg.qr(beta, mode="complete")[0][:, k:]
    zu, _, zvh = torch.linalg.svd(torch.matmul(b_perp.t(), a_perp), full_matrices=False)
    small = torch.matmul(zu, zvh)
    basis_a = torch.cat([alpha, a_perp], dim=1)
    basis_b = torch.cat([beta, b_perp], dim=1)
    sign = (torch.linalg.slogdet(basis_a)[0] * torch.linalg.slogdet(basis_b)[0]
            * torch.linalg.slogdet(small)[0])
    if float(sign) < 0:
        zu = zu.clone()
        zu[:, -1] = -zu[:, -1]
        small = torch.matmul(zu, zvh)
    return torch.matmul(beta, alpha.t()) + torch.matmul(torch.matmul(b_perp, small), a_perp.t())


def lowrank_procrustes(cross: torch.Tensor, rank: int) -> torch.Tensor:
    """Rotation from the top-k randomized SVD of a square Gram.

    torch.svd_lowrank returns only U_k, S_k, V_k. R acts as U_k V_k^T on that frame and is
    exactly I on the orthogonal complement of span(U_k, V_k). No n x n SVD is computed.
    """
    size = cross.shape[0]
    k = min(int(rank), size)
    if k >= size:
        return closest_rotation_svd(cross)
    left, _singular, right = torch.svd_lowrank(cross, q=k, niter=2)
    basis, factors = torch.linalg.qr(torch.cat([right, left], dim=1), mode="reduced")
    scale = factors.diagonal().abs()
    keep = max(k, int(torch.sum(scale > scale[0] * 1e-4).item()))
    keep = min(keep, basis.shape[1])
    basis = basis[:, :keep]
    small = _rotation_on_span(torch.matmul(basis.t(), right), torch.matmul(basis.t(), left))
    eye = torch.eye(size, device=cross.device, dtype=cross.dtype)
    eye_small = torch.eye(keep, device=cross.device, dtype=cross.dtype)
    return eye + torch.matmul(torch.matmul(basis, small - eye_small), basis.t())


def orthogonal_procrustes_right(expert: torch.Tensor, w0: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
    """R in SO(n) that minimises ||W0 R - M||_F (n = number of inputs).

    rank is None: today's rule, unchanged.
    --procrustes svd     : closest rotation to W0^T M from one det-corrected SVD (the usual code).
    --procrustes minimal : the same for tall and square layers; for wide layers (out < in) the
                           minimiser closest to I (see minimal_procrustes).
    rank = k < n: top-k randomized SVD of W0^T M, completed by lowrank_procrustes.
    k >= n falls back to the full det-corrected SVD because there is nothing to truncate.
    """
    cross = torch.matmul(w0.float().t(), expert.float())
    out_rows, inputs = w0.shape
    if rank is not None:
        k = int(rank)
        if k < 1:
            raise SystemExit(f"procrustes rank must be positive, got {k}")
        return lowrank_procrustes(cross, k)
    if PROCRUSTES["mode"] == "svd" or out_rows >= inputs:
        return closest_rotation_svd(cross)
    return minimal_procrustes(cross, out_rows)


def cayley_to_skew(rotation: torch.Tensor) -> torch.Tensor:
    """Q = (R - I)(R + I)^{-1}; Q^T = -Q."""
    eye = torch.eye(rotation.shape[0], device=rotation.device, dtype=torch.float32)
    rotation = rotation.float()
    return torch.linalg.solve((rotation + eye).t(), (rotation - eye).t()).t()


def cayley_from_skew(skew: torch.Tensor) -> torch.Tensor:
    """R = (I + Q)(I - Q)^{-1}."""
    eye = torch.eye(skew.shape[0], device=skew.device, dtype=torch.float32)
    skew = skew.float()
    return torch.linalg.solve((eye - skew).t(), (eye + skew).t()).t()


def procrustes_check() -> None:
    """The minimal choice keeps W0 R of plain SVD, is a rotation, and is I when the experts equal the base."""
    torch.manual_seed(1)
    w0 = torch.randn(5, 12)
    expert = w0 + 0.1 * torch.randn(5, 12)
    plain = closest_rotation_svd(torch.matmul(w0.t(), expert))
    minimal = minimal_procrustes(torch.matmul(w0.t(), expert), 5)
    if float((torch.matmul(w0, plain) - torch.matmul(w0, minimal)).abs().max()) > 1e-4:
        raise SystemExit("minimal Procrustes changes W0 R")
    if float((minimal.t() @ minimal - torch.eye(12)).abs().max()) > 1e-4 or float(torch.det(minimal)) < 0:
        raise SystemExit("minimal Procrustes is not a rotation")
    if float((minimal - torch.eye(12)).norm()) > float((plain - torch.eye(12)).norm()) + 1e-5:
        raise SystemExit("minimal Procrustes is farther from I than plain SVD")
    same = minimal_procrustes(torch.matmul(w0.t(), w0), 5)
    if float((same - torch.eye(12)).abs().max()) > 1e-4:
        raise SystemExit("minimal Procrustes of the base itself is not I")
    # rank=None is the branch above. A square layer has no null complement, so it is the det-corrected SVD.
    square_w0 = torch.randn(8, 8)
    square_expert = square_w0 + 0.1 * torch.randn(8, 8)
    square_cross = torch.matmul(square_w0.t(), square_expert)
    if float((orthogonal_procrustes_right(square_expert, square_w0) - closest_rotation_svd(square_cross)).abs().max()) > 1e-4:
        raise SystemExit("rank None changed the square Procrustes rotation")
    wide = orthogonal_procrustes_right(expert, w0)
    if float((wide - minimal).abs().max()) > 1e-4:
        raise SystemExit("rank None changed the wide Procrustes rotation")
    for k in (1, 4):
        rotation = lowrank_procrustes(torch.matmul(w0.t(), expert), k)
        if rotation.shape != (12, 12):
            raise SystemExit(f"R_{k} is {tuple(rotation.shape)}, not (12, 12)")
        if float((rotation.t() @ rotation - torch.eye(12)).abs().max()) > 1e-4:
            raise SystemExit(f"R_{k} is not orthogonal")
        if float(torch.linalg.slogdet(rotation)[0]) < 0:
            raise SystemExit(f"det R_{k} is not +1")
    completed = orthogonal_procrustes_right(expert, w0, rank=12)
    if float((completed - closest_rotation_svd(torch.matmul(w0.t(), expert))).abs().max()) > 1e-4:
        raise SystemExit("k >= n is not the full det-corrected SVD")
    log("PROCRUSTES_CHECK_OK")


class TensorCache:
    """Disk cache for per-layer tensors that several methods share (expert Procrustes rotations,
    fitted rotations of methods with the same rotation objective). Disabled when root is None."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".pt")

    def get(self, key: str):
        if self.root is None:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu")
        except Exception:  # a half-written file from an interrupted run
            return None

    def put(self, key: str, value) -> None:
        if self.root is None:
            return
        path = self._path(key)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{random.getrandbits(32):08x}.tmp")  # unique per process
        torch.save(value, tmp)
        os.replace(tmp, path)


CACHE = TensorCache(None)
# Fit signatures shared by more than one selected method; only these fitted rotations are cached.
SHARED_FITS: set = set()


def expert_rotations(name: str, w0_t: torch.Tensor, experts: Sequence[torch.Tensor],
                     rank: Optional[int] = None) -> List[torch.Tensor]:
    """Procrustes rotations R_i of the experts (cached on disk: the SVDs are the costly part).

    rank=None keeps the cache key and the rotation used by every existing method.
    """
    key = f"procrustes/{name}" if rank is None else f"procrustes/k{int(rank)}/{name}"
    token = prof_begin()
    cached = CACHE.get(key)
    prof_end(token, "disk_read")
    if cached is not None and len(cached) == len(experts):
        token = prof_begin()
        loaded = [rotation.to(w0_t.device) for rotation in cached]
        prof_end(token, "copy")
        return loaded
    token = prof_begin()
    rotations = [orthogonal_procrustes_right(expert, w0_t, rank=rank) for expert in experts]
    prof_end(token, "expert_tsvd")
    token = prof_begin()
    CACHE.put(key, [rotation.detach().cpu() for rotation in rotations])
    prof_end(token, "disk_write")
    return rotations


def spolar(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonal SVD factor U @ Vh, same convention as orthogonal_procrustes_torch_right."""
    u_factor, _singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return torch.matmul(u_factor, vh)


def plain_q_average(skews: Sequence[torch.Tensor]) -> torch.Tensor:
    """Q_merge = (1/N) sum_i Q_i. No magnitude correction."""
    return torch.stack([skew.float() for skew in skews], dim=0).mean(dim=0)


def weighted_q_merge(skews: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Direction of sum w_i Q_i, scaled to the weighted average of ||Q_i||_F.

    Q_merge = (G / ||G||_F) * (g / sum w_i),
    G = sum w_i Q_i, g = sum w_i ||Q_i||_F.
    A near-zero G returns the zero matrix.
    """
    stacked = torch.stack([skew.float() for skew in skews], dim=0)
    weight = stacked.new_tensor(list(weights))
    combined = (stacked * weight.view(-1, 1, 1)).sum(dim=0)
    norms = torch.stack([torch.linalg.matrix_norm(skew.float(), ord="fro") for skew in skews])
    magnitude = torch.dot(weight, norms)
    combined_norm = torch.linalg.matrix_norm(combined, ord="fro")
    if float(combined_norm) < NORM_EPS:
        return torch.zeros_like(combined)
    return combined / combined_norm * (magnitude / weight.sum())


def mean_residual(residuals: Sequence[torch.Tensor]) -> torch.Tensor:
    """rho_merge = (1/N) sum_i rho_i. Identical to the current OrthoMerge rule."""
    return torch.stack([residual.float() for residual in residuals], dim=0).mean(dim=0)


def weighted_residual(residuals: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """rho_merge = sum w_i rho_i / sum w_i."""
    stacked = torch.stack([residual.float() for residual in residuals], dim=0)
    weight = stacked.new_tensor(list(weights))
    return (stacked * weight.view(-1, 1, 1)).sum(dim=0) / weight.sum()


def activation_aware_rotation_input(
    w0: torch.Tensor,
    experts: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    iters: int,
    tol: float,
    starts: Optional[Sequence[torch.Tensor]] = None,
    solver: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """Input-side rotation R (in x in) for W = W0 @ R, fitted to all experts' inputs.

    Objective  f(R) = tr(R G R^T C0) - 2 <R, K>  (plus a constant), with
    G = sum_i G_i, C0 = W0^T W0, K = sum_i W0^T M_i G_i. Solved by rotation_fit,
    started from every rotation in `starts` and from the closest rotation to K; the best is
    returned, and the objective never increases, so the result is never worse than a start.
    Returns (R, total steps used).
    """
    rotation, info = rotation_fit(
        w0, experts, grams, grams, list(starts or []), iters=iters, tol=tol,
        solver=solver or OPTIONS["aa_solver"], polar=OPTIONS["aa_polar"],
    )
    return rotation, int(info["steps"])


def activation_output_impractical(tensor: torch.Tensor) -> bool:
    """True when R would be a vocab-scale (out, out) matrix."""
    return tensor.ndim == 2 and int(tensor.shape[0]) > MAX_ACTIVATION_OUTPUT


def activation_aware_weight(
    w0: torch.Tensor,
    experts: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Slide rule. No Procrustes residual.

    F = sum_i M_i G_i W0^T
    R = spolar(F)
    W = R @ W0
    """
    total = torch.zeros(w0.shape[0], w0.shape[0], device=w0.device, dtype=torch.float32)
    base = w0.float()
    for expert, gram in zip(experts, grams):
        mixed = torch.matmul(expert.float(), gram.float())
        total = total + torch.matmul(mixed, base.t())
    if float(torch.linalg.matrix_norm(total, ord="fro")) < NORM_EPS:
        rotation = torch.eye(base.shape[0], device=base.device, dtype=torch.float32)
    else:
        rotation = spolar(total)
    return torch.matmul(rotation, base)


def regmean_weight(
    experts: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    ridge: float,
) -> torch.Tensor:
    """PyTorch-layout RegMean.

    Papers often store W as (in, out) and use (sum G_i)^{-1} (sum G_i W_i).
    Our Linear weights are (out, in), so

        W = (sum_i M_i @ G_i) @ (sum_i G_i + lambda I)^{-1}.

    lambda = ridge * mean(diag(sum G_i)). This is the closed form. No Q or rho.
    """
    alpha = float(OPTIONS["regmean_alpha"])
    if alpha < 1.0:
        # Published RegMean: keep the diagonal, shrink off-diagonal entries by alpha.
        grams = [alpha * gram.float() + (1.0 - alpha) * torch.diag(torch.diagonal(gram.float())) for gram in grams]
    total_gram = torch.stack([gram.float() for gram in grams], dim=0).sum(dim=0)
    total_mixed = torch.stack(
        [torch.matmul(expert.float(), gram.float()) for expert, gram in zip(experts, grams)],
        dim=0,
    ).sum(dim=0)
    mean_diag = torch.trace(total_gram) / total_gram.shape[0]
    penalty = float(ridge) * torch.clamp(mean_diag, min=NORM_EPS)
    system = total_gram + penalty * torch.eye(total_gram.shape[0], device=total_gram.device, dtype=total_gram.dtype)
    solved = torch.linalg.solve(system, total_mixed.transpose(0, 1))
    return solved.transpose(0, 1).contiguous()


def uniform_mean(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([tensor.float() for tensor in tensors], dim=0).mean(dim=0)

def self_check() -> None:
    """Shape and identity checks on tiny tensors. No checkpoints are loaded."""
    torch.manual_seed(0)
    q_one = torch.randn(5, 5)
    q_one = 0.5 * (q_one - q_one.t())
    averaged = plain_q_average([q_one, 2 * q_one])
    if not torch.allclose(averaged, 1.5 * q_one):
        raise SystemExit("Plain Q average failed")
    weighted = weighted_q_merge([q_one, q_one, q_one, q_one], [14000, 5000, 10000, 4000])
    if not torch.allclose(weighted, q_one, atol=1e-5):
        raise SystemExit("Weighted Q merge failed on identical skew matrices")
    zeros = weighted_q_merge([torch.zeros(5, 5) for _ in range(4)], [1, 1, 1, 1])
    if float(torch.linalg.matrix_norm(zeros, ord="fro")) != 0.0:
        raise SystemExit("Weighted Q merge failed the zero-norm case")

    out, inn = 6, 4
    w0 = torch.randn(out, inn)
    expert = torch.randn(out, inn)
    eye = torch.eye(inn)
    product = torch.matmul(expert, w0.t())
    u_factor, _singular, vh = torch.linalg.svd(product, full_matrices=False)
    left_rotation = torch.matmul(u_factor, vh)
    left_procrustes = torch.matmul(left_rotation, w0)
    adapted = activation_aware_weight(w0, [expert], [eye])
    if left_rotation.shape != (out, out):
        raise SystemExit(f"Left Procrustes rotation shape {tuple(left_rotation.shape)} is not {(out, out)}")
    if adapted.shape != (out, inn) or not torch.allclose(adapted, left_procrustes, atol=1e-4):
        raise SystemExit("Activation-aware G=I does not match left Procrustes")

    experts = [torch.randn(out, inn) for _ in range(4)]
    merged = regmean_weight(experts, [eye for _ in experts], ridge=0.0)
    if not torch.allclose(merged, uniform_mean(experts), atol=1e-4):
        raise SystemExit("RegMean with G=I did not return the uniform mean")
    if merged.shape != (out, inn):
        raise SystemExit(f"RegMean shape {tuple(merged.shape)} is not {(out, inn)}")
    # EqualQ is OrthoMerge's formula: (c / N) sum Q_i with c = sum ||Q_i|| / ||sum Q_i||.
    skews = []
    for _ in range(4):
        raw = torch.randn(5, 5)
        skews.append(0.3 * (raw - raw.t()))
    total = torch.stack(skews).sum(dim=0)
    scale = sum(float(torch.linalg.matrix_norm(q, ord="fro")) for q in skews) / float(torch.linalg.matrix_norm(total, ord="fro"))
    if not torch.allclose(weighted_q_merge(skews, [1.0] * 4), scale * total / 4, atol=1e-5):
        raise SystemExit("EqualQ does not match the OrthoMerge formula")

    # Input-side activation-aware rotation.
    w0_in = torch.randn(6, 4)
    one = torch.randn(6, 4)
    rotation, _used = activation_aware_rotation_input(w0_in, [one], [torch.eye(4)], iters=0, tol=1e-9)
    if not torch.allclose(rotation, closest_rotation_svd(torch.matmul(w0_in.t(), one)), atol=1e-5):
        raise SystemExit("ActivationAwareInput with G=I, one expert, is not the right Procrustes rotation")
    experts_in = [torch.randn(6, 4) for _ in range(3)]
    grams_in = []
    for _ in range(3):
        rows = torch.randn(30, 4) * torch.rand(4)
        grams_in.append(torch.matmul(rows.t(), rows))
    def fit(rot: torch.Tensor) -> float:
        total_fit = 0.0
        for expert_in, gram_in in zip(experts_in, grams_in):
            diff = torch.matmul(w0_in, rot) - expert_in
            total_fit += float(torch.trace(torch.matmul(torch.matmul(diff, gram_in), diff.t())))
        return total_fit
    start, _ = activation_aware_rotation_input(w0_in, experts_in, grams_in, iters=0, tol=0.0)
    final, used = activation_aware_rotation_input(w0_in, experts_in, grams_in, iters=200, tol=0.0)
    if fit(final) > fit(start) + 1e-4:
        raise SystemExit("ActivationAwareInput majorization did not decrease the fit")
    if float(torch.linalg.matrix_norm(torch.matmul(final.t(), final) - torch.eye(4))) > 1e-4:
        raise SystemExit("ActivationAwareInput rotation is not orthogonal")
    raw = torch.randn(4, 4)
    other_start = spolar(torch.eye(4) + 0.2 * (raw - raw.t()))
    from_start, _ = activation_aware_rotation_input(w0_in, experts_in, grams_in, iters=50, tol=0.0, starts=[other_start])
    if fit(from_start) > fit(other_start) + 1e-4:
        raise SystemExit("ActivationAwareInput returned a rotation worse than its starting rotation")
    log(f"ActivationAwareInput self-check: fit {fit(start):.4f} -> {fit(final):.4f} in {used} steps")
    # Adaptive solver reaches the same fit as the safe one.
    safe_rot, _ = activation_aware_rotation_input(w0_in, experts_in, grams_in, iters=3000, tol=1e-12, solver="safe")
    fast_rot, _ = activation_aware_rotation_input(w0_in, experts_in, grams_in, iters=3000, tol=1e-12, solver="accel")
    if fit(fast_rot) > fit(safe_rot) * (1 + 1e-3) + 1e-6:
        raise SystemExit("AA_Fast solver did not reach the safe solver's fit")
    # Correction with a tiny ridge equals RegMean; a huge anchor returns the anchor.
    correction = activation_correction(torch.matmul(w0_in, final), experts_in, grams_in, grams_in, ridge=1e-10)
    if not torch.allclose(torch.matmul(w0_in, final) + correction, regmean_weight(experts_in, grams_in, ridge=1e-10), atol=1e-3):
        raise SystemExit("Full activation-aware correction does not equal RegMean")
    anchored, _ = rotation_fit(w0_in, experts_in, grams_in, grams_in, [other_start], anchor=other_start,
                                  anchor_weight=1e6, iters=50, solver="accel")
    if not torch.allclose(anchored, other_start, atol=1e-3):
        raise SystemExit("AA_Anchor with a huge weight does not return the anchor")
    ties = ties_merge([torch.tensor([0.8, 0, -0.3, 0.5]), torch.tensor([0.6, -0.4, 0, -0.2]),
                          torch.tensor([0, -0.7, 0.5, 0.4]), torch.tensor([-0.2, 0, 0.6, 0.3])], keep=1.0)
    if not torch.allclose(ties, torch.tensor([0.70, -0.55, 0.55, 0.40]), atol=1e-6):
        raise SystemExit("TIES merge example failed")
    big = torch.randn(300, 300) + 5 * torch.eye(300)
    if float((closest_rotation(big, method="ns") - closest_rotation_svd(big)).abs().max()) > 1e-4:
        raise SystemExit("Newton-Schulz closest rotation does not match SVD")
    log("SELF_CHECK_OK")

class GramCollector:
    """Online sum of x^T x for nn.Linear inputs. Padding rows are zeroed."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.grams: Dict[str, torch.Tensor] = {}
        self.tokens: Dict[str, int] = {}
        self.encoder_mask: Optional[torch.Tensor] = None
        self.decoder_mask: Optional[torch.Tensor] = None
        self.mask_mismatches = 0

    def _mask(self, module_name: str, length: int) -> Optional[torch.Tensor]:
        if self.model_type == "causal":
            mask = self.decoder_mask
        elif module_name.startswith("encoder.") or ".EncDecAttention.k" in module_name or ".EncDecAttention.v" in module_name:
            mask = self.encoder_mask
        else:
            mask = self.decoder_mask
        if mask is None or mask.shape[-1] != length:
            self.mask_mismatches += 1
            return None
        return mask

    def hook(self, module_name: str, _module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        hidden = inputs[0]
        if not torch.is_tensor(hidden) or not torch.is_floating_point(hidden):
            return
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        if hidden.ndim != 3:
            return
        mask = self._mask(module_name, hidden.shape[1])
        values = hidden.detach().float()
        if mask is not None:
            values = values * mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
            token_count = int(mask.sum().item())
        else:
            token_count = int(values.shape[0] * values.shape[1])
        flat = values.reshape(-1, values.shape[-1])
        gram = torch.matmul(flat.transpose(0, 1), flat)
        existing = self.grams.get(module_name)
        if existing is None:
            self.grams[module_name] = gram
            self.tokens[module_name] = token_count
        else:
            self.grams[module_name] = existing + gram
            self.tokens[module_name] += token_count


def linear_parameter_names(model: nn.Module) -> Dict[str, str]:
    """Map weight-parameter name to the module name for every nn.Linear."""
    mapping = {}
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mapping[f"{module_name}.weight"] = module_name
    return mapping

# ============================================================================
# One matrix: dispatch to the merge rule
# ============================================================================

def fit_signature(cfg: dict) -> str:
    """Methods whose rotation objective is the same share the fitted rotation (disk cache)."""
    key = {"balance": cfg["balance"], "fisher": cfg["fisher"], "anchor": float(cfg["anchor"] or 0.0),
           "target": cfg["residual"] if cfg["consistent"] else "experts", "solver": cfg["solver"],
           "iters": int(cfg["iters"]), "tol": OPTIONS["aa_tol"], "polar": OPTIONS["aa_polar"],
           "ties_keep": OPTIONS["aa_ties_keep"] if cfg["consistent"] else None}
    if cfg.get("procrustes_rank") is not None:
        key["procrustes_rank"] = int(cfg["procrustes_rank"])
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def aa_merge(
    method: str,
    name: str,
    w0_rows: torch.Tensor,
    task_rows: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    ridge: float,
    device: torch.device,
    extra: dict,
) -> torch.Tensor:
    """One matrix of the activation-aware family (input-side rotation), see AA_CONFIG.

    extra may hold, per task in DATASETS order:
        tokens : token counts of the Grams (for balance)
        fisher : diagonal output Fisher vectors (for AA_Fisher)
        seq    : {"gram_in", "cross", "tokens"} from the merged model (sequential variants)
    """
    cfg = aa_settings(method)
    w0_t = w0_rows.detach().float().to(device)
    experts = [tensor.detach().float().to(device) for tensor in task_rows]
    rotations_i = expert_rotations(name, w0_t, experts, rank=cfg.get("procrustes_rank"))
    residuals = [expert - torch.matmul(w0_t, rotation) for expert, rotation in zip(experts, rotations_i)]
    orthomerge_rotation = cayley_from_skew(
        weighted_q_merge([cayley_to_skew(rotation) for rotation in rotations_i], [1.0] * len(experts))
    )
    del rotations_i
    seq = extra.get("seq")
    if seq is not None:
        gram_in = [g.detach().float().to(device) for g in seq["gram_in"]]
        cross = [c.detach().float().to(device) for c in seq["cross"]]
        tokens = seq["tokens"]
    else:
        gram_in = [g.detach().float().to(device) for g in grams]
        cross = gram_in
        tokens = extra.get("tokens")
    if cfg["balance"]:
        if tokens is None:
            raise SystemExit(f"{method} needs token counts for {name}")
        scale = [1.0 / float(max(int(t), 1)) for t in tokens]
        gram_in = [g * c for g, c in zip(gram_in, scale)]
        cross = [x * c for x, c in zip(cross, scale)] if seq is not None else gram_in
    if cfg["residual"] == "mean":
        rho = mean_residual(residuals)
    elif cfg["residual"] == "ties":
        rho = ties_merge(residuals, keep=OPTIONS["aa_ties_keep"])
    else:
        rho = None
    del residuals
    targets = [expert - rho for expert in experts] if (cfg["consistent"] and rho is not None) else experts
    s_diag = None
    if cfg["fisher"]:
        fisher = extra.get("fisher")
        if fisher is None:
            raise SystemExit(f"{method} needs Fisher statistics for {name}")
        s_diag = []
        for vector in fisher:
            vector = vector.detach().float().to(device)
            s_diag.append(vector / vector.mean().clamp_min(1e-30) + 1e-3)
    signature = fit_signature(cfg) if seq is None else None
    signature = signature if signature in SHARED_FITS else None
    cached = CACHE.get(f"fit/{signature}/{name}") if signature else None
    if cached is not None:
        rotation = cached.to(device)
    else:
        rotation, _info = rotation_fit(
            w0_t, targets, gram_in, cross, [orthomerge_rotation], s_diag=s_diag,
            anchor=orthomerge_rotation if cfg["anchor"] else None, anchor_weight=float(cfg["anchor"] or 0.0),
            iters=int(cfg["iters"]), tol=OPTIONS["aa_tol"], solver=cfg["solver"], polar=OPTIONS["aa_polar"],
        )
        if signature:
            CACHE.put(f"fit/{signature}/{name}", rotation.detach().cpu())
    merged = torch.matmul(w0_t, rotation)
    if rho is not None:
        merged = merged + rho
    elif cfg["residual"] == "corr":
        rank = cfg["rank"] if cfg["rank"] else None
        merged = merged + float(cfg["alpha"]) * activation_correction(merged, experts, gram_in, cross, ridge, rank=rank)
    return merged.detach().cpu()


def merge_matrix(
    method: str,
    name: str,
    w0_rows: torch.Tensor,
    task_rows: Sequence[torch.Tensor],
    grams: Optional[Sequence[torch.Tensor]],
    ridge: float,
    device: torch.device,
    extra: Optional[dict] = None,
) -> torch.Tensor:
    """Merged weight of one Linear layer (out x in)."""
    weights = [TASK_WEIGHTS[task] for task in DATASETS]
    if method in AA_METHODS and grams is not None:
        return aa_merge(method, name, w0_rows, task_rows, grams, ridge, device, extra or {})
    if method in {"EqualQ", "PlainQ", "WeightedQ"} or method in AA_METHODS:
        w0_t = w0_rows.detach().float().to(device)
        experts = [tensor.detach().float().to(device) for tensor in task_rows]
        rotations = expert_rotations(name, w0_t, experts)
        skews = [cayley_to_skew(rotation) for rotation in rotations]
        residuals = [expert - torch.matmul(w0_t, rotation) for expert, rotation in zip(experts, rotations)]
        del rotations
        if method == "WeightedQ" and OPTIONS["weighted_residual"]:
            rho = weighted_residual(residuals, weights)
        else:
            rho = mean_residual(residuals)
        if method == "PlainQ":
            q_merged = plain_q_average(skews)
        elif method == "WeightedQ":
            q_merged = weighted_q_merge(skews, weights)
        else:  # EqualQ = OrthoMerge: Q = (sum ||Q_i|| / N) * sum Q_i / ||sum Q_i||
            q_merged = weighted_q_merge(skews, [1.0] * len(skews))
        return (torch.matmul(w0_t, cayley_from_skew(q_merged)) + rho).detach().cpu()
    if method == "ActivationAware" and grams is not None:
        if activation_output_impractical(w0_rows):
            return uniform_mean(task_rows).detach().cpu()
        w0_t = w0_rows.detach().float().to(device)
        experts = [tensor.detach().float().to(device) for tensor in task_rows]
        gram_t = [gram.detach().float().to(device) for gram in grams]
        return activation_aware_weight(w0_t, experts, gram_t).detach().cpu()
    if method == "RegMean" and grams is not None:
        experts = [tensor.detach().float().to(device) for tensor in task_rows]
        gram_t = [gram.detach().float().to(device) for gram in grams]
        return regmean_weight(experts, gram_t, ridge).detach().cpu()
    raise SystemExit(f"{method} needs Gram matrices for {name}")


# ============================================================================
# Data: CL_Benchmark files and the O-LoRA prompt (DataCollatorForUIE)
# ============================================================================

def load_instances(data_dir: Path, task: str, split: str) -> List[dict]:
    """Instances in the format of O-LoRA's uie_dataset (zero-shot, 'single' instruction)."""
    category, name = TASK_FOLDERS[task]
    folder = Path(data_dir) / category / name
    for needed in (f"{split}.json", "labels.json"):
        if not (folder / needed).exists():
            raise SystemExit(f"Missing {folder / needed}. --data-dir must be the CL_Benchmark folder "
                             "(TC/dbpedia, SC/amazon, TC/yahoo, TC/agnews).")
    with open(folder / f"{split}.json", encoding="utf-8") as handle:
        records = json.load(handle)
    with open(folder / "labels.json", encoding="utf-8") as handle:
        labels = json.load(handle)
    instruction = INSTRUCTIONS[category] + "Option: " + ", ".join(labels) + " \n" + "{0}" + "\nAnswer:"
    return [{
        "Task": category, "Dataset": name, "Samples": [], "subset": split,
        "Instance": {"id": str(index), "sentence": record["sentence"], "label": record["label"],
                     "ground_truth": record["label"], "instruction": instruction},
    } for index, record in enumerate(records)]


def subsample(instances: List[dict], count: Optional[int], seed: int) -> List[dict]:
    if count is None or count >= len(instances):
        return instances
    keep = sorted(random.Random(seed).sample(range(len(instances)), int(count)))
    return [instances[index] for index in keep]


class UIECollator:
    """O-LoRA's DataCollatorForUIE (add_task_name=True, add_dataset_name=True, zero-shot), with the
    model type given explicitly. Causal batches also carry answer_mask (1 on the answer tokens)."""

    def __init__(self, tokenizer, model, model_type: str, max_source_length: int = 512, max_target_length: int = 50,
                 pad_to_multiple_of: Optional[int] = 8, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.model = model
        self.model_type = model_type
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def _multiple(self, max_length: int) -> Optional[int]:
        """pad_to_multiple_of only when the truncation length is a multiple of it (transformers
        refuses the combination otherwise); padding is masked out everywhere, so this is harmless."""
        multiple = self.pad_to_multiple_of
        return multiple if multiple and max_length % multiple == 0 else None

    @staticmethod
    def instruction(instance: dict) -> str:
        prefix = "Task:" + instance["Task"] + "\n" + "Dataset:" + instance["Dataset"] + "\n"
        return (prefix + instance["Instance"]["instruction"]).format(instance["Instance"]["sentence"])

    def __call__(self, batch: List[dict]) -> dict:
        return self.seq2seq_call(batch) if self.model_type == "seq2seq" else self.decoder_call(batch)

    def seq2seq_call(self, batch: List[dict]) -> dict:
        tok = self.tokenizer
        sources, labels = [], []
        for instance in batch:
            labels.append(instance["Instance"]["label"])
            source = self.instruction(instance)
            ids = tok(source)["input_ids"]
            sources.append(source if len(ids) <= self.max_source_length
                           else tok.decode(ids[: self.max_source_length], skip_special_tokens=True))
        inputs = tok(sources, max_length=self.max_source_length, padding="longest", return_tensors="pt",
                     truncation=True, pad_to_multiple_of=self._multiple(self.max_source_length))
        targets = tok(text_target=labels, max_length=self.max_target_length, padding="longest", return_tensors="pt",
                      truncation=True, pad_to_multiple_of=self._multiple(self.max_target_length))
        label_ids = targets["input_ids"].masked_fill(~targets["attention_mask"].bool(), self.label_pad_token_id)
        model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": label_ids}
        model_inputs["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=label_ids)
        model_inputs["gold"] = labels
        return model_inputs

    def decoder_call(self, batch: List[dict]) -> dict:
        tok = self.tokenizer
        tok.padding_side = "left"
        train = batch[0]["subset"] == "train"
        limit = self.max_source_length + self.max_target_length if train else self.max_source_length
        sources, label_lens, golds = [], [], []
        for instance in batch:
            label = instance["Instance"]["label"]
            golds.append(label)
            task_input = tok.bos_token + self.instruction(instance)
            label_text = label + tok.eos_token
            input_ids = tok(task_input)["input_ids"]
            label_ids = tok(label_text)["input_ids"]
            if not train:
                label_lens.append(0)
                sources.append(task_input if len(input_ids) <= limit
                               else tok.decode(input_ids[:limit], skip_special_tokens=False))
            elif len(input_ids) + len(label_ids) <= limit:
                label_lens.append(len(label_ids))
                sources.append(task_input + label_text)
            else:
                sources.append(tok.decode((input_ids + label_ids)[:limit], skip_special_tokens=False))
                label_lens.append(max(0, limit - len(input_ids)))
        inputs = tok(sources, max_length=limit, padding="longest", return_tensors="pt", truncation=True,
                     pad_to_multiple_of=self._multiple(limit))
        mask = inputs["attention_mask"].bool()
        length = inputs["input_ids"].shape[1]
        answer = torch.zeros(inputs["input_ids"].shape, dtype=torch.float32)
        for row, label_len in enumerate(label_lens):
            if label_len > 0:
                answer[row, length - label_len:] = 1.0  # left padding: the answer ends the row
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                "labels": inputs["input_ids"].masked_fill(~mask, self.label_pad_token_id),
                "answer_mask": answer * mask.float(), "gold": golds}


def make_loader(instances: List[dict], collator: UIECollator, batch_size: int):
    from torch.utils.data import DataLoader
    return DataLoader(instances, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)


def forward_inputs(batch: dict, model_type: str, device: torch.device) -> dict:
    keys = ("input_ids", "attention_mask", "decoder_input_ids") if model_type == "seq2seq" else ("input_ids", "attention_mask")
    return {key: batch[key].to(device) for key in keys}


def batch_masks(model_type: str, batch: dict):
    """(encoder_mask, decoder_mask) for a collated batch: padding is excluded from all statistics."""
    if model_type == "causal":
        return None, batch["attention_mask"]
    return batch["attention_mask"], batch["labels"].ne(-100).to(dtype=torch.float32)


# ============================================================================
# Models and checkpoints
# ============================================================================

def model_class(model_type: str):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    return AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM


def load_model(path: str, model_type: str) -> nn.Module:
    """fp32 model on the CPU (works with transformers 4.x and 5.x)."""
    import transformers
    cls = model_class(model_type)
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = cls.from_pretrained(str(path), **{key: torch.float32})
    return model.float().eval()


def cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().float().cpu().clone() for key, value in model.state_dict().items()}


def infer_model_type(path: str) -> str:
    from transformers import AutoConfig
    return "seq2seq" if getattr(AutoConfig.from_pretrained(str(path)), "is_encoder_decoder", False) else "causal"


def load_tokenizer(paths: Sequence[str], model_type: str):
    from transformers import AutoTokenizer
    tokenizer = None
    for path in paths:
        local = Path(path).exists()
        if local and not (Path(path) / "tokenizer_config.json").exists():
            continue  # a local folder without tokenizer files
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True)
            log(f"Tokenizer from {path}")
            break
        except Exception as error:  # e.g. no network for a Hugging Face id
            log(f"Tokenizer from {path} failed: {error}")
    if tokenizer is None:
        raise SystemExit("No tokenizer found in the expert or base directories.")
    if model_type == "causal" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def align_base_rows(base: Dict[str, torch.Tensor], expert: Dict[str, torch.Tensor]) -> None:
    """If the experts have a resized vocabulary, give the base the same number of rows. Extra rows
    are zero in the base, so the merge takes them from the experts (mean, or mean leftover)."""
    for key, value in expert.items():
        if key not in base:
            raise SystemExit(f"Base model has no parameter {key}")
        if base[key].shape == value.shape:
            continue
        if base[key].ndim == value.ndim and base[key].shape[1:] == value.shape[1:]:
            rows = value.shape[0]
            fixed = torch.zeros_like(value)
            keep = min(rows, base[key].shape[0])
            fixed[:keep] = base[key][:keep]
            base[key] = fixed
            log(f"Base {key}: rows {tuple(base[key].shape)} aligned to the experts ({rows} rows)")
        else:
            raise SystemExit(f"Shape mismatch for {key}: base {tuple(base[key].shape)}, expert {tuple(value.shape)}")


def checkpoint_fingerprint(paths: Sequence[str]) -> str:
    """Changes when a checkpoint file changes (sizes and modification times)."""
    items = []
    for path in paths:
        folder = Path(path)
        if folder.is_dir():
            for file in sorted(folder.iterdir()):
                if file.suffix in {".safetensors", ".bin", ".pt"} or file.name.endswith(".index.json"):
                    stat = file.stat()
                    items.append((str(file.resolve()), stat.st_size, int(stat.st_mtime)))
        else:
            items.append((str(path), 0, 0))
    return hashlib.md5(json.dumps(items).encode()).hexdigest()[:12]


# ============================================================================
# Statistics: Gram matrices and Fisher, collected on the GPU
# ============================================================================

class StatStore:
    """Per-task, per-layer Gram matrices on disk (a Llama-3.2-1B task is about 6 GB)."""

    def __init__(self, root: Path):
        self.root = root

    def folder(self, task: str) -> Path:
        return self.root / task

    def complete(self, task: str) -> bool:
        return (self.folder(task) / "tokens.json").exists()

    def save(self, task: str, grams: Dict[str, torch.Tensor], tokens: Dict[str, int]) -> None:
        folder = self.folder(task)
        folder.mkdir(parents=True, exist_ok=True)
        for param, gram in grams.items():
            tmp = folder / f"{param}.{os.getpid()}.tmp"
            torch.save(gram.detach().cpu(), tmp)
            os.replace(tmp, folder / (param + ".pt"))
        tmp = folder / f"tokens.{os.getpid()}.tmp"
        with open(tmp, "w") as handle:
            json.dump(tokens, handle)
        os.replace(tmp, folder / "tokens.json")  # written last: marks the task as complete

    def gram(self, task: str, param: str) -> torch.Tensor:
        return torch.load(self.folder(task) / (param + ".pt"), map_location="cpu")

    def tokens(self, task: str) -> Dict[str, int]:
        with open(self.folder(task) / "tokens.json") as handle:
            return json.load(handle)


@torch.no_grad()
def collect_grams(task: str, state: Dict[str, torch.Tensor], worker: nn.Module, loader, model_type: str,
                  device: torch.device, store: StatStore) -> None:
    worker.load_state_dict(state, strict=True)
    collector = GramCollector(model_type)
    handles = []
    for name, module in worker.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(
                lambda mod, inputs, _out, name=name: collector.hook(name, mod, inputs)))
    started = time.time()
    try:
        for batch in loader:
            collector.encoder_mask, collector.decoder_mask = [None if m is None else m.to(device) for m in batch_masks(model_type, batch)]
            worker(**forward_inputs(batch, model_type, device))
    finally:
        for handle in handles:
            handle.remove()
    if collector.mask_mismatches:
        raise SystemExit(f"Gram collection for {task}: {collector.mask_mismatches} mask mismatches. Stopping.")
    grams = {f"{name}.weight": gram for name, gram in collector.grams.items()}
    tokens = {f"{name}.weight": count for name, count in collector.tokens.items()}
    store.save(task, grams, tokens)
    log(f"Grams {task}: {len(grams)} layers, {time.time() - started:.0f}s")
    collector = grams = None
    if device.type == "cuda":
        torch.cuda.empty_cache()


def collect_fisher(task: str, state: Dict[str, torch.Tensor], worker: nn.Module, loader, model_type: str,
                   device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    """Diagonal output Fisher of every Linear: sum over tokens of g^2, g = dLoss/d(output), labels
    sampled from the expert's own prediction (answer tokens only). One backward pass per batch."""
    worker.load_state_dict(state, strict=True)
    collector = FisherCollector(model_type)
    handles = [module.register_forward_hook(collector.forward_hook(name))
               for name, module in worker.named_modules() if isinstance(module, nn.Linear)]
    handles.append(enable_input_grad(worker))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    try:
        with torch.enable_grad():
            for batch in loader:
                collector.encoder_mask, collector.decoder_mask = [None if m is None else m.to(device) for m in batch_masks(model_type, batch)]
                inputs = forward_inputs(batch, model_type, device)
                logits = worker(**inputs).logits
                if model_type == "causal":
                    target = inputs["input_ids"].masked_fill(batch["answer_mask"].to(device) == 0, -100)
                else:
                    target = batch["labels"].to(device)
                loss = sampled_fisher_loss(model_type, logits, target, inputs.get("attention_mask"), generator)
                loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    return {f"{name}.weight": value.detach().cpu() for name, value in collector.sums.items()}


# ============================================================================
# Merging a whole model
# ============================================================================

def parameter_kind(name: str, tensor: torch.Tensor, linear_names: Dict[str, str], model_type: str, tied: bool) -> str:
    if model_type == "causal" and tied and name == "lm_head.weight":
        return "tied_lm_head"  # copied from the merged embedding
    if name in linear_names and tensor.ndim == 2:
        return "gram_linear"
    return "mean"  # embeddings, relative attention bias, norms


def tie_merged_state(state: Dict[str, torch.Tensor], model_type: str, tied: bool) -> None:
    """Keep tied matrices identical: Llama copies the merged embedding into lm_head; T5 (tied)
    copies the merged lm_head into the shared embeddings (as run_alternative_merges.py)."""
    if not tied:
        return
    if model_type == "causal":
        state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    elif "lm_head.weight" in state:
        for key in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
            if key in state and state[key].shape == state["lm_head.weight"].shape:
                state[key] = state["lm_head.weight"].clone()


def _aa_layer_inputs(method: str, name: str, ctx: dict) -> dict:
    """Per-layer inputs of ``aa_merge``, before the rotation search."""
    cfg = aa_settings(method)
    device = ctx["device"]
    if PROFILE.enabled:
        out_rows, in_cols = ctx["w0"][name].shape
        PROFILE.shape = f"R{int(in_cols)}x{int(in_cols)}|W{int(out_rows)}x{int(in_cols)}"
    token = prof_begin()
    w0_t = ctx["w0"][name].detach().float().to(device)
    experts = [ctx["task_states"][task][name].detach().float().to(device) for task in DATASETS]
    prof_end(token, "copy")
    rotations_i = expert_rotations(name, w0_t, experts, rank=cfg.get("procrustes_rank"))
    token = prof_begin()
    residuals = [expert - torch.matmul(w0_t, rotation) for expert, rotation in zip(experts, rotations_i)]
    prof_end(token, "copy")
    token = prof_begin()
    start = cayley_from_skew(weighted_q_merge([cayley_to_skew(rotation) for rotation in rotations_i], [1.0] * len(experts)))
    prof_end(token, "cayley")
    token = prof_begin()
    gram_cpu = [ctx["store"].gram(task, name) for task in DATASETS]
    prof_end(token, "disk_read")
    token = prof_begin()
    gram_in = [gram.detach().float().to(device) for gram in gram_cpu]
    prof_end(token, "copy")
    if cfg["balance"]:
        tokens = [ctx["tokens"][task].get(name, 1) for task in DATASETS]
        gram_in = [gram * (1.0 / float(max(int(count), 1))) for gram, count in zip(gram_in, tokens)]
    if cfg["residual"] == "mean":
        rho = mean_residual(residuals)
    elif cfg["residual"] == "ties":
        rho = ties_merge(residuals, keep=OPTIONS["aa_ties_keep"])
    else:
        rho = None
    targets = [expert - rho for expert in experts] if (cfg["consistent"] and rho is not None) else experts
    return {"name": name, "w0": w0_t, "experts": experts, "gram_in": gram_in, "start": start,
            "rho": rho, "targets": targets, "anchor": start if cfg["anchor"] else None}


def aa_merge_grouped(method: str, names: Sequence[str], ctx: dict) -> Dict[str, torch.Tensor]:
    """Search same-shaped layers together. Each layer still uses ``rotation_fit``'s update."""
    cfg = aa_settings(method)
    device = ctx["device"]
    signature = fit_signature(cfg)
    signature = signature if signature in SHARED_FITS else None
    pending: Dict[Tuple[int, int], List[str]] = {}
    merged: Dict[str, torch.Tensor] = {}
    for name in names:
        cached = CACHE.get(f"fit/{signature}/{name}") if signature else None
        if cached is None:
            shape = tuple(int(dim) for dim in ctx["w0"][name].shape)
            pending.setdefault(shape, []).append(name)
            continue
        merged[name] = aa_merge(method, name, ctx["w0"][name],
                                [ctx["task_states"][task][name] for task in DATASETS],
                                [ctx["store"].gram(task, name) for task in DATASETS],
                                ctx["ridge"], device,
                                {"tokens": [ctx["tokens"][task].get(name, 1) for task in DATASETS]})
    started = time.time()
    done = len(merged)
    total = len(names)
    for shape, group in pending.items():
        width = fit_batch_width(shape[1])
        for offset in range(0, len(group), width):
            chunk_names = group[offset:offset + width]
            wide_names = [name for name in chunk_names
                          if cfg.get("skip_fit_inputs") is not None
                          and int(shape[1]) >= int(cfg["skip_fit_inputs"])]
            if wide_names:
                prepared_wide = [_aa_layer_inputs(method, name, ctx) for name in wide_names]
                for item in prepared_wide:
                    weight = torch.matmul(item["w0"], item["start"])
                    if item["rho"] is not None:
                        weight = weight + item["rho"]
                    elif cfg["residual"] == "corr":
                        rank = cfg["rank"] if cfg["rank"] else None
                        weight = weight + float(cfg["alpha"]) * activation_correction(
                            weight, item["experts"], item["gram_in"], item["gram_in"], ctx["ridge"], rank=rank)
                    merged[item["name"]] = weight.detach().cpu()
                    SKIPPED_FIT["count"] += 1
                done += len(wide_names)
                log(f"  {method} skip rotation_fit {done}/{total} shape={shape} ({time.time() - started:.0f}s)")
                del prepared_wide
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            chunk_names = [name for name in chunk_names if name not in wide_names]
            if not chunk_names:
                continue
            prepared = [_aa_layer_inputs(method, name, ctx) for name in chunk_names]
            token = prof_begin()
            w0 = torch.stack([item["w0"] for item in prepared])
            targets = torch.stack([torch.stack(item["targets"]) for item in prepared])
            gram_in = torch.stack([torch.stack(item["gram_in"]) for item in prepared])
            starts = torch.stack([item["start"] for item in prepared])
            anchor = torch.stack([item["anchor"] for item in prepared]) if prepared[0]["anchor"] is not None else None
            prof_end(token, "copy")
            rotations = rotation_fit_batch(
                w0, targets, gram_in, gram_in, starts, iters=int(cfg["iters"]), tol=OPTIONS["aa_tol"],
                solver=cfg["solver"], polar=OPTIONS["aa_polar"], anchor=anchor,
                anchor_weight=float(cfg["anchor"] or 0.0),
            )
            for index, item in enumerate(prepared):
                if signature:
                    token = prof_begin()
                    CACHE.put(f"fit/{signature}/{item['name']}", rotations[index].detach().cpu())
                    prof_end(token, "disk_write")
                token = prof_begin()
                weight = torch.matmul(w0[index], rotations[index])
                prof_end(token, "apply")
                if item["rho"] is not None:
                    weight = weight + item["rho"]
                elif cfg["residual"] == "corr":
                    rank = cfg["rank"] if cfg["rank"] else None
                    token = prof_begin()
                    correction = activation_correction(
                        weight, item["experts"], item["gram_in"], item["gram_in"], ctx["ridge"], rank=rank)
                    prof_end(token, "correction")
                    weight = weight + float(cfg["alpha"]) * correction
                token = prof_begin()
                merged[item["name"]] = weight.detach().cpu()
                prof_end(token, "copy")
            done += len(chunk_names)
            log(f"  {method} grouped {done}/{total} shape={shape} ({time.time() - started:.0f}s)")
            del w0, targets, gram_in, starts, rotations, prepared
            if device.type == "cuda":
                token = prof_begin()
                torch.cuda.empty_cache()
                prof_end(token, "overhead_flush")
            if PROFILE.enabled:
                PROFILE.dump()
    return merged


def merge_model(method: str, ctx: dict) -> Dict[str, torch.Tensor]:
    model_type, device, ridge = ctx["model_type"], ctx["device"], ctx["ridge"]
    w0, task_states, store = ctx["w0"], ctx["task_states"], ctx["store"]
    merged: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    names = list(w0.keys())
    started = time.time()
    SKIPPED_FIT["count"] = 0
    cfg = aa_settings(method) if method in AA_METHODS else None
    if cfg is not None and not cfg["seq"] and not cfg["fisher"]:
        linear_names = []
        for name in names:
            kind = parameter_kind(name, w0[name], ctx["linear_names"], model_type, ctx["tied"])
            counts[kind] = counts.get(kind, 0) + 1
            rows = [task_states[task][name] for task in DATASETS]
            if kind == "tied_lm_head":
                continue
            if kind == "mean":
                token = prof_begin()
                merged[name] = uniform_mean(rows)
                prof_end(token, "copy")
                continue
            linear_names.append(name)
        merged.update(aa_merge_grouped(method, linear_names, ctx))
        if cfg.get("skip_fit_inputs") is not None:
            log(f"{method} rotation_fit skipped {SKIPPED_FIT['count']} layers")
        tie_merged_state(merged, model_type, ctx["tied"])
        for name in names:
            merged.setdefault(name, w0[name].clone())
        log(f"{method} parameter kinds: {counts}")
        return merged
    for index, name in enumerate(names, start=1):
        kind = parameter_kind(name, w0[name], ctx["linear_names"], model_type, ctx["tied"])
        counts[kind] = counts.get(kind, 0) + 1
        rows = [task_states[task][name] for task in DATASETS]
        if kind == "tied_lm_head":
            continue
        if kind == "mean":
            merged[name] = uniform_mean(rows)
            continue
        grams = [store.gram(task, name) for task in DATASETS] if method in GRAM_METHODS else None
        extra = None
        if method in AA_METHODS:
            extra = {"tokens": [ctx["tokens"][task].get(name, 1) for task in DATASETS],
                     "fisher": [ctx["fisher"][task][name] for task in DATASETS] if ctx.get("fisher") else None}
        merged[name] = merge_matrix(method, name, w0[name], rows, grams, ridge, device, extra)
        del grams
        if index % 25 == 0 or index == len(names):
            log(f"  {method} {index}/{len(names)} {name} ({time.time() - started:.0f}s)")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tie_merged_state(merged, model_type, ctx["tied"])
    for name in names:  # parameters skipped above (tied lm_head) and any left over
        merged.setdefault(name, w0[name].clone())
    log(f"{method} parameter kinds: {counts}")
    return merged


def sequential_refine(method: str, merged_state: Dict[str, torch.Tensor], ctx: dict) -> None:
    """Refit every block bottom-up on the inputs the merged model actually gives it (in place).

    For each transformer block, in forward order, the expert and the merged model run on the
    same batches (stopped after the block); the paired statistics Xe^T Xm and Xm^T Xm replace
    the expert-only Grams, the block's Linear layers are merged again and written into the
    merged model, and the next block sees the updated model.
    """
    model_type, device, ridge = ctx["model_type"], ctx["device"], ctx["ridge"]
    w0, task_states, store, worker = ctx["w0"], ctx["task_states"], ctx["store"], ctx["worker"]
    merged_model = ctx["merged_worker"]
    merged_model.load_state_dict(merged_state, strict=True)
    loaders = ctx["seq_loaders"]
    module_of = {param: module for param, module in ctx["linear_names"].items()
                 if parameter_kind(param, w0[param], ctx["linear_names"], model_type, ctx["tied"]) == "gram_linear"}
    first = forward_inputs(next(iter(loaders[DATASETS[0]])), model_type, device)
    order = linear_call_order(merged_model, list(module_of.values()), lambda: merged_model(**first))
    groups = group_by_block(order)
    log(f"{method} sequential: {len(groups)} blocks, {len(order)} Linear layers, up to {OPTIONS['seq_max_samples']} samples per task")
    started = time.time()
    for index, group in enumerate(groups, start=1):
        stats = {}
        for task in DATASETS:
            worker.load_state_dict(task_states[task], strict=True)
            collector = PairedGroupCollector(model_type, group, last_name=group[-1])
            handles = [worker.get_submodule(n).register_forward_hook(collector.hook(n)) for n in group]
            handles += [merged_model.get_submodule(n).register_forward_hook(collector.hook(n)) for n in group]
            try:
                for batch in loaders[task]:
                    collector.encoder_mask, collector.decoder_mask = [None if m is None else m.to(device) for m in batch_masks(model_type, batch)]
                    inputs = forward_inputs(batch, model_type, device)
                    collector.phase = "expert"
                    run_until_stop(worker, inputs)
                    collector.phase = "merged"
                    run_until_stop(merged_model, inputs)
                    collector.end_batch()
            finally:
                for handle in handles:
                    handle.remove()
            if collector.mismatches:
                raise SystemExit(f"{method} sequential: mask mismatches in block {group[0]} for {task}. Stopping.")
            stats[task] = {n: (collector.gram_in[n], collector.cross[n], collector.tokens[n]) for n in group}
            del collector
        for module_name in group:
            param = f"{module_name}.weight"
            seq = {"gram_in": [stats[task][module_name][0] for task in DATASETS],
                   "cross": [stats[task][module_name][1] for task in DATASETS],
                   "tokens": [stats[task][module_name][2] for task in DATASETS]}
            extra = {"seq": seq, "tokens": [ctx["tokens"][task].get(param, 1) for task in DATASETS],
                     "fisher": [ctx["fisher"][task][param] for task in DATASETS] if ctx.get("fisher") else None}
            merged = merge_matrix(method, param, w0[param], [task_states[task][param] for task in DATASETS],
                                  [store.gram(task, param) for task in DATASETS], ridge, device, extra)
            target = merged_model.get_submodule(module_name).weight
            target.data.copy_(merged.to(device=target.device, dtype=target.dtype))
            merged_state[param] = merged
        del stats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"  {method} sequential block {index}/{len(groups)} {group[0]} ({time.time() - started:.0f}s)")
    tie_merged_state(merged_state, model_type, ctx["tied"])


# ============================================================================
# Evaluation: generation + exact match (O-LoRA compute_metrics)
# ============================================================================

RUN = {"tf32_eval": False}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


@torch.no_grad()
def evaluate(model: nn.Module, tokenizer, ctx: dict, tasks: Sequence[str] = DATASETS,
             record: Optional[Path] = None) -> Dict[str, float]:
    from transformers import GenerationConfig
    model_type, device = ctx["model_type"], ctx["device"]
    gen = GenerationConfig(max_new_tokens=ctx["max_new_tokens"], do_sample=False, num_beams=1,
                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                           decoder_start_token_id=getattr(model.config, "decoder_start_token_id", None))
    scores = {}
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = RUN["tf32_eval"]
    try:
        scores = _evaluate_tasks(model, tokenizer, ctx, tasks, record, gen, model_type, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return scores


def _evaluate_tasks(model, tokenizer, ctx, tasks, record, gen, model_type, device) -> Dict[str, float]:
    scores = {}
    for task in tasks:
        correct, total, rows = 0, 0, []
        for batch in ctx["test_loaders"][task]:
            inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}
            output = model.generate(**inputs, generation_config=gen)
            texts = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if model_type == "causal":
                texts = [text.split(ANSWER_PREFIX)[-1].strip() if ANSWER_PREFIX in text else "" for text in texts]
            for prediction, gold in zip(texts, batch["gold"]):
                hit = normalize_answer(prediction) == normalize_answer(gold)
                correct += int(hit)
                total += 1
                if record is not None:
                    rows.append({"prediction": prediction, "gold": gold, "correct": hit})
        scores[task] = 100.0 * correct / max(total, 1)
        if record is not None:
            record.mkdir(parents=True, exist_ok=True)
            with open(record / f"{task}.jsonl", "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return scores


def score_state(state: Dict[str, torch.Tensor], ctx: dict, tasks: Sequence[str] = DATASETS, record: Optional[Path] = None) -> Dict[str, float]:
    model = ctx["eval_model"]
    model.load_state_dict(state, strict=True)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return evaluate(model, ctx["tokenizer"], ctx, tasks, record)


# ============================================================================
# Results table
# ============================================================================

def read_results(path: Path) -> Dict[str, dict]:
    if path.exists():
        with open(path) as handle:
            return json.load(handle)
    return {}


def ordered(results: Dict[str, dict]) -> Dict[str, dict]:
    """Reference rows first, then the methods in METHODS order."""
    rank = {name: index for index, name in enumerate(["Base", "Experts (own task)"] + [f"Expert {t}" for t in DATASETS] + list(METHODS))}
    return {name: results[name] for name in sorted(results, key=lambda n: (rank.get(n, len(rank)), n))}


def save_rows(out: Path, rows: Dict[str, dict]) -> Dict[str, dict]:
    """Add rows to results.json/csv under a file lock, keeping rows written by other processes
    (several GPUs may run different methods into the same output folder)."""
    import fcntl
    with open(out / "results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = read_results(out / "results.json")
        results.update(rows)
        results = ordered(results)
        write_results(results, out)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return results


def write_results(results: Dict[str, dict], out: Path) -> None:
    tmp = out / "results.json.tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=1)
    os.replace(tmp, out / "results.json")
    with open(out / "results.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *DATASETS, "average", "seconds"])
        for name, row in results.items():
            scores = row["scores"]
            values = [scores.get(task) for task in DATASETS]
            present = [value for value in values if value is not None]
            average = sum(present) / len(present) if len(present) == len(DATASETS) else None
            writer.writerow([name, *[("" if v is None else f"{v:.2f}") for v in values],
                             "" if average is None else f"{average:.2f}", f"{row.get('seconds', 0):.0f}"])


def print_table(results: Dict[str, dict]) -> None:
    header = f"{'method':26s}" + "".join(f"{task:>9s}" for task in DATASETS) + f"{'average':>9s}"
    print(header)
    print("-" * len(header))
    for name, row in results.items():
        scores = row["scores"]
        cells = "".join(f"{scores[task]:9.2f}" if task in scores else f"{'':9s}" for task in DATASETS)
        average = f"{sum(scores[t] for t in DATASETS) / len(DATASETS):9.2f}" if all(t in scores for t in DATASETS) else ""
        print(f"{name:26s}{cells}{average}")


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the DBpedia/Amazon/Yahoo/AG News experts on the GPU and evaluate.")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), help="t5 or llama: sets the default base model and model type.")
    parser.add_argument("--base", help="Base model path or Hugging Face id (default from --arch).")
    parser.add_argument("--model-type", choices=["seq2seq", "causal"], help="Default: from --arch or the base config.")
    parser.add_argument("--experts-root", help="Folder with <task>/<W_DBPedia|W_Amazon|W_Yahoo|W_AGNews> (project layout).")
    parser.add_argument("--expert", action="append", default=[], metavar="TASK=PATH", help="Expert checkpoint for one task (overrides --experts-root).")
    parser.add_argument("--data-dir", help="O-LoRA CL_Benchmark folder.")
    parser.add_argument("--output", help="Output folder (default: aa_gpu_results/<arch or model name>).")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42, help="Subsampling and Fisher sampling seed.")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cuda:N; 'cpu' only for small tests.")
    parser.add_argument("--train-samples", type=int, default=None, help="Cap per task for the Gram matrices (default: whole training set).")
    parser.add_argument("--eval-samples", type=int, default=None, help="Test examples per task (default: all).")
    parser.add_argument("--eval-split", default="test", choices=["test", "dev"])
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for Grams, Fisher and the sequential pass.")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--skip-reference", action="store_true", help="Do not evaluate the base model and the experts.")
    parser.add_argument("--experts-all-tasks", action="store_true", help="Evaluate every expert on all four tasks (default: own task).")
    parser.add_argument("--save-merged", action="store_true", help="Save each merged model (save_pretrained) under <output>/merged/<method>.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute methods already in results.json.")
    parser.add_argument("--force-stats", action="store_true", help="Recompute Grams and Fisher.")
    parser.add_argument("--no-cache", action="store_true", help="Do not cache Procrustes and fitted rotations on disk.")
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls during evaluation only (faster generation); merging stays fp32.")
    parser.add_argument("--self-check", action="store_true", help="Formula checks on small tensors, then exit.")
    parser.add_argument("--profile", action="store_true",
                        help="Time one merge and skip evaluation. Does not change the merge math.")
    parser.add_argument("--fast", action="store_true",
                        help="Shortcut for --aa-solver accel --aa-iters 30 --aap-iters 30 --eval-dtype bf16 --tf32 "
                             "(about 3x faster rotation search, faster evaluation; AA_Fast then equals ActivationAwareInput).")
    # merge-rule options (same names and defaults as run_alternative_merges_fixed.py)
    parser.add_argument("--procrustes", choices=["minimal", "svd"], default="minimal",
                        help="Expert rotations of wide layers (out < in): the one closest to I (default) or plain SVD.")
    parser.add_argument("--procrustes-rank", type=int, default=None,
                        help="Positive rank k. None (default) keeps every existing expert rotation. "
                             "AAPlus-TSVD-k already fixes its own k. Passing this together with a "
                             "named variant requires the same k, and it is rejected for OrthoMerge, "
                             "RegMean, and ActivationAwarePlus.")
    parser.add_argument("--regmean-ridge", type=float, default=1e-4, help="Fraction of the mean Gram diagonal.")
    parser.add_argument("--regmean-alpha", type=float, default=1.0, help="Shrink Gram off-diagonals (published RegMean).")
    parser.add_argument("--weighted-residual", action="store_true", help="WeightedQ: weight the residual like Q.")
    parser.add_argument("--aa-iters", type=int, default=100)
    parser.add_argument("--aa-tol", type=float, default=1e-6)
    parser.add_argument("--aa-alpha", type=float, default=0.5)
    parser.add_argument("--aa-solver", choices=["safe", "accel"], default="safe")
    parser.add_argument("--aa-polar", choices=["ns", "auto", "svd"], default="ns",
                        help="Closest rotation inside the rotation search: Newton-Schulz (default, matmuls only, SVD fallback), SVD, or auto (SVD up to 1024).")
    parser.add_argument("--aa-anchor", type=float, default=0.1)
    parser.add_argument("--aa-rank", type=int, default=64)
    parser.add_argument("--aa-ties-keep", type=float, default=0.2)
    parser.add_argument("--aap-alpha", type=float, default=0.3)
    parser.add_argument("--aap-iters", type=int, default=50)
    parser.add_argument("--seq-max-samples", type=int, default=512)
    parser.add_argument("--fisher-max-samples", type=int, default=512)
    return parser.parse_args()


def expert_paths_from(args) -> Dict[str, str]:
    paths = {}
    if args.experts_root:
        for task in DATASETS:
            paths[task] = str(Path(args.experts_root) / task / EXPERT_DIR[task])
    for item in args.expert:
        task, _, path = item.partition("=")
        if task not in DATASETS or not path:
            raise SystemExit(f"--expert must be TASK=PATH with TASK in {DATASETS}: {item}")
        paths[task] = path
    missing = [task for task in DATASETS if task not in paths]
    if missing:
        raise SystemExit(f"No expert for {missing}: give --experts-root or --expert TASK=PATH")
    for task, path in paths.items():
        if not Path(path).exists():
            raise SystemExit(f"Expert for {task} not found: {path}")
    return paths


def main() -> None:
    args = parse_args()
    PROFILE.enabled = bool(args.profile)
    if args.fast:
        args.aa_solver, args.aa_iters, args.aap_iters, args.eval_dtype, args.tf32 = "accel", 30, 30, "bf16", True
    OPTIONS.update({
        "weighted_residual": args.weighted_residual, "regmean_alpha": args.regmean_alpha,
        "aa_iters": args.aa_iters, "aa_tol": args.aa_tol, "aa_alpha": args.aa_alpha, "aa_solver": args.aa_solver,
        "aa_polar": args.aa_polar, "aa_anchor": args.aa_anchor, "aa_rank": args.aa_rank, "aa_ties_keep": args.aa_ties_keep,
        "aap_alpha": args.aap_alpha, "aap_iters": args.aap_iters, "seq_max_samples": args.seq_max_samples,
        "fisher_max_samples": args.fisher_max_samples, "batch_size": args.batch_size,
    })
    PROCRUSTES["mode"] = args.procrustes
    if args.procrustes_rank is not None and int(args.procrustes_rank) < 1:
        raise SystemExit("--procrustes-rank must be a positive integer")
    PROCRUSTES["rank"] = None if args.procrustes_rank is None else int(args.procrustes_rank)
    if args.self_check:
        self_check()
        procrustes_check()
        return
    if not args.data_dir:
        raise SystemExit("--data-dir (the CL_Benchmark folder) is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script is meant for a GPU (use --device cpu only for small tests).")
    torch.backends.cuda.matmul.allow_tf32 = False  # merging and statistics: full fp32
    torch.backends.cudnn.allow_tf32 = False
    RUN["tf32_eval"] = bool(args.tf32)
    arch = ARCHITECTURES.get(args.arch or "", {})
    base_path = args.base or arch.get("base")
    if not base_path:
        raise SystemExit("Give --arch or --base")
    paths = expert_paths_from(args)
    model_type = args.model_type or arch.get("model_type") or infer_model_type(paths[DATASETS[0]])
    out = Path(args.output or Path("aa_gpu_results") / (args.arch or Path(base_path).name))
    out.mkdir(parents=True, exist_ok=True)
    if args.profile:
        PROFILE.dump_path = out / "profile.json"
    methods = [method for method in METHODS if method in args.methods]
    if PROCRUSTES["rank"] is not None:
        protected = [method for method in methods if not str(method).startswith("AAPlus-TSVD-")]
        if protected:
            raise SystemExit(
                "--procrustes-rank does not change OrthoMerge, RegMean, or ActivationAwarePlus. "
                f"Remove it, or run only AAPlus-TSVD-k. Got {protected}"
            )
        mismatched = [method for method in methods if int(method.rpartition("-")[-1]) != PROCRUSTES["rank"]]
        if mismatched:
            raise SystemExit(
                f"--procrustes-rank {PROCRUSTES['rank']} does not match {mismatched}. "
                "Each AAPlus-TSVD-k uses the rank in its name."
            )
    log(f"model_type={model_type} base={base_path}")
    for task in DATASETS:
        log(f"expert {task}: {paths[task]}")

    # ---- data
    tokenizer = load_tokenizer([paths[DATASETS[0]], base_path], model_type)
    train = {task: subsample(load_instances(Path(args.data_dir), task, "train"), args.train_samples, args.seed) for task in DATASETS}
    test = {task: subsample(load_instances(Path(args.data_dir), task, args.eval_split), args.eval_samples, args.seed) for task in DATASETS}
    for task in DATASETS:
        log(f"data {task}: train {len(train[task])}, {args.eval_split} {len(test[task])}")

    # ---- checkpoints (CPU, fp32)
    log("Loading the experts and the base model")
    expert_model = None
    task_states = {}
    for task in DATASETS:
        expert_model = load_model(paths[task], model_type)
        task_states[task] = cpu_state(expert_model)
        if task != DATASETS[-1]:
            del expert_model
            gc.collect()
    for task in DATASETS[1:]:
        if task_states[task].keys() != task_states[DATASETS[0]].keys():
            raise SystemExit(f"Expert {task} has different parameters from {DATASETS[0]}")
    base_model = load_model(base_path, model_type)
    w0 = cpu_state(base_model)
    del base_model
    gc.collect()
    align_base_rows(w0, task_states[DATASETS[0]])
    w0 = {key: w0[key] for key in task_states[DATASETS[0]]}
    tied = bool(getattr(expert_model.config, "tie_word_embeddings", False))
    linear_names = {f"{name}.weight": name for name, module in expert_model.named_modules() if isinstance(module, nn.Linear)}

    # ---- GPU workers: one fp32 model for statistics, one for the merged model (sequential), one for evaluation
    worker = expert_model.to(device).eval()
    for parameter in worker.parameters():
        parameter.requires_grad_(False)
    eval_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.eval_dtype]
    eval_model = load_model(paths[DATASETS[0]], model_type).to(device=device, dtype=eval_dtype).eval()
    collator = UIECollator(tokenizer, worker, model_type, args.max_source_length, args.max_target_length)
    eval_collator = UIECollator(tokenizer, eval_model, model_type, args.max_source_length, args.max_target_length)
    eval_bs = args.eval_batch_size or arch.get("eval_batch_size", 32)
    ctx = {
        "model_type": model_type, "device": device, "ridge": args.regmean_ridge, "w0": w0, "task_states": task_states,
        "linear_names": linear_names, "tied": tied, "worker": worker, "eval_model": eval_model, "tokenizer": tokenizer,
        "max_new_tokens": args.max_new_tokens,
        "test_loaders": {task: make_loader(test[task], eval_collator, eval_bs) for task in DATASETS},
    }

    # ---- caches (keyed by the checkpoints and the statistics settings)
    stats_key = hashlib.md5(json.dumps({
        "checkpoints": checkpoint_fingerprint([base_path] + [paths[t] for t in DATASETS]),
        "train_samples": args.train_samples, "seed": args.seed, "src": args.max_source_length, "tgt": args.max_target_length,
        "fisher_max_samples": args.fisher_max_samples, "procrustes": args.procrustes,
    }, sort_keys=True).encode()).hexdigest()[:12]
    stats_root = out / "stats" / stats_key
    store = StatStore(stats_root / "grams")
    global CACHE
    CACHE = TensorCache(None if args.no_cache else stats_root / "rotation_cache")
    stats_root.mkdir(parents=True, exist_ok=True)
    with open(stats_root / "settings.json", "w") as handle:
        json.dump({"base": base_path, "experts": paths, "train_samples": args.train_samples, "seed": args.seed,
                   "max_source_length": args.max_source_length, "max_target_length": args.max_target_length,
                   "fisher_max_samples": args.fisher_max_samples}, handle, indent=1)

    signatures = [fit_signature(aa_settings(m)) for m in methods if m in AA_METHODS and not aa_settings(m)["seq"]]
    SHARED_FITS.update(sig for sig in set(signatures) if signatures.count(sig) > 1)
    results_path = out / "results.json"
    results = read_results(results_path)

    # ---- reference rows
    if not args.skip_reference:
        if "Base" not in results or args.force:
            started = time.time()
            results = save_rows(out, {"Base": {"scores": score_state(w0, ctx), "seconds": time.time() - started}})
            log(f"Base {results['Base']['scores']}")
        if "Experts (own task)" not in results or args.force:
            started = time.time()
            own = {}
            for task in DATASETS:
                if args.experts_all_tasks:
                    full = score_state(task_states[task], ctx)
                    results = save_rows(out, {f"Expert {task}": {"scores": full, "seconds": 0}})
                    own[task] = full[task]
                else:
                    own[task] = score_state(task_states[task], ctx, tasks=[task])[task]
            results = save_rows(out, {"Experts (own task)": {"scores": own, "seconds": time.time() - started}})
        own = results["Experts (own task)"]["scores"]
        log(f"Experts (own task) {own}")

    # ---- statistics
    todo = [method for method in methods if args.force or method not in results]
    need_grams = any(method in GRAM_METHODS for method in todo)
    need_fisher = any(method in FISHER_METHODS for method in todo)
    need_seq = any(method in SEQ_METHODS for method in todo)
    if need_grams:
        for task in DATASETS:
            if store.complete(task) and not args.force_stats:
                log(f"Reusing Grams {task}")
                continue
            collect_grams(task, task_states[task], worker, make_loader(train[task], collator, args.batch_size),
                          model_type, device, store)
        ctx["tokens"] = {task: store.tokens(task) for task in DATASETS}
    ctx["store"] = store
    ctx["fisher"] = None
    if need_fisher:
        fisher_path = stats_root / "fisher.pt"
        if fisher_path.exists() and not args.force_stats:
            ctx["fisher"] = torch.load(fisher_path, map_location="cpu")
            log("Reusing Fisher")
        else:
            ctx["fisher"] = {}
            for task in DATASETS:
                started = time.time()
                subset = subsample(train[task], args.fisher_max_samples, args.seed)
                ctx["fisher"][task] = collect_fisher(task, task_states[task], worker, make_loader(subset, collator, args.batch_size),
                                                     model_type, device, args.seed)
                log(f"Fisher {task}: {len(ctx['fisher'][task])} layers, {time.time() - started:.0f}s")
            torch.save(ctx["fisher"], fisher_path)
    if need_seq:
        ctx["seq_loaders"] = {task: make_loader(subsample(train[task], args.seq_max_samples, args.seed), collator, args.batch_size)
                              for task in DATASETS}
        ctx["merged_worker"] = load_model(paths[DATASETS[0]], model_type).to(device).eval()
        for parameter in ctx["merged_worker"].parameters():
            parameter.requires_grad_(False)

    # ---- methods
    for method in methods:
        results = read_results(results_path)  # another process may have finished it meanwhile
        if method in results and not args.force:
            log(f"{method}: already in results.json")
            continue
        started = time.time()
        log(f"=== {method}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fisher_ctx = ctx["fisher"] if method in FISHER_METHODS else None
        method_ctx = dict(ctx, fisher=fisher_ctx)
        merged = merge_model(method, method_ctx)
        if method in SEQ_METHODS:
            sequential_refine(method, merged, method_ctx)
        merge_seconds = time.time() - started
        peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        if args.profile:
            PROFILE.shape = ""
            PROFILE.add_time("merge_wall", merge_seconds)
            PROFILE.dump()
            log(f"{method} profile merge {merge_seconds:.0f}s, peak {peak_bytes / (1024 ** 2):.0f} MiB")
            log(f"Profile: {PROFILE.dump_path}")
            del merged
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        record = out / "predictions" / method if args.save_predictions else None
        scores = score_state(merged, ctx, record=record)
        row = {"scores": scores, "seconds": time.time() - started,
               "merge_seconds": merge_seconds, "peak_gpu_bytes": peak_bytes}
        if method in AA_METHODS and aa_settings(method).get("skip_fit_inputs") is not None:
            row["rotation_fit_skipped"] = int(SKIPPED_FIT["count"])
        results = save_rows(out, {method: row})
        log(f"{method} {scores} (merge {merge_seconds:.0f}s, total {time.time() - started:.0f}s, "
            f"peak {peak_bytes / (1024 ** 2):.0f} MiB)")
        if args.save_merged:
            dest = out / "merged" / method
            worker.load_state_dict(merged, strict=True)  # fp32 copy; the worker is reloaded before every later use
            worker.save_pretrained(dest)
            tokenizer.save_pretrained(dest)
            log(f"Saved {dest}")
        del merged
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print()
    print_table(results)
    log(f"Results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
