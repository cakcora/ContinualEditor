#!/usr/bin/env python3
"""Conflict-Budgeted RegMean: sequential, activation-aware expert merging.

For a PyTorch Linear weight (out_features, in_features), the task error is
    E_i(W) = tr[(W - W_i) H_i (W - W_i)^T],
    H_i = (G_i / tokens_i) / max(tr[W_i (G_i / tokens_i) W_i^T], eps).

At arrival t, minimize
    E_t(W) + lambda * ||W - M_(t-1)||_F^2
subject to
    E_i(W) <= E_i(M_(t-1)) + budget,   for each i < t.

With nonnegative dual multipliers mu_i, the inner solve is
    W = [W_t H_t + sum_i mu_i W_i H_i + lambda M_(t-1)]
        [H_t + sum_i mu_i H_i + lambda I]^{-1}.

Metrics are rescaled together inside the solve for better conditioning.
An L-BFGS-B dual solve adjusts the protection weights. A feasible segment
fallback guarantees budgets are respected within numerical tolerance even
when dual optimization reaches its iteration limit. The constraints concern
activation-weighted layer outputs, not actual downstream task accuracy.

Cached Grams and expert weights are read without retraining. Existing RegMean
files and checkpoint directories remain unchanged.

Run:
    python codes/run_conflict_budgeted_regmean.py --self-test
    python codes/run_conflict_budgeted_regmean.py --architecture t5 --seed 42 --evaluate
    python codes/run_conflict_budgeted_regmean.py --architecture llama --seed 42 \
        --order agnews yahoo amazon dbpedia --budget 0.001 --maxiter 30 --evaluate

Uses scipy, numpy, torch, and the repository's existing merge/evaluation code.
A CUDA GPU is needed for realistic model-sized experiments.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

DATASETS = ("dbpedia", "amazon", "yahoo", "agnews")
EPS = 1e-12


@dataclass
class LayerReport:
    status: str
    iterations: int
    feasible: bool
    max_violation: float
    new_error: float
    old_errors: List[float]
    old_limits: List[float]
    multipliers: List[float]
    conflicting_tasks: int
    step_fraction: float


def _metric(expert: torch.Tensor, gram: torch.Tensor, tokens: int) -> torch.Tensor:
    """Dimensionless task covariance. Token normalization is explicit."""
    if tokens <= 0:
        raise ValueError(f"Expected positive token count, got {tokens}")
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram must be square")
    if expert.ndim != 2 or expert.shape[-1] != gram.shape[0]:
        raise ValueError("Expert and Gram shapes do not agree")
    g = 0.5 * (gram + gram.T) / float(tokens)
    power = float(torch.sum((expert @ g) * expert).item())
    if not math.isfinite(power) or power <= EPS:
        raise ValueError("Expert output energy is zero or invalid; cannot normalize task error")
    return g / power


def _error(weight: torch.Tensor, expert: torch.Tensor, metric: torch.Tensor) -> float:
    delta = weight - expert
    value = float(torch.sum((delta @ metric) * delta).item())
    if not math.isfinite(value):
        raise ArithmeticError("Non-finite Gram-weighted task error")
    return max(0.0, value)


def _feasible_fraction(
    previous: torch.Tensor,
    proposal: torch.Tensor,
    old_experts: Sequence[torch.Tensor],
    old_metrics: Sequence[torch.Tensor],
    limits: Sequence[float],
) -> Tuple[torch.Tensor, float]:
    """Largest feasible fraction of a candidate update along one line segment.

    Each constraint is a convex quadratic in the step fraction. Previous
    weight is always feasible because every limit includes a nonnegative budget.
    """
    direction = proposal - previous
    fraction = 1.0
    for expert, metric, cap in zip(old_experts, old_metrics, limits):
        difference = previous - expert
        directional_metric = direction @ metric
        a = max(0.0, float(torch.sum(directional_metric * direction).item()))
        b = 2.0 * float(torch.sum(directional_metric * difference).item())
        slack = max(0.0, cap - _error(previous, expert, metric))
        if a <= EPS:
            bound = slack / b if b > EPS else 1.0
        else:
            discriminant = math.sqrt(max(0.0, b * b + 4.0 * a * slack))
            if b >= 0.0:
                bound = (2.0 * slack / (b + discriminant)) if b + discriminant > 0.0 else 0.0
            else:
                bound = (-b + discriminant) / (2.0 * a)
        fraction = min(fraction, max(0.0, bound))
    fraction = min(1.0, fraction)
    if fraction < 1.0:
        fraction *= 1.0 - 1e-5
    answer = previous + fraction * direction
    # Guard against roundoff or slightly non-PSD cached float32 Grams.
    if any(_error(answer, expert, metric) > cap + 1e-7 for expert, metric, cap in zip(old_experts, old_metrics, limits)):
        lo, hi = 0.0, fraction
        for _ in range(30):
            midpoint = (lo + hi) / 2.0
            trial = previous + midpoint * direction
            if all(_error(trial, expert, metric) <= cap + 1e-7 for expert, metric, cap in zip(old_experts, old_metrics, limits)):
                lo = midpoint
            else:
                hi = midpoint
        fraction = lo
        answer = previous + fraction * direction
    return answer, fraction


@torch.no_grad()
def conflict_budgeted_layer(
    previous: torch.Tensor,
    new_expert: torch.Tensor,
    new_gram: torch.Tensor,
    old_experts: Sequence[torch.Tensor],
    old_grams: Sequence[torch.Tensor],
    *,
    new_tokens: int = 1,
    old_tokens: Optional[Sequence[int]] = None,
    budget: float = 1e-3,
    ridge: float = 1e-4,
    maxiter: int = 30,
    solver: str = "dual",
    feasibility_tol: float = 1e-6,
) -> Tuple[torch.Tensor, LayerReport]:
    """Merge one Linear matrix while bounding old-task output regression.

    Pass all weights and Grams on the same device. Float32 is used by the
    full-checkpoint runner; float64 is supported for numerical checks.
    """
    if len(old_experts) != len(old_grams):
        raise ValueError("old_experts and old_grams differ in length")
    if old_tokens is None:
        old_tokens = [1] * len(old_experts)
    if len(old_tokens) != len(old_experts):
        raise ValueError("old_tokens differs in length")
    if budget < 0 or ridge <= 0 or maxiter < 0 or solver not in {"dual", "line"}:
        raise ValueError("Require budget >= 0, ridge > 0, maxiter >= 0, solver in {dual,line}")
    if previous.shape != new_expert.shape or any(x.shape != previous.shape for x in old_experts):
        raise ValueError("All Linear weights must have identical shapes")
    if any(x.device != previous.device for x in (new_expert, new_gram, *old_experts, *old_grams)):
        raise ValueError("All tensors must be on the same device")
    dtype = torch.float64 if previous.dtype == torch.float64 else torch.float32
    previous = previous.to(dtype)
    new_expert = new_expert.to(dtype)
    new_gram = new_gram.to(dtype)
    old_experts = [value.to(dtype) for value in old_experts]
    old_grams = [value.to(dtype) for value in old_grams]
    if not old_experts:
        return new_expert.contiguous(), LayerReport(
            "first_task", 0, True, 0.0, 0.0, [], [], [], 0, 1.0
        )

    new_metric = _metric(new_expert, new_gram, new_tokens)
    old_metrics = [
        _metric(expert, gram, tokens)
        for expert, gram, tokens in zip(old_experts, old_grams, old_tokens)
    ]
    baseline = [_error(previous, expert, metric) for expert, metric in zip(old_experts, old_metrics)]
    limits = [value + float(budget) for value in baseline]

    # Conflicts: opposite task-loss gradients at the previous solution.
    new_gradient = (previous - new_expert) @ new_metric
    conflicts = sum(
        float(torch.sum(new_gradient * ((previous - expert) @ metric)).item()) < 0.0
        for expert, metric in zip(old_experts, old_metrics)
    )

    # All Gram metrics are scaled by the same positive constant. This does
    # not alter the argmin. It makes the average diagonal of the new metric 1.
    scale = float(torch.trace(new_metric).item()) / new_metric.shape[0]
    if not math.isfinite(scale) or scale <= EPS:
        raise ValueError("New task metric has zero or invalid trace")
    new_scaled = new_metric / scale
    old_scaled = [metric / scale for metric in old_metrics]
    new_mixed = new_expert @ new_scaled
    old_mixed = [expert @ metric for expert, metric in zip(old_experts, old_scaled)]
    eye = torch.eye(new_scaled.shape[0], dtype=dtype, device=previous.device)
    fixed_system = new_scaled + float(ridge) * eye
    fixed_rhs = new_mixed + float(ridge) * previous

    def solve(mu: Sequence[float]) -> torch.Tensor:
        system = fixed_system.clone()
        rhs = fixed_rhs.clone()
        for coefficient, metric, mixed in zip(mu, old_scaled, old_mixed):
            if coefficient > 0:
                system.add_(metric, alpha=float(coefficient))
                rhs.add_(mixed, alpha=float(coefficient))
        return torch.linalg.solve(system, rhs.T).T.contiguous()

    def check(weight: torch.Tensor) -> Tuple[List[float], float, bool]:
        errors = [_error(weight, expert, metric) for expert, metric in zip(old_experts, old_metrics)]
        violation = max([0.0] + [error - cap for error, cap in zip(errors, limits)])
        return errors, violation, violation <= feasibility_tol

    proposal = solve([0.0] * len(old_experts))
    old_errors, violation, feasible = check(proposal)
    if feasible:
        return proposal, LayerReport(
            "unconstrained", 0, True, violation,
            _error(proposal, new_expert, new_metric), old_errors,
            limits, [0.0] * len(old_experts), conflicts, 1.0
        )

    multipliers = [0.0] * len(old_experts)
    iterations = 0
    status = "line_fallback"
    if solver == "dual" and maxiter > 0:
        try:
            from scipy.optimize import minimize

            def negative_dual(mu: np.ndarray) -> Tuple[float, np.ndarray]:
                candidate = solve(mu)
                errors = [
                    _error(candidate, expert, metric)
                    for expert, metric in zip(old_experts, old_metrics)
                ]
                # Envelope theorem: partial dual derivative = E_i(W(mu)) - cap_i.
                lagrangian = (
                    _error(candidate, new_expert, new_metric)
                    + float(ridge) * scale * float(torch.sum((candidate - previous) ** 2).item())
                    + sum(float(x) * (e - cap) for x, e, cap in zip(mu, errors, limits))
                )
                gradient = np.array([cap - e for e, cap in zip(errors, limits)], dtype=np.float64)
                return -lagrangian, gradient

            result = minimize(
                negative_dual,
                np.zeros(len(old_experts), dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, 1e6)] * len(old_experts),
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9},
            )
            iterations = int(result.nit)
            multipliers = [float(x) for x in result.x]
            proposal = solve(multipliers)
            old_errors, violation, feasible = check(proposal)
            if feasible:
                status = "dual_feasible"
        except (ArithmeticError, RuntimeError, ValueError, ImportError, np.linalg.LinAlgError):
            # The fallback is always defined because previous is feasible.
            multipliers = [0.0] * len(old_experts)
            proposal = solve(multipliers)

    fraction = 1.0
    if status != "dual_feasible":
        proposal, fraction = _feasible_fraction(previous, proposal, old_experts, old_metrics, limits)
        old_errors, violation, feasible = check(proposal)
        if not feasible:
            proposal = previous.clone()
            fraction = 0.0
            old_errors, violation, feasible = check(proposal)
        status = "line_fallback" if solver == "dual" else "line_only"
    return proposal, LayerReport(
        status, iterations, feasible, violation,
        _error(proposal, new_expert, new_metric), old_errors,
        limits, multipliers, conflicts, fraction,
    )


def self_test() -> None:
    """Tests normalization, feasibility, the first task, and both solvers."""
    torch.manual_seed(23)
    dtype = torch.float64
    eye = torch.eye(3, dtype=dtype)
    prior = torch.tensor([[1.0, -0.5, 2.0], [1.0, 0.1, -0.3]], dtype=dtype)
    fresh = torch.tensor([[3.0, 2.0, -1.0], [-2.0, 1.5, 0.5]], dtype=dtype)
    same, first = conflict_budgeted_layer(prior, fresh, eye, [], [])
    assert first.status == "first_task" and torch.equal(same, fresh)
    for method in ("dual", "line"):
        merged, report = conflict_budgeted_layer(
            prior, fresh, eye, [prior], [eye], budget=1e-3,
            old_tokens=[1], maxiter=40, solver=method,
        )
        metric = _metric(prior, eye, 1)
        assert report.feasible and _error(merged, prior, metric) <= 1e-3 + 1e-6
        assert torch.linalg.vector_norm(merged - prior) > 0
        # Multiplied Grams, with the token count changed accordingly, are identical.
        rescaled, _ = conflict_budgeted_layer(
            prior, fresh, 100 * eye, [prior], [100 * eye],
            old_tokens=[100], new_tokens=100, budget=1e-3, solver=method, maxiter=40,
        )
        assert torch.allclose(merged, rescaled, atol=1e-7, rtol=1e-7)
    old_two = torch.tensor([[0.0, 1.0, 0.5], [-1.0, 0.5, 0.5]], dtype=dtype)
    average = (prior + old_two) / 2
    grams = [eye, torch.diag(torch.tensor([4.0, 1.0, 2.0], dtype=dtype))]
    for method in ("dual", "line"):
        merged, report = conflict_budgeted_layer(
            average, fresh, eye, [prior, old_two], grams,
            budget=5e-4, solver=method, maxiter=50,
        )
        assert report.feasible and report.max_violation <= 1e-6
        assert all(e <= cap + 1e-6 for e, cap in zip(report.old_errors, report.old_limits))
    try:
        conflict_budgeted_layer(prior, fresh, eye, [prior], [eye], budget=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative budget must fail")
    print("CONFLICT_BUDGETED_REGMEAN_TESTS_OK")


def run_experiment(args: argparse.Namespace) -> None:
    """Use the existing full-finetune experts, cached Grams, and evaluators."""
    import run_merge
    from run_merge import EMBED_KEY, LM_HEAD_KEY, align_vocab_rows, refresh_tied_lm_head
    from run_regmean_variants import architecture, expert_paths
    from run_alternative_merges import (
        evaluate_checkpoint,
        has_full_checkpoint,
        linear_parameter_names,
        load_architecture,
        load_cpu_state,
        parameter_kind,
        save_merged_model,
    )

    order = list(args.order)
    if not order or len(set(order)) != len(order) or set(order) - set(DATASETS):
        raise ValueError("Order must list unique tasks from: " + ", ".join(DATASETS))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    spec = architecture(args.architecture, args.seed)
    out = (
        spec["output"] / "ConflictBudgetedRegMean"
        / ("_".join(order) + f"_budget{args.budget:g}_ridge{args.ridge:g}")
    ).resolve()
    allowed = (Path(__file__).resolve().parent / "results" / "regmean_variants").resolve()
    if allowed not in out.parents:
        raise RuntimeError("Refusing to write outside results/regmean_variants")
    run_merge.configure_run(args.seed, out, spec["learning_rate"], spec["model"], spec["model_type"])
    refresh_tied_lm_head()
    paths = expert_paths(spec)
    for task in order:
        if not has_full_checkpoint(paths[task]):
            raise FileNotFoundError(f"Missing expert checkpoint: {paths[task]}")
    payloads = {}
    for task in order:
        path = spec["grams"] / f"{task}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Gram cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("mask_mismatches", -1) != 0:
            raise ValueError(f"Invalid mask_mismatches in {path}")
        if not payload.get("tokens"):
            raise ValueError(f"Gram cache lacks token counts: {path}")
        payloads[task] = payload
    base_state = load_cpu_state(run_merge.W0)
    expert_states = {task: load_cpu_state(paths[task]) for task in order}
    model = load_architecture(paths[order[0]], torch.float32)
    linear_names = linear_parameter_names(model)
    del model
    gc.collect()
    merged_previous = None
    config = {
        "architecture": args.architecture,
        "seed": args.seed,
        "order": order,
        "budget": args.budget,
        "ridge": args.ridge,
        "maxiter": args.maxiter,
        "solver": args.solver,
        "model": str(spec["model"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "conflict_budgeted_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing output uses different settings; choose another output path")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    for index, task in enumerate(order):
        start = time.perf_counter()
        new_state = {}
        reports: List[LayerReport] = []
        gram_layers = 0
        for name, base in base_state.items():
            if spec["model_type"] == "causal" and name == LM_HEAD_KEY and run_merge.TIED_LM_HEAD:
                new_state[name] = base.clone()
                continue
            _, task_rows, prefix = align_vocab_rows(
                name, base, [expert_states[t][name] for t in order[: index + 1]]
            )
            current = task_rows[-1]
            if index == 0:
                result = current.float()
            else:
                assert merged_previous is not None
                previous = merged_previous[name]
                if prefix is not None:
                    previous = previous[:prefix]
                if parameter_kind(name, current, linear_names) == "gram_linear":
                    for previous_task in order[: index + 1]:
                        if name not in payloads[previous_task]["grams"]:
                            raise KeyError(f"No Gram for {name} in task {previous_task}")
                        if name not in payloads[previous_task]["tokens"]:
                            raise KeyError(f"No token count for {name} in task {previous_task}")
                    result_gpu, report = conflict_budgeted_layer(
                        previous.detach().float().to(device),
                        current.detach().float().to(device),
                        payloads[task]["grams"][name].to(device),
                        [row.detach().float().to(device) for row in task_rows[:-1]],
                        [payloads[t]["grams"][name].to(device) for t in order[:index]],
                        new_tokens=int(payloads[task]["tokens"][name]),
                        old_tokens=[int(payloads[t]["tokens"][name]) for t in order[:index]],
                        budget=args.budget,
                        ridge=args.ridge,
                        maxiter=args.maxiter,
                        solver=args.solver,
                    )
                    result = result_gpu.cpu()
                    reports.append(report)
                    gram_layers += 1
                    del result_gpu
                else:
                    # Non-Linear parameters have no input Gram; sequential
                    # arithmetic mean matches the batch mean for these keys.
                    result = previous.float() + (current.float() - previous.float()) / (index + 1)
            if prefix is not None:
                full = base.clone()
                full[:prefix] = result.to(dtype=full.dtype)
                result = full
            new_state[name] = result.to(dtype=base.dtype).cpu().contiguous()

        if spec["model_type"] == "causal" and run_merge.TIED_LM_HEAD:
            new_state[LM_HEAD_KEY] = new_state[EMBED_KEY].clone()
        elif LM_HEAD_KEY in new_state and LM_HEAD_KEY in payloads[task]["grams"]:
            tied = new_state[LM_HEAD_KEY]
            for key in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
                if key in new_state and tuple(new_state[key].shape) == tuple(tied.shape):
                    new_state[key] = tied.clone()

        target = out / f"step_{index+1}_{task}" / "W_MERGED_FINAL"
        if not has_full_checkpoint(target):
            save_merged_model(spec, new_state, target, "ConflictBudgetedRegMean", args.ridge,
                              LM_HEAD_KEY in payloads[task]["grams"], [])
        stats = {
            "step": index + 1,
            "added_task": task,
            "seen_tasks": order[: index + 1],
            "seconds": time.perf_counter() - start,
            "gram_layers": gram_layers,
            "status_counts": {key: sum(report.status == key for report in reports)
                              for key in ("unconstrained", "dual_feasible", "line_fallback", "line_only")},
            "max_violation": max([0.0] + [r.max_violation for r in reports]),
            "conflicting_task_layer_pairs": sum(r.conflicting_tasks for r in reports),
        }
        if any(not report.feasible for report in reports):
            raise RuntimeError(f"Preservation budgets were violated at step {index+1}")
        if args.evaluate:
            stats["scores"] = evaluate_checkpoint(
                target, order[: index + 1], target.parent / "eval", force=args.force_eval
            )
        stats_path = target.parent / "conflict_budgeted_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2), flush=True)
        merged_previous = new_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("COMPLETE", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only numerical checks")
    parser.add_argument("--architecture", choices=("t5", "llama"), default="t5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", nargs="+", default=list(DATASETS))
    parser.add_argument("--budget", type=float, default=1e-3,
                        help="Maximum increase of each old task's relative layer-output error")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--solver", choices=("dual", "line"), default="dual")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
