#!/usr/bin/env python3
"""RegMean variants for the existing T5 1e-4 and Llama 5e-6 experts.

Does not train. Grams are reused. Q/K/V/O drift is H0 sliced Wasserstein
between each expert and the pretrained base, cached under results/regmean_variants.
Existing RegMean checkpoints are only read.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import multiprocessing
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import run_merge
from regmean_variants import (
    DATASETS,
    RIDGE,
    agreement_alphas,
    anchor_regmean,
    cpu_checks,
    layer_epsilon,
    normalize_alpha,
    percentile_ranks,
    t5_attention_key,
    topo_raw_high,
    topo_raw_low,
    weighted_regmean,
)
from run_alternative_merges import (
    EXPERT_DIR,
    LLAMA_MODEL,
    T5_MODEL,
    evaluate_checkpoint,
    has_full_checkpoint,
    linear_parameter_names,
    load_architecture,
    matrix_batch_width,
    load_cpu_state,
    log,
    parameter_kind,
    regmean_weight_batch,
    save_merged_model,
)
from run_merge import LM_HEAD_KEY, align_vocab_rows, refresh_tied_lm_head

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HOME", "/data/kadir/hf-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
TDA_CODE = Path("/home/kadir/topo/numpy_weights/t5-continual-learning/codes")
OUT = ROOT / "results" / "regmean_variants"
SEEDS = (42, 123, 2026)
METHODS = (
    "RegMean",
    "AnchorRegMean",
    "TopoRegMean-High",
    "TopoRegMean-Low",
    "TopoRegMean-Anchor-g0.5",
    "TopoRegMean-Anchor-g1.0",
    "TopoRegMean-Anchor-g2.0",
    "AgreementRegMean",
)
TDA_WORKERS = 8
VERIFY_TOL = 1e-3


def attention_parameter(name: str) -> bool:
    if t5_attention_key(name) is not None:
        return True
    if ".self_attn." not in name or not name.endswith(".weight"):
        return False
    projection = name.split(".self_attn.")[1].split(".")[0]
    return projection in {"q_proj", "k_proj", "v_proj", "o_proj"}


def _sw_job(base_path: str, fine_path: str) -> float:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if str(TDA_CODE) not in sys.path:
        sys.path.insert(0, str(TDA_CODE))
    from train import compute_sliced_wasserstein, compute_vr_persistence

    base = np.load(base_path).astype(np.float32)
    fine = np.load(fine_path).astype(np.float32)
    base_h0 = compute_vr_persistence(base, maxdim=0)[0]
    fine_h0 = compute_vr_persistence(fine, maxdim=0)[0]
    return float(compute_sliced_wasserstein(base_h0, fine_h0))


def read_scores(path: Path) -> Dict[str, float]:
    rows: Dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["parameter"]] = float(row["sliced_wasserstein_h0"])
    return rows


def write_scores(path: Path, scores: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "sliced_wasserstein_h0"])
        writer.writeheader()
        for name in sorted(scores):
            writer.writerow({"parameter": name, "sliced_wasserstein_h0": scores[name]})


def compute_task_drift(base: Dict[str, torch.Tensor], expert: Dict[str, torch.Tensor], dest: Path) -> Dict[str, float]:
    if dest.is_file():
        scores = read_scores(dest)
        log(f"Reusing TDA {dest} layers={len(scores)}")
        return scores
    names = [name for name in base if attention_parameter(name)]
    missing = [name for name in names if name not in expert]
    if missing:
        raise SystemExit(f"Expert is missing Q/K/V/O weights: {missing[:4]}")
    work = dest.parent / "_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    jobs = []
    for name in names:
        safe = name.replace("/", "_")
        base_path = work / f"{safe}.base.npy"
        fine_path = work / f"{safe}.fine.npy"
        np.save(base_path, base[name].detach().float().numpy())
        np.save(fine_path, expert[name].detach().float().numpy())
        jobs.append((name, str(base_path), str(fine_path)))
    scores: Dict[str, float] = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=TDA_WORKERS, mp_context=ctx) as pool:
        futures = {pool.submit(_sw_job, base_path, fine_path): name for name, base_path, fine_path in jobs}
        done = 0
        for future in as_completed(futures):
            name = futures[future]
            scores[name] = future.result()
            done += 1
            if done % 32 == 0 or done == len(jobs):
                log(f"  TDA {done}/{len(jobs)} {dest.parent.name}")
    if len(scores) != len(names):
        raise SystemExit(f"TDA produced {len(scores)} scores for {len(names)} Q/K/V/O matrices")
    write_scores(dest, scores)
    shutil.rmtree(work)
    return scores


def architecture(name: str, seed: int) -> dict:
    if name == "t5":
        experts = ROOT / "results" / ("independent" if seed == 42 else f"seeds/seed_{seed}/independent")
        return {
            "name": "t5",
            "label": "t5-large",
            "model": T5_MODEL,
            "model_type": "seq2seq",
            "learning_rate": 1e-4,
            "seed": seed,
            "experts_root": experts,
            "grams": ROOT / "results" / "alternative_merges" / "t5-large" / f"seed_{seed}" / "grams",
            "official_model": (
                ROOT / "results" / "alternative_merges" / "t5-large" / f"seed_{seed}" / "RegMean" / "W_MERGED_FINAL"
            ),
            "official_scores": (
                ROOT / "results" / "alternative_merges" / "t5-large" / f"seed_{seed}" / "eval" / "RegMean" / "eval_scores.json"
            ),
            "output": OUT / "t5-large" / "lr_1e-4" / f"seed_{seed}",
        }
    if name == "llama":
        base = ROOT / "results" / "llama-3.2-1b" / "lr_5e-6" / f"seed_{seed}"
        official = ROOT / "results" / "alternative_merges" / "llama-3.2-1b" / "lr_5e-6" / f"seed_{seed}"
        return {
            "name": "llama",
            "label": "llama-3.2-1b",
            "model": LLAMA_MODEL,
            "model_type": "causal",
            "learning_rate": 5e-6,
            "seed": seed,
            "experts_root": base / "independent",
            "grams": base / "regmean_grams",
            "official_model": official / "RegMean" / "W_MERGED_FINAL",
            "official_scores": official / "official_eval" / "eval_scores.json",
            "output": OUT / "llama-3.2-1b" / "lr_5e-6" / f"seed_{seed}",
        }
    raise SystemExit(f"Unknown architecture {name}")


def expert_paths(spec: dict) -> Dict[str, Path]:
    return {task: spec["experts_root"] / task / EXPERT_DIR[task] for task in DATASETS}


def load_grams(spec: dict) -> Dict[str, Dict[str, torch.Tensor]]:
    loaded = {}
    for task in DATASETS:
        path = spec["grams"] / f"{task}.pt"
        if not path.is_file():
            raise SystemExit(f"Missing Gram cache {path}")
        payload = torch.load(path, map_location="cpu")
        if int(payload.get("mask_mismatches", -1)) != 0:
            raise SystemExit(f"Gram file {path} has mask_mismatches={payload.get('mask_mismatches')}")
        loaded[task] = payload["grams"]
        log(f"Gram {task}: layers={len(loaded[task])} {path}")
    return loaded


def drift_for_seed(spec: dict) -> Dict[str, Dict[str, float]]:
    paths = expert_paths(spec)
    for task, path in paths.items():
        if not has_full_checkpoint(path):
            raise SystemExit(f"Missing expert {path}")
    log(f"Loading base weights for TDA {spec['label']} seed={spec['seed']}")
    base = load_cpu_state(Path(spec["model"]) if Path(spec["model"]).exists() else run_merge.W0)
    by_task = {}
    for task in DATASETS:
        dest = spec["output"] / "tda" / task / "sw_scores.csv"
        log(f"TDA {spec['label']} seed={spec['seed']} {task}")
        expert = load_cpu_state(paths[task])
        by_task[task] = compute_task_drift(base, expert, dest)
        del expert
        gc.collect()
    del base
    gc.collect()
    return by_task


def gamma_of(method: str) -> Optional[float]:
    if not method.startswith("TopoRegMean-Anchor-g"):
        return None
    return float(method.rsplit("g", 1)[1])


def normalized_drift(drift: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    means = {}
    names = set.intersection(*(set(drift[task]) for task in DATASETS))
    for name in names:
        if not attention_parameter(name):
            continue
        means[name] = sum(drift[task][name] for task in DATASETS) / len(DATASETS)
    return percentile_ranks(means)


def alphas_for(
    method: str,
    name: str,
    w0: torch.Tensor,
    task_rows: Sequence[torch.Tensor],
    drift: Dict[str, Dict[str, float]],
    ranks: Dict[str, float],
) -> Tuple[str, object]:
    """Return ('regmean'|'weighted'|'anchor', payload). Missing TDA uses ordinary RegMean or scale 1."""
    if method == "RegMean":
        return "regmean", None
    if method == "AgreementRegMean":
        deltas = [expert.float() - w0.float() for expert in task_rows]
        return "weighted", agreement_alphas(deltas)
    if method == "AnchorRegMean":
        return "anchor", 1.0
    gamma = gamma_of(method)
    if gamma is not None:
        return "anchor", 1.0 + gamma * float(ranks.get(name, 0.0))
    distances = [drift[task].get(name) for task in DATASETS]
    if not attention_parameter(name) or any(distance is None for distance in distances):
        return "regmean", None
    values = [float(distance) for distance in distances]
    epsilon = layer_epsilon(values)
    raw = topo_raw_high(values, epsilon) if method == "TopoRegMean-High" else topo_raw_low(values, epsilon)
    return "weighted", normalize_alpha(raw)


def weighted_regmean_batch(
    experts: torch.Tensor,
    grams: torch.Tensor,
    alphas: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    scaled = grams * alphas[:, :, None, None]
    total_gram = scaled.sum(dim=1)
    total_mixed = torch.matmul(experts, scaled).sum(dim=1)
    unweighted = grams.sum(dim=1)
    mean_diag = torch.diagonal(unweighted, dim1=-2, dim2=-1).mean(dim=-1)
    penalty = float(ridge) * torch.clamp(mean_diag, min=1e-12)
    eye = torch.eye(total_gram.shape[-1], device=total_gram.device, dtype=total_gram.dtype)
    system = total_gram + penalty[:, None, None] * eye
    solved = torch.linalg.solve(system, total_mixed.transpose(-1, -2))
    return solved.transpose(-1, -2).contiguous()


def anchor_regmean_batch(
    w0: torch.Tensor,
    experts: torch.Tensor,
    grams: torch.Tensor,
    scales: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    deltas = experts - w0.unsqueeze(1)
    total_gram = grams.sum(dim=1)
    total_mixed = torch.matmul(deltas, grams).sum(dim=1)
    mean_diag = torch.diagonal(total_gram, dim1=-2, dim2=-1).mean(dim=-1)
    penalty = float(ridge) * torch.clamp(mean_diag, min=1e-12) * scales
    eye = torch.eye(total_gram.shape[-1], device=total_gram.device, dtype=total_gram.dtype)
    system = total_gram + penalty[:, None, None] * eye
    solved = torch.linalg.solve(system, total_mixed.transpose(-1, -2))
    return (w0 + solved.transpose(-1, -2)).contiguous()


def batch_matches_scalar() -> None:
    torch.manual_seed(1)
    experts = torch.randn(2, 4, 3, 2)
    raw = torch.randn(2, 4, 2, 2)
    grams = torch.matmul(raw.transpose(-1, -2), raw) + torch.eye(2)
    alphas = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.5, 1.5, 0.5, 1.5]])
    w0 = torch.randn(2, 3, 2)
    batched = weighted_regmean_batch(experts, grams, alphas, RIDGE)
    anchored = anchor_regmean_batch(w0, experts, grams, torch.tensor([1.0, 2.0]), RIDGE)
    for index in range(2):
        scalar = weighted_regmean(experts[index], grams[index], alphas[index].tolist(), RIDGE)
        if not torch.allclose(batched[index], scalar, atol=1e-5):
            raise SystemExit("Batched weighted RegMean does not match the scalar formula")
        scalar_anchor = anchor_regmean(w0[index], experts[index], grams[index], RIDGE, penalty_scale=float(index + 1))
        if not torch.allclose(anchored[index], scalar_anchor, atol=1e-5):
            raise SystemExit("Batched AnchorRegMean does not match the scalar formula")


def merge_variant(spec: dict, method: str, grams: Dict[str, Dict[str, torch.Tensor]], drift: Dict[str, Dict[str, float]]) -> dict:
    dest = spec["output"] / method / "W_MERGED_FINAL"
    if OUT.resolve() not in dest.resolve().parents:
        raise SystemExit(f"Refusing to write outside {OUT}: {dest}")
    stats_path = dest.parent / "run_stats.json"
    if has_full_checkpoint(dest) and stats_path.is_file():
        log(f"Reusing merge {dest}")
        return json.loads(stats_path.read_text())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    paths = expert_paths(spec)
    w0 = load_cpu_state(run_merge.W0)
    task_states = {task: load_cpu_state(path) for task, path in paths.items()}
    linear_model = load_architecture(paths["dbpedia"], torch.float32)
    linear_names = linear_parameter_names(linear_model)
    del linear_model
    gc.collect()
    ranks = normalized_drift(drift)
    merged_state = {key: value.clone() for key, value in w0.items()}
    lm_head_used_gram = False
    counts = {"regmean": 0, "weighted": 0, "anchor": 0, "mean": 0}
    batched: Dict[Tuple[str, int, int], List[dict]] = {}

    def store(item: dict, merged: torch.Tensor) -> None:
        value = merged.detach().cpu()
        if item["vocab_prefix"] is not None:
            full = item["source"].detach().float().clone()
            full[: item["vocab_prefix"]] = value
            value = full
        merged_state[item["name"]] = value.to(dtype=item["source"].dtype)

    def flush(group: List[dict]) -> None:
        if not group:
            return
        route = group[0]["route"]
        experts = torch.stack(
            [torch.stack([row.detach().float() for row in item["task_rows"]], dim=0) for item in group]
        ).to(device)
        gram_tensor = torch.stack(
            [torch.stack([gram.detach().float() for gram in item["grams"]], dim=0) for item in group]
        ).to(device)
        if route == "regmean":
            merged_batch = regmean_weight_batch(experts, gram_tensor, RIDGE)
        elif route == "weighted":
            alpha = torch.tensor([item["payload"] for item in group], dtype=torch.float32, device=device)
            merged_batch = weighted_regmean_batch(experts, gram_tensor, alpha, RIDGE)
        else:
            base = torch.stack([item["w0"].detach().float() for item in group]).to(device)
            scales = torch.tensor([item["payload"] for item in group], dtype=torch.float32, device=device)
            merged_batch = anchor_regmean_batch(base, experts, gram_tensor, scales, RIDGE)
        for row, item in enumerate(group):
            store(item, merged_batch[row])
        del experts, gram_tensor, merged_batch
        torch.cuda.empty_cache()

    for name in w0:
        if run_merge.MODEL_TYPE == "causal" and name == LM_HEAD_KEY and run_merge.TIED_LM_HEAD:
            continue
        w0_rows, task_rows, vocab_prefix = align_vocab_rows(
            name, w0[name], [task_states[task][name] for task in DATASETS]
        )
        kind = parameter_kind(name, w0_rows, linear_names)
        item = {"name": name, "source": w0[name], "vocab_prefix": vocab_prefix, "w0": w0_rows, "task_rows": task_rows}
        if kind == "gram_linear":
            if any(name not in grams[task] for task in DATASETS):
                raise SystemExit(f"Missing Gram for {name}")
            route, payload = alphas_for(method, name, w0_rows, task_rows, drift, ranks)
            item.update({"route": route, "payload": payload, "grams": [grams[task][name] for task in DATASETS]})
            counts[route] += 1
            if name == LM_HEAD_KEY:
                lm_head_used_gram = True
            key = (route, int(w0_rows.shape[0]), int(w0_rows.shape[1]))
            batched.setdefault(key, []).append(item)
        else:
            counts["mean"] += 1
            merged = torch.stack([row.float() for row in task_rows]).mean(dim=0)
            store({**item, "vocab_prefix": vocab_prefix}, merged)
    for group in batched.values():
        width = matrix_batch_width(int(group[0]["w0"].shape[-1]))
        for start in range(0, len(group), width):
            flush(group[start : start + width])
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    save_merged_model(spec, merged_state, dest, method, RIDGE, lm_head_used_gram, [])
    stats = {
        "method": method,
        "seconds": elapsed,
        "peak_bytes": peak,
        "counts": counts,
        "checkpoint": str(dest),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    del w0, task_states, merged_state
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Merged {method} in {elapsed:.1f}s peak={peak / (1024 ** 2):.0f} MiB counts={counts}")
    return stats


def max_diff(left: Path, right: Path) -> float:
    a = load_cpu_state(left)
    b = load_cpu_state(right)
    if a.keys() != b.keys():
        raise SystemExit("RegMean verification state dict keys differ from the official checkpoint")
    worst = 0.0
    for name in a:
        if not a[name].dtype.is_floating_point:
            continue
        worst = max(worst, float((a[name].float() - b[name].float()).abs().max()))
    del a, b
    gc.collect()
    return worst


def score_row(scores: Dict[str, float]) -> dict:
    values = [float(scores[task]) for task in DATASETS]
    row = {task: values[index] for index, task in enumerate(DATASETS)}
    row["average"] = sum(values) / len(values)
    return row


def run_one(spec: dict, grams: Dict[str, Dict[str, torch.Tensor]], drift: Dict[str, Dict[str, float]]) -> List[dict]:
    rows = []
    official = json.loads(spec["official_scores"].read_text())["scores"]
    for method in METHODS:
        log(f"START {spec['label']} seed={spec['seed']} {method}")
        stats = merge_variant(spec, method, grams, drift)
        if method == "RegMean":
            diff = max_diff(Path(stats["checkpoint"]), spec["official_model"])
            log(f"RegMean max abs diff vs official checkpoint = {diff:.3e}")
            if diff > VERIFY_TOL:
                raise SystemExit(f"RegMean verification failed for seed {spec['seed']}: max abs diff {diff}")
            stats["max_abs_diff"] = diff
        eval_dir = spec["output"] / method / "official_eval"
        scores = evaluate_checkpoint(Path(stats["checkpoint"]), DATASETS, eval_dir, force=False)
        scored = score_row(scores)
        row = {
            "architecture": spec["label"],
            "learning_rate": spec["learning_rate"],
            "seed": spec["seed"],
            "method": method,
            **scored,
            "merge_seconds": stats["seconds"],
            "peak_mib": stats["peak_bytes"] / (1024 ** 2),
            "official_average": sum(float(official[task]) for task in DATASETS) / len(DATASETS),
        }
        for task in DATASETS:
            row[f"delta_{task}"] = scored[task] - float(official[task])
        rows.append(row)
        log(
            f"SCORES {spec['label']} seed={spec['seed']} {method} "
            + " ".join(f"{task}={scored[task]:.4f}" for task in DATASETS)
            + f" avg={scored['average']:.4f}"
        )
    regmean_avg = next(row["average"] for row in rows if row["method"] == "RegMean")
    for row in rows:
        row["delta_vs_regmean"] = row["average"] - regmean_avg
    return rows


def write_table(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[dict]) -> List[dict]:
    summary = []
    architectures = []
    for row in rows:
        if row["architecture"] not in architectures:
            architectures.append(row["architecture"])
    for architecture in architectures:
        for method in METHODS:
            chosen = [row for row in rows if row["architecture"] == architecture and row["method"] == method]
            if len(chosen) != len(SEEDS):
                continue
            record = {"architecture": architecture, "method": method, "n": len(chosen)}
            for key in ("dbpedia", "amazon", "yahoo", "agnews", "average", "merge_seconds", "peak_mib", "delta_vs_regmean"):
                values = [float(row[key]) for row in chosen]
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(record)
    return summary


def main() -> None:
    cpu_checks()
    batch_matches_scalar()
    log("Formula checks passed")
    rows: List[dict] = []
    partial = OUT / "per_seed.csv"
    for name in ("t5", "llama"):
        for seed in SEEDS:
            spec = architecture(name, seed)
            run_merge.configure_run(
                seed, spec["output"], spec["learning_rate"], spec["model"], spec["model_type"]
            )
            refresh_tied_lm_head()
            for path in expert_paths(spec).values():
                if not has_full_checkpoint(path):
                    raise SystemExit(f"Missing expert {path}")
            if not has_full_checkpoint(spec["official_model"]):
                raise SystemExit(f"Missing official RegMean checkpoint {spec['official_model']}")
            drift = drift_for_seed(spec)
            grams = load_grams(spec)
            rows.extend(run_one(spec, grams, drift))
            write_table(partial, rows)
            write_table(OUT / "summary.csv", summarize(rows))
            del grams, drift
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_table(OUT / "summary.csv", summarize(rows))
    log(f"Finished. Summary -> {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
