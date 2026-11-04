#!/usr/bin/env python3
"""OrthoMerge on independently fine-tuned T5 models. No training.

Non-OFT pipeline is copied from the official Sphere-AI-Lab/OrthoMerge
implementation, specifically:

    merge/OrthoMerge_G_TA.py
        orthogonal_procrustes_torch_right
        cayley_to_skew
        cayley_from_skew
        merge_cayley_Q_list
        layerwise residual = expert - base @ R
        distributed_TA  (sum of residuals)
        final = (base @ R_merged) + (1/N) * sum(rho_i)

The official G_TA script hardcodes 0.2 * sum(rho) for their 5 Llama experts,
which is 1/N. We use 1/N for our 4 T5 experts.

Layer filter matches the official file:
    ndim == 2 and "weight" in name and "layernorm" not in name
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

ROOT = Path(__file__).resolve().parent
CL_CODES = Path("/home/kadir/topo/numpy_weights/t5-continual-learning/codes")
if str(CL_CODES) not in sys.path:
    sys.path.insert(0, str(CL_CODES))

import torch
import run_cl
import train as cl_train
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

RESULTS = ROOT / "results"
DATASETS = ("dbpedia", "amazon", "yahoo", "agnews")
INDEPENDENT_NAMES = {
    "dbpedia": "W_DBPedia",
    "amazon": "W_Amazon",
    "yahoo": "W_Yahoo",
    "agnews": "W_AGNews",
}
TASK_CATEGORY = {
    "dbpedia": "TC",
    "amazon": "SC",
    "yahoo": "TC",
    "agnews": "TC",
}
SEED = 42
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
MODEL_TYPE = "seq2seq"
W0 = Path(cl_train.INITIAL_MODEL)
EMBED_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"
# True only after refresh_tied_lm_head() reads the checkpoint. Llama-3.2-1B
# is tied, so the copy below stays. Llama-3.1-8B is not, so lm_head is merged.
TIED_LM_HEAD = True

def apply_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def configure_run(
    seed: int,
    results_root: Optional[Path],
    learning_rate: float = 1e-4,
    model: Optional[str] = None,
    model_type: Optional[str] = None,
) -> None:
    global SEED, RESULTS, W0, MODEL_TYPE
    run_cl.configure_run(seed, results_root, learning_rate, model, model_type)
    SEED = run_cl.SEED
    RESULTS = run_cl.RESULTS
    W0 = run_cl.W0
    MODEL_TYPE = run_cl.MODEL_TYPE
    apply_seed(SEED)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def has_full_checkpoint(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(
        (path / name).exists()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
    )


def single_task_config_dir(task: str) -> Path:
    dest = RESULTS / "_eval_configs" / task
    payload = {name: [] for name in ("SC", "TC", "NLI", "QQP", "WiC", "MultiRC", "COPA", "BoolQA")}
    payload[TASK_CATEGORY[task]] = [{"sampling strategy": "full", "dataset name": task}]
    empty = {name: [] for name in payload}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "train_tasks.json").write_text(json.dumps(payload, indent=4) + "\n")
    (dest / "test_tasks.json").write_text(json.dumps(payload, indent=4) + "\n")
    (dest / "dev_tasks.json").write_text(json.dumps(empty, indent=4) + "\n")
    return dest


def extract_accuracies(metrics: Dict[str, Any], tasks: Sequence[str]) -> Dict[str, Optional[float]]:
    return {
        task: float(metrics[f"predict_exact_match_for_{task}"]) if f"predict_exact_match_for_{task}" in metrics else None
        for task in tasks
    }


def is_orthomerge_weight(name: str, tensor: torch.Tensor) -> bool:
    """Official G_TA filter: 2D, name contains weight, not layernorm."""
    return tensor.ndim == 2 and "weight" in name and "layernorm" not in name


def align_vocab_rows(
    name: str,
    w0_tensor: torch.Tensor,
    task_tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[int]]:
    """Training calls resize_token_embeddings(len(tokenizer)), so saved
    embeddings are 32100 rows. Pretrained T5-Large stores 32128, and the
    extra 28 rows are not in the tokenizer. Merge the shared prefix and
    leave that tail equal to W0. Equations are unchanged on the prefix.
    """
    task_shapes = {tuple(tensor.shape) for tensor in task_tensors}
    if len(task_shapes) != 1:
        raise SystemExit(f"{name}: task shapes disagree: {sorted(task_shapes)}")
    task_shape = next(iter(task_shapes))
    if MODEL_TYPE == "causal" and task_shape != tuple(w0_tensor.shape):
        raise SystemExit(
            f"{name}: Llama shapes differ W0 {tuple(w0_tensor.shape)} vs {task_shape}. "
            "Refusing the T5 vocabulary crop."
        )
    if task_shape == tuple(w0_tensor.shape):
        return w0_tensor, task_tensors, None
    same_tail = tuple(w0_tensor.shape[1:]) == task_shape[1:]
    if w0_tensor.ndim == task_tensors[0].ndim and same_tail and w0_tensor.shape[0] > task_shape[0]:
        rows = task_shape[0]
        log(f"  {name}: merge rows [0:{rows}] of W0 {tuple(w0_tensor.shape)}; tail stays W0")
        return w0_tensor[:rows], task_tensors, rows
    raise SystemExit(f"{name}: cannot align W0 {tuple(w0_tensor.shape)} with {task_shape}")


# --- Official helpers from merge/OrthoMerge_G_TA.py (verbatim math) ---

def orthogonal_procrustes_torch_right(W1: torch.Tensor, W0: torch.Tensor) -> torch.Tensor:
    """Find R such that W0 @ R ≈ W1. Official OrthoMerge_G_TA.py."""
    gram = torch.matmul(W0.t(), W1)
    u, _s, vh = torch.linalg.svd(gram, full_matrices=False)
    return torch.matmul(u, vh)


def cayley_to_skew(rotation: torch.Tensor) -> torch.Tensor:
    """Official cayley_to_skew: solve(R+I, R-I), then skew-symmetrize."""
    eye = torch.eye(rotation.shape[-1], device=rotation.device, dtype=rotation.dtype)
    rotation_f = rotation.float()
    eye_f = eye.float()
    skew = torch.linalg.solve(rotation_f + eye_f, rotation_f - eye_f)
    skew = 0.5 * (skew - skew.transpose(-1, -2))
    return skew


def cayley_from_skew(skew: torch.Tensor) -> torch.Tensor:
    """Official cayley_from_skew: solve(I-A, I+A)."""
    eye = torch.eye(skew.shape[-1], device=skew.device, dtype=skew.dtype)
    skew_f = skew.float()
    eye_f = eye.float()
    return torch.linalg.solve(eye_f - skew_f, eye_f + skew_f)


def merge_cayley_Q_list(
    weights_list: List[torch.Tensor],
    theta_agg: str = "mean",
    direction_weight: str = "theta",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Official merge_cayley_Q_list. Returns Q_merged and the scalars used."""
    base_shape = weights_list[0].shape
    q_stack = torch.stack(weights_list, dim=0)
    n_tasks = q_stack.shape[0]
    q_flat = q_stack.reshape(n_tasks, -1)
    theta = torch.linalg.vector_norm(q_flat, dim=1)
    theta_clamped = torch.clamp(theta, min=1e-8)
    unit = q_flat / theta_clamped.unsqueeze(1)
    if direction_weight == "theta":
        weights = theta.clone()
    elif direction_weight == "uniform":
        weights = torch.ones_like(theta)
    else:
        raise ValueError(f"Unknown direction_weight: {direction_weight}")
    unit_sum = (unit * weights.unsqueeze(1)).sum(dim=0)
    unit_sum_norm = torch.linalg.vector_norm(unit_sum)
    if theta_agg == "mean":
        theta_merge = theta.mean()
    elif theta_agg == "median":
        theta_merge = theta.median()
    elif theta_agg == "max":
        theta_merge = theta.max()
    else:
        raise ValueError(f"Unknown theta_agg: {theta_agg}")
    if float(unit_sum_norm) < 1e-8:
        merged = torch.zeros(base_shape, device=q_flat.device, dtype=q_flat.dtype)
    else:
        unit_merge = unit_sum / unit_sum_norm
        merged = (unit_merge * theta_merge).reshape(base_shape)
    # The merged tensor does not depend on these labels. Four-task runs keep
    # the original theta_dbpedia/amazon/yahoo/agnews keys. A shorter list is
    # used only by task removal, which calls this same function.
    labels = ("dbpedia", "amazon", "yahoo", "agnews")
    stats = {
        "theta_merge": float(theta_merge),
        "direction_weight": direction_weight,
        "u_sum_norm": float(unit_sum_norm),
        "n_tasks": int(n_tasks),
    }
    for index in range(int(n_tasks)):
        label = labels[index] if int(n_tasks) == len(labels) else f"task{index}"
        stats[f"theta_{label}"] = float(theta[index])
    return merged, stats


def orthogonality_error(rotation: torch.Tensor) -> float:
    eye = torch.eye(rotation.shape[-1], device=rotation.device, dtype=rotation.dtype)
    gram = rotation.float().transpose(-1, -2) @ rotation.float()
    return float(torch.linalg.norm(gram - eye.float(), ord="fro"))


def skew_error(skew: torch.Tensor) -> float:
    return float(torch.linalg.norm(skew.float() + skew.float().transpose(-1, -2), ord="fro"))


def load_source(path: Path) -> str:
    if Path(path).exists():
        return str(path)
    return run_cl.model_source()


def load_architecture(path: Path, dtype: torch.dtype):
    source = load_source(path)
    if MODEL_TYPE == "causal":
        return AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype)
    return AutoModelForSeq2SeqLM.from_pretrained(source, torch_dtype=dtype)


def load_cpu_state(path: Path) -> Dict[str, torch.Tensor]:
    model = load_architecture(path, torch.float32)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    del model
    gc.collect()
    return state


def evaluate_checkpoint(model_path: Path, tasks: Sequence[str], output_dir: Path, force: bool) -> Dict[str, Optional[float]]:
    return run_cl.evaluate_checkpoint(model_path, tasks, output_dir, force)


def refresh_tied_lm_head() -> bool:
    """Read tie_word_embeddings from the checkpoint we are about to merge."""
    global TIED_LM_HEAD
    if MODEL_TYPE != "causal":
        TIED_LM_HEAD = False
        return False
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(load_source(W0))
    TIED_LM_HEAD = bool(getattr(config, "tie_word_embeddings", False))
    log(f"tie_word_embeddings={TIED_LM_HEAD} source={load_source(W0)}")
    return TIED_LM_HEAD


def merge_models(force: bool, discard_components: bool) -> Path:
    merge_dir = RESULTS / "merge"
    dest = merge_dir / "W_MERGED_FINAL"
    components_dir = merge_dir / "components"
    if not force and has_full_checkpoint(dest) and (merge_dir / "merge_checks.csv").exists():
        log(f"Merged model already exists -> {dest}")
        return dest

    missing = [
        task
        for task in DATASETS
        if not has_full_checkpoint(RESULTS / "independent" / task / INDEPENDENT_NAMES[task])
    ]
    if missing:
        raise SystemExit(f"Independent checkpoints missing: {missing}")

    log(f"Loading W0 {W0.resolve()} (no fine-tuning)")
    w0 = load_cpu_state(W0)
    task_states = {}
    for task in DATASETS:
        path = RESULTS / "independent" / task / INDEPENDENT_NAMES[task]
        log(f"Loading {task} {path.resolve()}")
        task_states[task] = load_cpu_state(path)

    n_tasks = len(DATASETS)
    residual_scale = 1.0 / n_tasks
    merged_state = {key: value.clone() for key, value in w0.items()}
    check_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    if not discard_components:
        components_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required so OrthoMerge SVDs are not run on CPU.")
    device = torch.device("cuda")
    log(f"State dicts stay on CPU; one matrix at a time is merged on {device}")

    names = list(w0.keys())
    for index, name in enumerate(names, start=1):
        if MODEL_TYPE == "causal" and name == LM_HEAD_KEY:
            log(f"  {name}: tied to {EMBED_KEY}; it is copied after the embedding merge")
            check_rows.append(
                {
                    "parameter_name": name,
                    "shape": "x".join(str(dim) for dim in w0[name].shape),
                    "orthogonality_error_task1": "",
                    "orthogonality_error_task2": "",
                    "orthogonality_error_task3": "",
                    "orthogonality_error_task4": "",
                    "skew_error_Q_task1": "",
                    "skew_error_Q_task2": "",
                    "skew_error_Q_task3": "",
                    "skew_error_Q_task4": "",
                    "orthogonality_error_merged_R": "",
                    "residual_scale": residual_scale,
                    "theta_merge": "",
                }
            )
            continue
        w0_rows, task_rows, vocab_prefix = align_vocab_rows(
            name,
            w0[name],
            [task_states[task][name] for task in DATASETS],
        )
        cpu_shape = tuple(w0_rows.shape)
        payload: Dict[str, Any] = {
            "parameter_name": name,
            "shape": list(cpu_shape),
            "orthomerge_2d": is_orthomerge_weight(name, w0_rows),
            "w0_tail_rows_kept": None if vocab_prefix is None else int(w0[name].shape[0] - vocab_prefix),
        }
        if is_orthomerge_weight(name, w0_rows):
            w0_t = w0_rows.detach().float().to(device)
            task_ts = [tensor.detach().float().to(device) for tensor in task_rows]
            rotations = []
            skews = []
            residuals = []
            ortho_errs = []
            skew_errs = []
            for task_t in task_ts:
                rotation = orthogonal_procrustes_torch_right(task_t, w0_t)
                skew = cayley_to_skew(rotation)
                residual = task_t - torch.matmul(w0_t, rotation)
                rotations.append(rotation)
                skews.append(skew)
                residuals.append(residual)
                ortho_errs.append(orthogonality_error(rotation))
                skew_errs.append(skew_error(skew))
            q_merged, q_stats = merge_cayley_Q_list(skews)
            r_merged = cayley_from_skew(q_merged)
            rho_merged = residual_scale * torch.stack(residuals, dim=0).sum(dim=0)
            merged = torch.matmul(w0_t, r_merged) + rho_merged
            merged_error = orthogonality_error(r_merged)
            payload.update(
                {
                    "R_dbpedia": rotations[0].detach().cpu(),
                    "R_amazon": rotations[1].detach().cpu(),
                    "R_yahoo": rotations[2].detach().cpu(),
                    "R_agnews": rotations[3].detach().cpu(),
                    "Q_dbpedia": skews[0].detach().cpu(),
                    "Q_amazon": skews[1].detach().cpu(),
                    "Q_yahoo": skews[2].detach().cpu(),
                    "Q_agnews": skews[3].detach().cpu(),
                    "rho_dbpedia": residuals[0].detach().cpu(),
                    "rho_amazon": residuals[1].detach().cpu(),
                    "rho_yahoo": residuals[2].detach().cpu(),
                    "rho_agnews": residuals[3].detach().cpu(),
                    "Q_merged": q_merged.detach().cpu(),
                    "R_merged": r_merged.detach().cpu(),
                    "rho_merged": rho_merged.detach().cpu(),
                    "residual_scale": residual_scale,
                    **q_stats,
                }
            )
            merged = merged.detach().cpu()
            del w0_t, task_ts, rotations, skews, residuals, q_merged, r_merged, rho_merged
            torch.cuda.empty_cache()
            if vocab_prefix is not None:
                full = w0[name].detach().float().clone()
                full[:vocab_prefix] = merged
                merged = full
            check_rows.append(
                {
                    "parameter_name": name,
                    "shape": "x".join(str(dim) for dim in cpu_shape),
                    "orthogonality_error_task1": ortho_errs[0],
                    "orthogonality_error_task2": ortho_errs[1],
                    "orthogonality_error_task3": ortho_errs[2],
                    "orthogonality_error_task4": ortho_errs[3],
                    "skew_error_Q_task1": skew_errs[0],
                    "skew_error_Q_task2": skew_errs[1],
                    "skew_error_Q_task3": skew_errs[2],
                    "skew_error_Q_task4": skew_errs[3],
                    "orthogonality_error_merged_R": merged_error,
                    "residual_scale": residual_scale,
                    "theta_merge": q_stats["theta_merge"],
                }
            )
        else:
            w0_t = w0_rows.float()
            task_ts = [tensor.float() for tensor in task_rows]
            residuals = [task_t - w0_t for task_t in task_ts]
            rho_merged = residual_scale * torch.stack(residuals, dim=0).sum(dim=0)
            merged = w0_t + rho_merged
            if vocab_prefix is not None:
                full = w0[name].detach().float().clone()
                full[:vocab_prefix] = merged
                merged = full
            payload.update(
                {
                    "rho_dbpedia": residuals[0],
                    "rho_amazon": residuals[1],
                    "rho_yahoo": residuals[2],
                    "rho_agnews": residuals[3],
                    "rho_merged": rho_merged,
                    "residual_scale": residual_scale,
                    "note": "Official G_TA leaves non-2D/LN tensors as base and adds TA residual.",
                }
            )
            check_rows.append(
                {
                    "parameter_name": name,
                    "shape": "x".join(str(dim) for dim in cpu_shape),
                    "orthogonality_error_task1": "",
                    "orthogonality_error_task2": "",
                    "orthogonality_error_task3": "",
                    "orthogonality_error_task4": "",
                    "skew_error_Q_task1": "",
                    "skew_error_Q_task2": "",
                    "skew_error_Q_task3": "",
                    "skew_error_Q_task4": "",
                    "orthogonality_error_merged_R": "",
                    "residual_scale": residual_scale,
                    "theta_merge": "",
                }
            )
        merged_state[name] = merged.to(dtype=w0[name].dtype)
        if not discard_components:
            safe = name.replace(".", "_")
            component_file = components_dir / f"{safe}.pt"
            torch.save(payload, component_file)
            index_rows.append(
                {
                    "parameter_name": name,
                    "shape": "x".join(str(dim) for dim in cpu_shape),
                    "component_file": str(component_file.resolve()),
                }
            )
        del payload
        if index % 25 == 0 or index == len(names):
            log(f"  merge {index}/{len(names)} {name}")

    if discard_components:
        log("Q/R/rho component tensors are not being written for this seed")
    else:
        write_csv(
            components_dir / "component_index.csv",
            ["parameter_name", "shape", "component_file"],
            index_rows,
        )
    write_csv(
        merge_dir / "merge_checks.csv",
        [
            "parameter_name",
            "shape",
            "orthogonality_error_task1",
            "orthogonality_error_task2",
            "orthogonality_error_task3",
            "orthogonality_error_task4",
            "skew_error_Q_task1",
            "skew_error_Q_task2",
            "skew_error_Q_task3",
            "skew_error_Q_task4",
            "orthogonality_error_merged_R",
            "residual_scale",
            "theta_merge",
        ],
        check_rows,
    )

    if MODEL_TYPE == "causal" and TIED_LM_HEAD:
        if EMBED_KEY not in merged_state:
            raise SystemExit(f"Llama merge is missing {EMBED_KEY}")
        merged_state[LM_HEAD_KEY] = merged_state[EMBED_KEY].detach().clone()
        log(f"Copied merged {EMBED_KEY} onto {LM_HEAD_KEY}")
    log(f"Loading {MODEL_TYPE} structure and writing W_MERGED_FINAL")
    model = load_architecture(W0, torch.float32)
    incompatible = model.load_state_dict(merged_state, strict=True)
    log(f"load_state_dict {incompatible}")
    if hasattr(model, "tie_weights"):
        model.tie_weights()
        log("Called model.tie_weights()")
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(load_source(W0), use_fast=True)
    if MODEL_TYPE == "causal" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(dest)
    if not has_full_checkpoint(dest):
        raise RuntimeError(f"W_MERGED_FINAL did not save a complete checkpoint at {dest}")
    reloaded = load_architecture(dest, torch.float32)
    if not hasattr(reloaded, "tie_weights"):
        raise RuntimeError(f"Reloaded merge at {dest} has no tie_weights()")
    reloaded.tie_weights()
    del reloaded
    gc.collect()
    log(f"Reloaded merged checkpoint successfully from {dest}")
    save_json(
        {
            "reference": "https://github.com/Sphere-AI-Lab/OrthoMerge/blob/main/merge/OrthoMerge_G_TA.py",
            "pipeline": "non-OFT Global Decoupling + Task Arithmetic residuals",
            "procrustes": "W0 @ R ≈ W_i, A = W0.T @ W_i, R = U @ Vh",
            "inverse_cayley": "solve(R+I, R-I) then 0.5*(A-A.T)",
            "cayley": "solve(I-A, I+A)",
            "q_merge": "merge_cayley_Q_list(theta_agg=mean, direction_weight=theta)",
            "residual": "rho_i = W_i - W0 @ R_i; rho_merged = (1/N) * sum(rho_i)",
            "reconstruction": "W = W0 @ R_merged + rho_merged",
            "residual_scale": residual_scale,
            "vocab_alignment": (
                "Llama embeddings are not cropped. lm_head is a copy of the merged embed_tokens."
                if MODEL_TYPE == "causal"
                else "Embeddings were resized to len(tokenizer)=32100 at training. OrthoMerge uses those rows. The remaining W0 rows are copied unchanged."
            ),
            "w0": str(W0.resolve()),
            "independent_inputs": {
                task: str((RESULTS / "independent" / task / INDEPENDENT_NAMES[task]).resolve())
                for task in DATASETS
            },
            "seed": SEED,
            "model": run_cl.MODEL_NAME,
            "model_type": MODEL_TYPE,
            "learning_rate": run_cl.LEARNING_RATE,
        },
        merge_dir / "merge_config.json",
    )
    del model, w0, task_states, merged_state
    gc.collect()
    log(f"Saved {dest}")
    return dest


def run_final_eval(force: bool) -> None:
    models = {
        "W_CL_O1": RESULTS / "cl" / "O1" / "W4_after_agnews",
        "W_CL_O2": RESULTS / "cl" / "O2" / "W4_after_yahoo",
        "W_CL_O3": RESULTS / "cl" / "O3" / "W4_after_dbpedia",
        "W_MERGED_FINAL": RESULTS / "merge" / "W_MERGED_FINAL",
    }
    missing = [name for name, path in models.items() if not has_full_checkpoint(path)]
    if missing:
        raise SystemExit(f"Cannot evaluate; missing checkpoints: {missing}")
    rows: List[Dict[str, Any]] = []
    for name, path in models.items():
        scores = evaluate_checkpoint(path, DATASETS, RESULTS / "final_eval" / name, force=force)
        values = [scores.get(task) for task in DATASETS]
        finite = [float(value) for value in values if value is not None]
        for task in DATASETS:
            rows.append({"model": name, "eval_dataset": task, "accuracy": scores.get(task, "")})
        rows.append(
            {
                "model": name,
                "eval_dataset": "average",
                "accuracy": (sum(finite) / len(finite)) if finite else "",
            }
        )
    write_csv(RESULTS / "final_comparison.csv", ["model", "eval_dataset", "accuracy"], rows)
    log(f"Wrote {RESULTS / 'final_comparison.csv'}")


def run_pretrained_baseline(force: bool) -> None:
    """One greedy eval of W0. The same scores are reused for every seed."""
    out = ROOT / "results" / "pretrained_baseline"
    scores = evaluate_checkpoint(W0, DATASETS, out, force=force)
    values = [scores.get(task) for task in DATASETS]
    finite = [float(value) for value in values if value is not None]
    rows = [{"model": "Pretrained", "eval_dataset": task, "accuracy": scores.get(task, "")} for task in DATASETS]
    rows.append({"model": "Pretrained", "eval_dataset": "average", "accuracy": (sum(finite) / len(finite)) if finite else ""})
    write_csv(ROOT / "results" / "pretrained_baseline.csv", ["model", "eval_dataset", "accuracy"], rows)
    log(f"Wrote {ROOT / 'results' / 'pretrained_baseline.csv'}")


def check_shapes() -> None:
    """Load W0 and print how OrthoMerge classifies the representative matrices."""
    log(f"Shape check {MODEL_TYPE} {load_source(W0)}")
    model = load_architecture(W0, torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    state = model.state_dict()
    markers = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head", "layernorm", "norm")
    print(f"{'parameter':<62} {'shape':<22} action")
    shown = 0
    for name, tensor in state.items():
        layer0 = ".layers.0." in name or name in {EMBED_KEY, LM_HEAD_KEY, "model.norm.weight"}
        if not layer0 or not any(marker in name for marker in markers):
            continue
        if MODEL_TYPE == "causal" and name == LM_HEAD_KEY and TIED_LM_HEAD:
            action = "copy merged embed_tokens; tie_weights()"
        elif is_orthomerge_weight(name, tensor):
            action = f"right Procrustes R ({tensor.shape[-1]}, {tensor.shape[-1]})"
        else:
            action = "non-2D / norm: base plus averaged residual"
        print(f"{name:<62} {str(tuple(tensor.shape)):<22} {action}")
        shown += 1
    embed = state[EMBED_KEY]
    head = state[LM_HEAD_KEY]
    print(f"embed_tokens {tuple(embed.shape)} lm_head {tuple(head.shape)} tied_config={model.config.tie_word_embeddings}")
    print(f"shown={shown}")
    if MODEL_TYPE == "causal":
        expected = {
            "model.layers.0.self_attn.q_proj.weight": (2048, 2048),
            "model.layers.0.self_attn.k_proj.weight": (512, 2048),
            "model.layers.0.self_attn.v_proj.weight": (512, 2048),
            "model.layers.0.self_attn.o_proj.weight": (2048, 2048),
            "model.layers.0.mlp.gate_proj.weight": (8192, 2048),
            "model.layers.0.mlp.up_proj.weight": (8192, 2048),
            "model.layers.0.mlp.down_proj.weight": (2048, 8192),
            EMBED_KEY: (128256, 2048),
        }
        for name, shape in expected.items():
            if name not in state or tuple(state[name].shape) != shape:
                raise SystemExit(f"Unexpected shape for {name}: {tuple(state[name].shape) if name in state else 'missing'} expected {shape}")
            if name != LM_HEAD_KEY and not is_orthomerge_weight(name, state[name]):
                raise SystemExit(f"{name} was not recognized as an OrthoMerge matrix")
        if is_orthomerge_weight("model.layers.0.input_layernorm.weight", state["model.layers.0.input_layernorm.weight"]):
            raise SystemExit("RMSNorm was classified as an OrthoMerge matrix")
    print("SHAPE_CHECK_OK")
    del model, state
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OrthoMerge non-OFT merge and final evaluation")
    parser.add_argument("--model", default=None, help="HF id or local path. Default: T5-Large W0.")
    parser.add_argument("--model-type", choices=["seq2seq", "causal"], default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--pretrained-only", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--check-shapes", action="store_true")
    parser.add_argument(
        "--keep-components",
        action="store_true",
        help="Write Q/R/rho dumps. New results roots skip them unless this is set.",
    )
    parser.add_argument(
        "--discard-components",
        action="store_true",
        help="Do not write Q/R/rho dumps. Refuses to run against the existing seed-42 results directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.discard_components and args.results_root is None and not args.check_shapes and not args.show_config:
        raise SystemExit("Refusing --discard-components without --results-root. The seed-42 component dump stays.")
    configure_run(args.seed, args.results_root, args.learning_rate, args.model, args.model_type)
    refresh_tied_lm_head()
    discard_components = bool(args.discard_components)
    if args.results_root is not None and not args.keep_components:
        discard_components = True
    if args.show_config:
        print(run_cl.describe_run())
        print(f"discard_components={discard_components}")
        return
    if args.check_shapes:
        check_shapes()
        return
    if args.pretrained_only:
        apply_seed(SEED)
        log(f"pretrained baseline only; eval seed={SEED} does not change greedy decoding")
        run_pretrained_baseline(force=args.force)
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    apply_seed(SEED)
    log(
        f"run_merge.py: zero fine-tuning seed={SEED} model_type={MODEL_TYPE} "
        f"results={RESULTS.resolve()} discard_components={discard_components}"
    )
    merge_models(force=args.force, discard_components=discard_components)
    if not args.skip_eval:
        run_final_eval(force=args.force)


if __name__ == "__main__":
    main()
