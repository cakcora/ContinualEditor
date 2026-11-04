#!/usr/bin/env python3
"""TDA-guided freezing for T5-Large Mono training.

One profiling fine-tune per task, then High1 through All.
High-k freezes the top k layers of each ranked projection inside the encoder,
inside the decoder, and inside cross-attention. All is k=24.
Every freezing run reloads the original pretrained checkpoint.
The H0 diagram and Sliced Wasserstein distance are the functions in
codes/train.py (compute_vr_persistence, compute_sliced_wasserstein,
extract_t5_weights).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CL_CODES = Path("/home/kadir/topo/numpy_weights/t5-continual-learning/codes")
if str(CL_CODES) not in sys.path:
    sys.path.insert(0, str(CL_CODES))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as cl_train
import run_cl

TASKS = ("dbpedia", "amazon", "yahoo", "agnews")
LAYER_DEPTH = cl_train.NUM_ENCODER_LAYERS
VARIANTS = tuple(f"high{k}" for k in range(1, LAYER_DEPTH)) + ("all",)
EXPERIMENT = ROOT / "results" / "tda_freeze" / "t5-large-per-block"
COMPONENTS = (
    ("enc_self", "enc", cl_train.NUM_ENCODER_LAYERS),
    ("dec_self", "dec", cl_train.NUM_DECODER_LAYERS),
    ("dec_cross", "cross", cl_train.NUM_DECODER_LAYERS),
)
PROJECTIONS = tuple(cl_train.FREEZE_PROJECTIONS)
TDA_WORKERS = 4
UPDATE_EPS = 1e-6
ACTIVE_LR = 2e-4
ACTIVE_PRIOR: Optional[Path] = None


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def status_path() -> Path:
    return EXPERIMENT / "status.json"


def update_status(key: str, state: str, detail: str = "") -> None:
    path = status_path()
    payload = read_json(path) if path.exists() else {}
    payload[key] = {"state": state, "detail": detail, "updated": datetime.now().isoformat(timespec="seconds")}
    save_json(path, payload)
    log(f"status {key} {state} {detail}")


def configure() -> Dict[str, Any]:
    run_cl.configure_run(run_cl.SEED, EXPERIMENT, ACTIVE_LR, str(cl_train.INITIAL_MODEL), "seq2seq")
    resolved = {
        "model": str(run_cl.W0),
        "model_type": run_cl.MODEL_TYPE,
        "learning_rate": run_cl.LEARNING_RATE,
        "batch_size": run_cl.BATCH_SIZE,
        "eval_batch_size": run_cl.EVAL_BATCH_SIZE,
        "gradient_accumulation": run_cl.GRAD_ACCUM,
        "effective_batch_size": run_cl.BATCH_SIZE * run_cl.GRAD_ACCUM,
        "epochs": run_cl.EPOCHS,
        "optimizer": "adamw_torch",
        "adam_beta1": run_cl.ADAM_BETA1,
        "adam_beta2": run_cl.ADAM_BETA2,
        "scheduler": run_cl.LR_SCHEDULER,
        "weight_decay": run_cl.WEIGHT_DECAY,
        "max_source_length": 512,
        "max_target_length": 50,
        "seed": run_cl.SEED,
        "data": str(cl_train.data_root_for_order(1)),
        "metric": "predict_exact_match",
        "update_epsilon": UPDATE_EPS,
        "tda": "train.compute_vr_persistence H0 + train.compute_sliced_wasserstein",
    }
    return resolved


def layer_key(component: str, layer_idx: int) -> str:
    return f"{component}:{int(layer_idx)}"


def parse_layer_key(key: str) -> Tuple[str, int]:
    component, idx = key.split(":")
    return component, int(idx)


def npy_name(component: str, layer_idx: int, proj: str) -> str:
    prefix = {"enc_self": "enc", "dec_self": "dec", "dec_cross": "cross"}[component]
    return f"{prefix}_layer{int(layer_idx)}_{proj}.npy"


def projection_weight(model, component: str, layer_idx: int, proj: str):
    if component == "enc_self":
        module = model.encoder.block[layer_idx].layer[0].SelfAttention
    elif component == "dec_self":
        module = model.decoder.block[layer_idx].layer[0].SelfAttention
    elif component == "dec_cross":
        module = model.decoder.block[layer_idx].layer[1].EncDecAttention
    else:
        raise KeyError(component)
    return getattr(module, proj).weight


def always_frozen_names(model) -> Dict[str, List[str]]:
    groups = {"ffn_wi": [], "ffn_wo": [], "embedding": []}
    for name, _param in model.named_parameters():
        if "DenseReluDense" in name and (".wi." in name or ".wi_" in name):
            groups["ffn_wi"].append(name)
        elif "DenseReluDense" in name and ".wo." in name:
            groups["ffn_wo"].append(name)
        elif name == "shared.weight" or name.endswith("embed_tokens.weight") or name == "lm_head.weight":
            groups["embedding"].append(name)
    return groups


def unique_parameters(model) -> List[Tuple[str, torch.nn.Parameter]]:
    seen = set()
    rows = []
    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        rows.append((name, param))
    return rows


def apply_freeze(model, frozen_qkvo: Sequence[Tuple[str, int, str]], always: Dict[str, List[str]]) -> None:
    run_cl.configure_full_finetune(model)
    for component, layer_idx, proj in frozen_qkvo:
        projection_weight(model, component, layer_idx, proj).requires_grad_(False)
    frozen_names = {name for names in always.values() for name in names}
    frozen_ptrs = set()
    for name, param in model.named_parameters():
        if name in frozen_names:
            param.requires_grad_(False)
            frozen_ptrs.add(param.data_ptr())
    for _name, param in model.named_parameters():
        if param.data_ptr() in frozen_ptrs:
            param.requires_grad_(False)


def parameter_report(model) -> Dict[str, float]:
    total = 0
    trainable = 0
    frozen = 0
    for _name, param in unique_parameters(model):
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        else:
            frozen += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": 100.0 * trainable / total,
        "frozen_parameters": frozen,
        "frozen_percent": 100.0 * frozen / total,
    }


def changed_report(model, before: Dict[int, torch.Tensor]) -> Dict[str, float]:
    total = 0
    changed = 0
    for _name, param in unique_parameters(model):
        total += int(param.numel())
        start = before[param.data_ptr()]
        diff = (param.detach().float().cpu() - start).abs()
        changed += int((diff > UPDATE_EPS).sum().item())
    return {
        "updated_parameters": changed,
        "updated_percent": 100.0 * changed / total,
        "update_epsilon": UPDATE_EPS,
    }


def snapshot(model) -> Dict[int, torch.Tensor]:
    return {param.data_ptr(): param.detach().float().cpu().clone() for _name, param in unique_parameters(model)}


def variant_limit(variant: str, depth: int = LAYER_DEPTH) -> Optional[int]:
    if variant == "all":
        return None
    if variant.startswith("high"):
        limit = int(variant[4:])
        if 1 <= limit <= depth:
            return limit
    raise KeyError(variant)


def ranked_rows(ranking: Dict[str, Any], family: str, variant: str) -> List[Dict[str, Any]]:
    """Top k layers inside each attention block, using that block's own order."""
    limit = variant_limit(variant)
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _prefix, _count in COMPONENTS}
    for row in ranking["rankings"][family]:
        component, _layer_idx = parse_layer_key(row["layer_id"])
        grouped[component].append(row)
    chosen = []
    for component, _prefix, count in COMPONENTS:
        group = grouped[component]
        if len(group) != count:
            raise RuntimeError(f"{family} {component} has {len(group)} layers, expected {count}")
        chosen.extend(group if limit is None else group[:limit])
    return chosen


def family_freeze_list(ranking: Dict[str, Any], variant: str) -> List[Tuple[str, int, str]]:
    chosen = []
    for family in ranking["low_drift_families"]:
        for row in ranking["rankings"][family]:
            component, layer_idx = parse_layer_key(row["layer_id"])
            chosen.append((component, layer_idx, family))
    for family in ranking["ranked_families"]:
        for row in ranked_rows(ranking, family, variant):
            component, layer_idx = parse_layer_key(row["layer_id"])
            chosen.append((component, layer_idx, family))
    return chosen


def tda_worker(task: Tuple[str, str, str]) -> Dict[str, Any]:
    baseline, finetuned, name = task
    w0 = np.load(baseline).astype(np.float32)
    w1 = np.load(finetuned).astype(np.float32)
    d0 = cl_train.compute_vr_persistence(w0, maxdim=0)[0]
    d1 = cl_train.compute_vr_persistence(w1, maxdim=0)[0]
    distance = cl_train.compute_sliced_wasserstein(d0, d1)
    return {"file": name, "distance": distance, "diagram_pretrained": d0, "diagram_finetuned": d1}


def compute_task_tda(pretrained_dir: Path, finetuned_dir: Path, diagram_dir: Path) -> List[Dict[str, Any]]:
    jobs = []
    for component, _prefix, count in COMPONENTS:
        for layer_idx in range(count):
            for proj in PROJECTIONS:
                name = npy_name(component, layer_idx, proj)
                jobs.append((str(pretrained_dir / name), str(finetuned_dir / name), name))
    diagram_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=TDA_WORKERS, mp_context=ctx) as pool:
        futures = [pool.submit(tda_worker, job) for job in jobs]
        done = 0
        for future in as_completed(futures):
            item = future.result()
            done += 1
            stem = Path(item["file"]).stem
            np.save(diagram_dir / f"{stem}_pretrained.npy", item["diagram_pretrained"])
            np.save(diagram_dir / f"{stem}_finetuned.npy", item["diagram_finetuned"])
            component = "enc_self" if stem.startswith("enc_") else "dec_self" if stem.startswith("dec_") else "dec_cross"
            layer_idx = int(stem.split("_layer")[1].split("_")[0])
            proj = stem.split("_")[-1]
            rows.append(
                {
                    "component": component,
                    "layer_idx": layer_idx,
                    "layer_id": layer_key(component, layer_idx),
                    "projection": proj,
                    "file": item["file"],
                    "sliced_wasserstein_h0": float(item["distance"]),
                }
            )
            if done % 48 == 0 or done == len(jobs):
                log(f"  TDA diagrams {done}/{len(jobs)}")
    rows.sort(key=lambda row: (row["projection"], row["component"], row["layer_idx"]))
    return rows


def build_ranking(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rankings = {}
    means = {}
    for proj in PROJECTIONS:
        group = [row for row in rows if row["projection"] == proj]
        group.sort(key=lambda row: (-row["sliced_wasserstein_h0"], row["layer_id"]))
        rankings[proj] = [
            {
                "rank": index + 1,
                "layer_id": row["layer_id"],
                "component": row["component"],
                "layer_idx": row["layer_idx"],
                "sliced_wasserstein_h0": row["sliced_wasserstein_h0"],
            }
            for index, row in enumerate(group)
        ]
        means[proj] = float(np.mean([row["sliced_wasserstein_h0"] for row in group]))
    ordered = sorted(PROJECTIONS, key=lambda proj: (means[proj], proj))
    return {
        "mean_sw_drift": means,
        "low_drift_families": list(ordered[:2]),
        "ranked_families": list(reversed(ordered[2:])),
        "rankings": rankings,
        "order": "highest sliced-Wasserstein H0 to lowest, independent per projection",
    }


def ranking_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    payload = read_json(path)
    if set(payload.get("low_drift_families", [])) | set(payload.get("ranked_families", [])) != set(PROJECTIONS):
        return False
    if len(payload["low_drift_families"]) != 2 or len(payload["ranked_families"]) != 2:
        return False
    expected = cl_train.NUM_ENCODER_LAYERS + 2 * cl_train.NUM_DECODER_LAYERS
    for proj in PROJECTIONS:
        rows = payload.get("rankings", {}).get(proj, [])
        if len(rows) != expected:
            return False
        drifts = [row["sliced_wasserstein_h0"] for row in rows]
        if drifts != sorted(drifts, reverse=True):
            return False
    return True


def result_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    payload = read_json(path)
    required = ("accuracy", "trainable_percent", "updated_percent", "train_seconds", "peak_gpu_allocated")
    return payload.get("status") == "completed" and all(key in payload for key in required)


def load_data(task: str, output_dir: Path):
    raw = cl_train.load_uie_datasets(
        data_dir=cl_train.data_root_for_order(1),
        task_config_dir=run_cl.single_task_config_dir(task),
        output_dir=output_dir / "dataset_cache",
        overwrite_cache=False,
    )
    if len(raw["test"]) == 0:
        raise RuntimeError(f"{task} loaded no official test rows")
    train_dataset, original, capped = cl_train.cap_train_dataset(
        raw["train"], max_train_samples=None, subsample_seed=run_cl.SEED, logger=None
    )
    if capped != original:
        raise RuntimeError(f"{task} training rows were capped: {capped}/{original}")
    return train_dataset, raw["test"], capped


def train_variant(task: str, variant: str, ranking: Optional[Dict[str, Any]], hyper: Dict[str, Any]) -> Dict[str, Any]:
    out = EXPERIMENT / task / variant
    result_path = out / "result.json"
    if result_is_valid(result_path):
        log(f"reuse {task} {variant}")
        return read_json(result_path)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / "model"
    run_cl.apply_seed(run_cl.SEED)
    if Path(run_cl.W0).resolve() != Path(cl_train.INITIAL_MODEL).resolve():
        raise RuntimeError(f"freezing run must start from the original pretrained model, got {run_cl.W0}")
    model, tokenizer = run_cl.load_run_model(str(run_cl.W0))
    init_names = {name for name, _param in model.named_parameters()}
    if "shared.weight" not in init_names:
        raise RuntimeError("pretrained T5 did not load shared.weight")
    always = always_frozen_names(model)
    frozen = []
    if variant != "profile":
        if ranking is None:
            raise RuntimeError("freezing run has no saved ranking")
        frozen = family_freeze_list(ranking, variant)
        apply_freeze(model, frozen, always)
    else:
        run_cl.configure_full_finetune(model)
    train_dataset, test_dataset, train_count = load_data(task, out)
    before = snapshot(model)
    trainer = run_cl.build_uie_trainer(model, tokenizer, train_dataset, test_dataset, out, do_train=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    peak_alloc = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
    counts = parameter_report(trainer.model)
    updated = changed_report(trainer.model, before)
    if variant == "profile":
        weight_dir = out / "finetuned_qkvo"
        saved = cl_train.extract_t5_weights(trainer.model, weight_dir)
        expected = (cl_train.NUM_ENCODER_LAYERS + 2 * cl_train.NUM_DECODER_LAYERS) * len(PROJECTIONS)
        if saved != expected:
            raise RuntimeError(f"extracted {saved} QKVO matrices, expected {expected}")
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(model_dir, safe_serialization=False)
    tokenizer.save_pretrained(model_dir)
    eval_started = time.perf_counter()
    prediction = trainer.predict(test_dataset, metric_key_prefix="predict", max_new_tokens=50, num_beams=1)
    eval_seconds = time.perf_counter() - eval_started
    scores = run_cl.extract_accuracies(dict(prediction.metrics), [task])
    accuracy = scores[task]
    if accuracy is None:
        raise RuntimeError(f"missing exact-match score for {task}")
    payload = {
        "status": "completed",
        "task": task,
        "variant": variant,
        "seed": hyper["seed"],
        "init_checkpoint": str(run_cl.W0),
        "accuracy": accuracy,
        "train_samples": train_count,
        "test_samples": len(test_dataset),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "peak_gpu_allocated": peak_alloc,
        "peak_gpu_reserved": peak_reserved,
        **counts,
        **updated,
        "low_drift_families": None if ranking is None else ranking["low_drift_families"],
        "ranked_families": None if ranking is None else ranking["ranked_families"],
        "ranked_layers_frozen": frozen_layer_ids(ranking, variant) if ranking else [],
        "always_frozen_families": ["ffn_wi", "ffn_wo", "shared_embedding_lm_head"] if variant != "profile" else [],
        "learning_rate": hyper["learning_rate"],
        "batch_size": hyper["batch_size"],
        "effective_batch_size": hyper["effective_batch_size"],
        "epochs": hyper["epochs"],
        "optimizer": hyper["optimizer"],
        "scheduler": hyper["scheduler"],
        "train_metrics": dict(train_result.metrics),
    }
    save_json(result_path, payload)
    trainer.model_wrapped = None
    trainer.model = None
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.accelerator = None
    del trainer, model, before, prediction, train_result, train_dataset, test_dataset
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"gpu allocated after release {torch.cuda.memory_allocated() / 2**20:.0f} MiB")
    return payload


def frozen_layer_ids(ranking: Dict[str, Any], variant: str) -> List[str]:
    if variant == "profile":
        return []
    return [f"{proj}:{layer_key(component, layer_idx)}" for component, layer_idx, proj in family_freeze_list(ranking, variant)]


def isolated_payload() -> Dict[str, Any]:
    return {
        "seed": run_cl.SEED,
        "learning_rate": ACTIVE_LR,
        "experiment": str(EXPERIMENT),
        "prior": "" if ACTIVE_PRIOR is None else str(ACTIVE_PRIOR),
    }


def run_isolated(payload: Dict[str, Any]) -> None:
    """One GPU step per process. Exiting the process releases every allocation."""
    env = os.environ.copy()
    env["TDA_FREEZE_WORKER"] = "1"
    env["TDA_FREEZE_PAYLOAD"] = json.dumps(payload)
    log(f"isolated {payload['action']} {payload.get('task', '')} {payload.get('variant', '')}".rstrip())
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"isolated {payload['action']} failed with exit {completed.returncode}")


def worker_main(payload: Dict[str, Any]) -> None:
    global EXPERIMENT, ACTIVE_LR, ACTIVE_PRIOR
    os.environ.pop("TDA_FREEZE_WORKER", None)
    run_cl.SEED = int(payload["seed"])
    ACTIVE_LR = float(payload["learning_rate"])
    ACTIVE_PRIOR = Path(payload["prior"]) if payload.get("prior") else None
    EXPERIMENT = Path(payload["experiment"])
    if payload["action"] == "extract":
        model, _tokenizer = run_cl.load_run_model(payload["checkpoint"])
        saved = cl_train.extract_t5_weights(model, Path(payload["dest"]))
        log(f"extracted {saved} matrices to {payload['dest']}")
        return
    hyper = configure()
    ranking = None
    if payload["variant"] != "profile":
        ranking = read_json(EXPERIMENT / payload["task"] / "ranking.json")
    train_variant(payload["task"], payload["variant"], ranking, hyper)


def profile_task(task: str, hyper: Dict[str, Any]) -> Dict[str, Any]:
    ranking_file = EXPERIMENT / task / "ranking.json"
    if ACTIVE_PRIOR is not None:
        prior = ACTIVE_PRIOR / task / "ranking.json"
        if not ranking_file.exists() and ranking_is_valid(prior):
            ranking_file.parent.mkdir(parents=True, exist_ok=True)
            ranking_file.symlink_to(prior)
    if ranking_is_valid(ranking_file):
        log(f"reuse TDA ranking {task}")
        update_status(f"{task}/profile", "completed", "ranking reused")
        return read_json(ranking_file)
    update_status(f"{task}/profile", "running")
    pretrained_dir = EXPERIMENT / "pretrained_qkvo"
    if not (pretrained_dir / "enc_layer0_q.npy").exists():
        log("extract pretrained Q/K/V/O once")
        run_isolated({**isolated_payload(), "action": "extract", "checkpoint": str(run_cl.W0), "dest": str(pretrained_dir)})
    run_isolated({**isolated_payload(), "action": "train", "task": task, "variant": "profile"})
    trained = read_json(EXPERIMENT / task / "profile" / "result.json")
    weight_dir = EXPERIMENT / task / "profile" / "finetuned_qkvo"
    if not (weight_dir / "enc_layer0_q.npy").exists():
        if not run_cl.has_full_checkpoint(EXPERIMENT / task / "profile" / "model"):
            raise RuntimeError(f"{task} profile checkpoint has no QKVO matrices")
        run_isolated(
            {
                **isolated_payload(),
                "action": "extract",
                "checkpoint": str(EXPERIMENT / task / "profile" / "model"),
                "dest": str(weight_dir),
            }
        )
    tda_started = time.perf_counter()
    rows = compute_task_tda(pretrained_dir, weight_dir, EXPERIMENT / task / "diagrams")
    tda_seconds = time.perf_counter() - tda_started
    save_json(EXPERIMENT / task / "sw_scores.json", rows)
    write_scores_csv(EXPERIMENT / task / "sw_scores.csv", rows)
    ranking = build_ranking(rows)
    ranking["task"] = task
    ranking["tda_seconds"] = tda_seconds
    ranking["profile_train_seconds"] = trained["train_seconds"]
    ranking["profile_accuracy"] = trained["accuracy"]
    save_json(ranking_file, ranking)
    if not ranking_is_valid(ranking_file):
        raise RuntimeError(f"{task} ranking failed validation")
    update_status(f"{task}/profile", "completed")
    return ranking


def write_scores_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["component", "layer_idx", "layer_id", "projection", "file", "sliced_wasserstein_h0"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")


def write_summaries() -> None:
    by_task = []
    combined = []
    tda_rows = []
    for task in TASKS:
        ranking_file = EXPERIMENT / task / "ranking.json"
        ranking = read_json(ranking_file) if ranking_file.exists() else None
        if ranking:
            order = sorted(PROJECTIONS, key=lambda proj: (-ranking["mean_sw_drift"][proj], proj))
            for rank, proj in enumerate(order, start=1):
                role = "low-drift frozen" if proj in ranking["low_drift_families"] else "ranked"
                tda_rows.append(
                    {
                        "task": task,
                        "projection": proj,
                        "mean_sw_drift": ranking["mean_sw_drift"][proj],
                        "rank": rank,
                        "role": role,
                    }
                )
        for variant in VARIANTS:
            result_file = EXPERIMENT / task / variant / "result.json"
            result = read_json(result_file) if result_is_valid(result_file) else {}
            layers = ""
            if ranking and result:
                shown = []
                for family in ranking["ranked_families"]:
                    ids = " ".join(row["layer_id"] for row in ranked_rows(ranking, family, variant))
                    shown.append(f"{family}: {ids}")
                layers = " | ".join(shown)
            by_task.append(
                {
                    "task": task,
                    "variant": variant,
                    "low_drift_families": "" if not ranking else " ".join(ranking["low_drift_families"]),
                    "ranked_families": "" if not ranking else " ".join(ranking["ranked_families"]),
                    "ranked_layers_frozen": layers,
                    "accuracy": result.get("accuracy", ""),
                    "trainable_percent": result.get("trainable_percent", ""),
                    "updated_percent": result.get("updated_percent", ""),
                    "train_seconds": result.get("train_seconds", ""),
                    "tda_seconds": "" if not ranking else ranking.get("tda_seconds", ""),
                    "peak_gpu_allocated": result.get("peak_gpu_allocated", ""),
                }
            )
            combined.append(
                {
                    "task": task,
                    "variant": variant,
                    "accuracy": result.get("accuracy", ""),
                    "trainable_percent": result.get("trainable_percent", ""),
                    "updated_percent": result.get("updated_percent", ""),
                    "train_seconds": result.get("train_seconds", ""),
                    "peak_gpu_allocated": result.get("peak_gpu_allocated", ""),
                }
            )
    write_table(EXPERIMENT / "summary_by_task.csv", by_task)
    write_table(EXPERIMENT / "summary_combined.csv", combined)
    write_table(EXPERIMENT / "tda_summary.csv", tda_rows)


def write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            handle.write(",".join(str(row[field]) for field in fields) + "\n")


def load_cpu_model():
    """Same checkpoint as training, kept off the GPU so a live job is left alone."""
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    source = str(cl_train.INITIAL_MODEL)
    config = AutoConfig.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(source, config=config, torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))
    return model


def run_sanity() -> None:
    hyper = configure()
    save_json(EXPERIMENT / "hyperparameters.json", hyper)
    log("resolved hyperparameters")
    for key, value in hyper.items():
        log(f"  {key}={value}")
    model = load_cpu_model()
    always = always_frozen_names(model)
    qkvo = []
    shapes: Dict[str, int] = {}
    for component, _prefix, count in COMPONENTS:
        for layer_idx in range(count):
            for proj in PROJECTIONS:
                weight = projection_weight(model, component, layer_idx, proj)
                qkvo.append((component, layer_idx, proj, tuple(weight.shape)))
                shapes[f"{component}.{proj}"] = shapes.get(f"{component}.{proj}", 0) + 1
    log(f"QKVO matrices {len(qkvo)}")
    for key, count in sorted(shapes.items()):
        log(f"  {key} count={count} example_shape={projection_weight(model, key.split('.')[0], 0, key.split('.')[1]).shape}")
    for group, names in always.items():
        log(f"  always {group} count={len(names)} example={names[:2]}")
    if len(qkvo) != (24 + 24 + 24) * 4:
        raise RuntimeError(f"unexpected QKVO count {len(qkvo)}")
    if len(always["ffn_wi"]) != 48 or len(always["ffn_wo"]) != 48:
        raise RuntimeError(f"unexpected FFN counts wi={len(always['ffn_wi'])} wo={len(always['ffn_wo'])}")
    if always["embedding"] != ["shared.weight"]:
        raise RuntimeError(f"shared embedding was not mapped once: {always['embedding']}")
    shared = dict(model.named_parameters())
    if model.lm_head.weight.data_ptr() != shared["shared.weight"].data_ptr():
        raise RuntimeError("lm_head.weight is not tied to shared.weight")
    if model.encoder.embed_tokens.weight.data_ptr() != shared["shared.weight"].data_ptr():
        raise RuntimeError("encoder embeddings are not tied to shared.weight")
    log("shared.weight, encoder/decoder embeddings, and lm_head.weight are one parameter")

    ranking = {
        "low_drift_families": ["q", "k"],
        "ranked_families": ["v", "o"],
        "rankings": {},
    }
    for proj in PROJECTIONS:
        ranking["rankings"][proj] = [
            {"layer_id": layer_key(component, layer_idx), "sliced_wasserstein_h0": float(1000 - layer_idx)}
            for component, _prefix, count in COMPONENTS
            for layer_idx in range(count)
        ]
    nested = []
    for variant in VARIANTS:
        frozen = family_freeze_list(ranking, variant)
        apply_freeze(model, frozen, always)
        for component, layer_idx, proj in frozen:
            if projection_weight(model, component, layer_idx, proj).requires_grad:
                raise RuntimeError(f"{variant} left {component}:{layer_idx}.{proj} trainable")
        for name in always["ffn_wi"] + always["ffn_wo"] + always["embedding"]:
            if shared[name].requires_grad:
                raise RuntimeError(f"{variant} left {name} trainable")
        nested.append(set(frozen_layer_ids(ranking, variant)))
    by_name = {variant: item for variant, item in zip(VARIANTS, nested)}
    if not (by_name["high1"] < by_name["high2"] < by_name["high3"] < by_name["all"]):
        raise RuntimeError("High1/High2/High3/All are not nested")
    for component in ("enc_self", "dec_self", "dec_cross"):
        if f"v:{component}:0" not in by_name["high1"] or f"o:{component}:0" not in by_name["high1"]:
            raise RuntimeError(f"High1 did not freeze {component} layer 0 of each ranked family")
        if f"v:{component}:1" in by_name["high1"] or f"v:{component}:1" not in by_name["high2"]:
            raise RuntimeError(f"High2 is not the next {component} layer after High1")
    ranked_in_all = [item for item in by_name["all"] if item.startswith("v:") or item.startswith("o:")]
    if len(ranked_in_all) != 2 * (24 + 24 + 24):
        raise RuntimeError(f"All froze {len(ranked_in_all)} ranked matrices, expected 144")
    apply_freeze(model, family_freeze_list(ranking, "high1"), always)
    before = snapshot(model)
    loss = sum(param.view(-1)[0] for _name, param in unique_parameters(model) if param.requires_grad)
    loss.backward()
    torch.optim.AdamW([param for _name, param in unique_parameters(model) if param.requires_grad], lr=1e-3).step()
    moved = 0
    stayed = 0
    for _name, param in unique_parameters(model):
        changed = bool((param.detach().float().cpu() - before[param.data_ptr()]).abs().max() > 0)
        if param.requires_grad:
            moved += int(changed)
        elif changed:
            raise RuntimeError("a frozen parameter changed after the optimizer step")
        else:
            stayed += 1
    if moved == 0:
        raise RuntimeError("no trainable parameter changed")
    log(f"sanity optimizer step moved {moved} trainable tensors and held {stayed} frozen tensors")
    save_json(EXPERIMENT / "sanity.json", {"status": "passed", "hyperparameters": hyper, "qkvo": len(qkvo)})
    del model


def lr_slug(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def apply_cli() -> argparse.Namespace:
    global EXPERIMENT, ACTIVE_LR, ACTIVE_PRIOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=run_cl.SEED)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--prior-ranking", type=Path, default=None)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    run_cl.SEED = args.seed
    ACTIVE_LR = args.learning_rate
    ACTIVE_PRIOR = args.prior_ranking
    if args.experiment is not None:
        EXPERIMENT = args.experiment
    else:
        EXPERIMENT = ROOT / "results" / "tda_freeze" / "t5-large-per-block" / f"lr_{lr_slug(args.learning_rate)}" / f"seed_{args.seed}"
    return args


def main() -> None:
    if os.environ.get("TDA_FREEZE_WORKER") == "1":
        worker_main(json.loads(os.environ["TDA_FREEZE_PAYLOAD"]))
        return
    args = apply_cli()
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if args.sanity_only:
        run_sanity()
        log("SANITY_PASSED")
        return
    sanity = EXPERIMENT / "sanity.json"
    if not sanity.exists() or read_json(sanity).get("status") != "passed":
        run_sanity()
    hyper = configure()
    save_json(EXPERIMENT / "hyperparameters.json", hyper)
    log("resolved hyperparameters")
    for key, value in hyper.items():
        log(f"  {key}={value}")
    failures = 0
    for task in TASKS:
        try:
            ranking = profile_task(task, hyper)
        except Exception:
            failures += 1
            err = traceback.format_exc()
            (EXPERIMENT / task).mkdir(parents=True, exist_ok=True)
            (EXPERIMENT / task / "profile_error.txt").write_text(err)
            update_status(f"{task}/profile", "failed", "see profile_error.txt")
            log(err)
            continue
        for variant in VARIANTS:
            key = f"{task}/{variant}"
            if result_is_valid(EXPERIMENT / task / variant / "result.json"):
                update_status(key, "completed", "already valid")
                continue
            update_status(key, "running")
            try:
                run_isolated({**isolated_payload(), "action": "train", "task": task, "variant": variant})
                if not result_is_valid(EXPERIMENT / task / variant / "result.json"):
                    raise RuntimeError(f"{key} result failed validation")
                update_status(key, "completed")
            except Exception:
                failures += 1
                err = traceback.format_exc()
                (EXPERIMENT / task / variant).mkdir(parents=True, exist_ok=True)
                (EXPERIMENT / task / variant / "error.txt").write_text(err)
                update_status(key, "failed", "see error.txt")
                log(err)
        write_summaries()
    write_summaries()
    if failures:
        log(f"TDA_FREEZE_DONE_WITH_FAILURES {failures}")
        raise SystemExit(1)
    log("TDA_FREEZE_DONE")


if __name__ == "__main__":
    main()
