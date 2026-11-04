#!/usr/bin/env python3
"""Four alternative merges of the existing full-finetune experts.

Does not train and does not write into the finished OrthoMerge trees.
PlainQ and WeightedQ follow run_merge.py: right Procrustes W0 @ R ≈ W_i,
inverse Cayley, and rho_i = W_i - W0 @ R_i.

ActivationAware is a separate left rotation with no rho:
    F = sum_i M_i G_i W0^T, R = spolar(F), W = R @ W0.
RegMean uses the same Grams and does not add rho.
G_i = X_i^T X_i, with X_i the Linear input stacked as rows.
Raw activations are not saved.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import run_merge
from run_merge import (
    DATASETS,
    EMBED_KEY,
    LM_HEAD_KEY,
    align_vocab_rows,
    cayley_from_skew,
    cayley_to_skew,
    configure_run,
    evaluate_checkpoint,
    has_full_checkpoint,
    is_orthomerge_weight,
    load_architecture,
    load_cpu_state,
    log,
    orthogonal_procrustes_torch_right,
    orthogonality_error,
    save_json,
    write_csv,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HOME", "/data/kadir/hf-cache")

ROOT = Path(__file__).resolve().parent
T5_MODEL = (
    "/data/cuneyt-topo/numpy_weights/t5-continual-learning/"
    "external_baselines/GORP/initial_model/t5-large"
)
LLAMA_MODEL = "meta-llama/Llama-3.2-1B"
LLAMA8B_MODEL = (
    "/data/kadir/hf-cache/hub/models--meta-llama--Llama-3.1-8B/"
    "snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)
TASK_WEIGHTS = {
    "dbpedia": 14000.0,
    "amazon": 5000.0,
    "yahoo": 10000.0,
    "agnews": 4000.0,
}
EXPERT_DIR = {
    "dbpedia": "W_DBPedia",
    "amazon": "W_Amazon",
    "yahoo": "W_Yahoo",
    "agnews": "W_AGNews",
}
METHODS = ("PlainQ", "WeightedQ", "RegMean", "ActivationAware")
METHOD_ORDER = ("PlainQ", "WeightedQ", "RegMean", "ActivationAware")
GRAM_METHODS = {"RegMean", "ActivationAware"}
SUMMARY_METHODS = ("OrthoMerge", "PlainQ", "WeightedQ", "RegMean", "ActivationAware")
NORM_EPS = 1e-8
# Left rotation R is (out, out). MLP outputs used here are at most 8192.
# Vocab matrices are about 32k or 128k on a side, so those rotations are skipped.
MAX_ACTIVATION_OUTPUT = 16384


def architecture_spec(name: str, seed: int) -> dict:
    """Point at the checkpoints that already exist for this architecture and seed."""
    results = ROOT / "results"
    seed = int(seed)
    if name == "t5":
        if seed == 42:
            experts_root = results / "independent"
            orthomerge_csv = results / "final_comparison.csv"
        else:
            experts_root = results / "seeds" / f"seed_{seed}" / "independent"
            orthomerge_csv = results / "seeds" / f"seed_{seed}" / "final_comparison.csv"
        label = "t5-large"
        return {
            "name": "t5",
            "label": label,
            "model": T5_MODEL,
            "model_type": "seq2seq",
            "learning_rate": 1e-4,
            "seed": seed,
            "experts_root": experts_root,
            "orthomerge_csv": orthomerge_csv,
            "output": results / "alternative_merges" / label / f"seed_{seed}",
        }
    if name == "llama":
        label = "llama-3.2-1b"
        base = results / label / "lr_2e-5" / f"seed_{seed}"
        return {
            "name": "llama",
            "label": label,
            "model": LLAMA_MODEL,
            "model_type": "causal",
            "learning_rate": 2e-5,
            "seed": seed,
            "experts_root": base / "independent",
            "orthomerge_csv": base / "final_comparison.csv",
            "output": results / "alternative_merges" / label / f"seed_{seed}",
        }
    if name == "llama8b":
        label = "llama-3.1-8b"
        base = results / label / "lr_1e-5" / f"seed_{seed}"
        return {
            "name": "llama8b",
            "label": label,
            "model": LLAMA8B_MODEL,
            "model_type": "causal",
            "learning_rate": 1e-5,
            "seed": seed,
            "experts_root": base / "independent",
            "orthomerge_csv": base / "final_comparison.csv",
            "output": base / "merge",
        }
    raise SystemExit(f"Unknown architecture {name}")


def expert_paths(spec: dict) -> Dict[str, Path]:
    paths = {}
    for task in DATASETS:
        path = spec["experts_root"] / task / EXPERT_DIR[task]
        paths[task] = path
    return paths


def assert_output_allowed(path: Path) -> None:
    resolved = path.resolve()
    results = (ROOT / "results").resolve()
    allowed = (results / "alternative_merges").resolve()
    llama8b = (results / "llama-3.1-8b").resolve()
    lr_sweep = (results / "lr_sweep").resolve()
    under_existing = resolved == allowed or allowed in resolved.parents
    under_llama8b = resolved == llama8b or llama8b in resolved.parents
    under_lr_sweep = resolved == lr_sweep or lr_sweep in resolved.parents
    if not under_existing and not under_llama8b and not under_lr_sweep:
        raise SystemExit(f"Refusing to write outside {allowed}: {resolved}")
    banned = [
        results / "independent",
        results / "merge",
        results / "cl",
        results / "seeds",
        results / "final_eval",
        results / "pretrained_baseline",
        results / "t5-large",
        results / "llama-3.2-1b",
    ]
    for blocked in banned:
        if resolved == blocked or blocked in resolved.parents:
            raise SystemExit(f"Refusing to write a finished result tree: {resolved}")


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


def matrix_batch_width(square: int) -> int:
    """How many same-shaped matrices to put in one GPU call.

    8192 SVDs and solves get a smaller batch so the workspace stays on the H200.
    """
    if square >= 12000:
        return 2
    if square >= 8192:
        return 8
    if square >= 4096:
        return 16
    return 32


def regmean_weight_batch(
    experts: torch.Tensor,
    grams: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Batched regmean_weight. experts (B, N, out, in), grams (B, N, in, in)."""
    total_gram = grams.sum(dim=1)
    total_mixed = torch.matmul(experts, grams).sum(dim=1)
    mean_diag = torch.diagonal(total_gram, dim1=-2, dim2=-1).mean(dim=-1)
    penalty = float(ridge) * torch.clamp(mean_diag, min=NORM_EPS)
    eye = torch.eye(total_gram.shape[-1], device=total_gram.device, dtype=total_gram.dtype)
    system = total_gram + penalty[:, None, None] * eye
    solved = torch.linalg.solve(system, total_mixed.transpose(-1, -2))
    return solved.transpose(-1, -2).contiguous()


def activation_aware_weight_batch(
    w0: torch.Tensor,
    experts: torch.Tensor,
    grams: torch.Tensor,
) -> torch.Tensor:
    """Batched activation_aware_weight. w0 (B, out, in), experts (B, N, out, in)."""
    base = w0.float()
    mixed = torch.matmul(experts.float(), grams.float())
    total = torch.matmul(mixed, base.unsqueeze(1).transpose(-1, -2)).sum(dim=1)
    norms = torch.linalg.matrix_norm(total, ord="fro")
    u_factor, _singular, vh = torch.linalg.svd(total, full_matrices=False)
    rotation = torch.matmul(u_factor, vh)
    eye = torch.eye(base.shape[-2], device=base.device, dtype=base.dtype)
    small = norms < NORM_EPS
    rotation = torch.where(small[:, None, None], eye.expand_as(rotation), rotation)
    return torch.matmul(rotation, base)


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

    batch_w0 = []
    batch_experts = []
    batch_grams = []
    scalar_reg = []
    scalar_act = []
    for _ in range(3):
        w0_i = torch.randn(out, inn)
        experts_i = [torch.randn(out, inn) for _ in range(4)]
        grams_i = []
        for _gram in range(4):
            raw = torch.randn(inn, inn)
            grams_i.append(raw @ raw.t())
        batch_w0.append(w0_i)
        batch_experts.append(torch.stack(experts_i))
        batch_grams.append(torch.stack(grams_i))
        scalar_reg.append(regmean_weight(experts_i, grams_i, ridge=1e-4))
        scalar_act.append(activation_aware_weight(w0_i, experts_i, grams_i))
    batched_reg = regmean_weight_batch(torch.stack(batch_experts), torch.stack(batch_grams), 1e-4)
    batched_act = activation_aware_weight_batch(
        torch.stack(batch_w0),
        torch.stack(batch_experts),
        torch.stack(batch_grams),
    )
    if not torch.allclose(batched_reg, torch.stack(scalar_reg), atol=1e-4):
        raise SystemExit("Batched RegMean does not match the single-matrix formula")
    if not torch.allclose(batched_act, torch.stack(scalar_act), atol=1e-4):
        raise SystemExit("Batched ActivationAware does not match the single-matrix formula")
    zero_expert = [torch.zeros(out, inn) for _ in range(4)]
    zero_act = activation_aware_weight_batch(
        w0.unsqueeze(0),
        torch.stack(zero_expert).unsqueeze(0),
        torch.stack([eye for _ in zero_expert]).unsqueeze(0),
    )
    if not torch.allclose(zero_act[0], w0, atol=1e-4):
        raise SystemExit("Batched ActivationAware did not return W0 for a zero F")
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


def parameter_kind(name: str, tensor: torch.Tensor, linear_names: Dict[str, str]) -> str:
    if run_merge.MODEL_TYPE == "causal" and name == LM_HEAD_KEY and run_merge.TIED_LM_HEAD:
        return "tied_lm_head"
    if name in linear_names and is_orthomerge_weight(name, tensor):
        return "gram_linear"
    if is_orthomerge_weight(name, tensor):
        return "orthomerge_without_input_gram"
    return "mean_residual"


def collect_task_grams(
    task: str,
    expert_path: Path,
    destination: Path,
    gram_batch_size: int,
    force: bool,
    data_dir: Optional[Path] = None,
    task_config_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        log(f"Reusing Grams {destination}")
        return torch.load(destination, map_location="cpu")

    from torch.utils.data import DataLoader
    from uie_collator import DataCollatorForUIE

    model, tokenizer = run_merge.run_cl.load_run_model(str(expert_path))
    model.eval()
    # Saved checkpoints store a local directory in _name_or_path. The GORP
    # collator only accepts a name containing "t5" or "llama".
    recorded_name = str(model.config._name_or_path)
    if run_merge.MODEL_TYPE == "causal":
        if "llama" not in recorded_name.lower():
            model.config._name_or_path = "meta-llama/Llama-3.2-1B"
    elif "t5" not in recorded_name.lower():
        model.config._name_or_path = "t5-large"
    log(f"Gram collator model name {recorded_name} -> {model.config._name_or_path}")
    linear_names = linear_parameter_names(model)
    collector = GramCollector(run_merge.MODEL_TYPE)
    handles = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(lambda _mod, inputs, _out, name=module_name: collector.hook(name, _mod, inputs)))
    raw = run_merge.cl_train.load_uie_datasets(
        data_dir=data_dir or run_merge.cl_train.data_root_for_order(1),
        task_config_dir=task_config_dir or run_merge.single_task_config_dir(task),
        output_dir=destination.parent.parent / "dataset_cache" / task,
        overwrite_cache=False,
    )
    train_dataset, original_size, capped_size = run_merge.cl_train.cap_train_dataset(
        raw["train"],
        max_train_samples=None,
        subsample_seed=run_merge.SEED,
        logger=None,
    )
    log(f"Gram forward {task} samples={capped_size}/{original_size} batch={gram_batch_size}")
    collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=512,
        max_target_length=50,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        add_task_name=True,
        add_dataset_name=True,
        num_examples=0,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=gram_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if run_merge.MODEL_TYPE == "causal":
                collector.decoder_mask = batch.get("attention_mask")
                collector.encoder_mask = None
            else:
                collector.encoder_mask = batch.get("attention_mask")
                labels = batch.get("labels")
                collector.decoder_mask = None if labels is None else labels.ne(-100).to(dtype=torch.float32)
            model_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels", "loss_mask", "decoder_input_ids"}
            }
            model(**model_inputs)
            if step == 1 or step % 100 == 0:
                log(f"  Gram {task} step {step}")
    for handle in handles:
        handle.remove()
    grams = {f"{name}.weight": tensor.detach().cpu() for name, tensor in collector.grams.items()}
    if collector.mask_mismatches != 0:
        raise SystemExit(
            f"Gram collection for {task} had mask_mismatches={collector.mask_mismatches}. Stopping."
        )
    log(f"Gram {task}: layers={len(grams)} mask_mismatches={collector.mask_mismatches}")
    payload = {
        "grams": grams,
        "tokens": {f"{name}.weight": count for name, count in collector.tokens.items()},
        "mask_mismatches": collector.mask_mismatches,
        "linear_names": sorted(linear_names),
    }
    torch.save(payload, destination)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def load_gram_payload(
    task: str,
    expert_path: Path,
    destination: Path,
    gram_batch_size: int,
    force: bool,
    data_dir: Optional[Path] = None,
    task_config_dir: Optional[Path] = None,
) -> dict:
    payload = collect_task_grams(
        task,
        expert_path,
        destination,
        gram_batch_size,
        force,
        data_dir=data_dir,
        task_config_dir=task_config_dir,
    )
    if "mask_mismatches" not in payload or "grams" not in payload:
        raise SystemExit(f"Gram file for {task} has no mask_mismatches record. Stopping.")
    mismatches = int(payload["mask_mismatches"])
    if mismatches != 0:
        raise SystemExit(f"Gram file for {task} has mask_mismatches={mismatches}. Stopping.")
    return payload


def require_grams(grams: Dict[str, Dict[str, torch.Tensor]], required_names: Sequence[str]) -> None:
    """Every Linear used by ActivationAware or RegMean must have a Gram on every task."""
    missing = []
    for task in DATASETS:
        present = grams.get(task, {})
        for name in required_names:
            if name not in present:
                missing.append(f"{task}:{name}")
    if missing:
        preview = ", ".join(missing[:8])
        raise SystemExit(
            f"Missing {len(missing)} required Gram matrices ({preview}). "
            "Refusing to fall back to another merge rule."
        )


def merge_matrix(
    method: str,
    name: str,
    w0_rows: torch.Tensor,
    task_rows: Sequence[torch.Tensor],
    grams: Optional[Sequence[torch.Tensor]],
    ridge: float,
    device: torch.device,
) -> torch.Tensor:
    weights = [TASK_WEIGHTS[task] for task in DATASETS]
    if method in {"PlainQ", "WeightedQ"} and is_orthomerge_weight(name, w0_rows):
        w0_t = w0_rows.detach().float().to(device)
        experts = [tensor.detach().float().to(device) for tensor in task_rows]
        skews = []
        residuals = []
        for expert in experts:
            rotation = orthogonal_procrustes_torch_right(expert, w0_t)
            skews.append(cayley_to_skew(rotation))
            residuals.append(expert - torch.matmul(w0_t, rotation))
        rho = mean_residual(residuals)
        if method == "PlainQ":
            q_merged = plain_q_average(skews)
            merged = torch.matmul(w0_t, cayley_from_skew(q_merged)) + rho
        else:
            q_merged = weighted_q_merge(skews, weights)
            merged = torch.matmul(w0_t, cayley_from_skew(q_merged)) + rho
        return merged.detach().cpu()
    if method == "ActivationAware" and grams is not None and is_orthomerge_weight(name, w0_rows):
        w0_t = w0_rows.detach().float().to(device)
        experts = [tensor.detach().float().to(device) for tensor in task_rows]
        gram_t = [gram.detach().float().to(device) for gram in grams]
        return activation_aware_weight(w0_t, experts, gram_t).detach().cpu()
    if method == "RegMean" and grams is not None and is_orthomerge_weight(name, w0_rows):
        experts = [tensor.detach().float().to(device) for tensor in task_rows]
        gram_t = [gram.detach().float().to(device) for gram in grams]
        return regmean_weight(experts, gram_t, ridge).detach().cpu()
    return uniform_mean(task_rows).detach().cpu()


def save_merged_model(
    spec: dict,
    merged_state: Dict[str, torch.Tensor],
    dest: Path,
    method: str,
    ridge: float,
    lm_head_used_gram: bool,
    activation_fallback_names: Sequence[str],
) -> None:
    if run_merge.MODEL_TYPE == "causal" and run_merge.TIED_LM_HEAD:
        merged_state[LM_HEAD_KEY] = merged_state[EMBED_KEY].detach().clone()
        log(f"Copied merged {EMBED_KEY} onto {LM_HEAD_KEY}")
    elif lm_head_used_gram and LM_HEAD_KEY in merged_state:
        # T5 tie_word_embeddings points lm_head at shared. Keep the Gram
        # solution for that one tied matrix; the embedding has no input Gram.
        tied = merged_state[LM_HEAD_KEY].detach().clone()
        for key in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
            if key in merged_state and tuple(merged_state[key].shape) == tuple(tied.shape):
                merged_state[key] = tied.clone()
        log("Copied Gram-merged lm_head onto the tied T5 embedding matrices")
    model = load_architecture(run_merge.W0, torch.float32)
    incompatible = model.load_state_dict(merged_state, strict=True)
    log(f"load_state_dict {incompatible}")
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest, safe_serialization=False)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(run_merge.load_source(run_merge.W0), use_fast=True)
    if run_merge.MODEL_TYPE == "causal" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(dest)
    if not has_full_checkpoint(dest):
        raise RuntimeError(f"Missing checkpoint at {dest}")
    reloaded = load_architecture(dest, torch.float32)
    reloaded.tie_weights()
    del model, reloaded
    gc.collect()
    save_json(
        {
            "method": method,
            "plain_q": "Q_merge = mean(Q_i); R = cayley(Q_merge); rho = mean(rho_i); W = W0 @ R + rho",
            "weighted_q": "Q_merge = normalize(sum w_i Q_i) * (sum w_i ||Q_i|| / sum w_i); rho = mean(rho_i)",
            "activation_aware": "R = spolar(sum_i M_i G_i W0^T); W = R @ W0; no rho residual is added",
            "activation_aware_fallback": (
                "uniform mean of the four expert weights when output dimension "
                f"> {MAX_ACTIVATION_OUTPUT} (vocab-scale left rotation)"
            ),
            "activation_aware_fallback_parameters": list(activation_fallback_names),
            "regmean": "W = (sum M_i G_i) (sum G_i + ridge * mean_diag * I)^{-1}",
            "weights": TASK_WEIGHTS,
            "regmean_ridge": ridge,
            "excluded_from_grams": "Embeddings, relative attention bias, norms, and Llama lm_head. No input Gram on the rotated axis.",
            "w0": str(run_merge.W0),
            "experts": {task: str(path) for task, path in expert_paths(spec).items()},
            "seed": spec["seed"],
            "learning_rate": spec["learning_rate"],
        },
        dest.parent / "merge_config.json",
    )


def merge_method(
    spec: dict,
    method: str,
    paths: Dict[str, Path],
    grams: Optional[Dict[str, Dict[str, torch.Tensor]]],
    ridge: float,
    force: bool,
) -> Path:
    dest = spec["output"] / method / "W_MERGED_FINAL"
    assert_output_allowed(dest)
    if not force and has_full_checkpoint(dest):
        log(f"Reusing {dest}")
        return dest
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the merge SVDs.")
    device = torch.device("cuda")
    log(f"Loading W0 for {method}")
    w0 = load_cpu_state(run_merge.W0)
    task_states = {task: load_cpu_state(path) for task, path in paths.items()}
    linear_model = load_architecture(paths["dbpedia"], torch.float32)
    linear_names = linear_parameter_names(linear_model)
    del linear_model
    gc.collect()
    merged_state = {key: value.clone() for key, value in w0.items()}
    counts = {"gram_linear": 0, "orthomerge_without_input_gram": 0, "mean_residual": 0, "tied_lm_head": 0}
    lm_head_used_gram = False
    activation_fallback_names: List[str] = []
    batched: Dict[Tuple[str, int, int], List[dict]] = {}
    names = list(w0.keys())
    total = len(names)

    def store_merged(name: str, merged: torch.Tensor, source: torch.Tensor, vocab_prefix: Optional[int]) -> None:
        if vocab_prefix is not None:
            full = source.detach().float().clone()
            full[:vocab_prefix] = merged.detach().cpu()
            merged = full
        merged_state[name] = merged.detach().cpu().to(dtype=source.dtype)

    def flush_batch(group: List[dict]) -> None:
        if not group:
            return
        route = group[0]["route"]
        experts = torch.stack(
            [torch.stack([row.detach().float() for row in item["task_rows"]], dim=0) for item in group],
            dim=0,
        ).to(device)
        gram_tensor = torch.stack(
            [torch.stack([gram.detach().float() for gram in item["use_grams"]], dim=0) for item in group],
            dim=0,
        ).to(device)
        if route == "RegMean":
            merged_batch = regmean_weight_batch(experts, gram_tensor, ridge)
        else:
            base = torch.stack([item["w0_rows"].detach().float() for item in group], dim=0).to(device)
            merged_batch = activation_aware_weight_batch(base, experts, gram_tensor)
        for row, item in enumerate(group):
            store_merged(item["name"], merged_batch[row], item["source"], item["vocab_prefix"])
        log(
            f"  {method} batch route={route} n={len(group)} shape={tuple(group[0]['w0_rows'].shape)} "
            f"finished={group[-1]['index']}/{total} {group[-1]['name']}"
        )
        del experts, gram_tensor, merged_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for index, name in enumerate(names, start=1):
        if run_merge.MODEL_TYPE == "causal" and name == LM_HEAD_KEY and run_merge.TIED_LM_HEAD:
            counts["tied_lm_head"] += 1
            if method == "ActivationAware":
                activation_fallback_names.append(name)
                log(f"ActivationAware fallback uniform mean via embed copy: {name} output={tuple(w0[name].shape)[0]}")
            continue
        w0_rows, task_rows, vocab_prefix = align_vocab_rows(
            name,
            w0[name],
            [task_states[task][name] for task in DATASETS],
        )
        kind = parameter_kind(name, w0_rows, linear_names)
        use_grams = None
        if kind == "gram_linear" and grams is not None:
            if any(name not in grams[task] for task in DATASETS):
                raise SystemExit(f"Missing Gram for {name}. Stopping. No PlainQ or RegMean fallback.")
            use_grams = [grams[task][name] for task in DATASETS]
            counts["gram_linear"] += 1
            if name == LM_HEAD_KEY and method == "RegMean":
                lm_head_used_gram = True
        elif kind == "gram_linear":
            counts["gram_linear"] += 1
        else:
            counts[kind] += 1
        if method == "ActivationAware" and activation_output_impractical(w0_rows):
            activation_fallback_names.append(name)
            log(f"ActivationAware fallback uniform mean: {name} output={w0_rows.shape[0]}")
            store_merged(name, uniform_mean(task_rows), w0[name], vocab_prefix)
            continue
        if method == "ActivationAware" and kind == "gram_linear" and use_grams is None:
            raise SystemExit(f"Missing Gram for ActivationAware parameter {name}. Stopping.")
        heavy = (
            method in {"RegMean", "ActivationAware"}
            and kind == "gram_linear"
            and use_grams is not None
            and is_orthomerge_weight(name, w0_rows)
        )
        if heavy:
            square = int(w0_rows.shape[0] if method == "ActivationAware" else w0_rows.shape[1])
            key = (method, int(w0_rows.shape[0]), int(w0_rows.shape[1]))
            group = batched.setdefault(key, [])
            group.append(
                {
                    "name": name,
                    "index": index,
                    "route": method,
                    "w0_rows": w0_rows,
                    "task_rows": task_rows,
                    "use_grams": use_grams,
                    "source": w0[name],
                    "vocab_prefix": vocab_prefix,
                }
            )
            if len(group) >= matrix_batch_width(square):
                flush_batch(group)
                batched[key] = []
            continue
        merged = merge_matrix(method, name, w0_rows, task_rows, use_grams, ridge, device)
        store_merged(name, merged, w0[name], vocab_prefix)
        if index % 25 == 0 or index == total:
            log(f"  {method} {index}/{total} {name}")
        del merged
    for group in batched.values():
        flush_batch(group)
    log(f"{method} parameter kinds: {counts}")
    save_merged_model(spec, merged_state, dest, method, ridge, lm_head_used_gram, activation_fallback_names)
    del w0, task_states, merged_state
    gc.collect()
    return dest


def read_orthomerge_scores(path: Path) -> Dict[str, float]:
    scores = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "W_MERGED_FINAL":
                scores[row["eval_dataset"]] = float(row["accuracy"])
    expected = {"dbpedia", "amazon", "yahoo", "agnews", "average"}
    if set(scores) != expected:
        raise SystemExit(f"OrthoMerge row in {path} is missing datasets: {sorted(expected - set(scores))}")
    return scores


def write_comparison(spec: dict, rows: List[Dict[str, object]]) -> None:
    destination = spec["output"] / "comparison.csv"
    assert_output_allowed(destination)
    write_csv(
        destination,
        ["Method", "DBpedia", "Amazon", "Yahoo", "AGNews", "Avg."],
        rows,
    )
    log(f"Wrote {destination}")


def load_comparison_table(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    table: Dict[str, Dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            table[row["Method"]] = {
                "dbpedia": float(row["DBpedia"]),
                "amazon": float(row["Amazon"]),
                "yahoo": float(row["Yahoo"]),
                "agnews": float(row["AGNews"]),
                "average": float(row["Avg."]),
            }
    return table


def upsert_comparison(spec: dict, method: str, scores: Dict[str, float]) -> None:
    table = load_comparison_table(spec["output"] / "comparison.csv")
    table["OrthoMerge"] = read_orthomerge_scores(spec["orthomerge_csv"])
    table[method] = scores
    rows = [score_row(name, table[name]) for name in SUMMARY_METHODS if name in table]
    write_comparison(spec, rows)


def update_three_seed_summary(label: str, seeds: Sequence[int]) -> None:
    """Write mean ± sample std only for methods present on every requested seed."""
    base = ROOT / "results" / "alternative_merges" / label
    assert_output_allowed(base / "three_seed_summary.csv")
    per_seed = []
    for seed in seeds:
        per_seed.append(load_comparison_table(base / f"seed_{seed}" / "comparison.csv"))
    rows = []
    for method in SUMMARY_METHODS:
        if not all(method in table for table in per_seed):
            continue
        cells = {"Method": method}
        for column, key in (
            ("DBpedia", "dbpedia"),
            ("Amazon", "amazon"),
            ("Yahoo", "yahoo"),
            ("AGNews", "agnews"),
            ("Avg.", "average"),
        ):
            values = [table[method][key] for table in per_seed]
            mean = statistics.fmean(values)
            std = statistics.stdev(values)
            cells[column] = f"{mean:.2f} ± {std:.2f}"
        rows.append(cells)
    if not rows:
        return
    destination = base / "three_seed_summary.csv"
    write_csv(destination, ["Method", "DBpedia", "Amazon", "Yahoo", "AGNews", "Avg."], rows)
    log(f"Wrote {destination}")


def score_row(method: str, scores: Dict[str, float]) -> Dict[str, object]:
    return {
        "Method": method,
        "DBpedia": scores["dbpedia"],
        "Amazon": scores["amazon"],
        "Yahoo": scores["yahoo"],
        "AGNews": scores["agnews"],
        "Avg.": scores["average"],
    }


def evaluate_method(spec: dict, method: str, checkpoint: Path, force: bool) -> Dict[str, float]:
    output = spec["output"] / "eval" / method
    assert_output_allowed(output)
    scores = evaluate_checkpoint(checkpoint, DATASETS, output, force=force)
    values = [float(scores[task]) for task in DATASETS]
    scores = dict(scores)
    scores["average"] = sum(values) / len(values)
    return scores


def ensure_inputs(spec: dict) -> Dict[str, Path]:
    assert_output_allowed(spec["output"])
    paths = expert_paths(spec)
    missing = [f"{task} {path}" for task, path in paths.items() if not has_full_checkpoint(path)]
    if missing:
        raise SystemExit(f"Missing experts for {spec['label']} seed {spec['seed']}: {missing}")
    if not spec["orthomerge_csv"].exists():
        raise SystemExit(f"Missing OrthoMerge csv for {spec['label']} seed {spec['seed']}: {spec['orthomerge_csv']}")
    read_orthomerge_scores(spec["orthomerge_csv"])
    return paths


def load_method_grams(spec: dict, paths: Dict[str, Path], gram_batch_size: int, force_grams: bool) -> Dict[str, Dict[str, torch.Tensor]]:
    grams = {}
    gram_dir = spec["output"] / "grams"
    assert_output_allowed(gram_dir)
    for task, path in paths.items():
        payload = load_gram_payload(task, path, gram_dir / f"{task}.pt", gram_batch_size, force_grams)
        grams[task] = payload["grams"]
    require_grams(grams, required_gram_names(spec))
    return grams


def run_one(
    spec: dict,
    method: str,
    gram_batch_size: int,
    ridge: float,
    force: bool,
    force_grams: bool,
) -> None:
    paths = ensure_inputs(spec)
    token_file = Path("/data/kadir/hf-cache/token")
    if token_file.exists():
        token = token_file.read_text().strip()
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    configure_run(spec["seed"], spec["output"], spec["learning_rate"], spec["model"], spec["model_type"])
    run_merge.refresh_tied_lm_head()
    log(
        f"START method={method} architecture={spec['label']} seed={spec['seed']} "
        f"stage=configure experts={spec['experts_root']}"
    )
    grams = None
    if method in GRAM_METHODS:
        log(
            f"START method={method} architecture={spec['label']} seed={spec['seed']} "
            f"stage=grams cache={spec['output'] / 'grams'}"
        )
        grams = load_method_grams(spec, paths, gram_batch_size, force_grams)
        log(f"END method={method} architecture={spec['label']} seed={spec['seed']} stage=grams")
    else:
        log(f"SKIP grams method={method} architecture={spec['label']} seed={spec['seed']}")
    log(f"START method={method} architecture={spec['label']} seed={spec['seed']} stage=merge")
    checkpoint = merge_method(spec, method, paths, grams, ridge, force)
    log(f"END method={method} architecture={spec['label']} seed={spec['seed']} stage=merge checkpoint={checkpoint}")
    log(f"START method={method} architecture={spec['label']} seed={spec['seed']} stage=eval")
    scores = evaluate_method(spec, method, checkpoint, force)
    log(
        f"SCORES method={method} architecture={spec['label']} seed={spec['seed']} "
        f"dbpedia={scores['dbpedia']} amazon={scores['amazon']} yahoo={scores['yahoo']} "
        f"agnews={scores['agnews']} avg={scores['average']}"
    )
    log(f"END method={method} architecture={spec['label']} seed={spec['seed']} stage=eval")
    upsert_comparison(spec, method, scores)


def meta_parameter_plan(spec: dict) -> dict:
    """Classify parameters without loading checkpoint weights."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    config = AutoConfig.from_pretrained(spec["model"])
    with torch.device("meta"):
        if spec["model_type"] == "causal":
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForSeq2SeqLM.from_config(config)
    previous_type = run_merge.MODEL_TYPE
    run_merge.MODEL_TYPE = spec["model_type"]
    linear_names = linear_parameter_names(model)
    required = []
    fallback = []
    counts = {"gram_linear": 0, "orthomerge_without_input_gram": 0, "mean_residual": 0, "tied_lm_head": 0}
    examples: Dict[str, List[str]] = {key: [] for key in counts}
    for name, tensor in model.state_dict().items():
        kind = parameter_kind(name, tensor, linear_names)
        counts[kind] += 1
        if len(examples[kind]) < 4:
            examples[kind].append(f"{name} {tuple(tensor.shape)}")
        if kind == "gram_linear":
            required.append(name)
        if activation_output_impractical(tensor) or (spec["model_type"] == "causal" and name == LM_HEAD_KEY):
            fallback.append(f"{name} output={int(tensor.shape[0])}")
    run_merge.MODEL_TYPE = previous_type
    del model
    return {
        "required_gram_names": required,
        "activation_fallback": fallback,
        "counts": counts,
        "examples": examples,
        "tie_word_embeddings": config.tie_word_embeddings,
    }


def required_gram_names(spec: dict) -> List[str]:
    return meta_parameter_plan(spec)["required_gram_names"]


def dry_run_architecture(spec: dict) -> None:
    paths = expert_paths(spec)
    log(f"DRY {spec['label']}")
    log(f"  w0={spec['model']}")
    for task, path in paths.items():
        log(f"  expert {task} exists={has_full_checkpoint(path)} {path.resolve()}")
    log(f"  orthomerge_csv={spec['orthomerge_csv'].resolve()} exists={spec['orthomerge_csv'].exists()}")
    log(f"  output={spec['output'].resolve()}")
    assert_output_allowed(spec["output"] / "PlainQ" / "W_MERGED_FINAL")
    plan = meta_parameter_plan(spec)
    log(f"  kinds={plan['counts']} tie_word_embeddings={plan['tie_word_embeddings']}")
    for kind, rows in plan["examples"].items():
        log(f"  examples {kind}: {rows}")
    gram_dir = (spec["output"] / "grams").resolve()
    log(f"  required Grams={len(plan['required_gram_names'])}")
    log(f"  ActivationAware fallback output>{MAX_ACTIVATION_OUTPUT}: {plan['activation_fallback']}")
    log("  PlainQ grams=no")
    log("  WeightedQ grams=no")
    log(f"  RegMean grams={gram_dir}")
    log(f"  ActivationAware grams={gram_dir} reuse=yes")
    if not all(has_full_checkpoint(path) for path in paths.values()):
        raise SystemExit(f"Dry-run expert check failed for {spec['label']} seed {spec['seed']}")
    read_orthomerge_scores(spec["orthomerge_csv"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PlainQ, WeightedQ, ActivationAware, and RegMean")
    parser.add_argument("--architectures", nargs="+", choices=["t5", "llama", "llama8b"], default=["t5", "llama"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHOD_ORDER))
    parser.add_argument("--gram-batch-size", type=int, default=4)
    parser.add_argument("--regmean-ridge", type=float, default=1e-4, help="Fraction of the mean Gram diagonal.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-grams", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Formula checks and module classification only.")
    return parser.parse_args()


def iter_specs(architectures: Sequence[str], seeds: Sequence[int]) -> List[dict]:
    return [architecture_spec(name, seed) for name in architectures for seed in seeds]


def main() -> None:
    args = parse_args()
    specs = iter_specs(args.architectures, args.seeds)
    if args.dry_run:
        self_check()
        expert_count = 0
        orthomerge_count = 0
        for spec in specs:
            dry_run_architecture(spec)
            expert_count += len(expert_paths(spec))
            orthomerge_count += 1
        log(f"DRY_RUN_OK experts={expert_count} orthomerge_csvs={orthomerge_count}")
        return
    for method in args.methods:
        for spec in specs:
            try:
                run_one(
                    spec,
                    method,
                    args.gram_batch_size,
                    args.regmean_ridge,
                    args.force,
                    args.force_grams,
                )
            except Exception as exc:
                log(
                    f"FAIL method={method} architecture={spec['label']} seed={spec['seed']} "
                    f"error={exc}. Later stages will not start."
                )
                raise
            if spec["name"] != "llama8b":
                update_three_seed_summary(spec["label"], args.seeds)
        log(f"END method={method} all requested architectures and seeds")


if __name__ == "__main__":
    main()
