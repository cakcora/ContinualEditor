#!/usr/bin/env python3
"""PGM (Posterior-Gated Merging) for the four full-finetune experts: merge, then evaluate.

This file contains only PGM.

Idea
    Each prompt names its task ("Task:TC Dataset:dbpedia ..."), so a merged model can tell the tasks
    apart from its own hidden states. PGM builds one model of the original size that
    (1) keeps, in every linear layer, the best single fixed matrix for all tasks (the static part), and
    (2) adds, in every feed-forward block, a few switched neurons per task: they turn on only for that
        task's tokens and move the merged model's hidden states back toward that task's expert.
    Everything is computed in closed form from the experts and unlabeled training prompts. No training,
    no task label at test time, no extra parameters (the switched neurons replace the neurons that
    contribute least on the four tasks). The derivation and proofs are in PGM_method.pdf.

Steps (one GPU)
    1. Static part. For every linear layer: S = (sum_i M_i G_i)(sum_i G_i + lam I)^-1, with M_i expert i's
       weight and G_i the average of x x^T over task i's tokens (x = the layer's input in expert i).
       Embeddings, norms and the output head: plain average.
    2. Switched part, block by block from the bottom (each block sees the corrected blocks below it):
       a. free the N*m feed-forward neurons that contribute least on the four tasks (N = 4 tasks);
       b. task switch: w_i = balanced least squares of +1 (task i) / -1 (other tasks) on the block's
          feed-forward input; switch s_i(x) = act(beta_i w_i^T x);
       c. target: expert i's block output minus the merged block output, on task i's tokens;
       d. rank-m correction V_i U_i^T by reduced-rank regression of the target on s_i(x) x;
       e. write it into task i's m neurons (gate rows beta_i w_i, up rows U_i, down columns V_i; T5's ReLU
          blocks use pairs of neurons), then refit all new down columns by least squares on the realized
          activations.
    3. Evaluate the static part ("PGM-static") and PGM on the four test sets.

Tasks (O-LoRA order 1): dbpedia (TC, 14 classes), amazon (SC, 5), yahoo (TC, 10), agnews (TC, 4)
Data: the CL_Benchmark folder of O-LoRA (https://github.com/cmnfriend/O-LoRA)
Score: greedy generation (50 new tokens), exact match after lower-casing and removing punctuation.

Inputs
    --experts-root DIR  with DIR/<task>/<W_DBPedia|W_Amazon|W_Yahoo|W_AGNews>, or --expert TASK=PATH (x4)
    --data-dir          the CL_Benchmark folder
    --arch t5|llama or --base PATH   (the base model only for the "Base" reference row)
Options
    --pgm-neurons m     switched neurons per task per block (default 16; T5 ReLU blocks use 2m)
    --calib-samples     unlabeled training prompts per task for the switched part (default 256)
    --gram-samples      prompts per task for the static part (default 1000; 0 = all)
    --static-from DIR   use a saved merged model as the static part instead of step 1 (e.g. AA+)
    --name NAME         row name (default PGM); give each setting its own name for ablations

Examples
    python pgm_merge_gpu.py --arch t5 --experts-root results/independent --data-dir CL_Benchmark
    python pgm_merge_gpu.py --arch llama --experts-root results/llama-3.2-1b/lr_2e-5/seed_42/independent \
        --data-dir CL_Benchmark --pgm-neurons 32 --name PGM-m32
    python pgm_merge_gpu.py --self-check        # algebra checks and PGM on tiny random models (CPU)
A run can be stopped and restarted: rows already in results.json are kept.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import random
import string
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
EXPERT_DIR = {"dbpedia": "W_DBPedia", "amazon": "W_Amazon", "yahoo": "W_Yahoo", "agnews": "W_AGNews"}
ARCHITECTURES = {
    "t5": {"base": "/data/cuneyt-topo/numpy_weights/t5-continual-learning/external_baselines/GORP/initial_model/t5-large",
           "model_type": "seq2seq", "eval_batch_size": 64},
    "llama": {"base": "meta-llama/Llama-3.2-1B", "model_type": "causal", "eval_batch_size": 32},
}

OPTIONS = {
    "neurons": 16,          # m: switched neurons per task per feed-forward block (ReLU blocks: 2m)
    "tau": 6.0,             # median switch pre-activation beta_i w_i^T x on task i's own tokens
    "ridge": 1e-4,          # relative ridge of the switched fits
    "readout_ridge": 1e-3,  # relative ridge of the task switch w_i
    "static_ridge": 1e-4,   # relative ridge of the static part
    "calib_samples": 256,   # unlabeled training prompts per task for the switched part
    "gram_samples": 1000,   # prompts per task for the static part (None: all)
    "batch_size": 16,
    "dtype": "bf16",        # expert / merged copies during the switched fit (GPU)
    "streams": "all",       # T5: all, encoder or decoder
    "offload": False,       # keep recorded block arguments on the CPU
}
# Settings that change the result; a results row records them.
SETTING_KEYS = ("neurons", "tau", "ridge", "readout_ridge", "static_ridge", "calib_samples", "gram_samples", "dtype", "streams")


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


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


# ============================================================================
# The starting merge
# ============================================================================

def average_state(task_states: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Plain average of the experts' weights, every parameter. A tied output head is the same matrix as
    the embedding in every expert, so the two averages are the same matrix too."""
    first = task_states[DATASETS[0]]
    return {key: torch.stack([task_states[task][key].float() for task in DATASETS], dim=0).mean(dim=0) for key in first}


def head_is_tied(model: nn.Module) -> bool:
    """Whether the output head IS the input embedding (shares its storage). The config flag is not
    enough: transformers 5 keeps tie_word_embeddings=True for T5 checkpoints whose head differs."""
    head, embedding = model.get_output_embeddings(), model.get_input_embeddings()
    return head is not None and head.weight.data_ptr() == embedding.weight.data_ptr()


# ============================================================================
# Small helpers used by the frame calibration
# ============================================================================

class StopForward(Exception):
    """Raised by a hook to stop a forward pass at a given module."""


def run_until_stop(model: nn.Module, inputs: dict) -> None:
    try:
        with torch.no_grad():
            model(**inputs)
    except StopForward:
        pass


# ============================================================================
# Static part: the best single fixed matrix for every linear layer
# ============================================================================
# For a linear layer with input x, expert weights M_i and task-i second moments G_i = E[x x^T | task i]
# (each task weighted equally), the risk  R(W) = sum_i tr((W - M_i) G_i (W - M_i)^T)  is minimised by
#     S = (sum_i M_i G_i)(sum_i G_i)^-1          (normal equations; a small ridge keeps it well posed)
# and every fixed W has R(W) = R(S) + tr((W - S) G (W - S)^T) >= R(S), G = sum_i G_i  (PGM_method.pdf, Thm 2).

def module_mask(model_type: str, module_name: str, encoder_mask, decoder_mask, length: int):
    """Which tokens a module's input rows belong to: T5 encoder modules and the cross-attention k, v read
    the encoder tokens; everything else reads the decoder tokens (the answer) or, for Llama, all tokens."""
    if model_type == "causal":
        mask = decoder_mask
    elif module_name.startswith("encoder.") or ".EncDecAttention.k" in module_name or ".EncDecAttention.v" in module_name:
        mask = encoder_mask
    else:
        mask = decoder_mask
    if mask is None or mask.shape[-1] != length:
        return None
    return mask


class GramCollector:
    """Sum of x x^T over the unpadded tokens for the input of every nn.Linear (not the output head).
    Modules that read the very same tensor (q, k, v; gate, up; T5's cross-attention k, v of all decoder
    blocks) share one sum: found on the first batch by object identity."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self.alias: Dict[str, str] = {}
        self.encoder_mask = self.decoder_mask = None
        self.first_batch = True
        self._seen: List[Tuple[torch.Tensor, str]] = []
        self.mismatches = 0

    def hook(self, name: str, inputs) -> None:
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_floating_point(x):
            return
        if self.first_batch:
            for tensor, owner in self._seen:
                if tensor is x:
                    self.alias[name] = owner
                    return
            self._seen.append((x, name))
        elif name in self.alias:
            return
        if x.ndim == 2:
            x = x.unsqueeze(1)
        mask = module_mask(self.model_type, name, self.encoder_mask, self.decoder_mask, x.shape[1])
        if mask is None:
            self.mismatches += 1
            return
        rows = x.detach()[mask].float()
        gram = torch.matmul(rows.t(), rows)
        if name in self.sums:
            self.sums[name] += gram
            self.counts[name] += rows.shape[0]
        else:
            self.sums[name] = gram
            self.counts[name] = rows.shape[0]

    def end_batch(self) -> None:
        self.first_batch = False
        self._seen = []


@torch.no_grad()
def collect_grams(model: nn.Module, loader, model_type: str, device: torch.device) -> dict:
    collector = GramCollector(model_type)
    handles = [module.register_forward_pre_hook(lambda _m, args, name=name: collector.hook(name, args))
               for name, module in model.named_modules() if isinstance(module, nn.Linear) and name != "lm_head"]
    try:
        for batch in loader:
            encoder_mask, decoder_mask = batch_masks(model_type, batch)
            collector.encoder_mask = None if encoder_mask is None else encoder_mask.to(device).bool()
            collector.decoder_mask = None if decoder_mask is None else decoder_mask.to(device).bool()
            model(**forward_inputs(batch, model_type, device))
            collector.end_batch()
    finally:
        for handle in handles:
            handle.remove()
    if collector.mismatches:
        raise SystemExit(f"Gram collection: {collector.mismatches} inputs without a matching token mask. Stopping.")
    return {"sums": collector.sums, "counts": collector.counts, "alias": collector.alias}


class StaticAccumulator:
    """Running G = sum_i G_i and C = sum_i M_i G_i (CPU), one task at a time, then S = C (G + lam I)^-1."""

    def __init__(self) -> None:
        self.gram: Dict[str, torch.Tensor] = {}
        self.cross: Dict[str, torch.Tensor] = {}
        self.alias: Optional[Dict[str, str]] = None

    def add(self, stats: dict, weights: Dict[str, torch.Tensor]) -> None:
        if self.alias is None:
            self.alias = dict(stats["alias"])
        elif self.alias != stats["alias"]:
            raise SystemExit("Gram collection: the experts' module input structure differs")
        members: Dict[str, List[str]] = {}
        for name, owner in self.alias.items():
            members.setdefault(owner, []).append(name)
        for key, total in stats["sums"].items():
            count = stats["counts"][key]
            if count == 0:
                continue
            gram = total / count
            self.gram[key] = self.gram.get(key, 0) + gram.cpu()
            for name in [key] + members.get(key, []):
                weight = weights[f"{name}.weight"].to(device=gram.device, dtype=torch.float32)
                product = torch.matmul(weight, gram).cpu()
                self.cross[name] = self.cross[name] + product if name in self.cross else product

    def solve(self, task_states: Dict[str, Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
        state = average_state(task_states)  # embeddings, norms, output head
        for name, cross in self.cross.items():
            gram = self.gram[self.alias.get(name, name)].to(device)
            ridge = OPTIONS["static_ridge"] * float(torch.trace(gram)) / gram.shape[0]
            eye = torch.eye(gram.shape[0], device=device)
            state[f"{name}.weight"] = torch.linalg.solve(gram + ridge * eye, cross.to(device).t()).t().cpu()
        return state


def static_part(task_states: Dict[str, Dict[str, torch.Tensor]], worker: nn.Module, loaders: dict, model_type: str,
                device: torch.device) -> Dict[str, torch.Tensor]:
    accumulator = StaticAccumulator()
    for task in DATASETS:
        started = time.time()
        worker.load_state_dict(task_states[task], strict=True)
        stats = collect_grams(worker, loaders[task], model_type, device)
        accumulator.add(stats, task_states[task])
        log(f"static part: {task} statistics for {len(stats['sums'])} inputs ({time.time() - started:.0f}s)")
        del stats
        if device.type == "cuda":
            torch.cuda.empty_cache()
    state = accumulator.solve(task_states, device)
    log(f"static part: {len(accumulator.cross)} linear layers solved")
    return state


# ============================================================================
# Switched part
# ============================================================================
# At every feed-forward block, for task i:
#   switch       s_i(x) = act(beta_i w_i^T x),  w_i: balanced least squares of +1 (task i) / -1 (others) on x
#   correction   s_i(x) V_i U_i^T x  =  output of m neurons with gate rows beta_i w_i, up rows U_i, down V_i
#   fit          reduced-rank regression of y = (expert block output - merged block output) on z = s_i(x) x
# The m*N neurons used are the ones that contribute least on the four tasks (their removal is part of y).
# ReLU blocks (T5): a pair  relu(a + b) - relu(a - b) = 2b if a >= |b|, 0 if a <= -|b|  gives the switch.

def pgm_layout(model: nn.Module, model_type: str) -> dict:
    modules = dict(model.named_modules())
    cfg = model.config
    d_model = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model"))

    def ffn(name: str) -> dict:
        module = modules.get(name)
        if module is None:
            raise SystemExit(f"PGM: no feed-forward module {name}")
        if all(hasattr(module, a) for a in ("gate_proj", "up_proj", "down_proj")):
            return {"name": name, "kind": "gated", "gate": f"{name}.gate_proj", "up": f"{name}.up_proj",
                    "down": f"{name}.down_proj", "act": module.act_fn}
        if all(hasattr(module, a) for a in ("wi_0", "wi_1", "wo")):
            return {"name": name, "kind": "gated", "gate": f"{name}.wi_0", "up": f"{name}.wi_1",
                    "down": f"{name}.wo", "act": module.act}
        if hasattr(module, "wi") and hasattr(module, "wo"):
            if not isinstance(module.act, nn.ReLU):
                raise SystemExit(f"PGM: {name} has a non-gated activation other than ReLU ({type(module.act).__name__})")
            return {"name": name, "kind": "relu", "gate": None, "up": f"{name}.wi", "down": f"{name}.wo", "act": module.act}
        raise SystemExit(f"PGM: unsupported feed-forward block {name}")

    streams = []
    if model_type == "causal":
        names = [f"model.layers.{k}" for k in range(len(model.model.layers))]
        streams.append({"name": "decoder", "blocks": names, "ffns": [ffn(f"{n}.mlp") for n in names],
                        "final_norm": "model.norm", "mask": "decoder"})
    else:
        enc = [f"encoder.block.{k}" for k in range(len(model.encoder.block))]
        dec = [f"decoder.block.{k}" for k in range(len(model.decoder.block))]
        streams.append({"name": "encoder", "blocks": enc, "ffns": [ffn(f"{n}.layer.1.DenseReluDense") for n in enc],
                        "final_norm": "encoder.final_layer_norm", "mask": "encoder"})
        streams.append({"name": "decoder", "blocks": dec, "ffns": [ffn(f"{n}.layer.2.DenseReluDense") for n in dec],
                        "final_norm": "decoder.final_layer_norm", "mask": "decoder"})
    return {"d": d_model, "streams": streams}


def _hidden_argument(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _move(obj, device: Optional[torch.device], memo: dict):
    """Tensors inside nested tuples/lists/dicts moved to `device` (None: unchanged); a tensor that
    appears several times (T5's position bias, shared by all blocks) is moved once."""
    if device is None:
        return obj
    if torch.is_tensor(obj):
        key = id(obj)
        if key not in memo:
            memo[key] = obj.to(device)
        return memo[key]
    if isinstance(obj, tuple):
        return tuple(_move(item, device, memo) for item in obj)
    if isinstance(obj, list):
        return [_move(item, device, memo) for item in obj]
    if isinstance(obj, dict):
        return {key: _move(value, device, memo) for key, value in obj.items()}
    return obj


def _record_stream(model: nn.Module, stream: dict, inputs: dict, offload: bool) -> dict:
    """One forward pass: the hidden state entering block 0, and the other arguments of every block call
    of the stream (masks, position bias or rotary embeddings, the encoder output for T5's decoder).
    None of these depend on the stream's hidden states, so a block can be called again later on other
    hidden states. The pass stops at the stream's final norm."""
    blocks = stream["blocks"]
    record = {"hidden": None, "positional": True, "calls": [None] * len(blocks), "offload": offload}
    memo: dict = {}
    store = torch.device("cpu") if offload else None

    def recorder(index: int):
        def hook(_module, args, kwargs):
            if index == 0:
                record["hidden"] = _hidden_argument(args, kwargs).detach()
                record["positional"] = bool(args)
            rest = tuple(args[1:]) if args else ()
            others = {key: value for key, value in kwargs.items() if args or key != "hidden_states"}
            record["calls"][index] = (_move(rest, store, memo), _move(others, store, memo))
        return hook

    def stop(_module, _args, _kwargs):
        raise StopForward()

    handles = [model.get_submodule(name).register_forward_pre_hook(recorder(index), with_kwargs=True)
               for index, name in enumerate(blocks)]
    handles.append(model.get_submodule(stream["final_norm"]).register_forward_pre_hook(stop, with_kwargs=True))
    try:
        run_until_stop(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    if record["hidden"] is None or any(call is None for call in record["calls"]):
        raise SystemExit(f"PGM: the {stream['name']} blocks were not all reached in the forward pass")
    return record


def _call_block(module: nn.Module, record: dict, index: int, device: torch.device) -> torch.Tensor:
    """Block `index` applied to the stored hidden state, with its recorded arguments (record unchanged)."""
    args, kwargs = record["calls"][index]
    if record["offload"]:
        memo: dict = {}
        args, kwargs = _move(args, device, memo), _move(kwargs, device, memo)
    if record["positional"]:
        out = module(record["hidden"], *args, **kwargs)
    else:
        out = module(*args, hidden_states=record["hidden"], **kwargs)
    return (out[0] if isinstance(out, (tuple, list)) else out).detach()


def _advance(module: nn.Module, record: dict, index: int, device: torch.device) -> None:
    record["hidden"] = _call_block(module, record, index, device)
    record["calls"][index] = None


def neuron_features(spec: dict, x: torch.Tensor, gate_w: Optional[torch.Tensor], up_w: torch.Tensor) -> torch.Tensor:
    """Activations (tokens x neurons) of the neurons with these gate / up rows (float32)."""
    if spec["kind"] == "gated":
        return spec["act"](torch.matmul(x, gate_w.t())) * torch.matmul(x, up_w.t())
    return F.relu(torch.matmul(x, up_w.t()))


def task_switches(xs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[float], List[dict]]:
    """w_i = balanced least squares of +1 on task i's tokens and -1 on the other tasks' tokens (each side
    weight 1/2, every task's tokens averaged); beta_i puts the median of beta_i w_i^T x on task i at tau."""
    n_tasks, d = len(xs), xs[0].shape[1]
    moments = [torch.matmul(x.t(), x) / x.shape[0] for x in xs]
    means = [x.mean(0) for x in xs]
    weights, betas, stats = [], [], []
    for i in range(n_tasks):
        others = [j for j in range(n_tasks) if j != i]
        gram = 0.5 * moments[i] + 0.5 * sum(moments[j] for j in others) / len(others)
        rhs = 0.5 * means[i] - 0.5 * sum(means[j] for j in others) / len(others)
        ridge = OPTIONS["readout_ridge"] * float(torch.trace(gram)) / d
        w = torch.linalg.solve(gram + ridge * torch.eye(d, device=gram.device), rhs)
        own = torch.matmul(xs[i], w)
        median = float(own.median())
        scale = median if median > 0 else max(float(own.abs().median()), 1e-12)
        beta = OPTIONS["tau"] / scale
        on = float((own > 0).float().mean())
        off = sum(float((torch.matmul(xs[j], w) < 0).float().mean()) for j in others) / len(others)
        weights.append(w)
        betas.append(beta)
        stats.append({"on_own": on, "off_others": off, "median_positive": median > 0})
    return weights, betas, stats


def reduced_rank_fit(zs: List[torch.Tensor], ys: List[Optional[torch.Tensor]], rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """min over rank(B) <= rank of  sum_j (1/n_j) ||Y_j - Z_j B^T||^2 + ridge ||B||^2  (Y_j = 0 when None).
    B_ls = (sum Y^T Z)(sum Z^T Z + lam I)^-1;  P = top eigenvectors of B_ls (Z^T Z + lam I) B_ls^T;
    B = P P^T B_ls = V U^T with V = P, U^T = P^T B_ls  (Anderson 1951, Izenman 1975; ridge = augmented data)."""
    d_in = zs[0].shape[1]
    zz = sum(torch.matmul(z.t(), z) / z.shape[0] for z in zs)
    yz = sum(torch.matmul(y.t(), z) / z.shape[0] for y, z in zip(ys, zs) if y is not None)
    ridge = OPTIONS["ridge"] * float(torch.trace(zz)) / d_in
    zz_r = zz + ridge * torch.eye(d_in, device=zz.device)
    b_ls = torch.linalg.solve(zz_r, yz.t()).t()                  # d_out x d_in
    fitted = torch.matmul(torch.matmul(b_ls, zz_r), b_ls.t())
    _evals, evecs = torch.linalg.eigh(fitted)
    p = evecs[:, -rank:]                                         # d_out x rank
    return p, torch.matmul(p.t(), b_ls).t()                      # V (d_out x rank), U (d_in x rank)


@torch.no_grad()
def pgm_block(stream: dict, index: int, runs: List[dict], workers: dict, state: Dict[str, torch.Tensor], ctx: dict) -> dict:
    """Switched neurons for one feed-forward block; advances the expert and merged records past it."""
    device, spec, block = ctx["device"], stream["ffns"][index], stream["blocks"][index]
    merged = workers["merged"]
    merged_block = merged.get_submodule(block)
    started = time.time()
    # 1. the merged block's feed-forward inputs x (unpadded tokens, float32), per task
    box: dict = {}
    handle = merged.get_submodule(spec["name"]).register_forward_pre_hook(lambda _m, args: box.__setitem__("x", args[0].detach()))
    xs: Dict[str, List[torch.Tensor]] = {task: [] for task in DATASETS}
    try:
        for run in runs:
            _call_block(merged_block, run["merged"], index, device)
            xs[run["task"]].append(box["x"][run["mask"]].float())
    finally:
        handle.remove()
    x_list = [torch.cat(xs[task]) for task in DATASETS]
    # 2. the neurons that contribute least on the four tasks
    down = merged.get_submodule(spec["down"])
    up = merged.get_submodule(spec["up"])
    gate = merged.get_submodule(spec["gate"]) if spec["gate"] else None
    up_w = up.weight.float()
    gate_w = gate.weight.float() if gate is not None else None
    width = up_w.shape[0]
    per_task = OPTIONS["neurons"] if spec["kind"] == "gated" else 2 * OPTIONS["neurons"]
    total = per_task * len(DATASETS)
    if total > width // 2:
        raise SystemExit(f"PGM: {total} switched neurons is more than half of {spec['name']} ({width})")
    energy = torch.zeros(width, device=device)
    for x in x_list:
        acc = torch.zeros(width, device=device)
        for start in range(0, x.shape[0], 4096):
            acc += neuron_features(spec, x[start:start + 4096], gate_w, up_w).square().sum(0)
        energy += acc / x.shape[0]
    energy *= down.weight.float().square().sum(0)
    spare = torch.argsort(energy)[:total]
    removed = float(energy[spare].sum() / energy.sum().clamp_min(1e-30))
    down.weight[:, spare] = 0
    # 3. targets: expert block output minus merged block output (spare neurons removed), per task
    ys: Dict[str, List[torch.Tensor]] = {task: [] for task in DATASETS}
    for run in runs:
        merged_out = _call_block(merged_block, run["merged"], index, device)
        _advance(workers["experts"][run["task"]].get_submodule(block), run["expert"], index, device)
        ys[run["task"]].append((run["expert"]["hidden"] - merged_out)[run["mask"]].float())
    y_list = [torch.cat(ys[task]) for task in DATASETS]
    # 4. task switches and the rank-m corrections
    switches, betas, gate_stats = task_switches(x_list)
    new_gate, new_up = [], []
    for i in range(len(DATASETS)):
        pre = [torch.matmul(x, switches[i]) for x in x_list]
        if spec["kind"] == "gated":
            values = [spec["act"](betas[i] * p) for p in pre]
        else:
            values = [(p > 0).float() for p in pre]
        zs = [v[:, None] * x for v, x in zip(values, x_list)]
        targets = [y_list[j] if j == i else None for j in range(len(DATASETS))]
        v_i, u_i = reduced_rank_fit(zs, targets, OPTIONS["neurons"])
        key = betas[i] * switches[i]
        if spec["kind"] == "gated":
            new_gate.append(key[None, :].expand(OPTIONS["neurons"], -1))
            new_up.append(u_i.t())
        else:  # pairs: relu(a + b/k) and relu(a - b/k), with |b|/k small next to a = beta w^T x on task i
            content = torch.matmul(x_list[i], u_i).abs()
            kappa = float(torch.quantile(content.flatten()[:1_000_000], 0.99)) / (0.25 * OPTIONS["tau"])
            kappa = kappa if kappa > 0 else 1.0
            for k in range(OPTIONS["neurons"]):
                new_up.append(torch.stack([key + u_i[:, k] / kappa, key - u_i[:, k] / kappa]))
    new_up = torch.cat(new_up)
    new_gate = torch.cat(new_gate) if new_gate else None
    # 5. all new down columns by least squares on the realized activations (each token's own target)
    feats = [neuron_features(spec, x, new_gate, new_up) for x in x_list]
    hh = sum(torch.matmul(f.t(), f) / f.shape[0] for f in feats)
    yh = sum(torch.matmul(y.t(), f) / f.shape[0] for y, f in zip(y_list, feats))
    ridge = OPTIONS["ridge"] * float(torch.trace(hh)) / hh.shape[0]
    new_down = torch.linalg.solve(hh + ridge * torch.eye(hh.shape[0], device=device), yh.t()).t()
    before = [float(y.square().sum(1).mean()) for y in y_list]
    after = [float((y - torch.matmul(f, new_down.t())).square().sum(1).mean()) for y, f in zip(y_list, feats)]
    # 6. write the neurons (merged worker and the float32 result)
    with torch.no_grad():
        up.weight[spare] = new_up.to(up.weight.dtype)
        down.weight[:, spare] = new_down.to(down.weight.dtype)
        if gate is not None:
            gate.weight[spare] = new_gate.to(gate.weight.dtype)
    for name, rows in ((spec["up"], new_up), (spec["gate"], new_gate)):
        if name:
            weight = state[f"{name}.weight"].clone()
            weight[spare.cpu()] = rows.cpu()
            state[f"{name}.weight"] = weight
    weight = state[f"{spec['down']}.weight"].clone()
    weight[:, spare.cpu()] = new_down.cpu()
    state[f"{spec['down']}.weight"] = weight
    # 7. move the merged records past this block with the new neurons
    for run in runs:
        _advance(merged_block, run["merged"], index, device)
    info = {"stream": stream["name"], "block": index, "neurons": total, "removed_energy": removed,
            "relative_error": sum(after) / max(sum(before), 1e-30),
            "per_task_relative_error": {task: a / max(b, 1e-30) for task, a, b in zip(DATASETS, after, before)},
            "switch_on_own": {task: s["on_own"] for task, s in zip(DATASETS, gate_stats)},
            "switch_off_others": {task: s["off_others"] for task, s in zip(DATASETS, gate_stats)},
            "seconds": time.time() - started}
    last = len(stream["blocks"]) - 1
    if index in (0, last) or index % 4 == 0:
        on = sum(info["switch_on_own"].values()) / len(DATASETS)
        off = sum(info["switch_off_others"].values()) / len(DATASETS)
        log(f"  PGM {stream['name']} {index}/{last}: block error x{info['relative_error']:.3f}, switch on own {on:.3f}, "
            f"off others {off:.3f}, removed energy {removed:.1e} ({info['seconds']:.0f}s)")
    return info


def pgm_workers(ctx: dict, static_state: Dict[str, torch.Tensor]) -> dict:
    """The four experts and one merged copy on the device, used to measure hidden states."""
    workers = ctx.get("pgm_workers")
    if workers is None:
        device = ctx["device"]
        dtype = torch.bfloat16 if (device.type == "cuda" and OPTIONS["dtype"] == "bf16") else torch.float32
        shell = load_model(ctx["model_path"], ctx["model_type"])
        experts = {}
        for task in DATASETS:
            expert = copy.deepcopy(shell)
            expert.load_state_dict(ctx["task_states"][task], strict=True)
            experts[task] = expert.to(device=device, dtype=dtype).eval()
        shell.load_state_dict(static_state, strict=True)
        workers = {"experts": experts, "merged": shell.to(device=device, dtype=dtype).eval(), "dtype": dtype}
        for model in list(experts.values()) + [workers["merged"]]:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        ctx["pgm_workers"] = workers
        log(f"PGM: experts and merged copy on {device} in {dtype}")
    return workers


@torch.no_grad()
def pgm_switched(static_state: Dict[str, torch.Tensor], ctx: dict) -> Tuple[Dict[str, torch.Tensor], List[dict]]:
    """Switched neurons in every feed-forward block, bottom-up (T5: encoder first, then decoder)."""
    layout, device, model_type = ctx["pgm_layout"], ctx["device"], ctx["model_type"]
    workers = pgm_workers(ctx, static_state)
    workers["merged"].load_state_dict(static_state, strict=True)
    state = dict(static_state)
    report: List[dict] = []
    started = time.time()
    for stream in layout["streams"]:
        if OPTIONS["streams"] not in ("all", stream["name"]):
            continue
        runs = []
        for task in DATASETS:
            for batch in ctx["pgm_loaders"][task]:
                inputs = dict(forward_inputs(batch, model_type, device), use_cache=False)
                encoder_mask, decoder_mask = batch_masks(model_type, batch)
                mask = (encoder_mask if stream["mask"] == "encoder" else decoder_mask).to(device).bool()
                runs.append({"task": task, "mask": mask,
                             "expert": _record_stream(workers["experts"][task], stream, inputs, OPTIONS["offload"]),
                             "merged": _record_stream(workers["merged"], stream, inputs, OPTIONS["offload"])})
        log(f"PGM {stream['name']}: {len(runs)} batches recorded ({time.time() - started:.0f}s)")
        for index in range(len(stream["blocks"])):
            report.append(pgm_block(stream, index, runs, workers, state, ctx))
        del runs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return state, report


def summarize(report: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for stream in sorted({r["stream"] for r in report}):
        rows = [r for r in report if r["stream"] == stream]
        out[stream] = {
            "blocks": len(rows),
            "mean_block_relative_error": sum(r["relative_error"] for r in rows) / len(rows),
            "mean_switch_on_own": sum(sum(r["switch_on_own"].values()) / len(DATASETS) for r in rows) / len(rows),
            "mean_switch_off_others": sum(sum(r["switch_off_others"].values()) / len(DATASETS) for r in rows) / len(rows),
            "neurons_per_block": rows[0]["neurons"],
            "per_block": rows}
    return out


# ============================================================================
# Self-checks (CPU, float64 algebra; tiny random models)
# ============================================================================

def pgm_algebra_check() -> None:
    """The identities and theorems of PGM_method.pdf, checked on random numbers."""
    torch.manual_seed(0)
    dt = torch.float64
    # Lemma: m neurons sharing the gate row beta*w output act(beta w^T x) * V U^T x
    d, m, n = 12, 3, 50
    x = torch.randn(n, d, dtype=dt)
    w, u, v = torch.randn(d, dtype=dt), torch.randn(d, m, dtype=dt), torch.randn(d, m, dtype=dt)
    for act in (F.silu, lambda z: F.gelu(z, approximate="tanh")):
        neurons = (act(torch.matmul(x, (2.0 * w)[:, None].expand(d, m))) * torch.matmul(x, u)) @ v.t()
        formula = act(2.0 * torch.matmul(x, w))[:, None] * (torch.matmul(x, u) @ v.t())
        if float((neurons - formula).abs().max()) > 1e-10:
            raise SystemExit("gated-neuron identity failed")
    # Lemma: relu(a + b) - relu(a - b) = 2b if a >= |b|, 0 if a <= -|b|, and never larger than 2|b|
    a, b = torch.randn(10000, dtype=dt) * 3, torch.randn(10000, dtype=dt)
    pair = F.relu(a + b) - F.relu(a - b)
    if float((pair - 2 * b)[a >= b.abs()].abs().max()) > 1e-12 or float(pair[a <= -b.abs()].abs().max()) > 1e-12 \
            or bool((pair.abs() > 2 * b.abs() + 1e-12).any()):
        raise SystemExit("ReLU pair identity failed")
    # Theorem: reduced-rank regression is the best rank-m fit (ridge = augmented data)
    z, y = torch.randn(200, d, dtype=dt), torch.randn(200, 8, dtype=dt)
    lam = 0.3
    zz = z.t() @ z + lam * torch.eye(d, dtype=dt)
    b_ls = torch.linalg.solve(zz, z.t() @ y).t()
    p = torch.linalg.eigh(b_ls @ zz @ b_ls.t())[1][:, -m:]
    best = p @ p.t() @ b_ls

    def loss(bm):
        return float((y - z @ bm.t()).square().sum() + lam * bm.square().sum())
    random_best = min(loss(torch.randn(8, m, dtype=dt) @ torch.randn(m, d, dtype=dt) * 0.1) for _ in range(2000))
    # local check: moving the rank-m factors V = P, U^T = P^T B_ls a little never lowers the loss
    left, right = p, p.t() @ b_ls
    perturbed = min(loss((left + 1e-3 * torch.randn(8, m, dtype=dt)) @ (right + 1e-3 * torch.randn(m, d, dtype=dt)))
                    for _ in range(500))
    if loss(best) > random_best + 1e-9 or loss(best) > perturbed + 1e-12:
        raise SystemExit("reduced-rank regression is not optimal")
    # Lemma: balanced +-1 least squares with an intercept gives the discriminant direction S_W^-1 (mu1 - mu0)
    x1, x0 = torch.randn(300, d, dtype=dt) + 1.0, torch.randn(900, d, dtype=dt) @ torch.randn(d, d, dtype=dt) * 0.3
    wts = torch.cat([torch.full((300,), 0.5 / 300, dtype=dt), torch.full((900,), 0.5 / 900, dtype=dt)])
    xx = torch.cat([torch.cat([x1, x0]), torch.ones(1200, 1, dtype=dt)], 1)
    target = torch.cat([torch.ones(300, dtype=dt), -torch.ones(900, dtype=dt)])
    coef = torch.linalg.solve((xx * wts[:, None]).t() @ xx, (xx * wts[:, None]).t() @ target)[:d]
    mu1, mu0 = x1.mean(0), x0.mean(0)
    s_w = 0.5 * torch.cov(x1.t(), correction=0) + 0.5 * torch.cov(x0.t(), correction=0)
    lda = torch.linalg.solve(s_w, mu1 - mu0)
    if float(F.cosine_similarity(coef, lda, dim=0)) < 1 - 1e-10:
        raise SystemExit("least squares / discriminant direction lemma failed")
    # Theorem: R(W) = R(S) + tr((W - S) G (W - S)^T) for the static part
    grams = [(lambda q: q @ q.t() / d)(torch.randn(d, d, dtype=dt)) for _ in range(4)]
    experts = [torch.randn(6, d, dtype=dt) for _ in range(4)]
    g = sum(grams)
    s = torch.linalg.solve(g, sum(mm @ gi for mm, gi in zip(experts, grams)).t()).t()

    def risk(wm):
        return float(sum(torch.trace((wm - mm) @ gi @ (wm - mm).t()) for mm, gi in zip(experts, grams)))
    for _ in range(5):
        wm = torch.randn(6, d, dtype=dt)
        if abs(risk(wm) - risk(s) - float(torch.trace((wm - s) @ g @ (wm - s).t()))) > 1e-8 * max(1.0, risk(wm)):
            raise SystemExit("static-part decomposition failed")
    log("PGM_ALGEBRA_CHECK_OK")


def _tiny_models(seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM, T5Config, T5ForConditionalGeneration
    torch.manual_seed(seed)
    cases = [
        ("causal", LlamaForCausalLM, LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=128, num_hidden_layers=3,
                                                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)),
        ("seq2seq", T5ForConditionalGeneration, T5Config(vocab_size=64, d_model=32, d_ff=128, d_kv=8, num_layers=2,
                                                          num_decoder_layers=2, num_heads=4, decoder_start_token_id=0)),
        ("seq2seq", T5ForConditionalGeneration, T5Config(vocab_size=64, d_model=32, d_ff=128, d_kv=8, num_layers=2,
                                                          num_decoder_layers=2, num_heads=4, decoder_start_token_id=0,
                                                          feed_forward_proj="gated-gelu", tie_word_embeddings=False)),
    ]
    out = []
    for model_type, cls, config in cases:
        base = cls(config).eval()
        with torch.no_grad():
            emb = base.get_input_embeddings().weight
            emb.add_(1.5 * emb.norm(dim=1).mean() * F.normalize(torch.randn(emb.shape[1]), dim=0)[None, :])
        states = {}
        for task in DATASETS:
            expert = copy.deepcopy(base)
            with torch.no_grad():
                for name, parameter in expert.named_parameters():
                    if parameter.ndim == 2 and "embed" not in name and "shared" not in name:
                        parameter.add_(0.3 * parameter.std() * torch.randn_like(parameter))
            states[task] = {k: t.detach().clone() for k, t in expert.state_dict().items()}
        out.append((model_type, cls, config, base, states))
    return out


def _tiny_task_batches(model_type: str, task_index: int, count: int, generator: torch.Generator) -> List[dict]:
    batches = []
    for _ in range(count):
        low = 8 + 12 * task_index
        ids = torch.cat([torch.full((4, 1), 2 + task_index), torch.randint(low, low + 24, (4, 11), generator=generator) % 64], 1)
        mask = torch.ones(4, 12, dtype=torch.long)
        batch = {"input_ids": ids, "attention_mask": mask}
        if model_type == "seq2seq":
            mask[1, -3:] = 0
            labels = torch.randint(8, 64, (4, 3), generator=generator)
            labels[2, 2:] = -100
            batch["labels"] = labels
            batch["decoder_input_ids"] = torch.cat([torch.zeros(4, 1, dtype=torch.long), labels[:, :-1].clamp_min(0)], 1)
        else:
            mask[1, :3] = 0
        batches.append(batch)
    return batches


@torch.no_grad()
def _logit_error(model: nn.Module, experts: Dict[str, nn.Module], loaders: dict, model_type: str) -> float:
    total = 0.0
    for task in DATASETS:
        num = den = 0.0
        for batch in loaders[task]:
            inputs = forward_inputs(batch, model_type, torch.device("cpu"))
            mine, ref = model(**inputs).logits, experts[task](**inputs).logits
            num += float((mine - ref).square().sum())
            den += float(ref.square().sum())
        total += num / den
    return total / len(DATASETS)


def pgm_model_check() -> None:
    """PGM on tiny random Llama and T5 models (ReLU and gated): every block's calibration error goes down,
    and the held-out logits move toward the experts'."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        log("PGM_MODEL_CHECK: transformers not installed, skipped")
        return
    saved = dict(OPTIONS)
    try:
        OPTIONS.update(neurons=2, dtype="fp32", streams="all", offload=False)
        for model_type, cls, config, base, states in _tiny_models():
            generator = torch.Generator().manual_seed(3)
            calib = {task: _tiny_task_batches(model_type, i, 6, generator) for i, task in enumerate(DATASETS)}
            held_out = {task: _tiny_task_batches(model_type, i, 3, generator) for i, task in enumerate(DATASETS)}
            worker = copy.deepcopy(base)
            static = static_part(states, worker, calib, model_type, torch.device("cpu"))
            experts = {}
            for task in DATASETS:
                experts[task] = copy.deepcopy(base)
                experts[task].load_state_dict(states[task])
                experts[task].eval()
            merged = copy.deepcopy(base)
            merged.load_state_dict(static)
            ctx = {"device": torch.device("cpu"), "model_type": model_type, "task_states": states,
                   "pgm_layout": pgm_layout(base, model_type), "pgm_loaders": calib,
                   "pgm_workers": {"experts": experts, "merged": merged, "dtype": torch.float32}}
            state, report = pgm_switched(static, ctx)
            worst = max(r["relative_error"] for r in report)
            if worst > 1.0 + 1e-6:
                raise SystemExit(f"PGM raised a block's calibration error ({cls.__name__}: x{worst:.3f})")
            check = copy.deepcopy(base)
            check.load_state_dict(static)
            static_error = _logit_error(check.eval(), experts, held_out, model_type)
            check.load_state_dict(state)
            pgm_error = _logit_error(check.eval(), experts, held_out, model_type)
            average = copy.deepcopy(base)
            average.load_state_dict(average_state(states))
            average_error = _logit_error(average.eval(), experts, held_out, model_type)
            kind = ctx["pgm_layout"]["streams"][0]["ffns"][0]["kind"]
            log(f"PGM check {cls.__name__} ({kind} feed-forward): worst block error x{worst:.3f}; held-out logit error "
                f"average {average_error:.4f}, static part {static_error:.4f}, PGM {pgm_error:.4f}")
    finally:
        OPTIONS.clear()
        OPTIONS.update(saved)
    log("PGM_MODEL_CHECK_OK")


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


def score_state(state: Dict[str, torch.Tensor], ctx: dict, tasks: Sequence[str] = DATASETS,
                record: Optional[Path] = None) -> Dict[str, float]:
    model = ctx["eval_model"]
    model.load_state_dict(state, strict=True)  # PGM changes only feed-forward rows; embeddings and head stay tied
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
    """Reference rows first, then the static part and the PGM rows by name."""
    rank = {name: index for index, name in enumerate(["Base", "Experts (own task)"] + [f"Expert {t}" for t in DATASETS] + ["PGM-static"])}
    return {name: results[name] for name in sorted(results, key=lambda n: (rank.get(n, len(rank)), n))}


def save_rows(out: Path, rows: Dict[str, dict]) -> Dict[str, dict]:
    """Add rows to results.json/csv under a file lock, keeping rows written by other processes."""
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


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PGM merge of the DBpedia/Amazon/Yahoo/AG News experts, then evaluation.")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), help="t5 or llama: sets the default base model and model type.")
    parser.add_argument("--base", help="Base model (only for the Base reference row; default from --arch).")
    parser.add_argument("--model-type", choices=["seq2seq", "causal"], help="Default: from --arch or the expert config.")
    parser.add_argument("--experts-root", help="Folder with <task>/<W_DBPedia|W_Amazon|W_Yahoo|W_AGNews> (project layout).")
    parser.add_argument("--expert", action="append", default=[], metavar="TASK=PATH", help="Expert checkpoint for one task (overrides --experts-root).")
    parser.add_argument("--data-dir", help="O-LoRA CL_Benchmark folder.")
    parser.add_argument("--output", help="Output folder (default: pgm_results/<arch or model type>).")
    parser.add_argument("--name", help="Row name of the merged model (default PGM). Give each setting its own name for ablations.")
    parser.add_argument("--static-from", metavar="DIR",
                        help="Use this saved merged model as the static part (e.g. AA+ saved with --save-merged).")
    parser.add_argument("--static-name", help="Row name of --static-from (default: its folder name).")
    parser.add_argument("--pgm-neurons", type=int, default=16, help="m: switched neurons per task per block (T5 ReLU blocks: 2m).")
    parser.add_argument("--pgm-tau", type=float, default=6.0, help="Median switch pre-activation on the task's own tokens.")
    parser.add_argument("--pgm-ridge", type=float, default=1e-4, help="Relative ridge of the switched fits.")
    parser.add_argument("--readout-ridge", type=float, default=1e-3, help="Relative ridge of the task switches.")
    parser.add_argument("--static-ridge", type=float, default=1e-4, help="Relative ridge of the static part.")
    parser.add_argument("--calib-samples", type=int, default=256, help="Unlabeled training prompts per task for the switched part.")
    parser.add_argument("--gram-samples", type=int, default=1000, help="Training prompts per task for the static part (0 = all).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="Expert / merged copies during the switched fit (GPU).")
    parser.add_argument("--streams", choices=["all", "encoder", "decoder"], default="all", help="T5: switched neurons in both stacks or one.")
    parser.add_argument("--offload", action="store_true", help="Keep recorded block arguments on the CPU.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for choosing the prompt subsets.")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cuda:N; 'cpu' only for small tests.")
    parser.add_argument("--eval-samples", type=int, default=None, help="Test examples per task (default: all).")
    parser.add_argument("--eval-split", default="test", choices=["test", "dev"])
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--skip-reference", action="store_true", help="Do not evaluate the base model and the experts.")
    parser.add_argument("--skip-static", action="store_true", help="Do not evaluate the static part on its own.")
    parser.add_argument("--static-only", action="store_true",
                        help="Score PGM-static and exit. Do not build or evaluate the switched model.")
    parser.add_argument("--experts-all-tasks", action="store_true", help="Evaluate every expert on all four tasks (default: own task).")
    parser.add_argument("--save-merged", action="store_true", help="Save the merged model under <output>/merged/<name>.")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute rows already in results.json.")
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls during evaluation only (faster generation).")
    parser.add_argument("--self-check", action="store_true", help="Algebra checks and PGM on tiny random models (CPU), then exit.")
    return parser.parse_args()


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    OPTIONS.update({
        "neurons": args.pgm_neurons, "tau": args.pgm_tau, "ridge": args.pgm_ridge, "readout_ridge": args.readout_ridge,
        "static_ridge": args.static_ridge, "calib_samples": args.calib_samples,
        "gram_samples": None if args.gram_samples == 0 else args.gram_samples, "batch_size": args.batch_size,
        "dtype": args.dtype, "streams": args.streams, "offload": args.offload,
    })
    if args.self_check:
        pgm_algebra_check()
        pgm_model_check()
        return
    if args.static_only and args.skip_static:
        raise SystemExit("--static-only scores PGM-static, so it cannot be combined with --skip-static")
    if not args.data_dir:
        raise SystemExit("--data-dir (the CL_Benchmark folder) is required")
    if args.pgm_neurons < 1:
        raise SystemExit("--pgm-neurons must be at least 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This script is meant for a GPU (use --device cpu only for small tests).")
    torch.backends.cuda.matmul.allow_tf32 = False  # statistics and fits: full fp32
    torch.backends.cudnn.allow_tf32 = False
    RUN["tf32_eval"] = bool(args.tf32)
    arch = ARCHITECTURES.get(args.arch or "", {})
    base_path = args.base or arch.get("base")
    paths = expert_paths_from(args)
    model_type = args.model_type or arch.get("model_type") or infer_model_type(paths[DATASETS[0]])
    if args.static_from and not Path(args.static_from).exists():
        raise SystemExit(f"--static-from not found: {args.static_from}")
    static_name = (args.static_name or Path(args.static_from).name) if args.static_from else "PGM-static"
    name = args.name or "PGM"
    if name == static_name or name in ("Base", "Experts (own task)"):
        raise SystemExit(f"--name {name} is already used by another row")
    settings = {"static": static_name, "seed": args.seed, **{key: OPTIONS[key] for key in SETTING_KEYS}}
    out = Path(args.output or Path("pgm_results") / (args.arch or model_type))
    out.mkdir(parents=True, exist_ok=True)
    results = read_results(out / "results.json")
    if name in results and not args.force and results[name].get("settings") not in (None, settings):
        raise SystemExit(f"results.json already has a row {name} with other settings ({results[name]['settings']}); "
                         "give another --name or use --force")
    if args.static_only and not args.force:
        done = results.get(static_name, {}).get("scores") or {}
        if "PGM" in results:
            raise SystemExit(
                f"{out / 'results.json'} already contains a switched PGM row; "
                "refusing to treat this directory as static-only"
            )
        if all(task in done for task in DATASETS):
            log(f"{static_name}: already complete in {out / 'results.json'}; switched part was not built")
            print()
            print_table(results)
            return
    log(f"model_type={model_type}, static={static_name}, row={name}, output={out}")
    for task in DATASETS:
        log(f"expert {task}: {paths[task]}")

    # ---- data: unlabeled training prompts for the statistics (the answer text is part of the prompt
    #      for Llama and the decoder input for T5, exactly as in training), test sets for evaluation
    tokenizer = load_tokenizer([paths[DATASETS[0]]] + ([base_path] if base_path else []), model_type)
    train = {task: load_instances(Path(args.data_dir), task, "train") for task in DATASETS}
    calibration = (
        {task: [] for task in DATASETS} if args.static_only
        else {task: subsample(train[task], OPTIONS["calib_samples"], args.seed) for task in DATASETS}
    )
    gram_set = {task: subsample(train[task], OPTIONS["gram_samples"], args.seed) for task in DATASETS}
    test = {task: subsample(load_instances(Path(args.data_dir), task, args.eval_split), args.eval_samples, args.seed)
            for task in DATASETS}
    for task in DATASETS:
        log(f"data {task}: switched part {len(calibration[task])}, static part {len(gram_set[task])}, "
            f"{args.eval_split} {len(test[task])}")

    # ---- the experts (CPU, fp32) and the evaluation copy (GPU)
    log("Loading the experts")
    task_states = {}
    for task in DATASETS:
        model = load_model(paths[task], model_type)
        task_states[task] = cpu_state(model)
        del model
        gc.collect()
    for task in DATASETS[1:]:
        if task_states[task].keys() != task_states[DATASETS[0]].keys():
            raise SystemExit(f"Expert {task} has different parameters from {DATASETS[0]}")
    eval_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.eval_dtype]
    eval_model = load_model(paths[DATASETS[0]], model_type).to(device=device, dtype=eval_dtype).eval()
    for parameter in eval_model.parameters():
        parameter.requires_grad_(False)
    collator = UIECollator(tokenizer, eval_model, model_type, args.max_source_length, args.max_target_length)
    eval_bs = args.eval_batch_size or arch.get("eval_batch_size", 32)
    ctx = {
        "model_type": model_type, "device": device, "task_states": task_states, "eval_model": eval_model,
        "tokenizer": tokenizer, "max_new_tokens": args.max_new_tokens, "model_path": paths[DATASETS[0]],
        "eval_dtype": eval_dtype,
        "test_loaders": {task: make_loader(test[task], collator, eval_bs) for task in DATASETS},
    }
    record_root = out / "predictions" if args.save_predictions else None

    # ---- reference rows
    if not args.skip_reference:
        if base_path and ("Base" not in results or args.force):
            started = time.time()
            base_model = load_model(base_path, model_type)
            w0 = cpu_state(base_model)
            del base_model
            align_base_rows(w0, task_states[DATASETS[0]])
            w0 = {key: w0[key] for key in task_states[DATASETS[0]]}
            results = save_rows(out, {"Base": {"scores": score_state(w0, ctx), "seconds": time.time() - started}})
            log(f"Base {results['Base']['scores']}")
            del w0
            gc.collect()
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
        log(f"Experts (own task) {results['Experts (own task)']['scores']}")

    if name in results and not args.force and (args.skip_static or static_name in results):
        log(f"{name}: already in results.json")
        print()
        print_table(results)
        return

    # ---- static part
    started = time.time()
    if args.static_from:
        model = load_model(args.static_from, model_type)
        static = cpu_state(model)
        del model
        if static.keys() != task_states[DATASETS[0]].keys():
            raise SystemExit(f"--static-from {args.static_from} has different parameters from the experts")
    else:
        worker = eval_model if eval_dtype == torch.float32 else load_model(paths[DATASETS[0]], model_type).to(device).eval()
        gram_loaders = {task: make_loader(gram_set[task], collator, OPTIONS["batch_size"]) for task in DATASETS}
        static = static_part(task_states, worker, gram_loaders, model_type, device)
        if worker is not eval_model:
            del worker
        free_memory(device)
    static_seconds = time.time() - started
    if not args.skip_static and (static_name not in results or args.force):
        record = record_root / static_name if record_root else None
        scores = score_state(static, ctx, record=record)
        results = save_rows(out, {static_name: {"scores": scores, "seconds": time.time() - started,
                                                "merge_seconds": static_seconds}})
        log(f"{static_name} {scores}")
    if args.static_only:
        print()
        print_table(results)
        log(f"Results: {out / 'results.csv'}")
        log("static-only: switched part was not built")
        return

    # ---- switched part
    if name in results and not args.force:
        log(f"{name}: already in results.json")
    else:
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        ctx["pgm_layout"] = pgm_layout(eval_model, model_type)
        ctx["pgm_loaders"] = {task: make_loader(calibration[task], collator, OPTIONS["batch_size"]) for task in DATASETS}
        merged, report = pgm_switched(static, ctx)
        switched_seconds = time.time() - started
        ctx.pop("pgm_workers", None)
        free_memory(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        record = record_root / name if record_root else None
        scores = score_state(merged, ctx, record=record)
        row = {"scores": scores, "seconds": time.time() - started, "merge_seconds": static_seconds + switched_seconds,
               "peak_gpu_bytes": peak_bytes, "settings": settings, "pgm": summarize(report)}
        results = save_rows(out, {name: row})
        log(f"{name} {scores} (switched part {switched_seconds:.0f}s, peak {peak_bytes / (1024 ** 2):.0f} MiB)")
        if args.save_merged:
            dest = out / "merged" / name
            eval_model.load_state_dict(merged, strict=True)
            eval_model.save_pretrained(dest)
            tokenizer.save_pretrained(dest)
            log(f"Saved {dest}")
    print()
    print_table(results)
    log(f"Results: {out / 'results.csv'}")


if __name__ == "__main__":
    main()
