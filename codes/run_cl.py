#!/usr/bin/env python3
"""Sequential CL (O1/O2/O3) and independent W0 -> W_task SeqFT.

Reuses the existing GORP/SeqFT UIE pipeline from codes/train.py.
No merge logic lives here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

ROOT = Path(__file__).resolve().parent
CL_CODES = Path("/home/kadir/topo/numpy_weights/t5-continual-learning/codes")
if str(CL_CODES) not in sys.path:
    sys.path.insert(0, str(CL_CODES))

import train as cl_train
from transformers import set_seed

RESULTS = ROOT / "results"
DATASETS = ("dbpedia", "amazon", "yahoo", "agnews")
CL_ORDERS = {
    "O1": 1,
    "O2": 2,
    "O3": 3,
}
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

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
GRAD_ACCUM = 1
EPOCHS = 1
SEED = 42
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
LR_SCHEDULER = "constant"
MODEL_TYPE = "seq2seq"
MODEL_NAME = "t5-large"
W0 = Path(cl_train.INITIAL_MODEL)
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-1B"


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
    """Resolve one run. Defaults are the completed T5-Large 1e-4 setup."""
    global SEED, RESULTS, LEARNING_RATE, W0, MODEL_TYPE, MODEL_NAME
    global BATCH_SIZE, EVAL_BATCH_SIZE, GRAD_ACCUM
    SEED = int(seed)
    LEARNING_RATE = float(learning_rate)
    if results_root is not None:
        RESULTS = Path(results_root)
    else:
        RESULTS = ROOT / "results"
    if model_type is None:
        model_type = "causal" if model and "llama" in model.lower() else "seq2seq"
    if model_type not in {"seq2seq", "causal"}:
        raise SystemExit(f"Unknown model type: {model_type}")
    MODEL_TYPE = model_type
    if model is None:
        if MODEL_TYPE == "causal":
            model = LLAMA_MODEL_ID
        else:
            model = str(cl_train.INITIAL_MODEL)
    MODEL_NAME = model
    W0 = Path(model) if Path(model).exists() else Path(model)
    if MODEL_TYPE == "seq2seq" and not W0.exists():
        W0 = Path(cl_train.INITIAL_MODEL)
    if MODEL_TYPE == "causal":
        # Physical 32 x accumulation 2. Effective batch stays 64.
        BATCH_SIZE = 32
        GRAD_ACCUM = 2
        EVAL_BATCH_SIZE = 8
    else:
        BATCH_SIZE = 64
        GRAD_ACCUM = 1
        EVAL_BATCH_SIZE = 128
    apply_seed(SEED)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


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


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
    scores: Dict[str, Optional[float]] = {}
    for task in tasks:
        key = f"predict_exact_match_for_{task}"
        scores[task] = float(metrics[key]) if key in metrics else None
    return scores


def configure_full_finetune(model) -> None:
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def model_source() -> str:
    """Local checkpoint path, or the Hugging Face id when it is not on disk yet."""
    if W0.exists():
        return str(W0.resolve())
    return MODEL_NAME


def load_causal_model(model_path: str):
    """Llama full fine-tune loader. Pad is EOS. Embeddings stay at vocab 128256."""
    from model.llama import LlamaForCausalLM_with_lossmask
    from transformers import AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = LlamaForCausalLM_with_lossmask.from_pretrained(model_path, torch_dtype=dtype)
    embed_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embed_rows:
        raise RuntimeError(
            f"Tokenizer length {len(tokenizer)} does not match embedding rows {embed_rows}. "
            "Refusing to add tokens or resize embeddings."
        )
    if "llama" not in str(model.config._name_or_path).lower():
        model.config._name_or_path = model_path
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def load_run_model(model_path: str):
    if MODEL_TYPE == "causal":
        return load_causal_model(model_path)
    return cl_train.load_model_and_tokenizer(Path(model_path))


def build_causal_trainer(model, tokenizer, train_dataset, predict_dataset, output_dir: Path, do_train: bool):
    """LlamaCLTrainer plus the UIE decoder collator. Exact match is the T5 metric."""
    from compute_metrics import compute_grouped_metrics, compute_metrics
    from train_llama_cl import LlamaCLTrainer
    from transformers import TrainingArguments
    from uie_collator import DataCollatorForUIE
    from uie_trainer_lora import skip_instructions

    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        examples = [dataset[i] for i in range(len(dataset))]
        references = [example["Instance"]["label"] for example in examples]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result.update(compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"]))
        result.update(compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Dataset"]))
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = float(np.mean(prediction_lens))
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            pred_path = output_dir / f"{save_prefix}_eval_predictions.jsonl"
            with pred_path.open("w", encoding="utf-8") as fout:
                for example, pred in zip(examples, decoded_preds):
                    fout.write(
                        json.dumps(
                            {
                                "Task": example["Task"],
                                "Dataset": example["Dataset"],
                                "Instance": example["Instance"],
                                "Prediction": pred,
                            }
                        )
                        + "\n"
                    )
        return result

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=do_train,
        do_eval=False,
        do_predict=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=0,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=512,
        max_target_length=50,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.bf16 else None,
        add_task_name=True,
        add_dataset_name=True,
        num_examples=0,
    )
    return LlamaCLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if do_train else None,
        eval_dataset=predict_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        rouge_metrics_fn=compute_rouge_metrics,
        max_new_tokens=50,
        num_beams=1,
    )


def build_uie_trainer(model, tokenizer, train_dataset, predict_dataset, output_dir: Path, do_train: bool):
    return cl_train.build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        predict_dataset=predict_dataset,
        output_dir=output_dir,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        do_train=do_train,
        do_predict=True,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        lr_scheduler_type=LR_SCHEDULER,
        gradient_accumulation_steps=GRAD_ACCUM,
    )


def train_task(
    *,
    init_path: Path,
    task: str,
    order: int,
    output_dir: Path,
    config_dir: Path,
    force: bool,
) -> Path:
    model_dir = output_dir / "seqft_model"
    predict_path = output_dir / "predict_results.json"
    if not force and has_full_checkpoint(model_dir) and predict_path.exists():
        log(f"Reusing {output_dir}")
        return model_dir

    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_run_model(str(init_path))
    configure_full_finetune(model)
    raw = cl_train.load_uie_datasets(
        data_dir=cl_train.data_root_for_order(order),
        task_config_dir=config_dir,
        output_dir=output_dir / "dataset_cache",
        overwrite_cache=False,
    )
    train_dataset, original_size, capped_size = cl_train.cap_train_dataset(
        raw["train"],
        max_train_samples=None,
        subsample_seed=SEED,
        logger=None,
    )
    if MODEL_TYPE == "causal":
        trainer = build_causal_trainer(model, tokenizer, train_dataset, raw["test"], output_dir, do_train=True)
    else:
        trainer = build_uie_trainer(model, tokenizer, train_dataset, raw["test"], output_dir, do_train=True)
    log(
        f"  train {task} init={init_path} samples={capped_size}/{original_size} "
        f"lr={LEARNING_RATE} seed={SEED} epochs={EPOCHS} "
        f"batch={BATCH_SIZE} accum={GRAD_ACCUM}"
    )
    train_result = trainer.train()
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(model_dir, safe_serialization=False)
    tokenizer.save_pretrained(model_dir)
    train_metrics = dict(train_result.metrics)
    train_metrics.update(
        {
            "train_samples": capped_size,
            "original_train_samples": original_size,
            "init_from": str(init_path.resolve()) if Path(init_path).exists() else str(init_path),
            "w0": model_source(),
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "model_type": MODEL_TYPE,
            "model": MODEL_NAME,
        }
    )
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    predict_results = trainer.predict(raw["test"], metric_key_prefix="predict", max_new_tokens=50, num_beams=1)
    metrics = dict(predict_results.metrics)
    metrics["predict_samples"] = len(raw["test"])
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    del trainer, model
    return model_dir


def evaluate_checkpoint(model_path: Path, tasks: Sequence[str], output_dir: Path, force: bool) -> Dict[str, Optional[float]]:
    scores_path = output_dir / "eval_scores.json"
    if not force and scores_path.exists():
        payload = json.loads(scores_path.read_text())
        if all(task in payload.get("scores", {}) for task in tasks):
            log(f"Reusing eval {scores_path}")
            return payload["scores"]

    set_seed(SEED)
    model, tokenizer = load_run_model(str(model_path))
    scores: Dict[str, Optional[float]] = {}
    for task in tasks:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        raw = cl_train.load_uie_datasets(
            data_dir=cl_train.data_root_for_order(1),
            task_config_dir=single_task_config_dir(task),
            output_dir=task_dir / "dataset_cache",
            overwrite_cache=False,
        )
        if MODEL_TYPE == "causal":
            trainer = build_causal_trainer(model, tokenizer, None, raw["test"], task_dir, do_train=False)
        else:
            trainer = build_uie_trainer(model, tokenizer, None, raw["test"], task_dir, do_train=False)
        log(f"  eval {model_path} on {task} n={len(raw['test'])}")
        predict_results = trainer.predict(raw["test"], metric_key_prefix="predict", max_new_tokens=50, num_beams=1)
        metrics = dict(predict_results.metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        scores.update(extract_accuracies(metrics, [task]))
        model = trainer.model
    save_json({"model_path": str(Path(model_path).resolve()), "scores": scores}, scores_path)
    del model
    return scores


def publish_checkpoint(src: Path, dest: Path) -> None:
    """Keep one permanent copy. seqft_model and W_<Task> are the same checkpoint."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))


def run_cl(force: bool) -> None:
    log("CL: sequential full fine-tune from the shared W0")
    log(f"W0={model_source()}")
    for order_name, order_id in CL_ORDERS.items():
        tasks = cl_train.tasks_for_order(order_id)
        order_dir = RESULTS / "cl" / order_name
        final_dir = order_dir / f"W{len(tasks)}_after_{tasks[-1]}"
        csv_path = order_dir / "results.csv"
        if not force and has_full_checkpoint(final_dir) and csv_path.exists():
            log(f"{order_name} already complete")
            continue

        order_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        init_path = W0
        for round_idx, task in enumerate(tasks, start=1):
            if round_idx == 1 and str(init_path) != str(W0):
                raise RuntimeError(f"{order_name} task 1 must start from W0")
            round_dir = order_dir / f"{round_idx}-{task}"
            stage_dir = order_dir / f"W{round_idx}_after_{task}"
            log(f"{order_name} stage {round_idx}/{len(tasks)} train={task} init={init_path}")
            predict_path = round_dir / "predict_results.json"
            if force or not (has_full_checkpoint(stage_dir) and predict_path.exists()):
                model_dir = train_task(
                    init_path=init_path,
                    task=task,
                    order=order_id,
                    output_dir=round_dir,
                    config_dir=cl_train.task_config_dir(order_id, task),
                    force=force,
                )
                if stage_dir.exists():
                    shutil.rmtree(stage_dir)
                shutil.move(str(model_dir), str(stage_dir))
            init_path = stage_dir
            metrics = json.loads(predict_path.read_text())
            seen = tasks[:round_idx]
            scores = extract_accuracies(metrics, seen)
            for eval_task in seen:
                rows.append(
                    {
                        "order": order_name,
                        "training_stage": round_idx,
                        "current_task": task,
                        "eval_dataset": eval_task,
                        "accuracy": scores.get(eval_task, ""),
                    }
                )

        write_csv(
            csv_path,
            ["order", "training_stage", "current_task", "eval_dataset", "accuracy"],
            rows,
        )
        log(f"{order_name} final weights -> {final_dir}")


def run_independent(force: bool) -> None:
    log("Independent: each task reloads the same W0")
    log(f"W0={model_source()}")
    score_rows: List[Dict[str, Any]] = []
    for task in DATASETS:
        task_dir = RESULTS / "independent" / task
        dest = task_dir / INDEPENDENT_NAMES[task]
        if not force and has_full_checkpoint(dest):
            log(f"Independent {task} already complete")
        else:
            log(f"Independent {task} init={model_source()}")
            model_dir = train_task(
                init_path=W0,
                task=task,
                order=1,
                output_dir=task_dir,
                config_dir=single_task_config_dir(task),
                force=force,
            )
            publish_checkpoint(model_dir, dest)
        scores = evaluate_checkpoint(dest, DATASETS, task_dir / "eval_all", force=force)
        for eval_task in DATASETS:
            score_rows.append(
                {
                    "model": INDEPENDENT_NAMES[task],
                    "eval_dataset": eval_task,
                    "accuracy": scores.get(eval_task, ""),
                }
            )
    write_csv(RESULTS / "independent" / "results.csv", ["model", "eval_dataset", "accuracy"], score_rows)


def describe_run() -> str:
    return (
        f"model={MODEL_NAME} model_type={MODEL_TYPE} seed={SEED} lr={LEARNING_RATE} "
        f"wd={WEIGHT_DECAY} batch={BATCH_SIZE} accum={GRAD_ACCUM} "
        f"effective_batch={BATCH_SIZE * GRAD_ACCUM} epochs={EPOCHS} "
        f"results={RESULTS.resolve()} w0={model_source()}"
    )


def check_llama_loss_mask() -> None:
    """Print a few left-padded training rows and require mask 1 only on the label."""
    from transformers import AutoTokenizer
    from uie_collator import DataCollatorForUIE

    if MODEL_TYPE != "causal":
        raise SystemExit("Loss-mask check is for --model-type causal.")
    source = model_source()
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    class _Config:
        _name_or_path = source if "llama" in source.lower() else "meta-llama/Llama-3.2-1B"

    class _Model:
        config = _Config()

    raw = cl_train.load_uie_datasets(
        data_dir=cl_train.data_root_for_order(1),
        task_config_dir=cl_train.task_config_dir(1, "dbpedia"),
        output_dir=Path("/tmp/topomerge_loss_mask_cache"),
        overwrite_cache=False,
    )
    picked = [0, 1, len(raw["train"]) // 2, len(raw["train"]) - 1]
    examples = []
    for index in picked:
        example = dict(raw["train"][int(index)])
        example["subset"] = "train"
        example["Samples"] = []
        examples.append(example)
    collator = DataCollatorForUIE(
        tokenizer,
        model=_Model(),
        padding="longest",
        max_source_length=512,
        max_target_length=50,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        add_task_name=True,
        add_dataset_name=True,
        num_examples=0,
    )
    batch = collator(examples)
    failures: List[str] = []
    print(f"loss-mask check model={source} rows={len(examples)} width={batch['input_ids'].shape[1]} pad_to_multiple_of=8")
    for row, example in enumerate(examples):
        ids = batch["input_ids"][row].tolist()
        mask = batch["loss_mask"][row].tolist()
        attn = batch["attention_mask"][row].tolist()
        label = example["Instance"]["label"]
        label_ids = tokenizer(label + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        one_ids = [token for token, bit, seen in zip(ids, mask, attn) if bit == 1]
        prompt_ids = [token for token, bit, seen in zip(ids, mask, attn) if bit == 0 and seen == 1]
        pad_ones = sum(1 for bit, seen in zip(mask, attn) if bit == 1 and seen == 0)
        trained = [token for index, token in enumerate(ids) if index > 0 and mask[index] == 1]
        ok = one_ids == label_ids and trained == label_ids and pad_ones == 0
        if not ok:
            failures.append(f"row {row} label={label!r}")
        tokens = tokenizer.convert_ids_to_tokens(ids)
        first_content = next(index for index, seen in enumerate(attn) if seen == 1)
        label_start = next((index for index, bit in enumerate(mask) if bit == 1), None)
        print(f"\n--- row {row} ok={ok} label={label!r}")
        print(
            f"prompt_token_count={len(prompt_ids)} label_token_count={len(one_ids)} "
            f"left_pad={first_content} label_start={label_start}"
        )
        print("pos  mask  attn  token")
        show = set(range(min(4, len(tokens))))
        if label_start is not None:
            show.update(range(max(0, label_start - 3), len(tokens)))
        else:
            show.update(range(max(0, len(tokens) - 6), len(tokens)))
        previous = -1
        for index in sorted(show):
            if previous >= 0 and index > previous + 1:
                print("...")
            print(f"{index:3d}  {int(mask[index]):4d}  {attn[index]:4d}  {tokens[index]}")
            previous = index
    if failures:
        raise SystemExit("Loss mask check failed: " + "; ".join(failures))
    print("\nLOSS_MASK_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TopoMerge continual-learning and independent SeqFT")
    parser.add_argument("--mode", choices=["cl", "independent", "all"], default=None)
    parser.add_argument("--model", default=None, help="HF id or local path. Default: T5-Large W0.")
    parser.add_argument("--model-type", choices=["seq2seq", "causal"], default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--check-loss-mask", action="store_true")
    args = parser.parse_args()
    if args.mode is None and not args.show_config and not args.check_loss_mask:
        parser.error("--mode is required unless --show-config or --check-loss-mask is set")
    return args


def main() -> None:
    args = parse_args()
    configure_run(args.seed, args.results_root, args.learning_rate, args.model, args.model_type)
    if args.show_config:
        print(describe_run())
        return
    if args.check_loss_mask:
        check_llama_loss_mask()
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    apply_seed(SEED)
    save_json(
        {
            "model": MODEL_NAME,
            "model_type": MODEL_TYPE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "w0": model_source(),
            "results_root": str(RESULTS.resolve()),
        },
        RESULTS / "run_config.json",
    )
    log(f"mode={args.mode} {describe_run()}")
    if args.mode in {"cl", "all"}:
        run_cl(force=args.force)
    if args.mode in {"independent", "all"}:
        run_independent(force=args.force)


if __name__ == "__main__":
    main()
