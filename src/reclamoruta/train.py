"""Train DistilBERT V1 or V2."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from reclamoruta.baseline import load_training_data, scores, write_json
from reclamoruta.predictor import write_manifest


def build_model(config: dict, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"], revision=config["revision"], num_labels=num_labels)
    if config["freeze_encoder"]:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    return model


def build_optimizer(model, lr: float) -> torch.optim.AdamW:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)


def _encode(tokenizer, texts: list[str], max_length: int, device):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt").to(device)


def build_scheduler(optimizer, config: dict, n_train: int):
    per_epoch = math.ceil(math.ceil(n_train / config["batch_size"]) / config["accumulation"])
    total = per_epoch * config["max_epochs"]
    return get_linear_schedule_with_warmup(optimizer, int(config["warmup_ratio"] * total), total)


def train_epoch(model, tokenizer, optimizer, scheduler, texts, labels, config: dict,
                generator) -> dict:
    model.train()
    order = torch.randperm(len(texts), generator=generator).tolist()
    starts = range(0, len(order), config["batch_size"])
    loss_sum = correct = 0.0
    for step, start in enumerate(starts, 1):
        rows = order[start:start + config["batch_size"]]
        target = torch.tensor([labels[i] for i in rows], device=model.device)
        batch = _encode(tokenizer, [texts[i] for i in rows], config["max_length"], model.device)
        logits = model(**batch).logits
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        (loss / config["accumulation"]).backward()
        if step % config["accumulation"] == 0 or step == len(starts):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss_sum += loss.item() * len(rows)
        correct += (logits.argmax(1) == target).sum().item()
    # measured with dropout active
    return {"loss": loss_sum / len(texts), "accuracy": correct / len(texts)}


@torch.no_grad()
def evaluate(model, tokenizer, texts, labels, max_length: int, batch_size: int = 64) -> dict:
    model.eval()
    loss_sum, predictions = 0.0, []
    for start in range(0, len(texts), batch_size):
        target = torch.tensor(labels[start:start + batch_size], device=model.device)
        batch = _encode(tokenizer, texts[start:start + batch_size], max_length, model.device)
        logits = model(**batch).logits
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        predictions += logits.argmax(1).tolist()
    return {"loss": loss_sum / len(texts), **scores(labels, predictions)}


def train_model(config_path: Path, data_dir: Path, output_dir: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    data = load_training_data(data_dir)
    catalog = data["catalog"]
    train_texts, train_labels = data["train"]
    val_texts, val_labels = data["validation_model"]
    # never overwrite an artifact
    output_dir.mkdir(parents=True)

    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])
    model = build_model(config, len(catalog)).to(device)
    optimizer = build_optimizer(model, config["lr"])
    scheduler = build_scheduler(optimizer, config, len(train_texts))
    lengths = [len(ids) for ids in tokenizer(train_texts + val_texts)["input_ids"]]

    curves, best = [], None
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        train = train_epoch(model, tokenizer, optimizer, scheduler, train_texts, train_labels,
                            config, generator)
        val = evaluate(model, tokenizer, val_texts, val_labels, config["max_length"])
        curves.append({
            "epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"], "seconds": time.perf_counter() - started,
        })
        if best is None or val["macro_f1"] > best["val_macro_f1"]:
            best = curves[-1]
            model.save_pretrained(output_dir)
        elif epoch - best["epoch"] >= config["patience"]:
            break

    tokenizer.save_pretrained(output_dir)
    summary = {
        "model_version": config["model_version"], "best": best, "curves": curves,
        "selection": "max val_macro_f1 on validation_model",
        "train_seconds": time.perf_counter() - started,
        "n_train": len(train_texts), "n_validation_model": len(val_texts),
        "truncated_texts": sum(length > config["max_length"] for length in lengths),
        "max_tokens": max(lengths),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "precision": "fp32",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    write_json(output_dir / "label_to_id.json", catalog)
    write_json(output_dir / "train_config.json", config)
    write_json(output_dir / "metrics.json", summary)
    write_manifest(output_dir, config["model_version"], "distilbert", data["manifest_sha256"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = train_model(args.config, args.data, args.out)
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({key: summary[key] for key in ("model_version", "best", "train_seconds")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
