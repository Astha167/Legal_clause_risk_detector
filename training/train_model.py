"""
STEP 7 - Fine-tune Contracts-BERT (nlpaueb/bert-base-uncased-contracts) for
three-class screening risk classification (0 = Low, 1 = Medium, 2 = High).

* Input  : clean original clause_text (NOT spaCy-processed text, NOT clause_type)
* Data   : the fixed contract-level split (same partitions as the baselines)
* Loss   : class-weighted cross-entropy (the label distribution is imbalanced)
* Select : best epoch by validation macro F1, with early stopping
* Device : CUDA if available, otherwise CPU (conservative CPU batch size)

The test split is NOT used here. Run training/evaluate_model.py afterwards.

Usage (from the project root):
    python training/train_model.py
    python training/train_model.py --epochs 3 --batch-size 8 --max-length 256
    python training/train_model.py --smoke-test     # quick pipeline check, NOT a real model
    python training/train_model.py --cpu-fast       # faster CPU preset (see README / SETUP_GUIDE)

CPU speed-ups
-------------
* Length-grouped batching (on by default): clauses of similar length are batched
  together, so far less compute is wasted on padding. Disable with --no-group-by-length.
* --freeze-layers N: keeps the embeddings and the lowest N of the 12 encoder layers
  fixed. Fewer weights are updated, so each step is faster (usually at some cost in accuracy).
* --cpu-fast: preset = --epochs 2 --max-length 128 --freeze-layers 6.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from training.common import SplitError, compute_metrics, load_split_frames, print_metrics  # noqa: E402

SMOKE_DIR = config.MODEL_DIR / "_smoke_test"


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune Contracts-BERT for clause risk screening.")
    ap.add_argument("--model-name", default=config.MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=None, help="default: BATCH_SIZE on CUDA, CPU_BATCH_SIZE on CPU")
    ap.add_argument("--max-length", type=int, default=config.MAX_LENGTH)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--warmup-ratio", type=float, default=config.WARMUP_RATIO)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--output-dir", default=str(config.BERT_MODEL_DIR))
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze embeddings + the lowest N encoder layers (0 = fine-tune everything)")
    ap.add_argument("--no-group-by-length", action="store_true",
                    help="disable length-grouped batching (it is on by default)")
    ap.add_argument("--cpu-fast", action="store_true",
                    help="CPU preset: --epochs 2 --max-length 128 --freeze-layers 6 (explicit flags still win)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Train on a tiny subset for 1 epoch to verify the pipeline (saved to model/_smoke_test).")
    args = ap.parse_args()
    if args.cpu_fast:
        # Apply preset values only where the user kept the default.
        if args.epochs == config.EPOCHS:
            args.epochs = 2
        if args.max_length == config.MAX_LENGTH:
            args.max_length = 128
        if args.freeze_layers == 0:
            args.freeze_layers = 6
    return args


class LengthGroupedBatchSampler:
    """Shuffled batches of similar-length texts (reduces padding, speeds up CPU training).

    Indices are shuffled, cut into pools of ``batch_size * pool_factor``, each pool is sorted
    by length and split into batches, and the batch order is shuffled again. Seeded, so the
    order is reproducible.
    """

    def __init__(self, lengths, batch_size, seed, pool_factor=50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool = batch_size * pool_factor
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(idx), self.pool):
            pool = idx[start:start + self.pool]
            pool = pool[np.argsort(self.lengths[pool], kind="stable")]
            batches.extend(pool[i:i + self.batch_size].tolist() for i in range(0, len(pool), self.batch_size))
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


def freeze_lower_layers(model, n_layers: int) -> int:
    """Freeze embeddings + the lowest n encoder layers. Returns the number of frozen layers."""
    if n_layers <= 0:
        return 0
    base = getattr(model, "base_model", model)
    emb = getattr(base, "embeddings", None)
    layers = getattr(getattr(base, "encoder", None), "layer", None)
    if emb is None or layers is None:
        print("WARNING: model architecture not recognised; --freeze-layers ignored.")
        return 0
    n = min(n_layers, len(layers))
    for p in emb.parameters():
        p.requires_grad = False
    for layer in layers[:n]:
        for p in layer.parameters():
            p.requires_grad = False
    return n


class ClauseDataset:
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_length):
    import torch

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return collate


def evaluate_loader(model, loader, device):
    import torch

    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(labels.tolist())
    return compute_metrics(gold, preds)


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). Run: pip install -r requirements.txt")
        return 1

    config.set_seed(args.seed)
    device = config.get_device()
    batch_size = args.batch_size or config.effective_batch_size(device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))

    try:
        frames, split_info = load_split_frames()
    except SplitError as exc:
        print(f"ERROR: {exc}")
        return 1
    train_df, val_df = frames["train"], frames["val"]
    output_dir = Path(args.output_dir)
    epochs = args.epochs
    if args.smoke_test:
        import pandas as pd

        train_df = pd.concat([g.sample(min(len(g), 16), random_state=args.seed)
                              for _, g in train_df.groupby("label")]).reset_index(drop=True)
        val_df = pd.concat([g.sample(min(len(g), 8), random_state=args.seed)
                            for _, g in val_df.groupby("label")]).reset_index(drop=True)
        epochs, output_dir = 1, SMOKE_DIR
        print("SMOKE TEST MODE: tiny subset, 1 epoch. The saved model is NOT a real result.")

    print(f"Model       : {args.model_name}")
    print(f"Train / Val : {len(train_df)} / {len(val_df)} clauses")
    print(f"Batch size  : {batch_size} | max_length {args.max_length} | lr {args.lr} | epochs {epochs}")
    print(f"Speed-ups   : group_by_length={not args.no_group_by_length} | freeze_layers={args.freeze_layers}")
    if device.type == "cpu" and not args.smoke_test and not args.cpu_fast and args.freeze_layers == 0:
        print("TIP: training on CPU. For a faster (somewhat less accurate) run use: "
              "python training/train_model.py --cpu-fast")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, id2label=config.ID2LABEL, label2id=config.LABEL2ID)
    except Exception as exc:
        print(f"ERROR: could not load '{args.model_name}'. An internet connection is needed the first time "
              f"so the pre-trained weights can be downloaded. Details: {exc}")
        return 1
    frozen = freeze_lower_layers(model, args.freeze_layers)
    if frozen:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"Frozen embeddings + {frozen} encoder layers: training {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M weights")
    model.to(device)

    counts = np.bincount(train_df["label"], minlength=len(config.LABELS)).astype(float)
    weights = counts.sum() / (len(config.LABELS) * np.maximum(counts, 1))
    print(f"Class counts {counts.astype(int).tolist()} -> loss weights {np.round(weights, 3).tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))

    gen = torch.Generator().manual_seed(args.seed)
    collate = make_collate(tokenizer, args.max_length)
    train_ds = ClauseDataset(train_df["clause_text"], train_df["label"])
    if args.no_group_by_length:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=gen)
    else:
        lengths = train_df["clause_text"].str.split().str.len().to_numpy()
        train_loader = DataLoader(train_ds, collate_fn=collate,
                                  batch_sampler=LengthGroupedBatchSampler(lengths, batch_size, args.seed))
    # Validation order does not affect metrics, so sort by length for speed.
    val_df = val_df.assign(_len=val_df["clause_text"].str.len()).sort_values("_len").drop(columns="_len")
    val_loader = DataLoader(ClauseDataset(val_df["clause_text"], val_df["label"]), batch_size=batch_size * 2,
                            shuffle=False, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight"]
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [
        {"params": [p for n, p in trainable if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in trainable if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * args.warmup_ratio), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = loss_fn(model(**batch).logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model(**batch).logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.GRAD_CLIP_NORM)
                optimizer.step()
            scheduler.step()
            running += loss.item()
            n += 1
            if step % 50 == 0 or step == len(train_loader):
                rate = (time.time() - t0) / step
                eta = rate * (len(train_loader) - step)
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / n:.4f} "
                      f"({rate:.2f}s/step, epoch ETA {eta / 60:.1f} min)")

        val_m = evaluate_loader(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": round(running / max(1, n), 4),
                        "val_macro_f1": val_m["f1_macro"], "val_accuracy": val_m["accuracy"]})
        print_metrics(f"Epoch {epoch} validation", val_m)

        if val_m["f1_macro"] > best_f1:
            best_f1, best_epoch, bad_epochs = val_m["f1_macro"], epoch, 0
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            best_val = val_m
            print(f"  New best model saved to {output_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epoch(s).")
                break

    elapsed = round(time.time() - start, 1)
    (output_dir / "label_mapping.json").write_text(json.dumps(
        {"id2label": {str(k): v for k, v in config.ID2LABEL.items()}, "label2id": config.LABEL2ID}, indent=2),
        encoding="utf-8")
    meta = {
        "model_name": args.model_name,
        "smoke_test": bool(args.smoke_test),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "device": device.type,
        "hyperparameters": {"max_length": args.max_length, "batch_size": batch_size, "learning_rate": args.lr,
                            "epochs_requested": epochs, "weight_decay": args.weight_decay,
                            "warmup_ratio": args.warmup_ratio, "early_stopping_patience": args.patience,
                            "class_weights": np.round(weights, 4).tolist(), "random_seed": args.seed,
                            "freeze_layers": frozen, "group_by_length": not args.no_group_by_length,
                            "cpu_fast_preset": bool(args.cpu_fast)},
        "dataset": {"train_size": len(train_df), "val_size": len(val_df), "test_size": len(frames["test"]),
                    "input": "clause_text (clean original legal text)", "split": "contract-level"},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_val_metrics": {k: v for k, v in best_val.items() if k != "classification_report_text"},
        "history": history,
        "training_time_seconds": elapsed,
        "split_info": split_info,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best epoch {best_epoch} (val macro F1 {best_f1:.4f}).")
    print(f"Model saved to: {output_dir}")
    print("Next: python training/evaluate_model.py" + (" --model-dir model/_smoke_test" if args.smoke_test else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
