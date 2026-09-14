#!/usr/bin/env python3
"""Diagnostic tests for the static Neuro-3D EEG 72-class decoder.

This script is intended to be used with train_static_eeg_spatiotemporal_trialsplit.py.
It does NOT touch the official test objects (08--09). It provides four diagnostics:

1) memorize
   Can the current network memorize 72 EEG samples (one sample per class)?
   If it cannot reach very high training accuracy, inspect optimization/model/labels first.

2) similarity
   Network-free trial reproducibility check. For each object, trial-0 EEG for each
   class is matched against all 72 classes in trial 1 using cosine similarity.

3) per_object
   Train on one object's trial 0 and validate on the same object's trial 1 (or reverse).
   This is a strict per-object trial-generalization diagnostic.

4) window_sweep
   Repeat the all-object same-object/different-trial experiment while keeping only
   selected temporal windows. Samples remain 250 points long; values outside the
   selected window are zeroed AFTER sample normalization.

Examples are printed with --help.
"""

import argparse
import csv
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


CLASS_COUNT = 72
CHANNELS = 64
SAMPLES = 250
SAMPLE_RATE = 250.0


def load_training_module(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Training script not found: {path}\n"
            "Pass --train_script /path/to/train_static_eeg_spatiotemporal_trialsplit.py"
        )
    spec = importlib.util.spec_from_file_location("static_eeg_train_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_model(mod, args, dropout=None):
    return mod.StaticMLP(
        input_dim=CHANNELS * SAMPLES,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout if dropout is None else dropout,
        use_transformer=True,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        input_representation="raw_waveform",
        conv_channels=args.conv_channels,
        temporal_kernel=args.temporal_kernel,
        separable_kernel=args.separable_kernel,
        temporal_pool1=args.temporal_pool1,
        temporal_pool2=args.temporal_pool2,
    )


def normalize_raw(mod, x, input_norm):
    return mod.normalize_eeg(np.asarray(x, dtype=np.float32), input_norm)


def get_one_object_trial(raw_train, object_idx, trial_idx, mod, input_norm):
    # raw_train: [1, class, object, trial, channel, time]
    x = np.asarray(raw_train)[0, :, object_idx, trial_idx]  # [72,64,250]
    x = normalize_raw(mod, x, input_norm)
    y = np.arange(CLASS_COUNT, dtype=np.int64)
    return x, y


def get_all_objects_trial(raw_train, trial_idx, mod, input_norm):
    # [class, object, channel, time]
    x = np.asarray(raw_train)[0, :, :, trial_idx]
    x = normalize_raw(mod, x, input_norm)
    x = x.reshape(CLASS_COUNT * 8, CHANNELS, SAMPLES)
    y = np.repeat(np.arange(CLASS_COUNT), 8).astype(np.int64)
    object_ids = np.tile(np.arange(8), CLASS_COUNT).astype(np.int64)
    return x, y, object_ids


def apply_window(x, start_ms, end_ms):
    if not (0 <= start_ms < end_ms <= 1000):
        raise ValueError(f"Invalid window {start_ms}-{end_ms} ms; expected 0 <= start < end <= 1000")
    start = int(round(start_ms * SAMPLE_RATE / 1000.0))
    end = int(round(end_ms * SAMPLE_RATE / 1000.0))
    start = max(0, min(SAMPLES, start))
    end = max(start + 1, min(SAMPLES, end))
    out = np.zeros_like(x, dtype=np.float32)
    out[..., start:end] = x[..., start:end]
    return out, start, end


def evaluate(model, x, y, device, batch_size=256):
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            n = yb.numel()
            total_loss += float(loss.item()) * n
            total_top1 += int((logits.argmax(1) == yb).sum().item())
            total_top5 += int(logits.topk(5, dim=1).indices.eq(yb[:, None]).any(1).sum().item())
            total_n += n
    return {
        "loss": total_loss / total_n,
        "top1": total_top1 / total_n,
        "top5": total_top5 / total_n,
        "n": total_n,
    }


def fit_model(
    mod,
    args,
    train_x,
    train_y,
    val_x=None,
    val_y=None,
    *,
    epochs,
    lr,
    weight_decay,
    dropout,
    batch_size,
    patience=None,
    print_every=10,
):
    device = resolve_device(args.device)
    model = make_model(mod, args, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(np.asarray(train_x, dtype=np.float32)),
        torch.from_numpy(np.asarray(train_y, dtype=np.int64)),
    )
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best_state = None
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n = yb.numel()
            loss_sum += float(loss.item()) * n
            n_sum += n

        train_stats = evaluate(model, train_x, train_y, device, batch_size=256)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_top1": train_stats["top1"],
            "train_top5": train_stats["top5"],
        }

        if val_x is not None:
            val_stats = evaluate(model, val_x, val_y, device, batch_size=256)
            row.update({
                "val_loss": val_stats["loss"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
            })
            if val_stats["loss"] < best_val - args.min_delta:
                best_val = val_stats["loss"]
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        else:
            # For memorization, best state is simply the latest state unless 100% is reached.
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(row)

        should_print = epoch == 1 or epoch % print_every == 0 or epoch == epochs
        if val_x is not None and patience is not None and stale >= patience:
            should_print = True
        if should_print:
            text = (
                f"[EPOCH {epoch:03d}] train_top1={row['train_top1']:.4%} "
                f"train_top5={row['train_top5']:.4%} train_loss={row['train_loss']:.4f}"
            )
            if val_x is not None:
                text += (
                    f" val_top1={row['val_top1']:.4%} val_top5={row['val_top5']:.4%} "
                    f"val_loss={row['val_loss']:.4f} stale={stale}/{patience}"
                )
            print(text, flush=True)

        if val_x is None and train_stats["top1"] >= args.memorize_target:
            print(
                f"[PASS] memorization target reached at epoch {epoch}: "
                f"top1={train_stats['top1']:.4%}",
                flush=True,
            )
            break

        if val_x is not None and patience is not None and stale >= patience:
            print(f"[EARLY-STOP] epoch={epoch}; best_epoch={best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = evaluate(model, train_x, train_y, device, batch_size=256)
    final_val = None if val_x is None else evaluate(model, val_x, val_y, device, batch_size=256)
    return model, history, final_train, final_val, best_epoch


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_memorize(mod, arrays, args, out):
    print("\n=== TEST 1: tiny-set memorization ===", flush=True)
    x, y = get_one_object_trial(
        arrays["train"], args.object_idx, args.trial_idx, mod, args.input_norm
    )
    print(
        f"[data] object={args.object_idx:02d} trial={args.trial_idx} "
        f"samples={len(y)} classes={len(np.unique(y))} shape={x.shape} norm={args.input_norm}",
        flush=True,
    )

    model, history, train_stats, _, best_epoch = fit_model(
        mod, args, x, y,
        epochs=args.memorize_epochs,
        lr=args.memorize_lr,
        weight_decay=0.0,
        dropout=0.0,
        batch_size=args.memorize_batch_size,
        patience=None,
        print_every=args.print_every,
    )

    passed = train_stats["top1"] >= args.memorize_target
    summary = {
        "test": "memorize",
        "object": args.object_idx,
        "trial": args.trial_idx,
        "samples": len(y),
        "input_norm": args.input_norm,
        "target_top1": args.memorize_target,
        "final_top1": train_stats["top1"],
        "final_top5": train_stats["top5"],
        "final_loss": train_stats["loss"],
        "passed": bool(passed),
        "best_epoch": best_epoch,
    }
    write_csv(out / "memorize_history.csv", history)
    torch.save({"model": model.state_dict(), "summary": summary}, out / "memorize_model.pt")
    (out / "memorize_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[RESULT] memorize top1={train_stats['top1']:.4%} top5={train_stats['top5']:.4%} "
        f"loss={train_stats['loss']:.4f} -> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return summary


def cosine_retrieval(a, b):
    # a,b: [72,64,250]
    a = a.reshape(CLASS_COUNT, -1).astype(np.float64)
    b = b.reshape(CLASS_COUNT, -1).astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    labels = np.arange(CLASS_COUNT)
    order = np.argsort(-sim, axis=1)
    top1 = float(np.mean(order[:, 0] == labels))
    top5 = float(np.mean(np.any(order[:, :5] == labels[:, None], axis=1)))
    diag = np.diag(sim)
    off_mask = ~np.eye(CLASS_COUNT, dtype=bool)
    off = sim[off_mask]
    # Per-row paired class minus best wrong class is a strict margin.
    wrong = sim.copy()
    wrong[np.arange(CLASS_COUNT), np.arange(CLASS_COUNT)] = -np.inf
    best_wrong = wrong.max(axis=1)
    margin = diag - best_wrong
    return {
        "top1": top1,
        "top5": top5,
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "off_mean": float(off.mean()),
        "off_std": float(off.std()),
        "diag_minus_off": float(diag.mean() - off.mean()),
        "strict_margin_mean": float(margin.mean()),
        "strict_margin_positive_rate": float(np.mean(margin > 0)),
    }


def run_similarity(mod, arrays, args, out):
    print("\n=== TEST 2: network-free trial similarity/retrieval ===", flush=True)
    rows = []
    for obj in range(8):
        x0, _ = get_one_object_trial(arrays["train"], obj, 0, mod, args.input_norm)
        x1, _ = get_one_object_trial(arrays["train"], obj, 1, mod, args.input_norm)
        stats01 = cosine_retrieval(x0, x1)
        stats10 = cosine_retrieval(x1, x0)
        row = {
            "object": obj,
            "t0_to_t1_top1": stats01["top1"],
            "t0_to_t1_top5": stats01["top5"],
            "t1_to_t0_top1": stats10["top1"],
            "t1_to_t0_top5": stats10["top5"],
            "diag_minus_off_01": stats01["diag_minus_off"],
            "diag_minus_off_10": stats10["diag_minus_off"],
            "strict_margin_positive_01": stats01["strict_margin_positive_rate"],
            "strict_margin_positive_10": stats10["strict_margin_positive_rate"],
        }
        rows.append(row)
        print(
            f"[object {obj:02d}] T0->T1 top1={stats01['top1']:.4%} top5={stats01['top5']:.4%} "
            f"diag-off={stats01['diag_minus_off']:+.6f}; "
            f"T1->T0 top1={stats10['top1']:.4%} top5={stats10['top5']:.4%} "
            f"diag-off={stats10['diag_minus_off']:+.6f}",
            flush=True,
        )

    avg = {
        "test": "similarity",
        "input_norm": args.input_norm,
        "chance_top1": 1.0 / CLASS_COUNT,
        "chance_top5": 5.0 / CLASS_COUNT,
        "mean_t0_to_t1_top1": float(np.mean([r["t0_to_t1_top1"] for r in rows])),
        "mean_t0_to_t1_top5": float(np.mean([r["t0_to_t1_top5"] for r in rows])),
        "mean_t1_to_t0_top1": float(np.mean([r["t1_to_t0_top1"] for r in rows])),
        "mean_t1_to_t0_top5": float(np.mean([r["t1_to_t0_top5"] for r in rows])),
        "mean_diag_minus_off_01": float(np.mean([r["diag_minus_off_01"] for r in rows])),
        "mean_diag_minus_off_10": float(np.mean([r["diag_minus_off_10"] for r in rows])),
    }
    write_csv(out / "similarity_by_object.csv", rows)
    (out / "similarity_summary.json").write_text(json.dumps(avg, indent=2), encoding="utf-8")
    print(
        f"[RESULT] mean T0->T1 top1={avg['mean_t0_to_t1_top1']:.4%} "
        f"top5={avg['mean_t0_to_t1_top5']:.4%}; "
        f"T1->T0 top1={avg['mean_t1_to_t0_top1']:.4%} "
        f"top5={avg['mean_t1_to_t0_top5']:.4%}",
        flush=True,
    )
    return avg


def run_per_object(mod, arrays, args, out):
    print("\n=== TEST 3: per-object trial generalization ===", flush=True)
    if args.per_object == "all":
        objects = list(range(8))
    else:
        objects = [int(args.per_object)]
        if objects[0] < 0 or objects[0] >= 8:
            raise ValueError("--per_object must be all or an integer 0..7")

    if args.direction == "both":
        directions = [(0, 1), (1, 0)]
    elif args.direction == "0to1":
        directions = [(0, 1)]
    else:
        directions = [(1, 0)]

    rows = []
    for obj in objects:
        for train_trial, val_trial in directions:
            seed_all(args.seed)
            train_x, train_y = get_one_object_trial(
                arrays["train"], obj, train_trial, mod, args.input_norm
            )
            val_x, val_y = get_one_object_trial(
                arrays["train"], obj, val_trial, mod, args.input_norm
            )
            print(
                f"\n[run] object={obj:02d} train_trial={train_trial} val_trial={val_trial}",
                flush=True,
            )
            model, history, train_stats, val_stats, best_epoch = fit_model(
                mod, args, train_x, train_y, val_x, val_y,
                epochs=args.diag_epochs,
                lr=args.diag_lr,
                weight_decay=args.diag_weight_decay,
                dropout=args.dropout,
                batch_size=args.batch_size,
                patience=args.patience,
                print_every=args.print_every,
            )
            row = {
                "object": obj,
                "train_trial": train_trial,
                "val_trial": val_trial,
                "best_epoch": best_epoch,
                "train_top1": train_stats["top1"],
                "train_top5": train_stats["top5"],
                "train_loss": train_stats["loss"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
                "val_loss": val_stats["loss"],
            }
            rows.append(row)
            write_csv(out / f"per_object_{obj:02d}_{train_trial}to{val_trial}_history.csv", history)
            torch.save(
                {"model": model.state_dict(), "result": row},
                out / f"per_object_{obj:02d}_{train_trial}to{val_trial}.pt",
            )
            print(
                f"[RESULT] object={obj:02d} {train_trial}->{val_trial} "
                f"train_top1={train_stats['top1']:.4%} val_top1={val_stats['top1']:.4%} "
                f"val_top5={val_stats['top5']:.4%}",
                flush=True,
            )

    write_csv(out / "per_object_summary.csv", rows)
    return rows


def parse_windows(text):
    windows = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid window '{token}', expected START-END in ms")
        a, b = token.split("-", 1)
        windows.append((int(a), int(b)))
    if not windows:
        raise ValueError("No windows parsed")
    return windows


def run_window_sweep(mod, arrays, args, out):
    print("\n=== TEST 4: temporal-window trial-split sweep ===", flush=True)
    windows = parse_windows(args.windows)
    directions = [(0, 1), (1, 0)] if args.direction == "both" else (
        [(0, 1)] if args.direction == "0to1" else [(1, 0)]
    )

    rows = []
    for start_ms, end_ms in windows:
        for train_trial, val_trial in directions:
            seed_all(args.seed)
            train_x, train_y, _ = get_all_objects_trial(
                arrays["train"], train_trial, mod, args.input_norm
            )
            val_x, val_y, _ = get_all_objects_trial(
                arrays["train"], val_trial, mod, args.input_norm
            )
            train_x, start_idx, end_idx = apply_window(train_x, start_ms, end_ms)
            val_x, _, _ = apply_window(val_x, start_ms, end_ms)
            print(
                f"\n[window] {start_ms}-{end_ms} ms samples=[{start_idx}:{end_idx}] "
                f"direction={train_trial}->{val_trial}",
                flush=True,
            )
            _, history, train_stats, val_stats, best_epoch = fit_model(
                mod, args, train_x, train_y, val_x, val_y,
                epochs=args.diag_epochs,
                lr=args.diag_lr,
                weight_decay=args.diag_weight_decay,
                dropout=args.dropout,
                batch_size=args.batch_size,
                patience=args.patience,
                print_every=args.print_every,
            )
            row = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_sample": start_idx,
                "end_sample": end_idx,
                "train_trial": train_trial,
                "val_trial": val_trial,
                "best_epoch": best_epoch,
                "train_top1": train_stats["top1"],
                "train_top5": train_stats["top5"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
                "val_loss": val_stats["loss"],
            }
            rows.append(row)
            write_csv(
                out / f"window_{start_ms}_{end_ms}_{train_trial}to{val_trial}_history.csv",
                history,
            )
            print(
                f"[RESULT] {start_ms}-{end_ms} ms {train_trial}->{val_trial}: "
                f"val_top1={val_stats['top1']:.4%} val_top5={val_stats['top5']:.4%} "
                f"val_loss={val_stats['loss']:.4f}",
                flush=True,
            )

    write_csv(out / "window_sweep_summary.csv", rows)
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["memorize", "similarity", "per_object", "window_sweep", "basic"],
        default="basic",
        help="basic runs memorize + similarity; expensive neural diagnostics are separate.",
    )
    p.add_argument("--train_script", default="./train_static_eeg_spatiotemporal_trialsplit.py")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--input_norm",
        choices=["none", "sample_minmax", "channel_zscore", "global_zscore"],
        default="global_zscore",
    )

    # Architecture: match the current spatiotemporal experiment by default.
    p.add_argument("--conv_channels", type=int, default=32)
    p.add_argument("--temporal_kernel", type=int, default=31)
    p.add_argument("--separable_kernel", type=int, default=15)
    p.add_argument("--temporal_pool1", type=int, default=4)
    p.add_argument("--temporal_pool2", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--transformer_layers", type=int, default=1)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--transformer_ff_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Memorization test.
    p.add_argument("--object_idx", type=int, choices=range(8), default=0)
    p.add_argument("--trial_idx", type=int, choices=[0, 1], default=0)
    p.add_argument("--memorize_epochs", type=int, default=300)
    p.add_argument("--memorize_lr", type=float, default=1e-3)
    p.add_argument("--memorize_batch_size", type=int, default=72)
    p.add_argument("--memorize_target", type=float, default=0.98)

    # General diagnostic training.
    p.add_argument("--diag_epochs", type=int, default=100)
    p.add_argument("--diag_lr", type=float, default=1e-4)
    p.add_argument("--diag_weight_decay", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=5e-4)
    p.add_argument("--print_every", type=int, default=10)

    # Per-object/window controls.
    p.add_argument("--per_object", default="0", help="all or one integer object id 0..7")
    p.add_argument("--direction", choices=["0to1", "1to0", "both"], default="both")
    p.add_argument(
        "--windows",
        default="0-250,100-350,200-500,300-700,500-1000,0-1000",
        help="comma-separated temporal windows in ms",
    )

    args = p.parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    mod = load_training_module(args.train_script)
    arrays, paths = mod.load_official_arrays(args.data_path, args.sub_id)
    print(f"[data] train={paths['train']}", flush=True)
    print(f"[data] shape={tuple(arrays['train'].shape)} norm={args.input_norm}", flush=True)
    print(f"[chance] top1={1/CLASS_COUNT:.4%} top5={5/CLASS_COUNT:.4%} CE={math.log(CLASS_COUNT):.6f}", flush=True)

    results = {"mode": args.mode, "data_paths": paths, "input_norm": args.input_norm}
    if args.mode in {"memorize", "basic"}:
        results["memorize"] = run_memorize(mod, arrays, args, out)
    if args.mode in {"similarity", "basic"}:
        results["similarity"] = run_similarity(mod, arrays, args, out)
    if args.mode == "per_object":
        results["per_object"] = run_per_object(mod, arrays, args, out)
    if args.mode == "window_sweep":
        results["window_sweep"] = run_window_sweep(mod, arrays, args, out)

    (out / "diagnostic_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] outputs={out}", flush=True)


if __name__ == "__main__":
    main()
