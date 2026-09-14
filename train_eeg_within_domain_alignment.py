#!/usr/bin/env python3
"""Two-stage 72-class EEG training with a leakage-safe object protocol.

Stage 1 (within_object): source objects 00--05, trial 0 for training and trial
1 for validation.  Stage 2 (domain_alignment): initialize from Stage 1, train
on both trials plus their mean from objects 00--05, and validate only on the
two-trial means of held-out objects 06--07.

The script depends only on train_static_eeg_mlp.py in the same directory.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from train_static_eeg_mlp import (
    CHANNELS,
    CLASS_COUNT,
    SAMPLES,
    StaticMLP,
    load_official_arrays,
    normalize_eeg,
    parse_objects,
    seed_all,
)


class EEGDomainDataset(Dataset):
    def __init__(self, eeg, class_labels, object_labels, noise_std=0.0, training=False):
        self.eeg = torch.as_tensor(np.asarray(eeg, dtype=np.float32))
        self.class_labels = torch.as_tensor(class_labels, dtype=torch.long)
        self.object_labels = torch.as_tensor(object_labels, dtype=torch.long)
        self.noise_std = float(noise_std)
        self.training = bool(training)
        expected = (len(self.class_labels), CHANNELS, SAMPLES)
        if tuple(self.eeg.shape) != expected:
            raise ValueError(f"Expected EEG {expected}, got {tuple(self.eeg.shape)}")
        if len(self.object_labels) != len(self.class_labels):
            raise ValueError("Object/class label length mismatch")

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, index):
        eeg = self.eeg[index]
        if self.training and self.noise_std > 0:
            eeg = eeg + torch.randn_like(eeg) * self.noise_std
        return eeg, self.class_labels[index], self.object_labels[index]


def _labels(objects, repetitions):
    class_labels = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int64), len(objects) * repetitions
    )
    object_labels = np.tile(
        np.repeat(np.asarray(objects, dtype=np.int64), repetitions), CLASS_COUNT
    )
    return class_labels, object_labels


def make_single_trial(raw, objects, trial, input_norm):
    selected = np.take(np.asarray(raw)[0], objects, axis=1)
    if trial < 0 or trial >= selected.shape[2]:
        raise ValueError(f"trial must be in 0..{selected.shape[2] - 1}")
    eeg = selected[:, :, trial]  # [class, object, channel, time]
    eeg = normalize_eeg(eeg, input_norm).reshape(-1, CHANNELS, SAMPLES)
    class_labels, object_labels = _labels(objects, repetitions=1)
    return eeg, class_labels, object_labels


def make_source_alignment_train(raw, objects, input_norm):
    selected = np.take(np.asarray(raw)[0], objects, axis=1)
    trial_mean = selected.mean(axis=2, keepdims=True)
    eeg = np.concatenate([selected, trial_mean], axis=2)
    eeg = normalize_eeg(eeg, input_norm).reshape(-1, CHANNELS, SAMPLES)
    class_labels, object_labels = _labels(objects, repetitions=eeg.shape[0] // (CLASS_COUNT * len(objects)))
    return eeg, class_labels, object_labels


def make_object_mean_validation(raw, objects, input_norm):
    selected = np.take(np.asarray(raw)[0], objects, axis=1)
    eeg = selected.mean(axis=2)
    eeg = normalize_eeg(eeg, input_norm).reshape(-1, CHANNELS, SAMPLES)
    class_labels, object_labels = _labels(objects, repetitions=1)
    return eeg, class_labels, object_labels


class ClassObjectBatchSampler(Sampler):
    """Every batch contains several objects for each selected class."""

    def __init__(self, class_labels, object_labels, batch_size=64,
                 objects_per_class=4, seed=0):
        self.batch_size = int(batch_size)
        self.objects_per_class = int(objects_per_class)
        self.seed = int(seed)
        self.epoch = 0
        if self.objects_per_class < 2:
            raise ValueError("objects_per_class must be >= 2")
        self.classes_per_batch = self.batch_size // self.objects_per_class
        if self.classes_per_batch < 1:
            raise ValueError("batch_size must be >= objects_per_class")
        self.groups = {}
        for index, (cls, obj) in enumerate(zip(class_labels, object_labels)):
            self.groups.setdefault(int(cls), {}).setdefault(int(obj), []).append(index)
        self.classes = sorted(self.groups)
        for cls in self.classes:
            if len(self.groups[cls]) < self.objects_per_class:
                raise ValueError(
                    f"Class {cls} has {len(self.groups[cls])} objects; "
                    f"need {self.objects_per_class}"
                )
        self.num_batches = math.ceil(len(class_labels) / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            if self.classes_per_batch <= len(self.classes):
                classes = rng.sample(self.classes, self.classes_per_batch)
            else:
                classes = [rng.choice(self.classes) for _ in range(self.classes_per_batch)]
            batch = []
            for cls in classes:
                domains = rng.sample(sorted(self.groups[cls]), self.objects_per_class)
                for domain in domains:
                    batch.append(rng.choice(self.groups[cls][domain]))
            rng.shuffle(batch)
            yield batch


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def reverse_gradient(x, coefficient):
    return GradientReverse.apply(x, coefficient)


class ObjectDiscriminator(nn.Module):
    def __init__(self, latent_dim, object_count=8, dropout=0.1):
        super().__init__()
        width = max(32, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, object_count),
        )

    def forward(self, latent):
        return self.net(latent)


def cross_object_alignment(latent, class_labels, object_labels):
    z = F.normalize(latent, dim=-1)
    same_class = class_labels[:, None].eq(class_labels[None, :])
    different_object = object_labels[:, None].ne(object_labels[None, :])
    upper = torch.triu(
        torch.ones_like(same_class, dtype=torch.bool), diagonal=1
    )
    mask = same_class & different_object & upper
    if not mask.any():
        return latent.new_zeros(())
    similarity = z @ z.T
    return (1.0 - similarity[mask]).mean()


def run_epoch(model, loader, device, optimizer=None, object_head=None,
              alignment_weight=0.0, domain_weight=0.0, grl_coefficient=0.0,
              label_smoothing=0.0):
    training = optimizer is not None
    model.train(training)
    if object_head is not None:
        object_head.train(training)
    totals = {
        "loss": 0.0, "ce": 0.0, "align": 0.0, "domain_ce": 0.0,
        "top1": 0, "top5": 0, "domain_correct": 0, "n": 0,
    }
    for eeg, class_labels, object_labels in loader:
        eeg = eeg.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)
        object_labels = object_labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits, latent = model(eeg)
            ce = F.cross_entropy(logits, class_labels, label_smoothing=label_smoothing)
            align = latent.new_zeros(())
            domain_ce = latent.new_zeros(())
            domain_logits = None
            if alignment_weight > 0:
                align = cross_object_alignment(latent, class_labels, object_labels)
            if object_head is not None and domain_weight > 0:
                domain_logits = object_head(reverse_gradient(latent, grl_coefficient))
                domain_ce = F.cross_entropy(domain_logits, object_labels)
            loss = ce + alignment_weight * align + domain_weight * domain_ce
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = list(model.parameters())
                if object_head is not None:
                    parameters += list(object_head.parameters())
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
        n = class_labels.numel()
        totals["loss"] += float(loss.item()) * n
        totals["ce"] += float(ce.item()) * n
        totals["align"] += float(align.item()) * n
        totals["domain_ce"] += float(domain_ce.item()) * n
        totals["top1"] += int(logits.argmax(1).eq(class_labels).sum().item())
        totals["top5"] += int(logits.topk(5, 1).indices.eq(class_labels[:, None]).any(1).sum().item())
        if domain_logits is not None:
            totals["domain_correct"] += int(domain_logits.argmax(1).eq(object_labels).sum().item())
        totals["n"] += n
    n = totals["n"]
    return {
        "loss": totals["loss"] / n,
        "ce": totals["ce"] / n,
        "alignment": totals["align"] / n,
        "domain_ce": totals["domain_ce"] / n,
        "top1": totals["top1"] / n,
        "top5": totals["top5"] / n,
        "domain_accuracy": totals["domain_correct"] / n,
        "n": n,
    }


def make_model(args, device):
    return StaticMLP(
        input_dim=CHANNELS * SAMPLES,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        use_transformer=True,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        input_representation="raw_waveform",
        conv_channels=args.conv_channels,
        conv_pooled_steps=args.conv_pooled_steps,
    ).to(device)


def improved(current, best, monitor, min_delta):
    if best is None:
        return True
    return current < best - min_delta if monitor == "val_loss" else current > best + min_delta


def train_loop(name, model, train_loader, val_loader, device, out_dir, args,
               epochs, patience, monitor, learning_rate,
               object_head=None, use_alignment=False):
    if object_head is not None:
        parameters = [
            {"params": model.parameters(), "lr": learning_rate},
            {"params": object_head.parameters(), "lr": args.domain_head_lr},
        ]
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_value = None
    best_model = None
    best_object_head = None
    best_epoch = 0
    best_stats = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        if use_alignment:
            ramp = min(1.0, epoch / max(1, args.alignment_warmup_epochs))
            progress = (epoch - 1) / max(1, epochs - 1)
            grl = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            alignment_weight = args.alignment_weight * ramp
            domain_weight = args.domain_weight * ramp
        else:
            grl = alignment_weight = domain_weight = 0.0

        train_stats = run_epoch(
            model, train_loader, device, optimizer=optimizer, object_head=object_head,
            alignment_weight=alignment_weight, domain_weight=domain_weight,
            grl_coefficient=grl, label_smoothing=args.label_smoothing,
        )
        val_stats = run_epoch(model, val_loader, device, label_smoothing=0.0)
        scheduler.step()
        metric_key = {
            "val_loss": "loss", "val_top1": "top1", "val_top5": "top5"
        }[monitor]
        current = val_stats[metric_key]
        if improved(current, best_value, monitor, args.min_delta):
            best_value = current
            best_epoch = epoch
            best_stats = dict(val_stats)
            stale = 0
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if object_head is not None:
                best_object_head = {
                    k: v.detach().cpu().clone() for k, v in object_head.state_dict().items()
                }
        else:
            stale += 1

        row = {
            "stage": name, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"], "train_ce": train_stats["ce"],
            "train_alignment": train_stats["alignment"],
            "train_domain_ce": train_stats["domain_ce"],
            "train_domain_accuracy": train_stats["domain_accuracy"],
            "train_top1": train_stats["top1"], "train_top5": train_stats["top5"],
            "val_loss": val_stats["loss"], "val_top1": val_stats["top1"],
            "val_top5": val_stats["top5"], "alignment_weight": alignment_weight,
            "domain_weight": domain_weight, "grl": grl, "stale": stale,
        }
        history.append(row)
        print(
            f"[{name}] epoch={epoch:03d} train_top1={train_stats['top1']:.4%} "
            f"train_top5={train_stats['top5']:.4%} val_top1={val_stats['top1']:.4%} "
            f"val_top5={val_stats['top5']:.4%} val_loss={val_stats['loss']:.4f} "
            f"align={train_stats['alignment']:.4f} domain_acc={train_stats['domain_accuracy']:.4%} "
            f"stale={stale}/{patience}", flush=True,
        )
        if stale >= patience:
            break

    if best_model is None:
        raise RuntimeError(f"{name}: no checkpoint was produced")
    model.load_state_dict(best_model)
    if object_head is not None and best_object_head is not None:
        object_head.load_state_dict(best_object_head)
    write_csv(out_dir / "train_log.csv", history)
    return best_epoch, best_stats, history


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_config(args):
    return {
        "input_dim": CHANNELS * SAMPLES,
        "input_shape": [CHANNELS, SAMPLES],
        "input_representation": "raw_waveform",
        "input_norm": args.input_norm,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "use_transformer": True,
        "token_dim": 50,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ff_dim": args.transformer_ff_dim,
        "conv_channels": args.conv_channels,
        "conv_pooled_steps": args.conv_pooled_steps,
    }


def save_checkpoint(path, model, args, best_epoch, best_stats, object_head=None):
    package = {
        "model": model.state_dict(),
        "config": checkpoint_config(args),
        "scaler_mean": np.zeros((1,), dtype=np.float32),
        "scaler_scale": np.ones((1,), dtype=np.float32),
        "class_count": CLASS_COUNT,
        "best_epoch": int(best_epoch),
        "best_validation": best_stats,
        "args": vars(args),
    }
    if object_head is not None:
        package["object_discriminator"] = object_head.state_dict()
    torch.save(package, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    parser.add_argument("--sub_id", default="sub01")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--source_objects", default="00,01,02,03,04,05")
    parser.add_argument("--target_val_objects", default="06,07")
    parser.add_argument("--within_train_trial", type=int, default=0)
    parser.add_argument("--within_val_trial", type=int, default=1)
    parser.add_argument(
        "--input_norm",
        choices=["none", "sample_minmax", "channel_zscore", "global_zscore"],
        default="sample_minmax",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--conv_channels", type=int, default=32)
    parser.add_argument("--conv_pooled_steps", type=int, default=4)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--within_lr", type=float, default=1e-4)
    parser.add_argument("--alignment_lr", type=float, default=5e-5)
    parser.add_argument("--domain_head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feature_noise_std", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--within_epochs", type=int, default=80)
    parser.add_argument("--within_patience", type=int, default=12)
    parser.add_argument("--within_monitor", choices=["val_loss", "val_top1", "val_top5"], default="val_top5")
    parser.add_argument("--within_min_top1", type=float, default=0.05)
    parser.add_argument("--within_min_top5", type=float, default=0.15)
    parser.add_argument("--alignment_epochs", type=int, default=40)
    parser.add_argument("--alignment_patience", type=int, default=6)
    parser.add_argument("--alignment_monitor", choices=["val_loss", "val_top1", "val_top5"], default="val_loss")
    parser.add_argument("--alignment_weight", type=float, default=0.05)
    parser.add_argument("--domain_weight", type=float, default=0.02)
    parser.add_argument("--alignment_warmup_epochs", type=int, default=5)
    parser.add_argument("--objects_per_class", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=5e-4)
    parser.add_argument("--force_alignment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    started = time.time()

    source_objects = parse_objects(args.source_objects)
    target_objects = parse_objects(args.target_val_objects)
    if set(source_objects) & set(target_objects):
        raise ValueError("source_objects and target_val_objects must be disjoint")
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    within_dir = out / "within_object"
    alignment_dir = out / "domain_alignment"
    within_dir.mkdir(parents=True)
    alignment_dir.mkdir(parents=True)
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arrays, paths = load_official_arrays(args.data_path, args.sub_id)
    raw = arrays["train"]

    within_train = make_single_trial(
        raw, source_objects, args.within_train_trial, args.input_norm
    )
    within_val = make_single_trial(
        raw, source_objects, args.within_val_trial, args.input_norm
    )
    train_dataset = EEGDomainDataset(*within_train, args.feature_noise_std, True)
    val_dataset = EEGDomainDataset(*within_val)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = make_model(args, device)
    print(
        f"[within_object] source={source_objects}; train_trial={args.within_train_trial}; "
        f"val_trial={args.within_val_trial}; train={len(train_dataset)}; val={len(val_dataset)}",
        flush=True,
    )
    within_epoch, within_stats, _ = train_loop(
        "WITHIN", model, train_loader, val_loader, device, within_dir, args,
        args.within_epochs, args.within_patience, args.within_monitor,
        args.within_lr,
    )
    save_checkpoint(within_dir / "best.pt", model, args, within_epoch, within_stats)
    qualified = (
        within_stats["top1"] >= args.within_min_top1
        or within_stats["top5"] >= args.within_min_top5
    )
    print(
        f"[within_result] best_epoch={within_epoch}; top1={within_stats['top1']:.4%}; "
        f"top5={within_stats['top5']:.4%}; qualified={qualified}", flush=True,
    )
    if not qualified and not args.force_alignment:
        summary = {
            "status": "stopped_after_within_object",
            "reason": "within-object transfer did not meet the learnability threshold",
            "within_validation": within_stats,
            "data_paths": paths,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[stop] Domain alignment was not started. Use --force_alignment only for diagnosis.")
        return

    alignment_train = make_source_alignment_train(raw, source_objects, args.input_norm)
    alignment_val = make_object_mean_validation(raw, target_objects, args.input_norm)
    train_dataset = EEGDomainDataset(*alignment_train, args.feature_noise_std, True)
    val_dataset = EEGDomainDataset(*alignment_val)
    batch_sampler = ClassObjectBatchSampler(
        alignment_train[1], alignment_train[2], args.batch_size,
        args.objects_per_class, args.seed + 1000,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    object_head = ObjectDiscriminator(args.latent_dim, dropout=args.dropout).to(device)
    print(
        f"[domain_alignment] source={source_objects}; target_validation={target_objects}; "
        f"train={len(train_dataset)}; val={len(val_dataset)}; "
        f"objects_per_class={args.objects_per_class}", flush=True,
    )
    alignment_epoch, alignment_stats, _ = train_loop(
        "ALIGN", model, train_loader, val_loader, device, alignment_dir, args,
        args.alignment_epochs, args.alignment_patience, args.alignment_monitor,
        args.alignment_lr,
        object_head=object_head, use_alignment=True,
    )
    save_checkpoint(
        alignment_dir / "best.pt", model, args, alignment_epoch,
        alignment_stats, object_head=object_head,
    )
    summary = {
        "status": "completed",
        "source_objects": source_objects,
        "target_validation_objects": target_objects,
        "within_best_epoch": within_epoch,
        "within_validation": within_stats,
        "alignment_best_epoch": alignment_epoch,
        "alignment_validation": alignment_stats,
        "data_paths": paths,
        "seconds": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[done] checkpoint={alignment_dir / 'best.pt'}; "
        f"val_top1={alignment_stats['top1']:.4%}; val_top5={alignment_stats['top5']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
