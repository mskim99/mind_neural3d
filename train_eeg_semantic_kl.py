#!/usr/bin/env python3
"""Trial-invariant, low-capacity EEG semantic learning with prototype KL.

q[c] = softmax(cosine(prototype[c], prototypes) / teacher_temperature)
L = lambda_kl * KL(q[label] || p_student)
  + lambda_trial * (1 - cosine(z_trial0, z_trial1))
No Oracle checkpoint, cross loss, denoising loss, or generation evaluation.
The two repetitions of each (class, object) pair are loaded together so that
trial-invariance is enforced directly. Validation prototype-KL is the primary
checkpoint metric, with strict patience-based early stopping.
Train: 00..06 except holdout; default holdout 06. 07/08/09 are not sampled.
All train/validation EEG is read from base.eeg_data with identical normalization;
base.__getitem__ is not called (avoids image/render loading).
The external dataset constructor is retained for EEG/text-feature loading.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def object_suffix(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[-1]
    return ""

def category_from_name(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[0]
    return key

def selected_object_columns(dataset, suffixes):
    wanted = set(suffixes)
    cols = []
    for o in range(int(dataset.obj_num)):
        seen = {
            object_suffix(dataset.name_list[c, o])
            for c in range(int(dataset.cls_num))
        }
        if len(seen) != 1:
            raise RuntimeError(
                f"Object column {o} has inconsistent suffixes: {seen}"
            )
        suffix = next(iter(seen))
        if suffix in wanted:
            cols.append(o)

    found = {object_suffix(dataset.name_list[0, o]) for o in cols}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Missing object suffixes: {sorted(missing)}")
    return cols

class EEGOnlyDataset(Dataset):
    """EEG-only train/validation view. mode='averaged' averages RAW trials."""

    def __init__(self, base, suffixes, mode="individual"):
        self.base = base
        self.mode = mode
        self.cols = selected_object_columns(base, suffixes)

        S = int(base.eeg_data.shape[0])
        C = int(base.cls_num)
        R = int(base.trails_num)

        self.items = []
        for s in range(S):
            for c in range(C):
                for o in self.cols:
                    if mode == "individual":
                        for r in range(R):
                            self.items.append((s, c, o, r))
                    elif mode == "averaged":
                        self.items.append((s, c, o, None))
                    else:
                        raise ValueError(mode)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, c, o, r = self.items[idx]
        # Read raw EEG only; both training and validation use this exact path.
        if r is None:
            raw = np.asarray(self.base.eeg_data[s, c, o, :], dtype=np.float32).mean(axis=0)
            trial_index = -1
        else:
            raw = np.asarray(self.base.eeg_data[s, c, o, r], dtype=np.float32)
            trial_index = int(r)
        eeg = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))
        if eeg.shape != (64, 600) or not torch.isfinite(eeg).all():
            raise RuntimeError(f"Invalid EEG at {(s,c,o,r)}: shape={tuple(eeg.shape)}")

        return {
            "eeg_data": eeg,
            "cls_index": int(c),
            "obj_index": int(o),
            "trial_index": trial_index,
            "name": str(self.base.name_list[c, o]),
        }


class TrialPairEEGDataset(Dataset):
    """Return the two repetitions of each (subject, class, object) item.

    Training on explicit trial pairs makes the nuisance-invariance objective
    well-defined even when the DataLoader does not place both repetitions in
    the same minibatch.
    """

    def __init__(self, base, suffixes):
        self.base = base
        self.cols = selected_object_columns(base, suffixes)
        self.items = []
        S = int(base.eeg_data.shape[0])
        C = int(base.cls_num)
        R = int(base.trails_num)
        if R != 2:
            raise RuntimeError(
                f"TrialPairEEGDataset expects exactly two trials, got {R}"
            )
        for s in range(S):
            for c in range(C):
                for o in self.cols:
                    self.items.append((s, c, o))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, c, o = self.items[idx]
        raw_a = np.asarray(self.base.eeg_data[s, c, o, 0], dtype=np.float32)
        raw_b = np.asarray(self.base.eeg_data[s, c, o, 1], dtype=np.float32)
        eeg_a = torch.from_numpy(np.array(raw_a, dtype=np.float32, copy=True))
        eeg_b = torch.from_numpy(np.array(raw_b, dtype=np.float32, copy=True))
        if (
            eeg_a.shape != (64, 600)
            or eeg_b.shape != (64, 600)
            or not torch.isfinite(eeg_a).all()
            or not torch.isfinite(eeg_b).all()
        ):
            raise RuntimeError(f"Invalid EEG trial pair at {(s, c, o)}")
        return {
            "eeg_a": eeg_a,
            "eeg_b": eeg_b,
            "cls_index": int(c),
            "obj_index": int(o),
        }

def to_flat_feature(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().reshape(-1)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)

def build_fixed_category_text_prototypes(
    base_dataset,
    source_suffixes,
    expected_dim=1024,
):
    source_cols = selected_object_columns(base_dataset, source_suffixes)
    prototypes, categories, source_objects = [], [], []

    for c in range(int(base_dataset.cls_num)):
        feats, cats, objs = [], [], []
        for o in source_cols:
            dataset_name = str(base_dataset.name_list[c, o])
            key = dataset_name[3:]
            cat = category_from_name(dataset_name)
            feat = to_flat_feature(base_dataset.clip_features[key]["text"])
            if feat.numel() != expected_dim:
                raise RuntimeError(
                    f"{key}: expected text dim {expected_dim}, got {feat.numel()}"
                )
            feats.append(F.normalize(feat, dim=0))
            cats.append(cat)
            objs.append(key)

        if len(set(cats)) != 1:
            raise RuntimeError(
                f"cls_index={c} maps to multiple categories: {set(cats)}"
            )

        prototypes.append(F.normalize(torch.stack(feats).mean(dim=0), dim=0))
        categories.append(cats[0])
        source_objects.append(objs)

    prototypes = torch.stack(prototypes, dim=0)
    if tuple(prototypes.shape) != (72, expected_dim):
        raise RuntimeError(
            f"Expected [72,{expected_dim}], got {tuple(prototypes.shape)}"
        )
    return prototypes, categories, source_objects

def save_prototypes(out_dir, prototypes, categories, source_objects, suffixes):
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "categories": categories,
            "source_objects": source_objects,
            "source_suffixes": list(suffixes),
            "construction": (
                "normalize each TRAIN-object text feature -> category mean -> normalize"
            ),
        },
        Path(out_dir) / "fixed_category_text_prototypes.pt",
    )

def normalize_eeg_channels(eeg: torch.Tensor) -> torch.Tensor:
    """eeg [B,64,600] -> deterministic per-sample/per-channel z-normalization."""
    eeg = eeg.float()
    if eeg.ndim != 3 or eeg.shape[1:] != (64, 600):
        raise RuntimeError(f"Expected EEG [B,64,600], got {tuple(eeg.shape)}")
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (eeg - mean) / std

class TemporalEEGSemanticStudent(nn.Module):
    """
    Low-capacity trainable temporal encoder.

    [B,64,600]
      -> depthwise temporal convolutions with a small channel width
      -> chronological mean/std bins
      -> low-dimensional feature projection
      -> small semantic adapter -> 72 logits

    The default width (feature_dim=128, hidden_dim=64) is intentionally small
    relative to the 864 training trials.  Dropout is applied only after the
    pooled representation and inside the adapter.
    """

    def __init__(
        self,
        feature_dim=128,
        hidden_dim=64,
        num_classes=72,
        temporal_bins=2,
        dropout=0.4,
    ):
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.temporal_bins = int(temporal_bins)
        if not 1 <= self.temporal_bins <= 150:
            raise ValueError("temporal_bins must be between 1 and 150")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        temporal_width = 96
        self.temporal = nn.Sequential(
            nn.Conv1d(
                64, 64, kernel_size=15, stride=2, padding=7,
                groups=64, bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, temporal_width, kernel_size=1, bias=False),
            nn.GroupNorm(12, temporal_width),
            nn.GELU(),
            nn.Conv1d(
                temporal_width, temporal_width, kernel_size=9, stride=2,
                padding=4, groups=temporal_width, bias=False,
            ),
            nn.GroupNorm(12, temporal_width),
            nn.GELU(),
        )

        pooled_dim = 2 * temporal_width * self.temporal_bins
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(feature_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        self.adapter = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def encode_temporal(self, eeg):
        x = normalize_eeg_channels(eeg)
        x = self.temporal(x)
        # Non-overlapping chronological bins preserve coarse temporal position.
        chunks = torch.tensor_split(x, self.temporal_bins, dim=-1)
        mean = torch.stack([z.mean(-1) for z in chunks], dim=-1).flatten(1)
        std = torch.stack([z.std(-1, unbiased=False) for z in chunks], dim=-1).flatten(1)
        pooled = torch.cat([mean, std], dim=-1)
        return self.feature_proj(pooled)

    def forward(self, eeg, temperature=0.5):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        feat = self.encode_temporal(eeg)
        logits = self.adapter(feat)
        probs = F.softmax(logits / float(temperature), dim=-1)
        return logits, probs, feat

def build_soft_semantic_table(prototypes, temperature):
    if temperature <= 0:
        raise ValueError("teacher semantic temperature must be > 0")
    p = F.normalize(prototypes.float(), dim=-1)
    sim = p @ p.t()
    return F.softmax(sim / float(temperature), dim=-1)

def soft_semantic_kl(logits, labels, q_table, student_temperature):
    target_q = q_table[labels]
    log_p = F.log_softmax(logits / float(student_temperature), dim=-1)
    return F.kl_div(log_p, target_q, reduction="batchmean")


def trial_consistency_loss(feat_a, feat_b):
    """Make the representation invariant to the repeated-trial nuisance."""
    a = F.normalize(feat_a.float(), dim=-1)
    b = F.normalize(feat_b.float(), dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()

def soft_weight_at_step(step, max_weight, decay_steps, min_weight=None):
    floor = float(max_weight if min_weight is None else min_weight)
    if not 0 <= floor <= max_weight:
        raise ValueError("Require 0 <= lambda_soft_min <= lambda_soft_sem")
    if decay_steps <= 0:
        return float(max_weight)
    frac = max(0.0, 1.0 - float(step) / float(decay_steps))
    return floor + (float(max_weight) - floor) * frac

@torch.no_grad()
def evaluate_semantic(student, dataset, q_table, args, device):
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.semantic_eval_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    was_training = student.training
    student.eval()

    total = 0
    top1_sum = 0
    kl_sum = 0.0
    entropy_sum = 0.0

    for batch in loader:
        eeg = batch["eeg_data"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        labels = batch["cls_index"].to(
            device, dtype=torch.long, non_blocking=True
        )
        logits, probs, _ = student(eeg, temperature=args.student_temperature)
        kl = soft_semantic_kl(
            logits, labels, q_table, args.student_temperature
        )
        n = labels.shape[0]
        total += n
        top1_sum += int((logits.argmax(dim=-1) == labels).sum().item())
        kl_sum += float(kl) * n
        ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_sum += float(ent.sum())

    if was_training:
        student.train()
    if total == 0:
        raise RuntimeError("Empty semantic validation dataset")
    return {
        "n_samples": total,
        "top1": top1_sum / total,
        "soft_kl": kl_sum / total,
        "entropy": entropy_sum / total,
    }

def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def build_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = min(int(warmup_steps), max(0, int(total_steps) - 1))

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        remain = max(1, int(total_steps) - warmup_steps)
        return max(0.0, float(int(total_steps) - step) / float(remain))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)

def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch

def parse_args():
    p = argparse.ArgumentParser(
        description="Trial-invariant low-capacity EEG prototype-KL training"
    )
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument("--rendered_view_path", default="/data/jionkim/neuro_3D/render_grid_v4",
                   help="Legacy dataset-constructor argument only; no rendered items are fetched")
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--holdout_suffix", default="06", choices=[f"{i:02d}" for i in range(7)])
    p.add_argument("--feature_dim", type=int, default=128,
                   help="Low-capacity pooled representation width")
    p.add_argument("--student_hidden_dim", type=int, default=64)
    p.add_argument("--temporal_bins", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--teacher_semantic_temperature", type=float, default=0.10)
    p.add_argument("--student_temperature", type=float, default=0.50)
    p.add_argument("--lambda_soft_sem", type=float, default=1.0,
                   help="Prototype-KL weight; kept active throughout training")
    p.add_argument("--lambda_soft_min", type=float, default=None,
                   help="Default: retain lambda_soft_sem throughout training")
    p.add_argument("--soft_sem_decay_steps", type=int, default=0)
    p.add_argument("--lambda_trial_consistency", type=float, default=0.1,
                   help="Weight for same-(class,object) repeated-trial invariance")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batchsize", type=int, default=16,
                   help="Number of explicit trial pairs per update")
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--semantic_eval_batch_size", type=int, default=64)
    p.add_argument("--student_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--validate_every", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=5,
                   help="Number of validation events without KL improvement")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                   help="Minimum validation KL improvement to reset patience")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensorboard", action="store_true")
    # Explicit compatibility, not parse_known_args: typos must still fail.
    for name in ("oracle_ckpt", "config", "lambda_distill", "lambda_cross",
                 "val_diff_batches", "val_ensemble_batches", "precision",
                 "gradient_every", "timestep_bins", "same_condition_mse_tol"):
        p.add_argument("--" + name, default=None, help="Deprecated: ignored in KL-only mode")
    args = p.parse_args()
    for name in ("oracle_ckpt", "config", "lambda_distill", "lambda_cross",
                 "val_diff_batches", "val_ensemble_batches", "precision",
                 "gradient_every", "timestep_bins", "same_condition_mse_tol"):
        if getattr(args, name) is not None:
            print(f"[prototype-KL] --{name} ignored")
        delattr(args, name)
    return args


def save_student_checkpoint(path, student, optimizer, scheduler, step, args, metadata):
    torch.save({
        "student": student.state_dict(),
        "student_arch": "trial_invariant_low_capacity_v1",
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "step": step, "args": vars(args),
        "training_mode": "trial_invariant_prototype_kl",
        "extra": metadata,
    }, path)


def main():
    args = parse_args()
    for name in ("max_steps", "batchsize", "accumulation_steps", "semantic_eval_batch_size",
                 "print_every", "validate_every", "save_every", "student_hidden_dim",
                 "feature_dim", "early_stop_patience"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.num_workers < 0 or args.soft_sem_decay_steps < 0:
        raise ValueError("num_workers and soft_sem_decay_steps must be nonnegative")
    if args.early_stop_min_delta < 0:
        raise ValueError("early_stop_min_delta must be nonnegative")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("dropout must be in [0, 1)")
    if args.lambda_trial_consistency < 0:
        raise ValueError("lambda_trial_consistency must be nonnegative")
    if args.student_temperature <= 0 or args.teacher_semantic_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    if args.lambda_soft_sem <= 0 or (args.lambda_soft_min is not None and args.lambda_soft_min <= 0):
        raise ValueError("KL is the only objective: its weight and floor must be positive")
    soft_weight_at_step(0, args.lambda_soft_sem, args.soft_sem_decay_steps, args.lambda_soft_min)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    if (out_dir / "train_log.csv").exists() or (out_dir / "checkpoints" / "last.pt").exists():
        raise FileExistsError("Use a new --out_dir to avoid mixing prior experiments")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(
        f"[trial-invariant prototype-KL] device={device}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    train_suffixes = tuple(f"{i:02d}" for i in range(7) if f"{i:02d}" != args.holdout_suffix)
    val_suffixes = (args.holdout_suffix,)
    # Lazy import: no MVDiffusion / diffusers / Oracle model is imported here.
    from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG
    base = AllDataFeatureTwoEEG(
        data_path=args.data_path, sub_list=[args.sub_id], train=True,
        test_mean=False, num_frames=6, rendered_view_path=args.rendered_view_path,
        aug_data=False, strict_rendered_views=False,
    )
    expected_shape = (int(base.cls_num), int(base.obj_num), int(base.trails_num), 64, 600)
    if tuple(base.eeg_data.shape[1:]) != expected_shape:
        raise RuntimeError(f"Expected EEG [subject,class,object,trial,64,600]; got {base.eeg_data.shape}")
    train_data = EEGOnlyDataset(base, train_suffixes, "individual")
    train_pairs = TrialPairEEGDataset(base, train_suffixes)
    val_data = EEGOnlyDataset(base, val_suffixes, "individual")
    val_avg = EEGOnlyDataset(base, val_suffixes, "averaged")
    if not len(train_data) or not len(train_pairs) or not len(val_data):
        raise RuntimeError("Empty train/validation split")
    print(
        f"[split] train={train_suffixes} ({len(train_data)} individual, "
        f"{len(train_pairs)} trial pairs); val={val_suffixes} ({len(val_data)}); "
        "07/08/09 not sampled"
    )
    prototypes, categories, source_objects = build_fixed_category_text_prototypes(base, train_suffixes)
    if not torch.isfinite(prototypes).all() or (prototypes.norm(dim=-1) < 1e-6).any():
        raise RuntimeError("Invalid text prototypes")
    save_prototypes(out_dir, prototypes, categories, source_objects, train_suffixes)
    q_table = build_soft_semantic_table(prototypes.to(device), args.teacher_semantic_temperature).detach()
    student = TemporalEEGSemanticStudent(
        feature_dim=args.feature_dim,
        hidden_dim=args.student_hidden_dim,
        num_classes=len(categories),
        temporal_bins=args.temporal_bins,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.max_steps, args.warmup_steps)
    loader = DataLoader(train_pairs, batch_size=args.batchsize, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0, drop_last=False)
    batches = cycle_loader(loader)
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out_dir / "logs"))
    metadata = {
        "training_mode": "trial_invariant_prototype_kl", "categories": categories,
        "prototypes": prototypes.cpu(), "train_suffixes": train_suffixes,
        "holdout_suffix": args.holdout_suffix,
        "selection_metric": "validation_individual_prototype_kl_min_then_top1",
        "trial_invariance": "explicit repeated-trial cosine consistency",
        "lambda_trial_consistency": args.lambda_trial_consistency,
        "low_capacity": {
            "feature_dim": args.feature_dim,
            "hidden_dim": args.student_hidden_dim,
            "dropout": args.dropout,
            "temporal_bins": args.temporal_bins,
        },
        "raw_eeg_preprocessing": "base.eeg_data -> per-channel z-normalization",
    }
    best_val_kl = float("inf")
    best_val_top1 = -float("inf")
    best_step = 0
    bad_validations = 0
    early_stopped = False
    window_n = window_correct = 0
    window_kl = window_consistency = window_loss = 0.0
    for step in range(1, args.max_steps + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        weight = soft_weight_at_step(step - 1, args.lambda_soft_sem,
                                     args.soft_sem_decay_steps, args.lambda_soft_min)
        # Normalize by actual trial pairs, including a partial final batch.
        step_batches = [next(batches) for _ in range(args.accumulation_steps)]
        step_n = sum(len(batch["cls_index"]) for batch in step_batches)
        for batch in step_batches:
            eeg_a = batch["eeg_a"].to(device, dtype=torch.float32)
            eeg_b = batch["eeg_b"].to(device, dtype=torch.float32)
            labels = batch["cls_index"].to(device, dtype=torch.long)
            logits_a, _, feat_a = student(eeg_a, args.student_temperature)
            logits_b, _, feat_b = student(eeg_b, args.student_temperature)
            kl_a = soft_semantic_kl(
                logits_a, labels, q_table, args.student_temperature
            )
            kl_b = soft_semantic_kl(
                logits_b, labels, q_table, args.student_temperature
            )
            kl = 0.5 * (kl_a + kl_b)
            consistency = trial_consistency_loss(feat_a, feat_b)
            loss = weight * kl + args.lambda_trial_consistency * consistency
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            n = len(labels)
            (loss * (n / step_n)).backward()
            window_n += n
            window_correct += int((logits_a.argmax(-1) == labels).sum())
            window_correct += int((logits_b.argmax(-1) == labels).sum())
            window_kl += float(kl.detach()) * n
            window_consistency += float(consistency.detach()) * n
            window_loss += float(loss.detach()) * n
        grad_norm = clip_grad_norm_(student.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"),
                                    error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.print_every == 0 or step == args.max_steps:
            train_metrics = {"loss": window_loss/window_n, "kl": window_kl/window_n,
                             "trial_consistency": window_consistency/window_n,
                             "top1": window_correct/(2.0 * window_n),
                             "soft_weight": weight,
                             "grad_weighted": float(grad_norm), "grad_raw": float(grad_norm)/weight}
            print(f"[TRAIN] step={step:06d} KL={train_metrics['kl']:.5f} "
                  f"Pair={train_metrics['trial_consistency']:.5f} "
                  f"L={train_metrics['loss']:.6f} Top1={train_metrics['top1']:.4f} "
                  f"SoftW={weight:.6f} Grad={float(grad_norm):.5f}")
            if writer:
                for k,v in train_metrics.items(): writer.add_scalar("train/"+k, v, step)
            window_n = window_correct = 0
            window_kl = window_consistency = window_loss = 0.0
        if step % args.validate_every == 0 or step == args.max_steps:
            # DataLoader iterator construction can consume Torch RNG even without shuffle.
            with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
                ind = evaluate_semantic(student, val_data, q_table, args, device)
                avg = evaluate_semantic(student, val_avg, q_table, args, device)
            if not all(np.isfinite(v) for metrics in (ind,avg) for v in metrics.values()):
                raise RuntimeError("Non-finite semantic validation metrics")
            row = {"step": step, **{"val_ind_"+k:v for k,v in ind.items()},
                   **{"val_rawavg_"+k:v for k,v in avg.items()}}
            append_csv(out_dir / "train_log.csv", row)
            print(f"[VAL] step={step:06d} IND Top1={ind['top1']:.4f} KL={ind['soft_kl']:.5f} "
                  f"RAWAVG Top1={avg['top1']:.4f} KL={avg['soft_kl']:.5f} N={ind['n_samples']}")
            if writer:
                for k,v in row.items():
                    if k != "step": writer.add_scalar(k, v, step)
            metadata["val"] = row
            save_student_checkpoint(ckpt_dir/"last.pt", student, optimizer, scheduler, step, args, metadata)
            improved = (
                ind["soft_kl"] < best_val_kl - args.early_stop_min_delta
                or (
                    abs(ind["soft_kl"] - best_val_kl) <= args.early_stop_min_delta
                    and ind["top1"] > best_val_top1
                )
            )
            if improved:
                best_val_kl = ind["soft_kl"]
                best_val_top1 = ind["top1"]
                best_step = step
                bad_validations = 0
                save_student_checkpoint(ckpt_dir/"best.pt", student, optimizer, scheduler, step, args, metadata)
                print(
                    f"[BEST] step={step}; validation KL={best_val_kl:.5f}; "
                    f"Top1={best_val_top1:.4f}"
                )
            else:
                bad_validations += 1
                print(
                    f"[EARLY-STOP] no validation KL improvement "
                    f"({bad_validations}/{args.early_stop_patience})"
                )
                if bad_validations >= args.early_stop_patience:
                    early_stopped = True
                    print(f"[EARLY-STOP] stopping at step={step}")
                    break
        if step % args.save_every == 0:
            save_student_checkpoint(ckpt_dir/f"step_{step:06d}.pt", student, optimizer, scheduler, step, args, metadata)
    if writer: writer.close()
    if best_step == 0:
        raise RuntimeError("No validation checkpoint was selected")
    summary = {"training_mode": "trial_invariant_prototype_kl",
               "student_arch": "trial_invariant_low_capacity_v1",
               "best_step": best_step, "best_top1": best_val_top1,
               "best_kl": best_val_kl, "early_stopped": early_stopped,
               "bad_validations_at_stop": bad_validations,
               "train_suffixes": train_suffixes, "holdout_suffix": args.holdout_suffix,
               "selection_metric": metadata["selection_metric"],
               "trainable_parameters": sum(p.numel() for p in student.parameters()),
               "best_checkpoint": str(ckpt_dir/"best.pt")}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best_step={best_step}; best={ckpt_dir/'best.pt'}")


if __name__ == "__main__":
    main()
