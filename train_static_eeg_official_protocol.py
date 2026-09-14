#!/usr/bin/env python3
"""Train the static EEG 72-class decoder with the official Neuro-3D data protocol.

Official static-only data protocol
----------------------------------
* train array: [1, 72 classes, 8 objects, 2 repetitions, 64 channels, 250 samples]
  - objects 00--07 are all used for training
  - the two repetitions are treated as INDEPENDENT training samples
  - total training samples: 72 * 8 * 2 = 1152
* test array: [1, 72 classes, 2 objects, 4 repetitions, 64 channels, 250 samples]
  - objects 08--09 are the official held-out test objects
  - the four repetitions are averaged BEFORE inference
  - total test samples: 72 * 2 = 144
* no additional z-score, min-max, or feature scaling is applied in this script.
  The released EEG arrays are assumed to already contain the official preprocessing.

Model
-----
The EEG encoder is the spatio-temporal architecture used in our previous diagnostic:
    temporal filtering
    -> full-electrode spatial mixing
    -> temporal compression
    -> separable temporal filtering
    -> temporal-token Transformer
    -> learnable attention pooling
    -> latent projection
    -> 72-way classifier

Important evaluation note
-------------------------
There is NO validation split in the official 8-train-object / 2-test-object protocol.
Therefore training runs for a fixed number of epochs and the default behavior evaluates
on the official test set only once, after the final epoch. Use --test_every N only for
reproduction/diagnostic reporting; test metrics are never used for checkpoint selection.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


CLASS_COUNT = 72
TRAIN_OBJECTS = 8
TEST_OBJECTS = 2
TRAIN_REPETITIONS = 2
TEST_REPETITIONS = 4
CHANNELS = 64
SAMPLES = 250


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_official_arrays(data_path: str, sub_id: str):
    """Load the released Neuro-3D static EEG train/test arrays."""
    root = Path(data_path).expanduser() / "EEGdata" / sub_id
    paths = {
        "train": root / f"{sub_id}_train_data_1s_250Hz.npy",
        "test": root / f"{sub_id}_test_data_1s_250Hz.npy",
    }

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing EEG arrays:\n" + "\n".join(missing))

    arrays = {key: np.load(path, mmap_mode="r") for key, path in paths.items()}
    expected = {
        "train": (1, CLASS_COUNT, TRAIN_OBJECTS, TRAIN_REPETITIONS, CHANNELS, SAMPLES),
        "test": (1, CLASS_COUNT, TEST_OBJECTS, TEST_REPETITIONS, CHANNELS, SAMPLES),
    }

    for key, shape in expected.items():
        value = arrays[key]
        # Some distributions omit a singleton subject dimension.
        if tuple(value.shape) == shape[1:]:
            value = value[None]
            arrays[key] = value
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{key}: expected {shape} or {shape[1:]}, got {tuple(value.shape)}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")

    return arrays, {key: str(path) for key, path in paths.items()}


def make_official_train_features(raw):
    """Use all 8 train objects and both repetitions as independent samples.

    Input:
        raw: [1,72,8,2,64,250]
    Output:
        x:   [1152,64,250]
        y:   [1152]

    Flattening is class-major, then object-major, then repetition-major.
    No additional input normalization is applied.
    """
    x = np.asarray(raw)[0].astype(np.float32, copy=False)
    if x.shape != (
        CLASS_COUNT,
        TRAIN_OBJECTS,
        TRAIN_REPETITIONS,
        CHANNELS,
        SAMPLES,
    ):
        raise ValueError(f"Unexpected train tensor shape: {x.shape}")

    x = x.reshape(-1, CHANNELS, SAMPLES)
    y = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int64),
        TRAIN_OBJECTS * TRAIN_REPETITIONS,
    )

    object_ids = np.tile(
        np.repeat(np.arange(TRAIN_OBJECTS, dtype=np.int64), TRAIN_REPETITIONS),
        CLASS_COUNT,
    )
    repetition_ids = np.tile(
        np.arange(TRAIN_REPETITIONS, dtype=np.int64),
        CLASS_COUNT * TRAIN_OBJECTS,
    )

    if len(x) != CLASS_COUNT * TRAIN_OBJECTS * TRAIN_REPETITIONS:
        raise RuntimeError("Official train sample count mismatch")

    return x, y, object_ids, repetition_ids


def make_official_test_features(raw):
    """Average the four official test repetitions before inference.

    Input:
        raw: [1,72,2,4,64,250]
    Output:
        x:   [144,64,250]
        y:   [144]

    Repetitions are averaged in raw/preprocessed EEG space, exactly once.
    No additional input normalization is applied afterward.
    """
    x = np.asarray(raw)[0].astype(np.float32, copy=False)
    if x.shape != (
        CLASS_COUNT,
        TEST_OBJECTS,
        TEST_REPETITIONS,
        CHANNELS,
        SAMPLES,
    ):
        raise ValueError(f"Unexpected test tensor shape: {x.shape}")

    # [class, object, repetition, channel, time]
    # -> [class, object, channel, time]
    x = x.mean(axis=2, dtype=np.float32)
    x = x.reshape(-1, CHANNELS, SAMPLES)
    y = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int64),
        TEST_OBJECTS,
    )
    object_ids = np.tile(
        np.arange(TEST_OBJECTS, dtype=np.int64) + TRAIN_OBJECTS,
        CLASS_COUNT,
    )

    if len(x) != CLASS_COUNT * TEST_OBJECTS:
        raise RuntimeError("Official test sample count mismatch")

    return x, y, object_ids


class FeatureDataset(Dataset):
    def __init__(self, features, labels, noise_std=0.0, training=False):
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.noise_std = float(noise_std)
        self.training = bool(training)

        if self.features.ndim != 3:
            raise ValueError(
                f"Expected EEG features [N,{CHANNELS},{SAMPLES}], got {self.features.shape}"
            )
        if tuple(self.features.shape[1:]) != (CHANNELS, SAMPLES):
            raise ValueError(
                f"Expected EEG features [N,{CHANNELS},{SAMPLES}], got {self.features.shape}"
            )
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Feature/label mismatch: {self.features.shape} vs {self.labels.shape}"
            )

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, index):
        value = self.features[index]
        if self.training and self.noise_std > 0:
            value = value + torch.randn_like(value) * self.noise_std
        return value, self.labels[index]


class AttentionPool(nn.Module):
    """Learn a soft importance weight over temporal EEG tokens."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens):
        # tokens: [B,T,D]
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


def _group_norm(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class StaticEEGSpatioTemporal(nn.Module):
    """Spatio-temporal static EEG classifier for [B,64,250] inputs."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=64,
        dropout=0.1,
        classes=CLASS_COUNT,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ff_dim=256,
        conv_channels=32,
        temporal_kernel=31,
        separable_kernel=15,
        temporal_pool1=4,
        temporal_pool2=4,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.transformer_heads = int(transformer_heads)
        self.transformer_layers = int(transformer_layers)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.conv_channels = int(conv_channels)
        self.temporal_kernel = int(temporal_kernel)
        self.separable_kernel = int(separable_kernel)
        self.temporal_pool1 = int(temporal_pool1)
        self.temporal_pool2 = int(temporal_pool2)

        if self.transformer_heads <= 0 or self.transformer_layers <= 0:
            raise ValueError("transformer_heads and transformer_layers must be positive")
        if self.transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if hidden_dim % self.transformer_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"transformer_heads ({self.transformer_heads})"
            )
        if self.conv_channels < 8 or self.conv_channels % 8 != 0:
            raise ValueError("conv_channels must be >= 8 and divisible by 8")
        if self.temporal_kernel <= 1 or self.temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be an odd integer > 1")
        if self.separable_kernel <= 1 or self.separable_kernel % 2 == 0:
            raise ValueError("separable_kernel must be an odd integer > 1")
        if self.temporal_pool1 <= 0 or self.temporal_pool2 <= 0:
            raise ValueError("temporal_pool1 and temporal_pool2 must be positive")

        stem_channels = self.conv_channels // 2

        # 1) Temporal filtering while retaining all electrodes.
        # [B,1,64,250] -> [B,Ct,64,250]
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(
                1,
                stem_channels,
                kernel_size=(1, self.temporal_kernel),
                padding=(0, self.temporal_kernel // 2),
                bias=False,
            ),
            _group_norm(stem_channels, 4),
            nn.GELU(),
        )

        # 2) Spatial electrode mixing before heavy temporal compression.
        # [B,Ct,64,250] -> [B,Cs,1,~62]
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                self.conv_channels,
                kernel_size=(CHANNELS, 1),
                groups=stem_channels,
                bias=False,
            ),
            _group_norm(self.conv_channels, 8),
            nn.GELU(),
            nn.AvgPool2d(
                kernel_size=(1, self.temporal_pool1),
                stride=(1, self.temporal_pool1),
            ),
            nn.Dropout(dropout),
        )

        # 3) Separable temporal refinement + projection to Transformer width.
        self.separable_temporal = nn.Sequential(
            nn.Conv2d(
                self.conv_channels,
                self.conv_channels,
                kernel_size=(1, self.separable_kernel),
                padding=(0, self.separable_kernel // 2),
                groups=self.conv_channels,
                bias=False,
            ),
            nn.Conv2d(
                self.conv_channels,
                hidden_dim,
                kernel_size=(1, 1),
                bias=False,
            ),
            _group_norm(hidden_dim, 8),
            nn.GELU(),
            nn.AvgPool2d(
                kernel_size=(1, self.temporal_pool2),
                stride=(1, self.temporal_pool2),
            ),
            nn.Dropout(dropout),
        )

        first_steps = SAMPLES // self.temporal_pool1
        self.token_count = first_steps // self.temporal_pool2
        if self.token_count < 2:
            raise ValueError(
                "Temporal pooling is too aggressive: "
                f"{SAMPLES} -> {first_steps} -> {self.token_count} tokens"
            )

        self.position = nn.Parameter(
            torch.zeros(1, self.token_count, hidden_dim)
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.transformer_layers,
        )
        self.pool = AttentionPool(hidden_dim)
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(latent_dim, classes)
        self.last_attention_weights = None

    def forward(self, x):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], CHANNELS, SAMPLES)
        if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS, SAMPLES):
            raise ValueError(
                f"Expected raw EEG [B,{CHANNELS},{SAMPLES}], got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)                     # [B,1,64,250]
        x = self.temporal_encoder(x)          # [B,Ct,64,250]
        x = self.spatial_encoder(x)           # [B,Cs,1,T1]
        x = self.separable_temporal(x)        # [B,H,1,T2]
        tokens = x.squeeze(2).transpose(1, 2) # [B,T2,H]

        if tokens.shape[1] != self.token_count:
            raise RuntimeError(
                f"Token count mismatch: expected {self.token_count}, "
                f"got {tokens.shape[1]}"
            )

        tokens = self.transformer(tokens + self.position)
        pooled, attention = self.pool(tokens)
        self.last_attention_weights = attention.detach()
        latent = self.latent_projection(pooled)
        logits = self.head(latent)
        return logits, latent


def prototype_targets(prototypes, labels, temperature):
    p = prototypes / np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True),
        1e-8,
    )
    similarity = p @ p.T
    logits = torch.as_tensor(similarity[labels], dtype=torch.float32) / temperature
    return torch.softmax(logits, dim=-1)


def load_prototypes(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix == ".npy":
        value = np.load(path)
    else:
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            for key in ("prototypes", "prototype", "embeddings"):
                if key in value:
                    value = value[key]
                    break
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != CLASS_COUNT:
        raise ValueError(
            f"Prototype file must have shape [{CLASS_COUNT},D], got {value.shape}"
        )
    return value


def same_class_consistency(latent, labels):
    """Cosine consistency among same-class examples within a batch."""
    if latent.shape[0] < 2:
        return latent.new_zeros(())

    z = F.normalize(latent, dim=-1)
    losses = []
    for label in labels.unique():
        indices = torch.where(labels == label)[0]
        if indices.numel() < 2:
            continue
        sim = z[indices] @ z[indices].T
        mask = ~torch.eye(
            indices.numel(),
            dtype=torch.bool,
            device=latent.device,
        )
        losses.append(1.0 - sim[mask].mean())

    return torch.stack(losses).mean() if losses else latent.new_zeros(())


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    prototype_array=None,
    prototype_weight=0.0,
    temperature=2.0,
    label_smoothing=0.0,
    consistency_weight=0.0,
    train=False,
):
    model.train(train)

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_consistency = 0.0
    total_top1 = 0
    total_top5 = 0
    total_n = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits, latent = model(features)
            ce = F.cross_entropy(
                logits,
                labels,
                label_smoothing=label_smoothing,
            )

            kl = torch.zeros((), device=device)
            if prototype_array is not None and prototype_weight > 0:
                target = prototype_targets(
                    prototype_array,
                    labels.detach().cpu().numpy(),
                    temperature,
                ).to(device)
                kl = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    target,
                    reduction="batchmean",
                ) * (temperature ** 2)

            consistency = torch.zeros((), device=device)
            if consistency_weight > 0:
                consistency = same_class_consistency(latent, labels)

            loss = (
                ce
                + prototype_weight * kl
                + consistency_weight * consistency
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        n = labels.numel()
        total_loss += float(loss.item()) * n
        total_ce += float(ce.item()) * n
        total_kl += float(kl.item()) * n
        total_consistency += float(consistency.item()) * n
        total_top1 += int((logits.argmax(1) == labels).sum().item())
        total_top5 += int(
            logits.topk(5, dim=1)
            .indices.eq(labels[:, None])
            .any(dim=1)
            .sum()
            .item()
        )
        total_n += n

    if total_n == 0:
        raise RuntimeError("Empty DataLoader")

    return {
        "loss": total_loss / total_n,
        "ce": total_ce / total_n,
        "kl": total_kl / total_n,
        "consistency": total_consistency / total_n,
        "top1": total_top1 / total_n,
        "top5": total_top5 / total_n,
        "n": total_n,
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clone_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(
    path,
    model,
    args,
    history,
    epoch,
    train_stats,
    test_stats,
    data_paths,
):
    config = {
        "protocol": "neuro3d_official_static_only",
        "input_representation": "raw_waveform",
        "input_shape": [CHANNELS, SAMPLES],
        "input_norm": "none",
        "additional_feature_scaling": "identity",
        "train_objects": list(range(8)),
        "train_repetitions": "independent_0_1",
        "test_objects": [8, 9],
        "test_repetitions": "mean_of_4_before_inference",
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ff_dim": args.transformer_ff_dim,
        "conv_channels": args.conv_channels,
        "temporal_kernel": args.temporal_kernel,
        "separable_kernel": args.separable_kernel,
        "temporal_pool1": args.temporal_pool1,
        "temporal_pool2": args.temporal_pool2,
        "prototype_weight": args.prototype_weight,
        "temperature": args.temperature,
        "label_smoothing": args.label_smoothing,
        "consistency_weight": args.consistency_weight,
    }

    torch.save(
        {
            "model": clone_state_dict(model),
            "config": config,
            # Kept for compatibility with prior inference utilities.
            "scaler_mean": np.zeros((1,), dtype=np.float32),
            "scaler_scale": np.ones((1,), dtype=np.float32),
            "class_count": CLASS_COUNT,
            "args": vars(args),
            "history": history,
            "epoch": epoch,
            "train_stats": train_stats,
            "test_stats": test_stats,
            "data_paths": data_paths,
        },
        path,
    )


def train(args):
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )

    arrays, data_paths = load_official_arrays(args.data_path, args.sub_id)
    train_x, train_y, train_object_ids, train_rep_ids = make_official_train_features(
        arrays["train"]
    )
    test_x, test_y, test_object_ids = make_official_test_features(arrays["test"])

    # Explicitly guarantee that this script does not apply any extra scaling.
    if not np.shares_memory(train_x, np.asarray(arrays["train"])):
        # astype(copy=False) followed by reshape usually still shares memory, but
        # this is not a correctness requirement. The important point is that the
        # values are not normalized or standardized here.
        pass

    train_loader = DataLoader(
        FeatureDataset(
            train_x,
            train_y,
            noise_std=args.feature_noise_std,
            training=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        FeatureDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        FeatureDataset(test_x, test_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = StaticEEGSpatioTemporal(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        conv_channels=args.conv_channels,
        temporal_kernel=args.temporal_kernel,
        separable_kernel=args.separable_kernel,
        temporal_pool1=args.temporal_pool1,
        temporal_pool2=args.temporal_pool2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    prototypes = load_prototypes(args.prototype_path)

    print(
        "[data] protocol=neuro3d_official_static_only "
        f"train_objects=00-07 train_repetitions=independent train_samples={len(train_y)}; "
        f"test_objects=08-09 test_repetitions=mean4 test_samples={len(test_y)}; "
        f"input=raw_waveform input_shape={(CHANNELS, SAMPLES)} "
        f"input_norm=none additional_feature_scaling=identity; device={device}",
        flush=True,
    )
    print(
        f"[raw-train] range=[{float(train_x.min()):.6f}, {float(train_x.max()):.6f}] "
        f"mean={float(train_x.mean()):.6f} std={float(train_x.std()):.6f}",
        flush=True,
    )
    print(
        f"[raw-test-mean4] range=[{float(test_x.min()):.6f}, {float(test_x.max()):.6f}] "
        f"mean={float(test_x.mean()):.6f} std={float(test_x.std()):.6f}",
        flush=True,
    )
    print(
        f"[chance] top1={1 / CLASS_COUNT:.4%} top5={5 / CLASS_COUNT:.4%}",
        flush=True,
    )

    history = []
    started = time.time()
    last_train_stats = None
    last_test_stats = None

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            prototype_array=prototypes,
            prototype_weight=args.prototype_weight,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            consistency_weight=args.consistency_weight,
            train=True,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_kl": train_stats["kl"],
            "train_consistency": train_stats["consistency"],
            "train_top1": train_stats["top1"],
            "train_top5": train_stats["top5"],
        }

        should_test = (
            args.test_every > 0
            and (epoch % args.test_every == 0 or epoch == args.epochs)
        )
        test_stats = None
        if should_test:
            test_stats = run_epoch(
                model,
                test_loader,
                None,
                device,
                prototype_array=prototypes,
                prototype_weight=args.prototype_weight,
                temperature=args.temperature,
                label_smoothing=args.label_smoothing,
                consistency_weight=0.0,
                train=False,
            )
            row.update(
                {
                    "test_loss": test_stats["loss"],
                    "test_ce": test_stats["ce"],
                    "test_kl": test_stats["kl"],
                    "test_top1": test_stats["top1"],
                    "test_top5": test_stats["top5"],
                }
            )

        history.append(row)
        last_train_stats = train_stats
        if test_stats is not None:
            last_test_stats = test_stats

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            message = (
                f"[EPOCH {epoch:03d}] "
                f"train_top1={train_stats['top1']:.4%} "
                f"train_top5={train_stats['top5']:.4%} "
                f"loss={train_stats['loss']:.4f}"
            )
            if test_stats is not None:
                message += (
                    f" test_top1={test_stats['top1']:.4%}"
                    f" test_top5={test_stats['top5']:.4%}"
                    f" test_loss={test_stats['loss']:.4f}"
                )
            print(message, flush=True)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                out / f"epoch_{epoch:03d}.pt",
                model,
                args,
                history,
                epoch,
                train_stats,
                test_stats,
                data_paths,
            )

    # Clean protocol: official test is evaluated once here unless it was already
    # evaluated on the last epoch via --test_every.
    if last_test_stats is None or not (
        args.test_every > 0 and args.epochs % args.test_every == 0
    ):
        last_test_stats = run_epoch(
            model,
            test_loader,
            None,
            device,
            prototype_array=prototypes,
            prototype_weight=args.prototype_weight,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            consistency_weight=0.0,
            train=False,
        )
        print(
            f"[TEST final] top1={last_test_stats['top1']:.4%} "
            f"top5={last_test_stats['top5']:.4%} "
            f"loss={last_test_stats['loss']:.4f}",
            flush=True,
        )

    # Evaluate deterministic train performance once with dropout disabled.
    final_train_eval = run_epoch(
        model,
        train_eval_loader,
        None,
        device,
        prototype_array=prototypes,
        prototype_weight=args.prototype_weight,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
        consistency_weight=0.0,
        train=False,
    )

    save_checkpoint(
        out / "last.pt",
        model,
        args,
        history,
        args.epochs,
        final_train_eval,
        last_test_stats,
        data_paths,
    )
    write_csv(out / "train_log.csv", history)

    # Save sample-index metadata so the exact official protocol can be audited.
    np.savez(
        out / "sample_index_metadata.npz",
        train_labels=train_y,
        train_object_ids=train_object_ids,
        train_repetition_ids=train_rep_ids,
        test_labels=test_y,
        test_object_ids=test_object_ids,
    )

    summary = {
        "protocol": "Neuro-3D official static-only data protocol",
        "subject": args.sub_id,
        "train": {
            "objects": [f"{i:02d}" for i in range(8)],
            "repetitions": [0, 1],
            "repetition_handling": "independent_samples",
            "samples": int(len(train_y)),
        },
        "test": {
            "objects": ["08", "09"],
            "repetitions": [0, 1, 2, 3],
            "repetition_handling": "mean_before_inference",
            "samples": int(len(test_y)),
        },
        "input": {
            "representation": "raw_waveform",
            "shape": [CHANNELS, SAMPLES],
            "additional_normalization": "none",
            "additional_feature_scaling": "identity",
        },
        "selection": {
            "validation_split": None,
            "early_stopping": False,
            "checkpoint_selection": "fixed_final_epoch",
            "test_every": args.test_every,
            "note": (
                "Periodic official-test evaluation, when requested, is reporting only "
                "and is never used to select the checkpoint."
            ),
        },
        "epochs": args.epochs,
        "final_train_eval": final_train_eval,
        "final_test": last_test_stats,
        "data_paths": data_paths,
        "seconds": time.time() - started,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"[done] checkpoint={out / 'last.pt'}; "
        f"final_train_top1={final_train_eval['top1']:.4%}; "
        f"final_test_top1={last_test_stats['top1']:.4%}; "
        f"final_test_top5={last_test_stats['top5']:.4%}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    parser.add_argument("--sub_id", default="sub01")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_ff_dim", type=int, default=256)
    parser.add_argument("--conv_channels", type=int, default=32)
    parser.add_argument("--temporal_kernel", type=int, default=31)
    parser.add_argument("--separable_kernel", type=int, default=15)
    parser.add_argument("--temporal_pool1", type=int, default=4)
    parser.add_argument("--temporal_pool2", type=int, default=4)

    parser.add_argument("--feature_noise_std", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--prototype_path", default=None)
    parser.add_argument("--prototype_weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument(
        "--test_every",
        type=int,
        default=0,
        help=(
            "0 = evaluate official test only after final epoch (recommended); "
            "N > 0 = additionally report official test every N epochs. "
            "Test metrics are never used for checkpoint selection."
        ),
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save epoch checkpoints every N epochs; 0 disables periodic saving.",
    )
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.print_every < 1:
        raise ValueError("--print_every must be >= 1")
    if args.test_every < 0:
        raise ValueError("--test_every must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save_every must be >= 0")

    train(args)


if __name__ == "__main__":
    main()
