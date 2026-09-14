#!/usr/bin/env python3
"""Train a leakage-safe static-EEG 72-class semantic decoder with an EEG spatial-temporal encoder.

Validation protocols:
  * object_disjoint (default): train on objects 00--05 and validate on 06--07
    by default; validation uses the two-trial mean for each held-out object.
  * trial_split: use the same objects 00--07 in train/validation, but train on
    one trial and validate on the other trial. This is a diagnostic protocol
    for separating trial generalization from unseen-object generalization.
  * objects 08--09 in the official test array remain untouched by both modes.

For trial_split, trial averaging is intentionally disabled because averaging
trial 0 and trial 1 would leak validation information into training.
Inference is implemented separately in inference_neuralpp.py.
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
CHANNELS = 64
SAMPLES = 250
FEATURE_BINS = 25


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_official_arrays(data_path, sub_id):
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
        "train": (1, CLASS_COUNT, 8, 2, CHANNELS, SAMPLES),
        "test": (1, CLASS_COUNT, 2, 4, CHANNELS, SAMPLES),
    }
    for key, shape in expected.items():
        value = arrays[key]
        if tuple(value.shape) == shape[1:]:
            value = value[None]
            arrays[key] = value
        if tuple(value.shape) != shape:
            raise ValueError(f"{key}: expected {shape} or {shape[1:]}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")
    return arrays, {key: str(value) for key, value in paths.items()}


def parse_objects(value):
    if value.strip().lower() in {"none", "all"}:
        return []
    try:
        values = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid object list: {value}") from exc
    if not values or len(set(values)) != len(values) or any(i < 0 or i >= 8 for i in values):
        raise ValueError(f"Objects must be unique values in 00..07: {value}")
    return values


def normalize_eeg(x, mode="channel_zscore"):
    """Normalize each input without using statistics from another split."""
    x = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return x
    if mode == "sample_minmax":
        # One scale per EEG example over all channels and time points.  This
        # reproduces the earlier explicit [0,1] normalization while retaining
        # relative amplitudes between channels within an example.
        minimum = x.min(axis=(-2, -1), keepdims=True)
        maximum = x.max(axis=(-2, -1), keepdims=True)
        return ((x - minimum) / np.maximum(maximum - minimum, 1e-6)).astype(np.float32)
    if mode == "channel_zscore":
        mean = x.mean(axis=-1, keepdims=True)
        scale = x.std(axis=-1, keepdims=True)
    elif mode == "global_zscore":
        mean = x.mean(axis=(-2, -1), keepdims=True)
        scale = x.std(axis=(-2, -1), keepdims=True)
    else:
        raise ValueError(f"Unknown input normalization: {mode}")
    return ((x - mean) / np.maximum(scale, 1e-6)).astype(np.float32)


def temporal_features(x, bins=FEATURE_BINS):
    """Convert [...,64,250] EEG to [...,3200] mean/std features."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-2:] != (CHANNELS, SAMPLES):
        raise ValueError(f"Expected [...,{CHANNELS},{SAMPLES}], got {x.shape}")
    if SAMPLES % bins != 0:
        raise ValueError(f"bins={bins} must divide {SAMPLES}")
    blocks = x.reshape(*x.shape[:-1], bins, x.shape[-1] // bins)
    means = blocks.mean(axis=-1)
    stds = blocks.std(axis=-1)
    return np.concatenate([means, stds], axis=-1).reshape(*x.shape[:-2], -1)


def preprocess_eeg(x, input_norm="channel_zscore", bins=FEATURE_BINS):
    return temporal_features(normalize_eeg(x, input_norm), bins=bins)


def fit_scaler(features):
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


def apply_scaler(features, mean, scale):
    return ((features - mean) / scale).astype(np.float32)


def make_train_features(raw, objects, train_mode="individual", input_norm="channel_zscore",
                        input_representation="temporal_stats"):
    """Create individual-trial, trial-mean, or combined training examples."""
    if train_mode not in {"individual", "mean", "both"}:
        raise ValueError(f"Invalid train_mode: {train_mode}")
    selected = np.take(np.asarray(raw)[0], objects, axis=1)
    # selected: [class, object, trial, channel, time]
    pieces = []
    if train_mode in {"individual", "both"}:
        pieces.append(normalize_eeg(selected, input_norm) if input_representation == "raw_waveform"
                      else preprocess_eeg(selected, input_norm=input_norm))
    if train_mode in {"mean", "both"}:
        mean_raw = selected.mean(axis=2, keepdims=True)
        pieces.append(normalize_eeg(mean_raw, input_norm) if input_representation == "raw_waveform"
                      else preprocess_eeg(mean_raw, input_norm=input_norm))
    if input_representation not in {"raw_waveform", "temporal_stats"}:
        raise ValueError(f"Unknown input representation: {input_representation}")
    features = np.concatenate(pieces, axis=2)
    repetitions = features.shape[2]
    x = features.reshape(-1, *features.shape[3:])
    y = np.repeat(np.arange(CLASS_COUNT), len(objects) * repetitions).astype(np.int64)
    return x, y


def make_trial_mean_features(raw, objects, input_norm="channel_zscore",
                             input_representation="temporal_stats"):
    selected = np.take(np.asarray(raw)[0], objects, axis=1)
    mean_raw = selected.mean(axis=2)
    features = (normalize_eeg(mean_raw, input_norm)
                if input_representation == "raw_waveform"
                else preprocess_eeg(mean_raw, input_norm=input_norm))
    if input_representation not in {"raw_waveform", "temporal_stats"}:
        raise ValueError(f"Unknown input representation: {input_representation}")
    x = features.reshape(-1, *features.shape[2:])
    y = np.repeat(np.arange(CLASS_COUNT), len(objects)).astype(np.int64)
    object_ids = np.tile(np.asarray(objects), CLASS_COUNT)
    return x, y, object_ids


def make_trial_split_features(raw, trial_idx, objects=None, input_norm="channel_zscore",
                              input_representation="raw_waveform"):
    """Create same-object/different-trial examples without trial leakage.

    raw is expected to have shape [1, class, object, trial, channel, time].
    A single trial is selected, while every requested object is retained.
    """
    if trial_idx not in {0, 1}:
        raise ValueError(f"trial_idx must be 0 or 1, got {trial_idx}")
    if objects is None:
        objects = list(range(8))
    objects = list(objects)
    if not objects or len(set(objects)) != len(objects) or any(i < 0 or i >= 8 for i in objects):
        raise ValueError(f"objects must be unique values in 00..07, got {objects}")

    # [class, object, channel, time]
    selected = np.take(np.asarray(raw)[0, :, :, trial_idx], objects, axis=1)
    if input_representation == "raw_waveform":
        features = normalize_eeg(selected, input_norm)
    elif input_representation == "temporal_stats":
        features = preprocess_eeg(selected, input_norm=input_norm)
    else:
        raise ValueError(f"Unknown input representation: {input_representation}")

    # Flatten class/object while preserving class-major ordering.
    # x: [72 * len(objects), ...]
    x = features.reshape(-1, *features.shape[2:])
    y = np.repeat(np.arange(CLASS_COUNT), len(objects)).astype(np.int64)
    object_ids = np.tile(np.asarray(objects, dtype=np.int64), CLASS_COUNT)
    trial_ids = np.full(y.shape, trial_idx, dtype=np.int64)
    return x, y, object_ids, trial_ids


class FeatureDataset(Dataset):
    def __init__(self, features, labels, noise_std=0.0, training=False):
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.noise_std = float(noise_std)
        self.training = bool(training)
        if self.features.ndim not in {2, 3} or self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(f"Feature/label mismatch: {self.features.shape}, {self.labels.shape}")

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, index):
        value = self.features[index]
        if self.training and self.noise_std > 0:
            value = value + torch.randn_like(value) * self.noise_std
        return value, self.labels[index]


class AttentionPool(nn.Module):
    """Learn a soft importance weight over EEG tokens instead of uniform averaging."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens):
        # tokens: [B, T, D]
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


def _group_norm(channels, max_groups=8):
    """Choose the largest valid GroupNorm group count up to max_groups."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class StaticMLP(nn.Module):
    """Static EEG decoder.

    For raw EEG, the encoder follows an EEG-specific order:
      temporal filtering -> full-electrode spatial mixing -> temporal compression
      -> separable temporal filtering -> time-token Transformer -> attention pool.

    This avoids independently compressing each electrode before spatial interaction.
    """
    def __init__(self, input_dim=3200, hidden_dim=128, latent_dim=64,
                 dropout=0.4, classes=CLASS_COUNT, use_transformer=False,
                 token_dim=50, transformer_heads=4, transformer_layers=2,
                 transformer_ff_dim=256, input_representation="temporal_stats",
                 conv_channels=32, conv_pooled_steps=4,
                 temporal_kernel=31, separable_kernel=15,
                 temporal_pool1=4, temporal_pool2=4):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_transformer = bool(use_transformer)
        self.input_representation = str(input_representation)
        self.token_dim = int(token_dim)
        self.transformer_heads = int(transformer_heads)
        self.transformer_layers = int(transformer_layers)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.conv_channels = int(conv_channels)
        # Kept in the checkpoint for backward CLI compatibility. Raw EEG no longer
        # uses AdaptiveAvgPool(..., conv_pooled_steps).
        self.conv_pooled_steps = int(conv_pooled_steps)
        self.temporal_kernel = int(temporal_kernel)
        self.separable_kernel = int(separable_kernel)
        self.temporal_pool1 = int(temporal_pool1)
        self.temporal_pool2 = int(temporal_pool2)

        if self.use_transformer:
            if self.transformer_heads <= 0 or self.transformer_layers <= 0:
                raise ValueError("transformer_heads and transformer_layers must be positive")
            if self.transformer_ff_dim <= 0:
                raise ValueError("transformer_ff_dim must be positive")
            if hidden_dim % self.transformer_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by transformer_heads "
                    f"({self.transformer_heads})"
                )

            if self.input_representation == "raw_waveform":
                if input_dim != CHANNELS * SAMPLES:
                    raise ValueError(
                        f"raw_waveform requires input_dim={CHANNELS * SAMPLES}, got {input_dim}"
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
                if self.conv_channels % stem_channels != 0:
                    raise ValueError("conv_channels must be divisible by stem_channels")

                # 1) Temporal filtering while preserving all 64 electrodes.
                #    Input [B,1,64,250] -> [B,Ct,64,250].
                self.temporal_encoder = nn.Sequential(
                    nn.Conv2d(
                        1, stem_channels,
                        kernel_size=(1, self.temporal_kernel),
                        padding=(0, self.temporal_kernel // 2),
                        bias=False,
                    ),
                    _group_norm(stem_channels, 4),
                    nn.GELU(),
                )

                # 2) EEG spatial mixing before heavy temporal pooling.
                #    Depthwise spatial filters learn a different electrode
                #    combination for each temporal feature map.
                #    [B,Ct,64,250] -> [B,conv_channels,1,250].
                self.spatial_encoder = nn.Sequential(
                    nn.Conv2d(
                        stem_channels, self.conv_channels,
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

                # 3) Separable temporal refinement after electrodes have been mixed.
                #    Pointwise projection moves features to Transformer hidden_dim.
                self.separable_temporal = nn.Sequential(
                    nn.Conv2d(
                        self.conv_channels, self.conv_channels,
                        kernel_size=(1, self.separable_kernel),
                        padding=(0, self.separable_kernel // 2),
                        groups=self.conv_channels,
                        bias=False,
                    ),
                    nn.Conv2d(
                        self.conv_channels, hidden_dim,
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

                # Pooling is floor division because kernel_size == stride.
                first_steps = SAMPLES // self.temporal_pool1
                self.token_count = first_steps // self.temporal_pool2
                if self.token_count < 2:
                    raise ValueError(
                        "Temporal pooling is too aggressive: "
                        f"250 -> {first_steps} -> {self.token_count} tokens"
                    )
                self.encoder = None

            elif self.input_representation == "temporal_stats":
                if self.token_dim <= 0 or input_dim % self.token_dim != 0:
                    raise ValueError(
                        f"input_dim ({input_dim}) must be divisible by token_dim ({self.token_dim})"
                    )
                self.token_count = input_dim // self.token_dim
                self.encoder = nn.Sequential(
                    nn.Linear(self.token_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            else:
                raise ValueError(f"Unknown input representation: {self.input_representation}")

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
                layer, num_layers=self.transformer_layers
            )
            self.pool = AttentionPool(hidden_dim)
            self.latent_projection = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            if self.input_representation != "temporal_stats":
                raise ValueError("raw_waveform requires use_transformer=True")
            self.token_count = 1
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.head = nn.Linear(latent_dim, classes)
        self.last_attention_weights = None

    def forward(self, x):
        if self.use_transformer:
            if self.input_representation == "raw_waveform":
                if x.ndim == 2:
                    x = x.reshape(x.shape[0], CHANNELS, SAMPLES)
                if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS, SAMPLES):
                    raise ValueError(
                        f"Expected raw EEG [B,{CHANNELS},{SAMPLES}], got {tuple(x.shape)}"
                    )
                x = x.unsqueeze(1)                         # [B,1,64,250]
                x = self.temporal_encoder(x)              # [B,Ct,64,250]
                x = self.spatial_encoder(x)               # [B,Cs,1,~62]
                x = self.separable_temporal(x)            # [B,H,1,~15]
                tokens = x.squeeze(2).transpose(1, 2)     # [B,T,H]
            else:
                tokens = x.reshape(x.shape[0], self.token_count, self.token_dim)
                tokens = self.encoder(tokens)

            if tokens.shape[1] != self.token_count:
                raise RuntimeError(
                    f"Token count mismatch: expected {self.token_count}, got {tokens.shape[1]}"
                )
            tokens = self.transformer(tokens + self.position)
            pooled, attention = self.pool(tokens)
            self.last_attention_weights = attention.detach()
            latent = self.latent_projection(pooled)
        else:
            latent = self.encoder(x)
        return self.head(latent), latent


def prototype_targets(prototypes, labels, temperature):
    p = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    similarity = p @ p.T
    logits = torch.as_tensor(similarity[labels], dtype=torch.float32) / temperature
    return torch.softmax(logits, dim=-1)


def load_prototypes(path):
    if path is None:
        return None
    path = Path(path).expanduser()
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
        raise ValueError(f"Prototype file must have shape [72,D], got {value.shape}")
    return value


def same_class_consistency(latent, labels):
    """Cosine consistency for repeated examples of the same class in a batch."""
    if latent.shape[0] < 2:
        return latent.new_zeros(())
    z = F.normalize(latent, dim=-1)
    losses = []
    for label in labels.unique():
        indices = torch.where(labels == label)[0]
        if indices.numel() < 2:
            continue
        sim = z[indices] @ z[indices].T
        mask = ~torch.eye(indices.numel(), dtype=torch.bool, device=latent.device)
        losses.append(1.0 - sim[mask].mean())
    return torch.stack(losses).mean() if losses else latent.new_zeros(())


def run_epoch(model, loader, optimizer, device, prototype_array=None,
              prototype_weight=0.0, temperature=2.0, label_smoothing=0.1,
              consistency_weight=0.05, train=False):
    model.train(train)
    total_loss = total_ce = total_kl = total_consistency = 0.0
    total_top1 = total_top5 = total_n = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits, latent = model(features)
            ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            kl = torch.zeros((), device=device)
            if prototype_array is not None and prototype_weight > 0:
                target = prototype_targets(
                    prototype_array, labels.detach().cpu().numpy(), temperature
                ).to(device)
                kl = F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1),
                    target,
                    reduction="batchmean",
                ) * (temperature ** 2)
            consistency = same_class_consistency(latent, labels)
            loss = ce + prototype_weight * kl + consistency_weight * consistency
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
        total_top5 += int(logits.topk(5, 1).indices.eq(labels[:, None]).any(1).sum().item())
        total_n += n
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


def metric_improved(current, best, monitor, min_delta):
    if best is None:
        return True
    if monitor == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def train(args):
    if args.final_train_all and args.validation_protocol == "trial_split":
        raise ValueError(
            "--final_train_all is incompatible with --validation_protocol trial_split. "
            "Use object_disjoint for final training on all objects."
        )

    out = Path(args.out_dir).expanduser().resolve()
    # if out.exists() and any(out.iterdir()):
        # raise FileExistsError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    )
    arrays, paths = load_official_arrays(args.data_path, args.sub_id)

    val_loader = None
    val_x = val_y = None
    train_trial = None
    val_trial = None

    if args.validation_protocol == "trial_split":
        # Same objects in both splits; only trial identity differs.
        train_objects = list(range(8))
        val_objects = list(range(8))
        train_trial = int(args.train_trial)
        val_trial = 1 - train_trial
        effective_train_mode = "single_trial"

        train_x_raw, train_y, _, _ = make_trial_split_features(
            arrays["train"],
            trial_idx=train_trial,
            objects=train_objects,
            input_norm=args.input_norm,
            input_representation=args.input_representation,
        )
        val_x_raw, val_y, _, _ = make_trial_split_features(
            arrays["train"],
            trial_idx=val_trial,
            objects=val_objects,
            input_norm=args.input_norm,
            input_representation=args.input_representation,
        )
    else:
        if args.final_train_all:
            val_objects = []
            train_objects = list(range(8))
        else:
            val_objects = parse_objects(args.val_objects)
            train_objects = [i for i in range(8) if i not in val_objects]
            if not train_objects or not val_objects:
                raise ValueError("Validation training needs non-empty train and val objects")
        effective_train_mode = args.train_mode

        train_x_raw, train_y = make_train_features(
            arrays["train"], train_objects, args.train_mode, args.input_norm,
            args.input_representation,
        )
        if val_objects:
            val_x_raw, val_y, _ = make_trial_mean_features(
                arrays["train"], val_objects, input_norm=args.input_norm,
                input_representation=args.input_representation,
            )

    # Fit feature-wise scaling on training data only. Raw waveform already uses
    # sample-level normalization and therefore keeps identity scaling here.
    if args.input_representation == "raw_waveform":
        scaler_mean = np.zeros((1,), dtype=np.float32)
        scaler_scale = np.ones((1,), dtype=np.float32)
    else:
        scaler_mean, scaler_scale = fit_scaler(train_x_raw)

    train_x = apply_scaler(train_x_raw, scaler_mean, scaler_scale)
    train_loader = DataLoader(
        FeatureDataset(
            train_x, train_y,
            noise_std=args.feature_noise_std,
            training=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if val_y is not None:
        val_x = apply_scaler(val_x_raw, scaler_mean, scaler_scale)
        val_loader = DataLoader(
            FeatureDataset(val_x, val_y),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    prototypes = load_prototypes(args.prototype_path)
    model = StaticMLP(
        input_dim=int(np.prod(train_x.shape[1:])),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        use_transformer=args.use_transformer,
        token_dim=args.token_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        input_representation=args.input_representation,
        conv_channels=args.conv_channels,
        conv_pooled_steps=args.conv_pooled_steps,
        temporal_kernel=args.temporal_kernel,
        separable_kernel=args.separable_kernel,
        temporal_pool1=args.temporal_pool1,
        temporal_pool2=args.temporal_pool2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    best_state = None
    best_value = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()

    if args.validation_protocol == "trial_split":
        print(
            f"[data] protocol=trial_split "
            f"train_objects={','.join(f'{i:02d}' for i in train_objects)} train_trial={train_trial} "
            f"train_samples={len(train_y)}; "
            f"val_objects={','.join(f'{i:02d}' for i in val_objects)} val_trial={val_trial} "
            f"val_samples={len(val_y)}; "
            f"input={args.input_representation} input_shape={tuple(train_x.shape[1:])} "
            f"input_norm={args.input_norm}; device={device}",
            flush=True,
        )
    else:
        print(
            f"[data] protocol=object_disjoint "
            f"train_objects={','.join(f'{i:02d}' for i in train_objects)} "
            f"train_mode={args.train_mode} train_samples={len(train_y)}; "
            f"val_objects={','.join(f'{i:02d}' for i in val_objects) if val_objects else 'none'} "
            f"val_samples={len(val_y) if val_y is not None else 0}; "
            f"input={args.input_representation} input_shape={tuple(train_x.shape[1:])} "
            f"input_norm={args.input_norm}; device={device}",
            flush=True,
        )

    if args.input_representation == "raw_waveform":
        print(
            f"[raw] range=[{float(train_x.min()):.6f}, {float(train_x.max()):.6f}]; "
            "additional feature scaling=identity",
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model, train_loader, optimizer, device, prototypes,
            args.prototype_weight, args.temperature, args.label_smoothing,
            args.consistency_weight, train=True,
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

        if val_loader is not None:
            val_stats = run_epoch(
                model, val_loader, None, device, prototypes,
                args.prototype_weight, args.temperature, args.label_smoothing,
                0.0, train=False,
            )
            row.update({
                "val_loss": val_stats["loss"],
                "val_ce": val_stats["ce"],
                "val_kl": val_stats["kl"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
            })
            current = {
                "val_loss": val_stats["loss"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
            }[args.monitor]
            if metric_improved(current, best_value, args.monitor, args.min_delta):
                best_value = current
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(row)
        should_print = (
            epoch == 1
            or epoch % args.print_every == 0
            or epoch == args.epochs
            or (val_loader is not None and stale >= args.patience)
        )
        if should_print:
            extra = ""
            if val_loader is not None:
                extra = (
                    f" val_top1={row['val_top1']:.4%}"
                    f" val_top5={row['val_top5']:.4%}"
                    f" val_loss={row['val_loss']:.4f}"
                    f" stale={stale}/{args.patience}"
                )
            print(
                f"[EPOCH {epoch:03d}] train_top1={row['train_top1']:.4%}"
                f" train_top5={row['train_top5']:.4%}"
                f" loss={row['train_loss']:.4f}{extra}",
                flush=True,
            )
        if val_loader is not None and stale >= args.patience:
            print(
                f"[EARLY-STOP] epoch={epoch}; best_epoch={best_epoch}; "
                f"monitor={args.monitor}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint state was produced")
    model.load_state_dict(best_state)

    checkpoint_config = {
        "input_dim": int(np.prod(train_x.shape[1:])),
        "input_shape": list(train_x.shape[1:]),
        "input_representation": args.input_representation,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "use_transformer": args.use_transformer,
        "token_dim": args.token_dim,
        "transformer_heads": args.transformer_heads,
        "transformer_layers": args.transformer_layers,
        "transformer_ff_dim": args.transformer_ff_dim,
        "conv_channels": args.conv_channels,
        "conv_pooled_steps": args.conv_pooled_steps,
        "temporal_kernel": args.temporal_kernel,
        "separable_kernel": args.separable_kernel,
        "temporal_pool1": args.temporal_pool1,
        "temporal_pool2": args.temporal_pool2,
        "feature_bins": FEATURE_BINS,
        "input_norm": args.input_norm,
        "train_mode": effective_train_mode,
        "include_trial_mean": (
            args.validation_protocol == "object_disjoint" and args.train_mode == "both"
        ),
        "validation_protocol": args.validation_protocol,
        "train_trial": train_trial,
        "val_trial": val_trial,
        "prototype_weight": args.prototype_weight,
        "temperature": args.temperature,
        "label_smoothing": args.label_smoothing,
        "consistency_weight": args.consistency_weight,
    }
    torch.save({
        "model": model.state_dict(),
        "config": checkpoint_config,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "class_count": CLASS_COUNT,
        "args": vars(args),
        "history": history,
        "best_epoch": best_epoch,
    }, out / "best.pt")
    write_csv(out / "train_log.csv", history)

    if args.validation_protocol == "trial_split":
        protocol_description = (
            "static 1s EEG; same-object/different-trial validation; "
            "official test objects remain untouched"
        )
    else:
        protocol_description = (
            "static 1s EEG; object-disjoint validation; test inference is separate"
        )

    summary = {
        "protocol": protocol_description,
        "validation_protocol": args.validation_protocol,
        "train_objects": [f"{i:02d}" for i in train_objects],
        "validation_objects": [f"{i:02d}" for i in val_objects],
        "train_trial": train_trial,
        "validation_trial": val_trial,
        "train_mode": effective_train_mode,
        "input_representation": args.input_representation,
        "input_norm": args.input_norm,
        "monitor": args.monitor if val_loader is not None else "none",
        "best_epoch": best_epoch,
        "final_train_all": args.final_train_all,
        "prototype_path": args.prototype_path,
        "prototype_weight": args.prototype_weight,
        "data_paths": paths,
        "seconds": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] checkpoint={out / 'best.pt'}; best_epoch={best_epoch}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    parser.add_argument("--sub_id", default="sub01")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--validation_protocol", "--validation-protocol",
        choices=["object_disjoint", "trial_split"],
        default="object_disjoint",
        help=(
            "object_disjoint: held-out objects for validation; "
            "trial_split: same objects but different trials for train/validation."
        ),
    )
    parser.add_argument(
        "--val_objects", default="06,07",
        help="Validation objects for object_disjoint mode only.",
    )
    parser.add_argument(
        "--train_trial", "--train-trial", type=int, choices=[0, 1], default=0,
        help="Training trial for trial_split mode; validation uses 1-train_trial.",
    )
    parser.add_argument("--final_train_all", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--input_representation", "--input-representation",
        choices=["raw_waveform", "temporal_stats"], default="raw_waveform",
        help="raw_waveform keeps [64,250] EEG and uses the spatial-temporal encoder.",
    )
    parser.add_argument(
        "--use_transformer", "--use-transformer", dest="use_transformer",
        action=argparse.BooleanOptionalAction, default=True,
        help="Use the Transformer stage after EEG feature extraction.",
    )
    parser.add_argument(
        "--token_dim", "--token-dim", type=int, default=50,
        help="Values per token for temporal_stats; 50 = 25 means + 25 stds per channel.",
    )
    parser.add_argument("--transformer_heads", "--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer_layers", "--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer_ff_dim", "--transformer-ff-dim", type=int, default=256)
    parser.add_argument(
        "--conv_channels", "--conv-channels", type=int, default=32,
        help="Spatial feature channels after full-electrode convolution.",
    )
    parser.add_argument(
        "--conv_pooled_steps", "--conv-pooled-steps", type=int, default=4,
        help="Deprecated for raw_waveform; kept for checkpoint/CLI compatibility.",
    )
    parser.add_argument(
        "--temporal_kernel", "--temporal-kernel", type=int, default=31,
        help="First temporal Conv2D kernel (odd samples).",
    )
    parser.add_argument(
        "--separable_kernel", "--separable-kernel", type=int, default=15,
        help="Depthwise separable temporal kernel (odd samples).",
    )
    parser.add_argument(
        "--temporal_pool1", "--temporal-pool1", type=int, default=4,
        help="Temporal pooling factor after spatial electrode mixing.",
    )
    parser.add_argument(
        "--temporal_pool2", "--temporal-pool2", type=int, default=4,
        help="Second temporal pooling factor before the Transformer.",
    )
    parser.add_argument(
        "--train_mode", choices=["individual", "mean", "both"], default="individual",
        help=(
            "Training examples for object_disjoint mode. Ignored in trial_split, "
            "where exactly one trial is used to prevent leakage."
        ),
    )
    parser.add_argument(
        "--include_trial_mean", "--include-trial-mean", dest="include_trial_mean",
        action=argparse.BooleanOptionalAction, default=None,
        help="Legacy alias affecting object_disjoint mode only.",
    )
    parser.add_argument(
        "--input_norm",
        choices=["none", "sample_minmax", "channel_zscore", "global_zscore"],
        default="sample_minmax",
    )
    parser.add_argument("--feature_noise_std", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--consistency_weight", type=float, default=0.0)
    parser.add_argument("--prototype_path", default=None)
    parser.add_argument("--prototype_weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--monitor", choices=["val_loss", "val_top1", "val_top5"], default="val_loss"
    )
    parser.add_argument("--patience", type=int, default=10000)
    parser.add_argument("--min_delta", type=float, default=5e-4)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.include_trial_mean is not None:
        if args.validation_protocol == "trial_split":
            raise ValueError(
                "--include_trial_mean cannot be used with trial_split because it mixes train/val trials."
            )
        if args.train_mode != "individual":
            raise ValueError("Use either --train_mode or --include_trial_mean, not both")
        args.train_mode = "both" if args.include_trial_mean else "individual"

    if args.print_every < 1:
        raise ValueError("--print_every must be >= 1")
    train(args)


if __name__ == "__main__":
    main()
