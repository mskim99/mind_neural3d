#!/usr/bin/env python3
"""
Two-stage Neuro-3D training with object-invariant semantic supervision.

Why this revision exists
------------------------
The previous 4-way diagnostic showed:

    train individual  -> ~93% Top-1
    train averaged    -> ~69% Top-1
    test individual   -> ~chance
    test averaged     -> ~chance + prediction collapse

Therefore the dominant failure is unseen-object semantic generalization, while
trial averaging is a secondary distribution shift.  This trainer keeps the
existing EEG->Zero123++ architecture but changes the semantic objective so that
the encoder is explicitly encouraged to learn category-invariant features.

Stage 1: fixed-category semantic warm-up
    EEG -> semantic / variation / bias
          |
          +-> CLIP image/text alignment
          +-> FIXED 72-way CLIP-text prototype CE       (PRIMARY)
          +-> learned 72-way CE                          (weak auxiliary)
          +-> cross-object supervised contrastive loss   (weak auxiliary)
          +-> individual<->trial-average consistency
          +-> orthogonality
    Zero123++ LoRA / generation heads are frozen.

Stage 2: joint generation
    semantic  -> Zero123++ cross-attention
    variation -> Zero123++ spatial cond_lat
    + diffusion
    + all semantic objectives above

Key design details
------------------
1. Cross-object positive:
       anchor EEG and a DIFFERENT object from the SAME semantic class.
2. SupCon memory bank:
       supports batch_size=1 training by providing negatives from prior steps.
3. Averaging consistency:
       raw trials of the anchor object are averaged BEFORE the EEG encoder and
       the resulting semantic feature is pulled toward the individual feature.
4. Fixed semantic probe:
       periodically evaluates train-individual / train-averaged /
       test-individual / test-averaged using train semantic prototypes.
       The most important monitored metric is test-individual prototype margin.

Recommended first run on a 24-GB GPU:
    CUDA_VISIBLE_DEVICES=0 python train_neural3d_pp_semantic_invariant.py \
        --config ./configs/mind3d_pp.yaml \
        --sub_id sub01 \
        --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
        --batchsize 1 \
        --accumulation_steps 2 \
        --semantic_stage_steps 5000 \
        --max_steps 60000 \
        --out_dir stage2_semantic_invariant
"""

# -------------------------------------------------------------------------
# GPU selection MUST happen before importing torch.
# -------------------------------------------------------------------------
import os
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf

from src.mvdiffusion_var_semantic_cls_sg import MVDiffusion, unscale_image
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG
from src.utils import set_random_seed


# =========================================================================
# Utilities
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="EEG->Zero123++ training with fixed category semantic prototypes"
    )

    p.add_argument("--config", default="./configs/mind3d_pp.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument("--rendered_view_path",
                   default="/data/jionkim/neuro_3D/eeg3d_training")
    p.add_argument("--sub_id", default="0001")
    p.add_argument("--out_dir", default="stage2_semantic_fixedproto")

    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help=(
            "LoRA/diffusion learning rate. If omitted, the script searches "
            "the YAML for learning_rate / training.learning_rate / "
            "model.learning_rate / model.base_learning_rate. "
            "Falls back to 1e-5 only if none is present."
        ),
    )

    # Optimizer-step counts, NOT micro-batch iterations.
    p.add_argument("--semantic_stage_steps", type=int, default=15000)
    p.add_argument("--max_steps", type=int, default=60000)

    # ------------------------------------------------------------------
    # Semantic-objective weights.
    #
    # Previous setting:
    #   CLIP=0.5, CE=1.0
    # made the hard 72-way classifier dominate despite very poor unseen-
    # object generalization.  The revised default makes CLIP + invariance
    # objectives primary and keeps CE only as an auxiliary boundary loss.
    # ------------------------------------------------------------------
    p.add_argument("--lambda_diff", type=float, default=1.0)
    p.add_argument("--lambda_clip", type=float, default=1.0)
    p.add_argument("--lambda_cls", type=float, default=0.10)
    p.add_argument("--lambda_proto", type=float, default=0.75)
    p.add_argument("--lambda_supcon", type=float, default=0.20)
    p.add_argument("--lambda_avg_consistency", type=float, default=0.25)
    p.add_argument("--lambda_ortho", type=float, default=0.05)
    p.add_argument("--cls_label_smoothing", type=float, default=0.05)

    # Fixed 72-way category geometry. Prototypes are built ONLY from the
    # training split's precomputed CLIP text features and never optimized.
    p.add_argument("--prototype_temperature", type=float, default=0.07)
    p.add_argument(
        "--prototype_anchor_only",
        action="store_true",
        help=(
            "Apply fixed-prototype CE only to the anchor semantic feature. "
            "Default applies it to anchor + cross-object positive + trial-average "
            "features so all semantic views share the same fixed category space."
        ),
    )

    # SupCon is retained only as a secondary local invariance objective.
    p.add_argument("--supcon_temperature", type=float, default=0.07)
    p.add_argument("--supcon_queue_size", type=int, default=2048)

    p.add_argument("--grad_clip", type=float, default=1.0)

    # Logging / validation.
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--validate_every", type=int, default=200)
    p.add_argument("--visualize_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--vis_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=4.0)

    # Full EEG-only semantic probe.  0 disables it.
    p.add_argument("--semantic_probe_every", type=int, default=2000)
    p.add_argument("--semantic_probe_batchsize", type=int, default=64)

    # Stage-2 diagnostic: block ONLY diffusion-loss gradients from flowing
    # into fmri_encoder. Semantic objectives still update the EEG encoder.
    p.add_argument(
        "--stop_diffusion_grad_to_eeg",
        action="store_true",
        help=(
            "In joint stage, detach semantic/variation EEG features only for "
            "the diffusion-conditioning branch. CLIP/CE/SupCon/AvgCons/Ortho "
            "still update fmri_encoder."
        ),
    )

    p.add_argument(
        "--resume_model_only",
        action="store_true",
        help=(
            "When changing the semantic objective, load model weights and step "
            "from --resume but reset optimizer/scheduler/SupCon-bank state. "
            "Recommended for a quick continuation from the old 15k checkpoint."
        ),
    )
    p.add_argument("--resume", default="",
                   help="Resume from a checkpoint produced by the invariant/stop-grad trainer.")
    p.add_argument("--aug_data", action="store_true")
    p.add_argument("--audit_only", action="store_true")

    return p.parse_args()


def unwrap(model):
    return model



def _find_key_recursive(cfg, target_key):
    """
    Recursively find every occurrence of `target_key` in an OmegaConf tree.

    Returns:
        list[(path, value)]
    """
    container = OmegaConf.to_container(cfg, resolve=False)
    matches = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == target_key:
                    matches.append((child_path, value))
                walk(value, child_path)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                child_path = f"{path}[{idx}]"
                walk(value, child_path)

    walk(container)
    return matches


def resolve_model_configs(cfg, config_path, cli_learning_rate=None):
    """
    Resolve the MinD-3D++ / Zero123++ training config.

    IMPORTANT:
      configs/mind3d.yaml    -> original MinD-3D config (WRONG for this trainer)
      configs/mind3d_pp.yaml -> MinD-3D++ Zero123++ config (CORRECT)

    The expected structure is:
        learning_rate: ...
        model:
          params:
            stable_diffusion_config: ...
            fmri_encoder_config: ...

    Fail explicitly instead of trying to reinterpret the original MinD-3D YAML.
    """
    top_keys = list(cfg.keys()) if OmegaConf.is_dict(cfg) else []
    print(f"[config] loaded: {config_path}")
    print(f"[config] top-level keys: {top_keys}")

    stable_cfg = OmegaConf.select(
        cfg,
        "model.params.stable_diffusion_config",
        default=None,
    )
    fmri_cfg = OmegaConf.select(
        cfg,
        "model.params.fmri_encoder_config",
        default=None,
    )

    if stable_cfg is None:
        # Detect the original MinD-3D YAML from its characteristic keys.
        original_mind3d_keys = {
            "diff_prior_config_path",
            "fmri_model",
            "3d_model",
        }
        if original_mind3d_keys.intersection(set(top_keys)):
            raise KeyError(
                "\nWrong config file for EEG -> Zero123++ training.\n\n"
                f"Loaded: {config_path}\n"
                f"Top-level keys: {top_keys}\n\n"
                "This is the original MinD-3D config (`mind3d.yaml`). "
                "The current MVDiffusion/Zero123++ trainer requires "
                "`configs/mind3d_pp.yaml`, which contains:\n"
                "  model.params.stable_diffusion_config\n"
                "  model.params.fmri_encoder_config\n\n"
                "Run with:\n"
                "  --config ./configs/mind3d_pp.yaml"
            )

        raise KeyError(
            "\n`model.params.stable_diffusion_config` is missing.\n"
            f"Loaded config: {config_path}\n"
            f"Top-level keys: {top_keys}\n"
            "Use the MinD-3D++ config: ./configs/mind3d_pp.yaml"
        )

    if fmri_cfg is None:
        # Current EEG disentangling encoder does not consume this config directly,
        # but preserve the official interface and warn rather than crash.
        print(
            "[config warning] model.params.fmri_encoder_config not found; "
            "passing None because the current EEG encoder is constructed explicitly."
        )

    # Preserve the original project's behavior:
    # MVDiffusion reads `args.learning_rate` from the first argument.
    if cli_learning_rate is not None:
        cfg.learning_rate = float(cli_learning_rate)
        print(
            f"[config] overriding learning_rate with CLI: "
            f"{cfg.learning_rate}"
        )
    elif OmegaConf.select(cfg, "learning_rate", default=None) is None:
        # mind3d_pp.yaml should contain this. Fail rather than silently inventing it.
        raise KeyError(
            "`learning_rate` is missing from the MinD-3D++ config. "
            "Pass --learning_rate explicitly."
        )

    print(
        "[config] stable_diffusion_config: "
        "model.params.stable_diffusion_config"
    )
    print(
        "[config] fmri_encoder_config    : "
        "model.params.fmri_encoder_config"
    )
    print(f"[config] learning_rate          : {cfg.learning_rate}")

    return stable_cfg, fmri_cfg, cfg


def dump_dataset_mapping(dataset, path):
    """
    Save the exact EEG class-row -> rendered object mapping before training.
    This is a preflight audit artifact, not a learned prediction.
    """
    rows = []
    for cls_index in range(dataset.name_list.shape[0]):
        for obj_index in range(dataset.name_list.shape[1]):
            name = str(dataset.name_list[cls_index, obj_index])
            rows.append({
                "cls_index": cls_index,
                "obj_index": obj_index,
                "class_prefix": name[:3],
                "dataset_name": name,
                "label": name[3:],
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cls_index", "obj_index", "class_prefix",
                "dataset_name", "label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)



# =========================================================================
# Fixed category semantic space
# =========================================================================

def _category_from_dataset_name(name):
    """Example: 001_airplane_00 -> airplane."""
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[0]
    return key


def _to_flat_feature(x):
    if torch.is_tensor(x):
        t = x.detach().float().cpu()
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return t.reshape(-1)


def build_fixed_category_text_prototypes(train_dataset, device, expected_dim=1024):
    """
    Build ONE frozen CLIP-text prototype per semantic class using ONLY train
    objects (00..07).  Each object-level text feature is L2-normalized first,
    then the class mean is normalized again.

    This makes the target semantic geometry external/fixed instead of allowing
    the EEG encoder and a learned classifier to invent a train-object-specific
    geometry together.
    """
    prototypes = []
    categories = []
    source_objects = []

    for c in range(int(train_dataset.cls_num)):
        feats = []
        cat_names = []
        object_names = []

        for o in range(int(train_dataset.obj_num)):
            dataset_name = str(train_dataset.name_list[c, o])
            key = dataset_name[3:]
            category = _category_from_dataset_name(dataset_name)

            # Guard against accidental final-test semantic leakage.
            suffix = key.rsplit("_", 1)[-1] if "_" in key else ""
            if suffix in {"08", "09"}:
                raise RuntimeError(
                    f"Fixed prototype builder saw held-out object {key!r} in "
                    "the training dataset. Check the train/test split."
                )

            if not hasattr(train_dataset, "clip_features"):
                raise AttributeError(
                    "train_dataset must expose `clip_features` so fixed text "
                    "prototypes can be built in the exact CLIP space already "
                    "used by this project."
                )
            if key not in train_dataset.clip_features:
                raise KeyError(f"Missing CLIP feature for train object {key!r}.")

            feat = _to_flat_feature(train_dataset.clip_features[key]["text"])
            if feat.numel() != expected_dim:
                raise RuntimeError(
                    f"Expected CLIP text dim={expected_dim}, got {feat.numel()} "
                    f"for {key!r}."
                )
            feats.append(F.normalize(feat, dim=0))
            cat_names.append(category)
            object_names.append(key)

        if len(set(cat_names)) != 1:
            raise RuntimeError(
                f"cls_index={c} maps to multiple categories: {sorted(set(cat_names))}"
            )

        proto = F.normalize(torch.stack(feats, dim=0).mean(dim=0), dim=0)
        prototypes.append(proto)
        categories.append(cat_names[0])
        source_objects.append(object_names)

    prototypes = torch.stack(prototypes, dim=0).to(
        device=device, dtype=torch.float32
    )
    prototypes = prototypes.detach()
    prototypes.requires_grad_(False)

    if prototypes.shape != (72, expected_dim):
        raise RuntimeError(
            f"Expected fixed prototype matrix [72,{expected_dim}], "
            f"got {tuple(prototypes.shape)}"
        )

    return prototypes, categories, source_objects


def save_fixed_category_prototypes(
    out_dir, prototypes, categories, source_objects
):
    out_dir = Path(out_dir)
    torch.save(
        {
            "prototypes": prototypes.detach().cpu(),
            "categories": list(categories),
            "source_objects": list(source_objects),
            "construction": (
                "L2-normalize each TRAIN object CLIP text feature -> class mean "
                "-> L2-normalize; no test 08/09 features"
            ),
        },
        out_dir / "fixed_category_text_prototypes.pt",
    )

    with open(
        out_dir / "fixed_category_text_prototypes.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cls_index", "category", "source_train_objects"],
        )
        writer.writeheader()
        for c, (cat, objs) in enumerate(zip(categories, source_objects)):
            writer.writerow({
                "cls_index": c,
                "category": cat,
                "source_train_objects": ";".join(objs),
            })


def fixed_category_prototype_loss(
    semantic_features, labels, prototypes, temperature
):
    """72-way CE against NON-TRAINABLE category text prototypes."""
    if temperature <= 0:
        raise ValueError("prototype_temperature must be > 0")

    z = F.normalize(semantic_features.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    y = labels.long()

    cosine = z @ p.t()
    logits = cosine / float(temperature)
    loss = F.cross_entropy(logits, y)

    with torch.no_grad():
        pred = cosine.argmax(dim=1)
        rows = torch.arange(len(y), device=y.device)
        correct = cosine[rows, y]
        wrong = cosine.clone()
        wrong[rows, y] = -torch.inf
        best_wrong = wrong.max(dim=1).values
        margin = correct - best_wrong
        metrics = {
            "top1": float((pred == y).float().mean().item()),
            "correct_cosine": float(correct.mean().item()),
            "margin": float(margin.mean().item()),
            "positive_margin_rate": float((margin > 0).float().mean().item()),
        }

    return loss, metrics


# =========================================================================
# Object-invariant semantic training helpers
# =========================================================================

class SemanticFeatureCapture:
    """
    Forward-hook that captures the semantic feature produced by fmri_encoder
    during the NORMAL model(batch) forward.

    This avoids re-running the anchor EEG through the encoder just to compute
    SupCon / consistency losses.
    """

    def __init__(self, fmri_encoder):
        self.last_semantic = None
        self.handle = fmri_encoder.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 1:
            raise RuntimeError(
                "fmri_encoder hook expected (semantic, variation, bias)."
            )
        self.last_semantic = output[0]

    def clear(self):
        self.last_semantic = None

    def close(self):
        self.handle.remove()


class SupConMemoryBank:
    """
    Supervised-contrastive memory bank.

    Why a memory bank?
    ------------------
    The diffusion model is normally trained with batch_size=1.  Standard
    in-batch SupCon would have almost no negatives.  We therefore contrast
    the current anchor/positive pair against detached semantic features from
    previous micro-batches.

    Same-class queued samples are treated as additional POSITIVES, never false
    negatives.
    """

    def __init__(self, dim, queue_size, temperature, device):
        if queue_size < 0:
            raise ValueError("supcon_queue_size must be >= 0")
        if temperature <= 0:
            raise ValueError("supcon_temperature must be > 0")

        self.dim = int(dim)
        self.queue_size = int(queue_size)
        self.temperature = float(temperature)
        self.device = torch.device(device)

        self.features = torch.zeros(
            self.queue_size, self.dim,
            device=self.device, dtype=torch.float32,
        )
        self.labels = torch.full(
            (self.queue_size,),
            -1,
            device=self.device, dtype=torch.long,
        )
        self.ptr = 0
        self.count = 0

    def __len__(self):
        return int(self.count)

    def compute_loss(self, anchor, positive, labels):
        """
        anchor, positive: [B,D], both trainable.
        labels: [B]

        Current anchor and its cross-object positive are always a positive pair.
        The bank contributes detached positives/negatives.
        """
        anchor = F.normalize(anchor.float(), dim=-1)
        positive = F.normalize(positive.float(), dim=-1)
        labels = labels.long()

        current = torch.cat([anchor, positive], dim=0)  # [2B,D]
        current_labels = torch.cat([labels, labels], dim=0)

        if self.count > 0:
            bank_feat = self.features[:self.count].detach()
            bank_labels = self.labels[:self.count].detach()
            contrast = torch.cat([current, bank_feat], dim=0)
            contrast_labels = torch.cat(
                [current_labels, bank_labels], dim=0
            )
        else:
            contrast = current
            contrast_labels = current_labels

        logits = current @ contrast.t()
        logits = logits / self.temperature

        n_current = current.shape[0]
        valid = torch.ones_like(logits, dtype=torch.bool)

        # Exclude only exact self-comparisons for the current vectors.
        idx = torch.arange(n_current, device=logits.device)
        valid[idx, idx] = False

        positive_mask = (
            current_labels[:, None] == contrast_labels[None, :]
        ) & valid

        positive_count = positive_mask.sum(dim=1)
        if torch.any(positive_count == 0):
            raise RuntimeError(
                "SupCon anchor has no positive. The cross-object paired view "
                "should guarantee at least one positive per anchor."
            )

        masked_logits = logits.masked_fill(~valid, -torch.inf)
        log_denom = torch.logsumexp(masked_logits, dim=1)
        log_prob = logits - log_denom[:, None]

        pos_log_prob = torch.where(
            positive_mask,
            log_prob,
            torch.zeros_like(log_prob),
        )
        mean_log_prob_pos = (
            pos_log_prob.sum(dim=1) / positive_count.float()
        )

        return -mean_log_prob_pos.mean()

    @torch.no_grad()
    def enqueue(self, features, labels):
        if self.queue_size == 0:
            return

        features = F.normalize(
            features.detach().float(), dim=-1
        )
        labels = labels.detach().long()

        if features.shape[0] != labels.shape[0]:
            raise ValueError("features / labels batch mismatch")

        # If a single update is larger than the entire queue, keep its tail.
        if features.shape[0] >= self.queue_size:
            features = features[-self.queue_size:]
            labels = labels[-self.queue_size:]
            self.features.copy_(features)
            self.labels.copy_(labels)
            self.ptr = 0
            self.count = self.queue_size
            return

        n = features.shape[0]
        end = self.ptr + n

        if end <= self.queue_size:
            self.features[self.ptr:end] = features
            self.labels[self.ptr:end] = labels
        else:
            first = self.queue_size - self.ptr
            self.features[self.ptr:] = features[:first]
            self.labels[self.ptr:] = labels[:first]
            remain = n - first
            self.features[:remain] = features[first:]
            self.labels[:remain] = labels[first:]

        self.ptr = (self.ptr + n) % self.queue_size
        self.count = min(self.queue_size, self.count + n)

    def state_dict(self):
        return {
            "features": self.features[:self.count].detach().cpu(),
            "labels": self.labels[:self.count].detach().cpu(),
            "ptr": int(self.ptr),
            "count": int(self.count),
            "queue_size": int(self.queue_size),
            "dim": int(self.dim),
            "temperature": float(self.temperature),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if not state:
            return

        feat = state.get("features", None)
        lab = state.get("labels", None)
        if feat is None or lab is None:
            return

        feat = feat.float().to(self.device)
        lab = lab.long().to(self.device)

        n = min(len(feat), self.queue_size)
        if n > 0:
            self.features[:n] = feat[-n:]
            self.labels[:n] = lab[-n:]

        self.count = n
        self.ptr = n % max(self.queue_size, 1) if self.queue_size > 0 else 0


def _batch_index_numpy(batch, key, batch_size, default=0):
    if key not in batch:
        return np.full((batch_size,), int(default), dtype=np.int64)

    value = batch[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    value = np.asarray(value).reshape(-1).astype(np.int64)
    if len(value) != batch_size:
        raise RuntimeError(
            f"Batch metadata `{key}` has {len(value)} items, "
            f"expected {batch_size}."
        )
    return value


def build_semantic_auxiliary_views(train_dataset, batch, device):
    """
    Build TWO extra EEG views per anchor without touching the diffusion input.

    positive_eeg:
        same cls_index, DIFFERENT object instance, random trial.

    averaged_eeg:
        same object as anchor, raw trials averaged before encoder.

    This is the key object/trial invariance intervention.
    """
    if "cls_index" not in batch:
        raise KeyError("Training batch must contain `cls_index`.")
    if "obj_index" not in batch:
        raise KeyError(
            "Training batch must contain `obj_index` for cross-object SupCon."
        )

    labels = batch["cls_index"]
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    labels = labels.long().to(device)
    B = int(labels.shape[0])

    cls_idx = _batch_index_numpy(batch, "cls_index", B)
    obj_idx = _batch_index_numpy(batch, "obj_index", B)
    sub_idx = _batch_index_numpy(batch, "subject_index", B, default=0)

    raw = train_dataset.eeg_data
    if raw.ndim != 6:
        raise RuntimeError(
            "Expected train_dataset.eeg_data [S,C,O,R,64,600], "
            f"got {raw.shape}."
        )

    num_objects = int(raw.shape[2])
    num_trials = int(raw.shape[3])

    if num_objects < 2:
        raise RuntimeError(
            "Cross-object positive sampling requires >=2 train objects/class."
        )

    positive = []
    averaged = []

    for b in range(B):
        s = int(sub_idx[b])
        c = int(cls_idx[b])
        o = int(obj_idx[b])

        # Guaranteed DIFFERENT object from the same category.
        offset = np.random.randint(1, num_objects)
        pos_obj = (o + offset) % num_objects
        pos_trial = np.random.randint(0, num_trials)

        pos = np.asarray(
            raw[s, c, pos_obj, pos_trial],
            dtype=np.float32,
        )

        avg = np.asarray(
            raw[s, c, o, :],
            dtype=np.float32,
        ).mean(axis=0, dtype=np.float32)

        positive.append(pos)
        averaged.append(avg)

    positive = torch.from_numpy(
        np.stack(positive, axis=0)
    ).to(device=device, dtype=torch.float32, non_blocking=True)

    averaged = torch.from_numpy(
        np.stack(averaged, axis=0)
    ).to(device=device, dtype=torch.float32, non_blocking=True)

    return positive, averaged, labels


def cosine_consistency_loss(anchor_sem, averaged_sem):
    """
    Pull trial-averaged EEG semantics toward the individual-trial semantics.
    Norm is intentionally removed so the objective acts on semantic direction.
    """
    return (
        1.0
        - F.cosine_similarity(
            anchor_sem.float(),
            averaged_sem.float(),
            dim=-1,
        )
    ).mean()


# =========================================================================
# Full semantic probe (encoder-only; no VAE / UNet)
# =========================================================================

@torch.no_grad()
def _encode_raw_condition(
    model,
    dataset,
    mode,
    device,
    batch_size,
):
    """
    mode:
      individual -> each raw trial separately
      averaged   -> raw trial mean before encoder
    """
    raw = dataset.eeg_data
    if raw.ndim != 6:
        raise RuntimeError(
            f"Expected [S,C,O,R,64,600], got {raw.shape}"
        )

    S, C, O, R, E, T = map(int, raw.shape)

    indices = []
    for s in range(S):
        for c in range(C):
            for o in range(O):
                if mode == "individual":
                    for r in range(R):
                        indices.append((s, c, o, r))
                elif mode == "averaged":
                    indices.append((s, c, o, None))
                else:
                    raise ValueError(mode)

    features = []
    targets = []

    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        eeg_np = []
        y = []

        for s, c, o, r in chunk:
            if r is None:
                x = np.asarray(
                    raw[s, c, o, :],
                    dtype=np.float32,
                ).mean(axis=0, dtype=np.float32)
            else:
                x = np.asarray(
                    raw[s, c, o, r],
                    dtype=np.float32,
                )
            eeg_np.append(x)
            y.append(c)

        eeg = torch.from_numpy(
            np.stack(eeg_np, axis=0)
        ).to(device=device, dtype=torch.float32)

        sem, _, _ = model.fmri_encoder(eeg)
        features.append(sem.float().cpu())
        targets.append(
            torch.tensor(y, dtype=torch.long)
        )

    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _build_semantic_prototypes(features, targets, num_classes=72):
    proto = []
    for c in range(num_classes):
        x = features[targets == c].float()
        if len(x) == 0:
            raise RuntimeError(f"Probe has no train samples for class {c}.")
        p = F.normalize(x.mean(dim=0), dim=-1)
        proto.append(p)
    return torch.stack(proto, dim=0)


def _prototype_probe_metrics(features, targets, prototypes):
    features = features.float()
    targets = targets.long()

    logits = (
        F.normalize(features, dim=-1)
        @ F.normalize(prototypes.float(), dim=-1).t()
    )

    pred = logits.argmax(dim=1)
    top1 = (pred == targets).float().mean()

    top5_idx = logits.topk(
        k=min(5, logits.shape[1]), dim=1
    ).indices
    top5 = (
        top5_idx == targets[:, None]
    ).any(dim=1).float().mean()

    rows = torch.arange(len(targets))
    correct = logits[rows, targets]

    wrong = logits.clone()
    wrong[rows, targets] = -torch.inf
    best_wrong = wrong.max(dim=1).values
    margin = correct - best_wrong

    return {
        "top1": float(top1.item()),
        "top5": float(top5.item()),
        "correct_cosine": float(correct.mean().item()),
        "margin": float(margin.mean().item()),
        "positive_margin_rate": float((margin > 0).float().mean().item()),
        "semantic_norm": float(features.norm(dim=-1).mean().item()),
    }


@torch.no_grad()
def run_semantic_probe(
    model,
    train_dataset,
    test_raw_dataset,
    device,
    batch_size,
    fixed_text_prototypes=None,
):
    """
    Rebuild CURRENT train semantic prototypes, then evaluate all four EEG
    conditions.  This directly tracks the failure discovered by the 4-way audit.
    """
    was_training = model.training
    model.eval()

    train_ind_f, train_ind_y = _encode_raw_condition(
        model, train_dataset, "individual",
        device, batch_size,
    )
    prototypes = _build_semantic_prototypes(
        train_ind_f, train_ind_y, num_classes=72
    )

    train_avg_f, train_avg_y = _encode_raw_condition(
        model, train_dataset, "averaged",
        device, batch_size,
    )
    test_ind_f, test_ind_y = _encode_raw_condition(
        model, test_raw_dataset, "individual",
        device, batch_size,
    )
    test_avg_f, test_avg_y = _encode_raw_condition(
        model, test_raw_dataset, "averaged",
        device, batch_size,
    )

    train_semantic_result = {
        "train_individual": _prototype_probe_metrics(
            train_ind_f, train_ind_y, prototypes
        ),
        "train_averaged": _prototype_probe_metrics(
            train_avg_f, train_avg_y, prototypes
        ),
        "test_individual": _prototype_probe_metrics(
            test_ind_f, test_ind_y, prototypes
        ),
        "test_averaged": _prototype_probe_metrics(
            test_avg_f, test_avg_y, prototypes
        ),
    }

    result = {"train_semantic": train_semantic_result}

    if fixed_text_prototypes is not None:
        fixed_cpu = fixed_text_prototypes.detach().float().cpu()
        result["fixed_text"] = {
            "train_individual": _prototype_probe_metrics(
                train_ind_f, train_ind_y, fixed_cpu
            ),
            "train_averaged": _prototype_probe_metrics(
                train_avg_f, train_avg_y, fixed_cpu
            ),
            "test_individual": _prototype_probe_metrics(
                test_ind_f, test_ind_y, fixed_cpu
            ),
            "test_averaged": _prototype_probe_metrics(
                test_avg_f, test_avg_y, fixed_cpu
            ),
        }

    if was_training:
        model.train()

    return result


def log_semantic_probe(writer, probe, optimizer_step):
    for space_name, conditions in probe.items():
        for condition, metrics in conditions.items():
            for key, value in metrics.items():
                writer.add_scalar(
                    f"probe/{space_name}/{condition}/{key}",
                    float(value),
                    optimizer_step,
                )



def save_checkpoint(
    path,
    model,
    optimizer_step,
    micro_step,
    stage,
    args,
    supcon_bank=None,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": model.opt.state_dict(),
        "scheduler": model.sche.state_dict(),
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "stage": str(stage),
        "args": vars(args),
    }
    if supcon_bank is not None:
        payload["supcon_bank"] = supcon_bank.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path, model, supcon_bank=None, load_training_state=True
):
    ckpt = torch.load(path, map_location="cpu")

    if "model" not in ckpt:
        raise RuntimeError(
            "--resume expects a checkpoint produced by this training script "
            "with a top-level `model` entry."
        )

    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[RESUME MODEL MISMATCH]")
        print("missing:", incompatible.missing_keys[:30])
        print("unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError("Resume checkpoint architecture mismatch.")

    if load_training_state:
        if "optimizer" in ckpt:
            model.opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            model.sche.load_state_dict(ckpt["scheduler"])
        if supcon_bank is not None and "supcon_bank" in ckpt:
            supcon_bank.load_state_dict(ckpt["supcon_bank"])
            print(
                f"[resume] restored SupCon bank: {len(supcon_bank)} entries"
            )
    else:
        print(
            "[resume-model-only] optimizer/scheduler/SupCon bank reset; "
            "model weights + optimizer_step are retained."
        )

    return (
        int(ckpt.get("optimizer_step", 0)),
        int(ckpt.get("micro_step", 0)),
    )


@torch.no_grad()
def validate_semantics(model, batch):
    metrics = model.semantic_validation(batch)
    return {
        k: float(v.item())
        for k, v in metrics.items()
    }


@torch.no_grad()
def generate_validation_grid(
    model,
    batch,
    save_path,
    num_steps=50,
    cfg_scale=4.0,
):
    """
    Generate the 3x2 grid with the SAME semantic/variation conditioning used
    in training and save GT above prediction.
    """
    was_training = model.training
    model.eval()

    cond_eeg, target_imgs = model.prepare_batch_data(batch)
    B = cond_eeg.shape[0]

    _, prompt_cond, latent_cond = model.encode_embed_fmri_condition_fmri(
        cond_eeg,
        drop_condition=False,
    )
    _, prompt_uncond, latent_uncond = model.encode_embed_fmri_condition_fmri(
        cond_eeg,
        drop_condition=True,
    )

    prompt_embeds = torch.cat([prompt_uncond, prompt_cond], dim=0)
    cond_latents = torch.cat([latent_uncond, latent_cond], dim=0)

    scheduler = model.pipeline.scheduler
    scheduler.set_timesteps(num_steps, device=cond_eeg.device)

    dtype = next(model.pipeline.unet.parameters()).dtype
    latents = torch.randn(
        (B, 4, 120, 80),
        device=cond_eeg.device,
        dtype=dtype,
    )
    latents = latents * scheduler.init_noise_sigma

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in scheduler.timesteps:
            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = scheduler.scale_model_input(latent_in, t)

            noise_pred = model.forward_unet(
                latent_in,
                t,
                prompt_embeds,
                cond_latents,
            )
            pred_u, pred_c = noise_pred.chunk(2)
            noise_pred = pred_u + cfg_scale * (pred_c - pred_u)

            latents = scheduler.step(
                noise_pred,
                t,
                latents,
            ).prev_sample

        images_pred = model.pipeline.vae.decode(
            latents / model.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images_pred = unscale_image(images_pred)
        images_pred = (images_pred * 0.5 + 0.5).clamp(0, 1)

    # Each item is already one [3,960,640] 3x2 view grid.
    comparison = make_grid(
        torch.cat(
            [
                target_imgs.detach().float().cpu(),
                images_pred.detach().float().cpu(),
            ],
            dim=0,
        ),
        nrow=B,
        normalize=True,
        value_range=(0, 1),
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(comparison, save_path)

    if was_training:
        model.train()


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    if args.semantic_stage_steps < 0:
        raise ValueError("--semantic_stage_steps must be >= 0")
    if args.max_steps <= args.semantic_stage_steps:
        raise ValueError(
            "--max_steps must be larger than --semantic_stage_steps "
            "so Stage-2 joint training actually runs."
        )
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be >= 1")
    if (
        args.lambda_proto < 0
        or args.lambda_supcon < 0
        or args.lambda_avg_consistency < 0
    ):
        raise ValueError("Semantic auxiliary loss weights must be >= 0")
    if args.prototype_temperature <= 0:
        raise ValueError("--prototype_temperature must be > 0")
    if args.supcon_temperature <= 0:
        raise ValueError("--supcon_temperature must be > 0")
    if args.supcon_queue_size < 0:
        raise ValueError("--supcon_queue_size must be >= 0")
    if args.semantic_probe_every < 0:
        raise ValueError("--semantic_probe_every must be >= 0")

    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda:0")
    print(
        "[device]",
        device,
        "CUDA_VISIBLE_DEVICES=",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )

    cfg = OmegaConf.load(config_path)
    stable_cfg, fmri_cfg, model_args = resolve_model_configs(
        cfg,
        config_path,
        cli_learning_rate=args.learning_rate,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    shutil.copyfile(
        config_path,
        out_dir / "config.yaml",
    )
    with open(out_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------
    # Dataset: explicit class labels are mandatory.
    # ---------------------------------------------------------------------
    sub_list = [args.sub_id]

    train_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=sub_list,
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=args.aug_data,
        strict_rendered_views=True,
    )

    # Keep the historical averaged validation stream for direct comparability.
    val_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=sub_list,
        train=False,
        test_mean=True,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )

    # Additional RAW test split used only by the encoder-only semantic probe.
    # No images are consumed by the probe, so strict rendered-view validation
    # is unnecessary here.
    test_raw_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=sub_list,
        train=False,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=False,
    )

    dump_dataset_mapping(
        train_dataset,
        out_dir / "mapping_train.csv",
    )
    dump_dataset_mapping(
        val_dataset,
        out_dir / "mapping_test.csv",
    )

    print(
        f"[mapping] train name_list={train_dataset.name_list.shape}, "
        f"test name_list={val_dataset.name_list.shape}"
    )
    print(
        f"[mapping] audit CSVs: {out_dir / 'mapping_train.csv'}, "
        f"{out_dir / 'mapping_test.csv'}"
    )

    fixed_text_prototypes, fixed_categories, fixed_source_objects = (
        build_fixed_category_text_prototypes(
            train_dataset, device, expected_dim=1024
        )
    )
    save_fixed_category_prototypes(
        out_dir,
        fixed_text_prototypes,
        fixed_categories,
        fixed_source_objects,
    )
    print(
        "[fixed-prototype] "
        f"shape={tuple(fixed_text_prototypes.shape)}, "
        f"temperature={args.prototype_temperature}, "
        "source=train-only CLIP text features (00..07)"
    )

    if args.audit_only:
        print("[audit-only] Dataset mapping validated. No training started.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------------------------------------------------
    # Model.
    # ---------------------------------------------------------------------
    model = MVDiffusion(
        model_args,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(out_dir),
        num_classes=72,
        cls_label_smoothing=args.cls_label_smoothing,
    ).to(device)

    # This flag does NOT freeze fmri_encoder. It only detaches the feature
    # copies that enter the Stage-2 diffusion branch.
    model.stop_diffusion_grad_to_eeg = bool(
        args.stop_diffusion_grad_to_eeg
    )
    print(
        "[diff-stopgrad] "
        f"enabled={model.stop_diffusion_grad_to_eeg}"
    )

    supcon_bank = SupConMemoryBank(
        dim=1024,
        queue_size=args.supcon_queue_size,
        temperature=args.supcon_temperature,
        device=device,
    )

    optimizer_step = 0
    micro_step = 0

    if args.resume:
        optimizer_step, micro_step = load_checkpoint(
            args.resume,
            model,
            supcon_bank=supcon_bank,
            load_training_state=not args.resume_model_only,
        )
        print(
            f"[resume] optimizer_step={optimizer_step}, "
            f"micro_step={micro_step}"
        )

    semantic_capture = SemanticFeatureCapture(model.fmri_encoder)

    # Determine current stage from the optimizer step, not from the checkpoint
    # text field, so resuming is deterministic.
    stage = (
        "semantic"
        if optimizer_step < args.semantic_stage_steps
        else "joint"
    )
    model.set_training_stage(stage)
    model.train()

    # Stage-1 does NOT step the diffusion scheduler. Thus when Stage-2 starts,
    # the original 300-step LR warm-up begins at its first joint optimizer step.
    model.opt.zero_grad(set_to_none=True)

    writer = SummaryWriter(str(out_dir / "logs"))
    val_iter = iter(val_loader)

    start_time = time.time()
    epoch = 0
    accum_counter = 0

    print(
        f"[training] Stage-1 semantic: 0 -> {args.semantic_stage_steps} optimizer steps"
    )
    print(
        f"[training] Stage-2 joint   : {args.semantic_stage_steps} -> "
        f"{args.max_steps} optimizer steps"
    )
    print(
        "[weights] "
        f"diff={args.lambda_diff}, clip={args.lambda_clip}, "
        f"cls={args.lambda_cls}, proto={args.lambda_proto}, "
        f"supcon={args.lambda_supcon}, "
        f"avg_consistency={args.lambda_avg_consistency}, "
        f"ortho={args.lambda_ortho}"
    )
    print(
        "[fixed-prototype] "
        f"temperature={args.prototype_temperature}, "
        f"anchor_only={args.prototype_anchor_only}"
    )
    print(
        "[supcon] "
        f"temperature={args.supcon_temperature}, "
        f"queue_size={args.supcon_queue_size}"
    )
    print(
        "[probe] "
        f"every={args.semantic_probe_every}, "
        f"batchsize={args.semantic_probe_batchsize}"
    )
    print(
        "[joint-gradient] diffusion -> fmri_encoder: "
        + ("BLOCKED" if args.stop_diffusion_grad_to_eeg else "ENABLED")
    )

    while optimizer_step < args.max_steps:
        epoch += 1

        for batch in train_loader:
            if optimizer_step >= args.max_steps:
                break

            wanted_stage = (
                "semantic"
                if optimizer_step < args.semantic_stage_steps
                else "joint"
            )
            if wanted_stage != stage:
                stage = wanted_stage
                model.set_training_stage(stage)
                model.opt.zero_grad(set_to_none=True)
                accum_counter = 0
                print(
                    f"\n[stage transition] optimizer_step={optimizer_step} "
                    f"-> {stage}\n"
                )

            micro_step += 1
            accum_counter += 1

            # -------------------------------------------------------------
            # Main model forward.
            # SemanticFeatureCapture obtains the exact semantic feature used
            # inside MVDiffusion._semantic_losses without a second anchor pass.
            # -------------------------------------------------------------
            semantic_capture.clear()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(
                    batch,
                    stage=stage,
                )

            anchor_sem = semantic_capture.last_semantic
            if anchor_sem is None:
                raise RuntimeError(
                    "Failed to capture anchor semantic feature from fmri_encoder."
                )

            # -------------------------------------------------------------
            # NEW: category/object/trial invariant semantic objectives.
            #
            # We concatenate the two auxiliary EEG views and run ONE additional
            # encoder pass:
            #   [same-class different-object positive,
            #    same-object raw-trial average]
            # -------------------------------------------------------------
            positive_eeg, averaged_eeg, semantic_labels = (
                build_semantic_auxiliary_views(
                    train_dataset,
                    batch,
                    device,
                )
            )

            with torch.autocast("cuda", enabled=False):
                auxiliary_eeg = torch.cat(
                    [positive_eeg, averaged_eeg],
                    dim=0,
                )
                auxiliary_sem, _, _ = model.fmri_encoder(
                    auxiliary_eeg.float()
                )

            B_sem = anchor_sem.shape[0]
            positive_sem = auxiliary_sem[:B_sem]
            averaged_sem = auxiliary_sem[B_sem:]

            supcon_loss = supcon_bank.compute_loss(
                anchor_sem,
                positive_sem,
                semantic_labels,
            )
            avg_consistency_loss = cosine_consistency_loss(
                anchor_sem,
                averaged_sem,
            )

            # -------------------------------------------------------------
            # PRIMARY semantic intervention: every semantic view is classified
            # against the SAME NON-TRAINABLE 72-way category text geometry.
            # This prevents a learned classifier + encoder from co-adapting to
            # a seen-object-only semantic space.
            # -------------------------------------------------------------
            if args.prototype_anchor_only:
                proto_features = anchor_sem
                proto_labels = semantic_labels
            else:
                proto_features = torch.cat(
                    [anchor_sem, positive_sem, averaged_sem], dim=0
                )
                proto_labels = torch.cat(
                    [semantic_labels, semantic_labels, semantic_labels], dim=0
                )

            proto_loss, proto_metrics = fixed_category_prototype_loss(
                proto_features,
                proto_labels,
                fixed_text_prototypes,
                temperature=args.prototype_temperature,
            )

            # Contrastive bank receives detached current features AFTER the
            # current loss was computed, so they cannot become self-negatives.
            supcon_bank.enqueue(
                torch.cat(
                    [anchor_sem, positive_sem],
                    dim=0,
                ),
                torch.cat(
                    [semantic_labels, semantic_labels],
                    dim=0,
                ),
            )

            # -------------------------------------------------------------
            # Revised total semantic objective:
            #   stronger CLIP alignment,
            #   much weaker hard 72-way CE,
            #   explicit cross-object SupCon,
            #   individual<->average consistency.
            # -------------------------------------------------------------
            total_raw = (
                args.lambda_clip * losses["clip_loss"]
                + args.lambda_cls * losses["cls_loss"]
                + args.lambda_proto * proto_loss
                + args.lambda_supcon * supcon_loss
                + args.lambda_avg_consistency * avg_consistency_loss
                + args.lambda_ortho * losses["ortho_loss"]
            )

            if stage == "joint":
                total_raw = (
                    total_raw
                    + args.lambda_diff * losses["diff_loss"]
                )

            loss_for_backward = (
                total_raw / args.accumulation_steps
            )

            loss_for_backward.backward()

            if accum_counter < args.accumulation_steps:
                continue

            # One optimizer update.
            if args.grad_clip > 0:
                grad_norm = clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            else:
                grad_norm = torch.tensor(0.0, device=device)

            model.opt.step()
            model.opt.zero_grad(set_to_none=True)

            # Diffusion/LoRA scheduler begins ONLY in joint stage.
            if stage == "joint":
                model.sche.step()

            accum_counter = 0
            optimizer_step += 1

            # -------------------------------------------------------------
            # Logging.
            # -------------------------------------------------------------
            writer.add_scalar(
                "train/total_loss",
                float(total_raw.detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/diff_loss",
                float(losses["diff_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/clip_loss",
                float(losses["clip_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/cls_loss",
                float(losses["cls_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/cls_acc",
                float(losses["cls_acc"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/ortho_loss",
                float(losses["ortho_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/fixed_proto_loss",
                float(proto_loss.detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/fixed_proto_top1",
                float(proto_metrics["top1"]),
                optimizer_step,
            )
            writer.add_scalar(
                "train/fixed_proto_margin",
                float(proto_metrics["margin"]),
                optimizer_step,
            )
            writer.add_scalar(
                "train/supcon_loss",
                float(supcon_loss.detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/avg_consistency_loss",
                float(avg_consistency_loss.detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/supcon_queue_count",
                float(len(supcon_bank)),
                optimizer_step,
            )
            writer.add_scalar(
                "train/grad_norm",
                float(grad_norm),
                optimizer_step,
            )
            writer.add_scalar(
                "train/stage",
                0 if stage == "semantic" else 1,
                optimizer_step,
            )

            if stage == "joint":
                writer.add_scalar(
                    "train/lr_lora",
                    model.opt.param_groups[0]["lr"],
                    optimizer_step,
                )

            if (
                args.print_every > 0
                and optimizer_step % args.print_every == 0
            ):
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch:03d}] "
                    f"step={optimizer_step:06d} "
                    f"stage={stage:<8s} "
                    f"Total={total_raw.item():.4f} "
                    f"Diff={losses['diff_loss'].item():.4f} "
                    f"CLIP={losses['clip_loss'].item():.4f} "
                    f"Cls={losses['cls_loss'].item():.4f} "
                    f"Proto={proto_loss.item():.4f} "
                    f"PAcc={proto_metrics['top1']:.3f} "
                    f"PMargin={proto_metrics['margin']:+.4f} "
                    f"SupCon={supcon_loss.item():.4f} "
                    f"AvgCons={avg_consistency_loss.item():.4f} "
                    f"Acc={losses['cls_acc'].item():.3f} "
                    f"Ortho={losses['ortho_loss'].item():.4f} "
                    f"Q={len(supcon_bank)} "
                    f"time={elapsed / 60.0:.1f}m"
                )

            # -------------------------------------------------------------
            # Semantic validation.
            # -------------------------------------------------------------
            if (
                args.validate_every > 0
                and optimizer_step % args.validate_every == 0
            ):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                val_metrics = validate_semantics(
                    model,
                    val_batch,
                )

                for key, value in val_metrics.items():
                    writer.add_scalar(
                        f"val/{key}",
                        value,
                        optimizer_step,
                    )

                print(
                    f"[VAL semantic] step={optimizer_step:06d} "
                    f"CLIP={val_metrics['clip_loss']:.4f} "
                    f"Cls={val_metrics['cls_loss']:.4f} "
                    f"Acc={val_metrics['cls_acc']:.3f} "
                    f"Ortho={val_metrics['ortho_loss']:.4f}"
                )

            # -------------------------------------------------------------
            # NEW: encoder-only four-way semantic probe.
            #
            # This is the validation signal that directly measures the failure
            # found in diagnosis.  It is deliberately less frequent than the
            # cheap one-batch semantic validation.
            # -------------------------------------------------------------
            if (
                args.semantic_probe_every > 0
                and optimizer_step % args.semantic_probe_every == 0
            ):
                probe = run_semantic_probe(
                    model,
                    train_dataset,
                    test_raw_dataset,
                    device,
                    batch_size=args.semantic_probe_batchsize,
                    fixed_text_prototypes=fixed_text_prototypes,
                )
                log_semantic_probe(
                    writer,
                    probe,
                    optimizer_step,
                )

                train_space = probe["train_semantic"]
                fixed_space = probe["fixed_text"]

                ti = train_space["train_individual"]
                ta = train_space["train_averaged"]
                vi = train_space["test_individual"]
                va = train_space["test_averaged"]

                fti = fixed_space["train_individual"]
                fta = fixed_space["train_averaged"]
                fvi = fixed_space["test_individual"]
                fva = fixed_space["test_averaged"]

                print(
                    f"[SEMANTIC PROBE / TRAIN-PROTOTYPE] step={optimizer_step:06d}\n"
                    f"  train individual: top1={ti['top1']:.3f}, "
                    f"margin={ti['margin']:+.4f}\n"
                    f"  train averaged  : top1={ta['top1']:.3f}, "
                    f"margin={ta['margin']:+.4f}\n"
                    f"  test individual : top1={vi['top1']:.3f}, "
                    f"margin={vi['margin']:+.4f}\n"
                    f"  test averaged   : top1={va['top1']:.3f}, "
                    f"margin={va['margin']:+.4f}"
                )
                print(
                    f"[SEMANTIC PROBE / FIXED-TEXT] step={optimizer_step:06d}\n"
                    f"  train individual: top1={fti['top1']:.3f}, "
                    f"margin={fti['margin']:+.4f}\n"
                    f"  train averaged  : top1={fta['top1']:.3f}, "
                    f"margin={fta['margin']:+.4f}\n"
                    f"  test individual : top1={fvi['top1']:.3f}, "
                    f"margin={fvi['margin']:+.4f}  <-- PRIMARY\n"
                    f"  test averaged   : top1={fva['top1']:.3f}, "
                    f"margin={fva['margin']:+.4f}"
                )

                # Probe switches eval/train internally; restore stage-specific
                # requires_grad configuration explicitly for clarity.
                model.set_training_stage(stage)
                model.train()

            # -------------------------------------------------------------
            # Expensive generation visualization only after Stage-2 starts.
            # -------------------------------------------------------------
            if (
                stage == "joint"
                and args.visualize_every > 0
                and optimizer_step % args.visualize_every == 0
            ):
                try:
                    vis_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    vis_batch = next(val_iter)

                generate_validation_grid(
                    model,
                    vis_batch,
                    str(
                        out_dir
                        / "images"
                        / f"val_{optimizer_step:06d}.png"
                    ),
                    num_steps=args.vis_steps,
                    cfg_scale=args.cfg_scale,
                )
                model.train()

            # -------------------------------------------------------------
            # Checkpoint.
            # -------------------------------------------------------------
            if (
                args.save_every > 0
                and optimizer_step % args.save_every == 0
            ):
                ckpt_path = (
                    out_dir
                    / "checkpoints"
                    / f"model_{optimizer_step:06d}.pt"
                )
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer_step,
                    micro_step,
                    stage,
                    args,
                    supcon_bank=supcon_bank,
                )
                print(f"[checkpoint] {ckpt_path}")

            # Release references to large Stage-2 / auxiliary tensors.
            del (
                losses,
                total_raw,
                loss_for_backward,
                supcon_loss,
                avg_consistency_loss,
                anchor_sem,
                positive_sem,
                averaged_sem,
                positive_eeg,
                averaged_eeg,
                auxiliary_eeg,
                auxiliary_sem,
                semantic_labels,
                batch,
            )

    final_path = out_dir / "checkpoints" / "model_final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer_step,
        micro_step,
        stage,
        args,
        supcon_bank=supcon_bank,
    )

    semantic_capture.close()
    writer.close()
    print(f"[done] final checkpoint: {final_path}")


if __name__ == "__main__":
    main()