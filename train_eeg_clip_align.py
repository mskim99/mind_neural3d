#!/usr/bin/env python3
"""Train MinD-3D EEG -> CLIP with fMRI-style train/validation metrics.

Unchanged from the supplied EEG trainer:
  * EEG features = mean/std over 250 time samples, AFTER averaging two trials.
  * EEGCLIPAligner architecture and symmetric in-batch CLIP contrastive loss.
  * Frozen OpenCLIP ViT-B-32 (OpenAI) category-text supervision.

New evaluation (not an additional training loss):
  * UID-disjoint-by-object validation: object 00..05 train, 06..07 val.
  * CLIP image features pooled over six GT rendered views, per object.
  * Exact expected 2-way/10-way identification against all objects in each split.
  * Full-gallery CE and Top-1/5 for object-image and category-text retrieval.
  * Separate best-object-CE and best-10way checkpoints (raw state_dict files).

NOTE: fMRI object targets are Cap3D text, while these EEG object targets are
      CLIP image prototypes. Their identically named scores are not directly
      comparable across modalities/datasets or to 3D reconstruction metrics.
"""

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# Preserve the model architecture/state_dict keys from train_eeg_clip_align.py.
class EEGCLIPAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim, clip_dim=512, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        features = self.net(x)
        return F.normalize(features, dim=-1)


def parse_objects(spec: str, name: str) -> List[int]:
    try:
        values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers: {spec!r}") from exc
    if not values or len(values) != len(set(values)) or any(x < 0 or x > 7 for x in values):
        raise ValueError(f"{name} must contain unique object IDs in [0,7], got {spec!r}")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="EEG-CLIP with fMRI-style metrics")
    # Existing parameters remain valid.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    # New parameters for a genuinely held-out validation set and object metrics.
    parser.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    parser.add_argument("--sub_id", default="sub01")
    parser.add_argument("--rendered_view_path", default=None,
                        help="GT six-view root; default <data_path>/render_grid_v4")
    parser.add_argument("--train_objects", default="0,1,2,3,4,5")
    parser.add_argument("--val_objects", default="6,7")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--clip_batch_size", type=int, default=48)
    parser.add_argument("--image_cache", default="",
                        help="Optional shared cache for CLIP six-view object prototypes")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_names(name_list: np.ndarray) -> Tuple[List[str], List[str]]:
    """Use the exact [3:] mapping from the original training script."""
    if tuple(name_list.shape) != (72, 8):
        raise ValueError(f"Expected name_list=(72,8), got {name_list.shape}")
    object_labels = [str(name_list[c, o])[3:] for c in range(72) for o in range(8)]
    category_prompts = [
        f"a photo of a {str(name_list[c, 0])[3:].replace('_', ' ')}"
        for c in range(72)
    ]
    if any(not lab for lab in object_labels):
        raise ValueError("Empty object label in dataset.name_list")
    if len(object_labels) != len(set(object_labels)):
        # A duplicate object label would make the ground-truth image gallery ambiguous.
        duplicates = sorted({x for x in object_labels if object_labels.count(x) > 1})
        raise ValueError(f"Non-unique object names in name_list: {duplicates[:5]}")
    return object_labels, category_prompts


@torch.no_grad()
def encode_category_texts(clip_model, tokenizer, prompts, device) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    feat = clip_model.encode_text(tokens).float()
    return F.normalize(feat, dim=-1)


@torch.no_grad()
def build_object_image_bank(
    object_labels: Sequence[str], root: Path, clip_model, preprocess, device,
    batch_size: int, cache_path: Path,
) -> torch.Tensor:
    """Return [576,D] CLIP image prototypes, one per EEG class/object pair.

    Each object prototype is the L2-normalized mean of six individually
    L2-normalized view encodings. This is an EVALUATION target only.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    clip_key = "ViT-B-32/openai/six-views-normalized-mean/v1"
    root = root.expanduser().resolve()
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (cached.get("root") == str(root)
                and cached.get("labels") == list(object_labels)
                and cached.get("clip_key") == clip_key
                and tuple(cached["features"].shape)[0] == len(object_labels)):
            print(f"[*] Loaded object CLIP image prototypes from cache: {cache_path}")
            return F.normalize(cached["features"].float().to(device), dim=-1)
        print("[Warning] Object image cache metadata mismatch; rebuilding cache.")

    if batch_size < 1:
        raise ValueError("--clip_batch_size must be >= 1")
    files = []
    missing = []
    for idx, label in enumerate(object_labels):
        for k in range(6):
            image_path = root / label / f"{k:02d}.png"
            if not image_path.is_file():
                missing.append(str(image_path))
            else:
                files.append((idx, image_path))
    if missing:
        msg = "\n".join(f"  {x}" for x in missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} GT rendered view(s). First examples:\n{msg}\n"
            "Required layout: <rendered_view_path>/<dataset.name_list[c,o][3:]>/00.png...05.png. "
            "Check --rendered_view_path and dataset.name_list before computing object metrics."
        )

    sums = None
    counts = torch.zeros(len(object_labels), device=device, dtype=torch.int64)
    for start in tqdm(range(0, len(files), batch_size), desc="Encoding GT object views"):
        chunk = files[start:start + batch_size]
        images = []
        for _, path in chunk:
            with Image.open(path) as pil:
                images.append(preprocess(pil.convert("RGB")))
        batch_images = torch.stack(images, dim=0).to(device)
        embeds = F.normalize(clip_model.encode_image(batch_images).float(), dim=-1)
        if sums is None:
            sums = torch.zeros(len(object_labels), embeds.size(1), device=device)
        indices = torch.tensor([idx for idx, _ in chunk], dtype=torch.long, device=device)
        sums.index_add_(0, indices, embeds)
        counts.index_add_(0, indices, torch.ones_like(indices))

    if (counts != 6).any():
        raise RuntimeError(f"Expected six views per object, count histogram={torch.unique(counts, return_counts=True)}")
    bank = F.normalize(sums / 6.0, dim=-1)
    torch.save(
        {"root": str(root), "labels": list(object_labels), "clip_key": clip_key,
         "features": bank.cpu()}, cache_path,
    )
    print(f"[*] Object CLIP image bank: {tuple(bank.shape)}; saved: {cache_path}")
    return bank


def load_static_eeg_features(data_root: Path, sub_id: str) -> torch.Tensor:
    path = data_root / "EEGdata" / sub_id / f"{sub_id}_train_data_1s_250Hz.npy"
    static_train = np.load(path, mmap_mode="r")
    expected = (1, 72, 8, 2, 64, 250)
    if tuple(static_train.shape) == expected[1:]:
        static_train = static_train[None]
    elif tuple(static_train.shape) != expected:
        raise ValueError(f"Expected static EEG shape {expected} (or without subject axis), got {static_train.shape}")

    # EXACTLY matches the provided trainer: average both repeats first, then
    # compute temporal mean/std for each channel. No trial augmentation.
    x = np.asarray(static_train).mean(axis=3)[0]  # [72,8,64,250]
    means = x.mean(axis=-1)  # [72,8,64]
    stds = x.std(axis=-1)    # [72,8,64]
    features = np.concatenate((means, stds), axis=-1).astype("float32")
    if not np.isfinite(features).all():
        raise ValueError("NaN/Inf found in EEG mean/std features")
    print(f"[*] Static EEG: {path} | raw={expected}, trial-averaged features={features.shape}")
    return torch.from_numpy(features)


def make_split_dataset(features: torch.Tensor, objects: Sequence[int]) -> Tuple[TensorDataset, List[int]]:
    split_ids = [c * 8 + o for c in range(72) for o in objects]
    cat = torch.tensor([c for c in range(72) for _ in objects], dtype=torch.long)
    flat = features.reshape(72 * 8, -1)
    dataset = TensorDataset(flat[split_ids], cat, torch.tensor(split_ids, dtype=torch.long))
    return dataset, split_ids


def topk(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    return float((logits.topk(min(k, logits.size(1)), dim=1).indices == target[:, None]).any(dim=1).float().mean().item())


def expected_nway_top1(scores: torch.Tensor, targets: torch.Tensor, n_way: int) -> float:
    """Exact random-distractor N-way expectation (same method as fMRI ablation)."""
    scores = scores.detach().cpu().double()
    targets = targets.detach().cpu().long()
    n_queries, n_candidates = scores.shape
    if not 2 <= n_way <= n_candidates:
        raise ValueError(f"N-way {n_way} requires >={n_way} candidate objects (got {n_candidates})")
    choose_k = n_way - 1
    denominator = math.comb(n_candidates - 1, choose_k)
    wins = []
    for i in range(n_queries):
        target = int(targets[i])
        positive = scores[i, target]
        others = torch.cat([scores[i, :target], scores[i, target + 1:]])
        higher = int((others > positive + 1e-12).sum())
        equal = int(((others - positive).abs() <= 1e-12).sum())
        lower = (n_candidates - 1) - higher - equal
        ways = 0.0
        for j in range(max(0, choose_k - lower), min(equal, choose_k) + 1):
            k = choose_k - j
            if k <= lower:
                ways += math.comb(equal, j) * math.comb(lower, k) / (j + 1)
        wins.append(ways / denominator)
    return float(np.mean(wins))


@torch.no_grad()
def evaluate(
    model: EEGCLIPAligner, loader: DataLoader, split_ids: Sequence[int],
    all_object_images: torch.Tensor, category_texts: torch.Tensor, device: torch.device,
) -> Dict[str, float]:
    """Evaluate against disjoint *split-specific* objects and 72 categories."""
    model.eval()
    embeddings, categories, object_ids = [], [], []
    for x, c, objid in loader:
        embeddings.append(model(x.to(device)).detach())
        categories.append(c.to(device))
        object_ids.append(objid.to(device))
    brain = torch.cat(embeddings, 0)
    cats = torch.cat(categories, 0)
    ids = torch.cat(object_ids, 0)
    id_to_idx = {objid: j for j, objid in enumerate(split_ids)}
    targets = torch.tensor([id_to_idx[int(i)] for i in ids.cpu().tolist()], device=device)
    scale = model.logit_scale.detach().exp().clamp(max=100.0)
    obj_logits = scale * brain @ all_object_images[split_ids].T
    cat_logits = scale * brain @ category_texts.T
    return {
        "ObjCE": float(F.cross_entropy(obj_logits, targets).item()),
        "CatCE": float(F.cross_entropy(cat_logits, cats).item()),
        "Obj@1": topk(obj_logits, targets, 1),
        "Obj@5": topk(obj_logits, targets, 5),
        "Cat@1": topk(cat_logits, cats, 1),
        "Cat@5": topk(cat_logits, cats, 5),
        "2Way": expected_nway_top1(obj_logits, targets, 2),
        "10Way": expected_nway_top1(obj_logits, targets, 10),
    }


def save_state(model: nn.Module, path: Path):
    # Preserve the original EEG checkpoint representation: plain state_dict.
    torch.save(model.state_dict(), path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("Require epochs>=1, batch_size>=2, eval_batch_size>=1")
    train_objects = parse_objects(args.train_objects, "--train_objects")
    val_objects = parse_objects(args.val_objects, "--val_objects")
    if set(train_objects).intersection(val_objects):
        raise ValueError("train_objects and val_objects must not overlap")
    root = Path(args.data_path).expanduser()
    render_root = Path(args.rendered_view_path) if args.rendered_view_path else root / "render_grid_v4"
    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.image_cache) if args.image_cache else output / "clip_image_targets.pt"

    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[*] Device={device}, subject={args.sub_id}, seed={args.seed}")
    print(f"[*] Objects: train={train_objects}, val={val_objects}; official test objects 08,09 never used")
    print("[*] Evaluation: Obj*=six-view CLIP IMAGE retrieval; Cat*=72 CLIP TEXT retrieval")
    print("[*] 2Way/10Way: exact expected object-level identification over split-specific image candidates")
    print("[*] Training objective unchanged: symmetric category-text Batch NCE")

    # Only name_list is used here: EEG signals are loaded directly from the
    # same static file as in the original trainer, not from rendered dataset items.
    if hasattr(AllDataFeatureTwoEEG, "_validate_rendered_dataset"):
        AllDataFeatureTwoEEG._validate_rendered_dataset = lambda self: None
    naming_dataset = AllDataFeatureTwoEEG(
        data_path=str(root) + '/', sub_list=[args.sub_id], train=True,
        num_frames=6, rendered_view_path=str(render_root),
    )
    object_labels, category_prompts = build_names(np.asarray(naming_dataset.name_list))
    del naming_dataset

    print("[*] Loading frozen OpenCLIP ViT-B-32 (openai)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    clip_model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    cat_bank = encode_category_texts(clip_model, tokenizer, category_prompts, device)
    obj_bank = build_object_image_bank(
        object_labels, render_root, clip_model, preprocess, device,
        args.clip_batch_size, cache,
    )
    if cat_bank.size(1) != obj_bank.size(1):
        raise ValueError("CLIP text and image feature dimensions differ")
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    eeg_features = load_static_eeg_features(root, args.sub_id)
    train_dataset, train_ids = make_split_dataset(eeg_features, train_objects)
    val_dataset, val_ids = make_split_dataset(eeg_features, val_objects)
    batch_rng = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
        generator=batch_rng,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    print(f"[*] Splits: train={len(train_dataset)} samples/objects, val={len(val_dataset)}; "
          f"cat=72, clip_dim={cat_bank.size(1)}")
    print(f"[*] Chance levels (val gallery): Obj@1={1/len(val_ids):.4f}, "
          f"Cat@1={1/72:.4f}, 2Way=0.5, 10Way=0.1")

    model = EEGCLIPAligner(
        input_dim=eeg_features.size(-1), hidden_dim=args.hidden_dim,
        clip_dim=cat_bank.size(1), dropout=args.dropout,
    ).to(device)
    print(f"[*] EEG aligner parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    saved_config = dict(vars(args))
    saved_config.update(
        train_objects=train_objects, val_objects=val_objects,
        image_target_source="six-view OpenCLIP image embedding",
        category_target_source="OpenCLIP text prompt from name_list[c,0]",
        training_trials="two-repeat mean (unchanged)",
    )
    (output / "config.json").write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    keys = [
        "Epoch", "Loss", "NCE", "TrainBatchAcc", "TrainFullObjCE",
        "TrainCatCE", "TrainObj@1", "TrainCat@1", "ValObjCE",
        "ValCatCE", "Val2Way", "Val10Way", "ValObj@1", "ValObj@5",
        "ValCat@1", "ValCat@5", "Scale", "LR",
    ]
    log_path = output / "training_log.csv"
    best_ce = float("inf")
    best_10way = -1.0
    best_ce_epoch = -1
    best_10way_epoch = -1
    best_ce_metrics = {}
    best_10way_metrics = {}

    with log_path.open("w", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=keys)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for x, cls, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                x = x.to(device)
                cls = cls.to(device)
                optimizer.zero_grad(set_to_none=True)
                eeg_embed = model(x)
                category_target = cat_bank[cls]
                logits = model.logit_scale.exp() * (eeg_embed @ category_target.T)
                labels = torch.arange(x.size(0), device=device)
                nce = 0.5 * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)
                )
                nce.backward()
                optimizer.step()
                loss_sum += float(nce.detach()) * x.size(0)
                correct += int((logits.argmax(1) == labels).sum())
                total += x.size(0)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            train_metrics = evaluate(model, train_eval_loader, train_ids, obj_bank, cat_bank, device)
            val_metrics = evaluate(model, val_loader, val_ids, obj_bank, cat_bank, device)
            avg_nce = loss_sum / total
            train_batch_acc = correct / total
            scale = float(model.logit_scale.detach().exp().item())
            line = {
                "Epoch": epoch, "Loss": avg_nce, "NCE": avg_nce,
                "TrainBatchAcc": train_batch_acc,
                "TrainFullObjCE": train_metrics["ObjCE"],
                "TrainCatCE": train_metrics["CatCE"],
                "TrainObj@1": train_metrics["Obj@1"],
                "TrainCat@1": train_metrics["Cat@1"],
                "ValObjCE": val_metrics["ObjCE"],
                "ValCatCE": val_metrics["CatCE"],
                "Val2Way": val_metrics["2Way"],
                "Val10Way": val_metrics["10Way"],
                "ValObj@1": val_metrics["Obj@1"],
                "ValObj@5": val_metrics["Obj@5"],
                "ValCat@1": val_metrics["Cat@1"],
                "ValCat@5": val_metrics["Cat@5"],
                "Scale": scale,
                "LR": current_lr,
            }
            writer.writerow(line)
            log.flush()
            print(
                f"Epoch {epoch:03d} | Loss={avg_nce:.4f} | NCE={avg_nce:.4f} | "
                f"TrainBatchAcc={train_batch_acc:.4f} | "
                f"TrainFullObjCE={train_metrics['ObjCE']:.4f} | "
                f"TrainCatCE={train_metrics['CatCE']:.4f} | "
                f"TrainObj@1={train_metrics['Obj@1']:.4f} | "
                f"TrainCat@1={train_metrics['Cat@1']:.4f} | "
                f"ValObjCE={val_metrics['ObjCE']:.4f} | "
                f"ValCatCE={val_metrics['CatCE']:.4f} | "
                f"Val2Way={val_metrics['2Way']:.4f} | "
                f"Val10Way={val_metrics['10Way']:.4f} | "
                f"ValObj@1={val_metrics['Obj@1']:.4f} | "
                f"ValObj@5={val_metrics['Obj@5']:.4f} | "
                f"ValCat@1={val_metrics['Cat@1']:.4f} | "
                f"ValCat@5={val_metrics['Cat@5']:.4f} | "
                f"Scale={scale:.4f} | LR={current_lr:.6g}"
            )
            if val_metrics["ObjCE"] < best_ce:
                best_ce = val_metrics["ObjCE"]
                best_ce_epoch = epoch
                best_ce_metrics = dict(val_metrics)
                save_state(model, output / "eeg_clip_aligner_best_objce.pt")
                save_state(model, output / "eeg_clip_aligner_best.pt")
                print(f"[*] BestObjCE updated: epoch={epoch}, ValObjCE={best_ce:.6f}")
            if val_metrics["10Way"] > best_10way:
                best_10way = val_metrics["10Way"]
                best_10way_epoch = epoch
                best_10way_metrics = dict(val_metrics)
                save_state(model, output / "eeg_clip_aligner_best_10way.pt")
                print(f"[*] Best10Way updated: epoch={epoch}, Val10Way={best_10way:.6f}")

    save_state(model, output / "eeg_clip_aligner_last.pt")
    summary = {
        "best_objce": {"epoch": best_ce_epoch, **best_ce_metrics},
        "best_10way": {"epoch": best_10way_epoch, **best_10way_metrics},
        "note": "Obj*=CLIP image six-view retrieval; Cat*=CLIP category text retrieval",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for label, epoch, path in [
        ("best-objce", best_ce_epoch, output / "eeg_clip_aligner_best_objce.pt"),
        ("best-10way", best_10way_epoch, output / "eeg_clip_aligner_best_10way.pt"),
    ]:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        val = evaluate(model, val_loader, val_ids, obj_bank, cat_bank, device)
        train = evaluate(model, train_eval_loader, train_ids, obj_bank, cat_bank, device)
        for group, metrics in [("VAL", val), ("TRAIN", train)]:
            print(
                f"[{group} {label}] epoch={epoch} "
                f"ObjCE={metrics['ObjCE']:.4f} CatCE={metrics['CatCE']:.4f} "
                f"2Way={metrics['2Way']:.2%} 10Way={metrics['10Way']:.2%} "
                f"Obj@1={metrics['Obj@1']:.2%} Obj@5={metrics['Obj@5']:.2%} "
                f"Cat@1={metrics['Cat@1']:.2%} Cat@5={metrics['Cat@5']:.2%}"
            )

    print(f"[*] Best ObjCE: {output / 'eeg_clip_aligner_best_objce.pt'}")
    print(f"[*] Best 10Way: {output / 'eeg_clip_aligner_best_10way.pt'}")
    print(f"[*] Legacy best (ObjCE selection): {output / 'eeg_clip_aligner_best.pt'}")
    print(f"[*] Training log: {log_path}")


if __name__ == "__main__":
    main()
