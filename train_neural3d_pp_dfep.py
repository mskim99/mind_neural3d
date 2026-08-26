#!/usr/bin/env python3
"""
Frozen-encoder + projection-only diagnostic for EEG -> fixed CLIP category space.

Purpose
-------
Given a trained semantic checkpoint (e.g. step 30000):
  1) load the exact EEG encoder and FREEZE it completely
  2) cache EEG semantic features for train/test
  3) train ONLY a small semantic_to_clip projection
  4) use ONLY one training objective:
         72-way CE against the frozen fixed-text prototypes
  5) report raw-vs-projected retrieval on:
         train individual / train averaged
         test individual  / test averaged

No diffusion loss.
No CLIP image/text cosine loss.
No learned classifier CE.
No SupCon / memory bank.
No AvgCons.
No orthogonality loss.

Interpretation
--------------
- train projected high + test projected high:
    mainly a coordinate-alignment problem.
- train projected high + test projected ~chance:
    encoder contains seen-object structure but does not generalize to unseen objects.
- linear train remains low:
    try --head mlp. If MLP fixes train but not test, this still points to encoder generalization.
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
from omegaconf import OmegaConf

from src.mvdiffusion_var_semantic_cls_sg import MVDiffusion
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Frozen EEG encoder -> projection-only fixed-text diagnostic"
    )
    p.add_argument("--ckpt_path", required=True,
                   help="step-30000 checkpoint from fixed-prototype trainer")
    p.add_argument("--prototype_ckpt", default="",
                   help="fixed_category_text_prototypes.pt; default: run root")
    p.add_argument("--config", default="",
                   help="mind3d_pp config; default: run_root/config.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument("--rendered_view_path",
                   default="/data/jionkim/neuro_3D/render_grid_v4")
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", default="./projection_only_diagnostic")

    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--encode_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # Train on individual raw trials by default. This flag additionally adds
    # the train trial-averaged features to the projection fitting set.
    p.add_argument("--include_train_averaged", action="store_true")

    return p.parse_args()


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Config / checkpoint
# -------------------------------------------------------------------------
def resolve_paths(args):
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    # Current trainer layout:
    # run_root/
    #   config.yaml
    #   fixed_category_text_prototypes.pt
    #   checkpoints/model_030000.pt
    if ckpt_path.parent.name == "checkpoints":
        run_root = ckpt_path.parent.parent
    else:
        run_root = ckpt_path.parent

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else run_root / "config.yaml"
    )
    prototype_path = (
        Path(args.prototype_ckpt).expanduser().resolve()
        if args.prototype_ckpt
        else run_root / "fixed_category_text_prototypes.pt"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Pass --config ./configs/mind3d_pp.yaml explicitly."
        )
    if not prototype_path.is_file():
        raise FileNotFoundError(
            f"Prototype file not found: {prototype_path}\n"
            "Pass --prototype_ckpt explicitly."
        )
    return ckpt_path, run_root, config_path, prototype_path


def resolve_model_configs(cfg, config_path):
    stable_cfg = OmegaConf.select(
        cfg, "model.params.stable_diffusion_config", default=None
    )
    fmri_cfg = OmegaConf.select(
        cfg, "model.params.fmri_encoder_config", default=None
    )
    if stable_cfg is None:
        raise KeyError(
            f"{config_path} does not contain "
            "`model.params.stable_diffusion_config`. "
            "Use the MinD-3D++ mind3d_pp config."
        )

    # MVDiffusion expects args.learning_rate.
    if OmegaConf.select(cfg, "learning_rate", default=None) is None:
        # It is irrelevant to this diagnostic because the original model
        # optimizer is never stepped, but the constructor requires it.
        cfg.learning_rate = 1e-5

    return stable_cfg, fmri_cfg, cfg


def load_model(ckpt_path, config_path, device, out_dir):
    cfg = OmegaConf.load(config_path)
    stable_cfg, fmri_cfg, model_args = resolve_model_configs(cfg, config_path)

    model = MVDiffusion(
        model_args,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(out_dir),
        num_classes=72,
        cls_label_smoothing=0.05,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise RuntimeError(
            "Expected a fixed-prototype trainer checkpoint with top-level `model`."
        )

    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[CHECKPOINT MISMATCH]")
        print(" missing   :", incompatible.missing_keys[:30])
        print(" unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError(
            "Checkpoint/model mismatch. Use the same "
            "mvdiffusion_var_semantic_cls_stopgrad architecture as training."
        )

    # The whole model is irrelevant after feature extraction, but freeze all
    # parameters to make accidental updates impossible.
    model.requires_grad_(False)
    model.eval()

    # Explicit sanity check: BN running stats also stay fixed in eval mode.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"Frozen model still has {trainable} trainable params.")

    print(
        f"[checkpoint] loaded step={ckpt.get('optimizer_step', 'unknown')} "
        f"from {ckpt_path}"
    )
    print("[encoder] FULLY FROZEN; model.eval(); BN running stats fixed")
    return model


# -------------------------------------------------------------------------
# Dataset + encoder caching
# -------------------------------------------------------------------------
def make_datasets(args):
    common = dict(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
    )

    train_ds = AllDataFeatureTwoEEG(
        train=True,
        test_mean=False,
        strict_rendered_views=True,
        **common,
    )

    # Raw test retains all four trials for individual-vs-average diagnostic.
    test_raw_ds = AllDataFeatureTwoEEG(
        train=False,
        test_mean=False,
        strict_rendered_views=False,
        **common,
    )
    return train_ds, test_raw_ds


@torch.no_grad()
def encode_raw_condition(model, dataset, mode, device, batch_size):
    """
    Exactly mirrors the semantic probe:
      individual: every raw EEG trial
      averaged:   mean raw trials BEFORE the EEG encoder

    Returns CPU tensors:
      features [N,1024]
      labels   [N]
    """
    raw = dataset.eeg_data
    if raw.ndim != 6:
        raise RuntimeError(
            f"Expected eeg_data [S,C,O,R,64,600], got {raw.shape}"
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

    all_features, all_labels = [], []

    model.eval()
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        xs, ys = [], []

        for s, c, o, r in chunk:
            if r is None:
                x = np.asarray(
                    raw[s, c, o, :], dtype=np.float32
                ).mean(axis=0, dtype=np.float32)
            else:
                x = np.asarray(raw[s, c, o, r], dtype=np.float32)
            xs.append(x)
            ys.append(c)

        eeg = torch.from_numpy(np.stack(xs)).to(
            device=device, dtype=torch.float32
        )
        sem, _, _ = model.fmri_encoder(eeg)

        all_features.append(sem.detach().float().cpu())
        all_labels.append(torch.tensor(ys, dtype=torch.long))

    return torch.cat(all_features), torch.cat(all_labels)


# -------------------------------------------------------------------------
# Projection head
# -------------------------------------------------------------------------
class LinearProjection(nn.Module):
    """
    Most diagnostic option: one linear coordinate transform only.
    Identity initialization means step-0 metrics exactly reflect the frozen
    encoder's raw relation to the CLIP prototype space.
    """
    def __init__(self, dim=1024):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class MLPProjection(nn.Module):
    """
    Use only after the linear diagnostic if linear cannot fit the train set.
    Residual + zero-init final layer starts as identity.
    """
    def __init__(self, dim=1024, hidden=512):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def build_head(kind):
    if kind == "linear":
        return LinearProjection(1024)
    if kind == "mlp":
        return MLPProjection(1024, 512)
    raise ValueError(kind)


# -------------------------------------------------------------------------
# ONE AND ONLY training loss
# -------------------------------------------------------------------------
def prototype_ce(projected, labels, prototypes, temperature):
    z = F.normalize(projected.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    logits = z @ p.T / float(temperature)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def retrieval_metrics(features, labels, prototypes, head=None, device=None):
    if head is None:
        z = features.float()
    else:
        z = []
        head.eval()
        bs = 512
        for s in range(0, len(features), bs):
            xb = features[s:s + bs].to(device)
            z.append(head(xb).float().cpu())
        z = torch.cat(z, dim=0)

    z = F.normalize(z, dim=-1)
    p = F.normalize(prototypes.float().cpu(), dim=-1)
    sim = z @ p.T

    pred = sim.argmax(dim=1)
    top1 = (pred == labels).float().mean().item()

    top5_idx = sim.topk(k=5, dim=1).indices
    top5 = (top5_idx == labels[:, None]).any(dim=1).float().mean().item()

    rows = torch.arange(len(labels))
    correct = sim[rows, labels]
    wrong = sim.clone()
    wrong[rows, labels] = -torch.inf
    best_wrong = wrong.max(dim=1).values
    margin = correct - best_wrong

    return {
        "top1": float(top1),
        "top5": float(top5),
        "correct_cosine": float(correct.mean().item()),
        "margin": float(margin.mean().item()),
        "positive_margin_rate": float((margin > 0).float().mean().item()),
        "feature_norm": float(z.norm(dim=-1).mean().item()),
    }


def print_metrics(title, metrics):
    print(
        f"{title:<24s} "
        f"top1={metrics['top1']:.4f} "
        f"top5={metrics['top5']:.4f} "
        f"margin={metrics['margin']:+.5f} "
        f"pos_margin={metrics['positive_margin_rate']:.4f} "
        f"correct_cos={metrics['correct_cosine']:.4f}"
    )


def save_prediction_csv(path, features, labels, prototypes, categories, head, device):
    head.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(features), 512):
            outs.append(head(features[s:s+512].to(device)).float().cpu())
    z = F.normalize(torch.cat(outs), dim=-1)
    p = F.normalize(prototypes.float().cpu(), dim=-1)
    sim = z @ p.T
    pred = sim.argmax(dim=1)

    rows_idx = torch.arange(len(labels))
    correct = sim[rows_idx, labels]
    wrong = sim.clone()
    wrong[rows_idx, labels] = -torch.inf
    best_wrong, best_wrong_idx = wrong.max(dim=1)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_index", "target_cls", "target_category",
            "pred_cls", "pred_category",
            "correct_cos", "best_wrong_cls", "best_wrong_category",
            "best_wrong_cos", "margin",
        ])
        for i in range(len(labels)):
            y = int(labels[i])
            pr = int(pred[i])
            bw = int(best_wrong_idx[i])
            w.writerow([
                i, y, categories[y],
                pr, categories[pr],
                float(correct[i]), bw, categories[bw],
                float(best_wrong[i]),
                float(correct[i] - best_wrong[i]),
            ])


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.batch_size <= 0 or args.encode_batch_size <= 0:
        raise ValueError("batch sizes must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, run_root, config_path, prototype_path = resolve_paths(args)

    print(f"[device]     {device}")
    print(f"[run root]   {run_root}")
    print(f"[config]     {config_path}")
    print(f"[checkpoint] {ckpt_path}")
    print(f"[prototype]  {prototype_path}")

    # Frozen target semantic geometry.
    proto_pack = torch.load(prototype_path, map_location="cpu")
    prototypes = proto_pack["prototypes"].detach().float().cpu()
    prototypes = F.normalize(prototypes, dim=-1)

    if tuple(prototypes.shape) != (72, 1024):
        raise RuntimeError(
            f"Expected fixed prototypes [72,1024], got {tuple(prototypes.shape)}"
        )

    categories = list(proto_pack.get(
        "categories", [str(i) for i in range(72)]
    ))
    if len(categories) != 72:
        raise RuntimeError("Prototype category list must have length 72.")

    model = load_model(
        ckpt_path=ckpt_path,
        config_path=config_path,
        device=device,
        out_dir=out_dir,
    )
    train_ds, test_ds = make_datasets(args)

    print("\n[caching frozen encoder features]")
    train_ind_x, train_ind_y = encode_raw_condition(
        model, train_ds, "individual", device, args.encode_batch_size
    )
    train_avg_x, train_avg_y = encode_raw_condition(
        model, train_ds, "averaged", device, args.encode_batch_size
    )
    test_ind_x, test_ind_y = encode_raw_condition(
        model, test_ds, "individual", device, args.encode_batch_size
    )
    test_avg_x, test_avg_y = encode_raw_condition(
        model, test_ds, "averaged", device, args.encode_batch_size
    )

    print(
        f"  train individual: {tuple(train_ind_x.shape)}\n"
        f"  train averaged  : {tuple(train_avg_x.shape)}\n"
        f"  test individual : {tuple(test_ind_x.shape)}\n"
        f"  test averaged   : {tuple(test_avg_x.shape)}"
    )

    # Encoder can now be deleted entirely; projection training sees only cached
    # detached features. This guarantees zero encoder / BN updates.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n[RAW frozen-encoder -> fixed-text]")
    raw_metrics = {
        "train_individual": retrieval_metrics(
            train_ind_x, train_ind_y, prototypes
        ),
        "train_averaged": retrieval_metrics(
            train_avg_x, train_avg_y, prototypes
        ),
        "test_individual": retrieval_metrics(
            test_ind_x, test_ind_y, prototypes
        ),
        "test_averaged": retrieval_metrics(
            test_avg_x, test_avg_y, prototypes
        ),
    }
    for k, v in raw_metrics.items():
        print_metrics("  " + k, v)

    # ------------------------------------------------------------------
    # Projection-only optimization
    # ------------------------------------------------------------------
    head = build_head(args.head).to(device)
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(
        f"\n[projection] head={args.head}, trainable={trainable/1e6:.3f}M"
    )
    print("[loss] ONLY fixed-prototype 72-way CE")

    if args.include_train_averaged:
        fit_x = torch.cat([train_ind_x, train_avg_x], dim=0)
        fit_y = torch.cat([train_ind_y, train_avg_y], dim=0)
        print("[fit data] train individual + train averaged")
    else:
        fit_x, fit_y = train_ind_x, train_ind_y
        print("[fit data] train individual only")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    proto_dev = prototypes.to(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed)

    head.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(
            low=0,
            high=len(fit_x),
            size=(args.batch_size,),
            generator=g,
        )

        xb = fit_x[idx].to(device=device, dtype=torch.float32)
        yb = fit_y[idx].to(device=device, dtype=torch.long)

        projected = head(xb)
        loss = prototype_ce(
            projected, yb, proto_dev, args.temperature
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            # IMPORTANT: only TRAIN is monitored during fitting.
            # Held-out objects 08/09 are evaluated only after the fixed step count.
            tr = retrieval_metrics(
                train_ind_x, train_ind_y, prototypes,
                head=head, device=device
            )
            print(
                f"[step {step:05d}/{args.steps}] "
                f"loss={loss.item():.4f} "
                f"train_top1={tr['top1']:.4f} "
                f"train_margin={tr['margin']:+.5f}"
            )
            head.train()

    # ------------------------------------------------------------------
    # Final fixed-protocol evaluation
    # ------------------------------------------------------------------
    print("\n[PROJECTED -> fixed-text | FINAL]")
    projected_metrics = {
        "train_individual": retrieval_metrics(
            train_ind_x, train_ind_y, prototypes, head, device
        ),
        "train_averaged": retrieval_metrics(
            train_avg_x, train_avg_y, prototypes, head, device
        ),
        "test_individual": retrieval_metrics(
            test_ind_x, test_ind_y, prototypes, head, device
        ),
        "test_averaged": retrieval_metrics(
            test_avg_x, test_avg_y, prototypes, head, device
        ),
    }
    for k, v in projected_metrics.items():
        print_metrics("  " + k, v)

    # Changes in primary test metrics.
    delta_test_top1 = (
        projected_metrics["test_individual"]["top1"]
        - raw_metrics["test_individual"]["top1"]
    )
    delta_test_margin = (
        projected_metrics["test_individual"]["margin"]
        - raw_metrics["test_individual"]["margin"]
    )

    print("\n[PRIMARY delta: test individual]")
    print(f"  Top1   : {delta_test_top1:+.4f}")
    print(f"  Margin : {delta_test_margin:+.5f}")

    # Simple interpretation, intentionally based only on final metrics.
    tr_top1 = projected_metrics["train_individual"]["top1"]
    te_top1 = projected_metrics["test_individual"]["top1"]
    te_margin = projected_metrics["test_individual"]["margin"]

    if tr_top1 >= 0.70 and te_top1 >= 0.10 and te_margin > 0:
        verdict = (
            "ALIGNMENT_SUPPORTED: a frozen encoder can be mapped to the fixed "
            "CLIP space and this mapping transfers meaningfully to unseen objects."
        )
    elif tr_top1 >= 0.70 and te_top1 < 0.05:
        verdict = (
            "ENCODER_GENERALIZATION_FAILURE: projection can fit seen objects but "
            "does not transfer to held-out objects 08/09."
        )
    elif tr_top1 < 0.30 and args.head == "linear":
        verdict = (
            "LINEAR_ALIGNMENT_INSUFFICIENT: the frozen space is not easily "
            "linearly mapped. Run the same diagnostic with --head mlp before "
            "changing the encoder."
        )
    else:
        verdict = (
            "MIXED: inspect train/test gaps. If MLP fits train but test stays near "
            "chance, treat this as an encoder generalization problem rather than "
            "a projection problem."
        )

    print("\n[VERDICT]")
    print(" ", verdict)

    # Save artifacts.
    torch.save(
        {
            "head_type": args.head,
            "projection_state_dict": head.state_dict(),
            "prototype_ckpt": str(prototype_path),
            "source_checkpoint": str(ckpt_path),
            "steps": args.steps,
            "temperature": args.temperature,
            "categories": categories,
        },
        out_dir / "projection_only.pt",
    )

    summary = {
        "source_checkpoint": str(ckpt_path),
        "prototype_checkpoint": str(prototype_path),
        "config": str(config_path),
        "head": args.head,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "include_train_averaged": bool(args.include_train_averaged),
        "training_objective": "fixed prototype CE only",
        "removed_losses": [
            "diffusion",
            "clip image/text cosine",
            "learned classifier CE",
            "SupCon",
            "AvgCons",
            "orthogonality",
        ],
        "raw": raw_metrics,
        "projected": projected_metrics,
        "primary_delta": {
            "test_individual_top1": delta_test_top1,
            "test_individual_margin": delta_test_margin,
        },
        "verdict": verdict,
    }

    with (out_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_prediction_csv(
        out_dir / "test_individual_predictions.csv",
        test_ind_x, test_ind_y, prototypes, categories, head, device
    )
    save_prediction_csv(
        out_dir / "test_averaged_predictions.csv",
        test_avg_x, test_avg_y, prototypes, categories, head, device
    )

    print("\n[saved]")
    print(" ", out_dir / "projection_only.pt")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "test_individual_predictions.csv")
    print(" ", out_dir / "test_averaged_predictions.csv")


if __name__ == "__main__":
    main()
