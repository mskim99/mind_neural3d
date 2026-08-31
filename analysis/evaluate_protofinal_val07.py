#!/usr/bin/env python3
"""
Recover VAL07 semantic metrics from saved ProtoFinal checkpoints WITHOUT retraining.

This uses the exact helper classes/functions from the user's trainer file, so the
evaluation protocol matches training:
  train prototype source: objects 00..06
  validation object:      07
  final test 08..09:      never loaded

Example:
CUDA_VISIBLE_DEVICES=0 python evaluate_protofinal_val07.py \
  --trainer_path /home/jionkim/MinD-3D/train_neural3d_pp_protofinal.py \
  --config /home/jionkim/MinD-3D/configs/mind3d_pp.yaml \
  --data_path /data/jionkim/neuro_3D/ \
  --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
  --sub_id sub01 \
  --checkpoint_dir /data/jionkim/mind_3d_output/protofinal_dev/checkpoints \
  --out_csv /data/jionkim/mind_3d_output/protofinal_dev/val07_recovered.csv
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf


def load_module_from_path(path):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("protofinal_trainer_runtime", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_step(path):
    ckpt = torch.load(path, map_location="cpu")
    return int(ckpt.get("step", -1)), ckpt.get("stage", "unknown")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trainer_path", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--rendered_view_path", required=True)
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--ckpt", action="append", default=[],
                   help="Specific checkpoint(s). Can be repeated.")
    p.add_argument("--out_csv", default="./val07_recovered.csv")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.07)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = load_module_from_path(args.trainer_path)

    required = [
        "resolve_model_configs",
        "SemanticObjectDataset",
        "build_fixed_category_text_prototypes",
        "evaluate_semantic",
        "MVDiffusion",
        "AllDataFeatureTwoEEG",
    ]
    missing = [x for x in required if not hasattr(T, x)]
    if missing:
        raise RuntimeError(
            f"Trainer is missing expected ProtoFinal helpers: {missing}"
        )

    config_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    stable_cfg, fmri_cfg, model_args = T.resolve_model_configs(
        cfg, config_path
    )

    # Only train=True dataset is loaded. This includes 00..07.
    # We explicitly split 00..06 vs 07 below. Test 08..09 is never instantiated.
    base_train = T.AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )

    train_suffixes = [f"{i:02d}" for i in range(7)]  # 00..06
    val_suffixes = ["07"]

    val_ind = T.SemanticObjectDataset(
        base_train, val_suffixes, mode="individual"
    )
    val_avg = T.SemanticObjectDataset(
        base_train, val_suffixes, mode="averaged"
    )

    prototypes, categories, source_objects = (
        T.build_fixed_category_text_prototypes(
            base_train, train_suffixes
        )
    )

    print(
        f"[protocol] prototype source={train_suffixes}, "
        f"VAL={val_suffixes}, final test=NOT LOADED"
    )
    print(
        f"[samples] val_ind={len(val_ind)}, val_avg={len(val_avg)}, "
        f"prototype_shape={tuple(prototypes.shape)}"
    )

    model = T.MVDiffusion(
        model_args,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(Path(args.out_csv).expanduser().resolve().parent),
        num_classes=72,
    ).to(device)

    # Find checkpoints.
    paths = [Path(x).expanduser().resolve() for x in args.ckpt]
    if args.checkpoint_dir:
        d = Path(args.checkpoint_dir).expanduser().resolve()
        # Prefer Stage-1 snapshots, but also include selected/last if present.
        paths += sorted(d.glob("semantic_*.pt"))
        for name in ["best_semantic.pt", "semantic_last.pt"]:
            q = d / name
            if q.is_file():
                paths.append(q)

    # Deduplicate while preserving order.
    uniq = []
    seen = set()
    for q in paths:
        if q.is_file() and str(q) not in seen:
            seen.add(str(q))
            uniq.append(q)
    paths = uniq

    if not paths:
        raise FileNotFoundError(
            "No checkpoints found. Pass --checkpoint_dir or --ckpt."
        )

    # Sort by checkpoint step when possible, keeping filename for deterministic order.
    meta = []
    for q in paths:
        step, stage = checkpoint_step(q)
        meta.append((step, q.name, stage, q))
    meta.sort(key=lambda x: (x[0], x[1]))

    rows = []
    best = None

    for step, name, stage, q in meta:
        ckpt = torch.load(q, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"[SKIP] {name}: architecture mismatch")
            print(" missing:", incompatible.missing_keys[:10])
            print(" unexpected:", incompatible.unexpected_keys[:10])
            continue

        # Fixed prototypes are not trusted from checkpoint; rebuild 00..06 and reset.
        model.set_fixed_prototypes(prototypes.to(device))
        model.eval()

        vi = T.evaluate_semantic(
            model,
            val_ind,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            temperature=args.temperature,
        )
        va = T.evaluate_semantic(
            model,
            val_avg,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            temperature=args.temperature,
        )

        row = {
            "checkpoint": str(q),
            "stage": stage,
            "step": step,
            "val07_ind_proto": vi["proto_loss"],
            "val07_ind_top1": vi["top1"],
            "val07_ind_margin": vi["margin"],
            "val07_ind_cos": vi["correct_cosine"],
            "val07_ind_ortho": vi["ortho"],
            "val07_avg_proto": va["proto_loss"],
            "val07_avg_top1": va["top1"],
            "val07_avg_margin": va["margin"],
            "val07_avg_cos": va["correct_cosine"],
            "val07_avg_ortho": va["ortho"],
        }
        rows.append(row)

        print(
            f"[VAL07] step={step:6d} {name:<24s} "
            f"IND top1={vi['top1']:.4f} margin={vi['margin']:+.5f} "
            f"proto={vi['proto_loss']:.4f} | "
            f"AVG top1={va['top1']:.4f} margin={va['margin']:+.5f}"
        )

        # Primary selection: individual margin, tie-break top1.
        key = (vi["margin"], vi["top1"])
        if best is None or key > best[0]:
            best = (key, row)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n[BEST RECOVERED S1]")
    if best is None:
        print("No compatible checkpoints evaluated.")
    else:
        r = best[1]
        print(" checkpoint:", r["checkpoint"])
        print(" step      :", r["step"])
        print(" IND Top1  :", f"{r['val07_ind_top1']:.4f}")
        print(" IND Margin:", f"{r['val07_ind_margin']:+.5f}")
        print(" IND Proto :", f"{r['val07_ind_proto']:.4f}")
        print(" AVG Top1  :", f"{r['val07_avg_top1']:.4f}")
        print(" AVG Margin:", f"{r['val07_avg_margin']:+.5f}")

    print("\n[saved]", out_csv)


if __name__ == "__main__":
    main()
