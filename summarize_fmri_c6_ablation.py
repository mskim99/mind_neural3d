#!/usr/bin/env python3
"""Summarize E3-V050 C6 one-factor-at-a-time ablations across seeds.

Uses only validation and diagnostic training metrics from each summary.json.
Does not touch the official test split. Missing seeds remain missing (no fake SD=0).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = (
    "C6B0", "C6T_CENTER", "C6T_UNIFORM", "C6N_SAMPLE", "C6N_FRAME",
    "C6R_DROP04", "C6R_WD05",
)
SELECTS = ("objce", "10way")
KEYS = (
    "val_object_ce", "val_10way", "val_2way", "val_category_top1",
    "val_category_top5", "val_object_top5", "val_visual_10way",
    "train_category_top1", "train_10way", "cat_top1_train_minus_val",
)
PCT = {"val_10way", "val_2way", "val_category_top1", "val_category_top5",
       "val_object_top5", "val_visual_10way", "train_category_top1",
       "train_10way", "cat_top1_train_minus_val"}


def load_record(path: Path, run: str, seed: int, sel: str):
    if not path.is_file():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    item = blob.get("best_objce" if sel == "objce" else "best_10way")
    if not isinstance(item, dict) or "validation" not in item:
        raise ValueError(f"Invalid summary structure: {path} ({sel})")
    v = item["validation"]
    t = item.get("train_full_retrieval") or {}
    row = {"run": run, "seed": seed, "selection": sel, "epoch": item.get("epoch")}
    mapping = {
        "val_object_ce": (v, "object_ce"),
        "val_10way": (v, "object_10way_top1"),
        "val_2way": (v, "object_2way_top1"),
        "val_category_top1": (v, "category_top1"),
        "val_category_top5": (v, "category_top5"),
        "val_object_top5": (v, "object_top5"),
        "val_visual_10way": (v, "visual_10way"),
        "train_category_top1": (t, "category_top1"),
        "train_10way": (t, "object_10way_top1"),
    }
    for name, (src, key) in mapping.items():
        value = src.get(key) if isinstance(src, dict) else None
        row[name] = float(value) if value is not None else None
    row["cat_top1_train_minus_val"] = (
        row["train_category_top1"] - row["val_category_top1"]
        if row["train_category_top1"] is not None and row["val_category_top1"] is not None
        else None
    )
    return row


def fmt(vals, key, signed=False):
    valid = [v for v in vals if v is not None and math.isfinite(v)]
    if not valid:
        return "n/a"
    is_pct = key in PCT
    scale = 100 if is_pct else 1
    precision = 2 if is_pct else 4
    avg = statistics.mean(valid) * scale
    value = f"{avg:+.{precision}f}" if signed else f"{avg:.{precision}f}"
    suffix = ("pp" if signed else "%") if is_pct else ""
    if len(valid) == 1:
        return f"{value}{suffix} (n=1)"
    sd = statistics.stdev(valid) * scale
    return f"{value}±{sd:.{precision}f}{suffix}"


def csv_write(path, rows, names):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] Saved {path} ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(
        "/data/jionkim/mind_3d_output/fmri_c6_ablation"))
    p.add_argument("--runs", default=",".join(RUNS))
    p.add_argument("--seeds", default="0,1,2")
    a = p.parse_args()
    root = a.root.expanduser()
    runs = [x.strip() for x in a.runs.replace(" ", ",").split(",") if x.strip()]
    seeds = [int(x.strip()) for x in a.seeds.replace(" ", ",").split(",") if x.strip()]
    out = root / "C6_summary"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in runs:
        for seed in seeds:
            path = root / f"{run}_seed{seed}" / "summary.json"
            if not path.is_file():
                print(f"[Missing] {run} seed={seed}: {path}")
                continue
            for sel in SELECTS:
                rows.append(load_record(path, run, seed, sel))
    if not rows:
        print("[ERROR] No results found under", root)
        return 1

    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["run"], r["selection"])].append(r)

    # Paired C6 treatment - same-seed C6B0. Values are in native units (fractions for accuracies).
    indexed = {(r["run"], r["seed"], r["selection"]): r for r in rows}
    paired = []
    for r in rows:
        if r["run"] == "C6B0":
            continue
        b = indexed.get(("C6B0", r["seed"], r["selection"]))
        if b is None:
            continue
        new = {"run": r["run"], "seed": r["seed"], "selection": r["selection"]}
        for k in KEYS:
            new[k] = r[k] - b[k] if r[k] is not None and b[k] is not None else None
        paired.append(new)
    by_delta = defaultdict(list)
    for r in paired:
        by_delta[(r["run"], r["selection"])].append(r)

    print("\n=== C6 Validation: mean±SD over available seeds ===")
    print("run | selection | n | ValObjCE↓ | Val10Way↑ | ValCat@1↑ | ValCat@5↑ | ValVisual10Way↑ | Train-Val Cat@1 gap↓")
    for key in sorted(by_group):
        grp = by_group[key]
        items = [fmt([r.get(k) for r in grp],k) for k in (
            "val_object_ce", "val_10way", "val_category_top1", "val_category_top5",
            "val_visual_10way", "cat_top1_train_minus_val")]
        print(" | ".join([*key, str(len(grp)), *items]))

    print("\n=== C6 Paired change vs C6B0 (same seed, selection; accuracy in pp) ===")
    print("run | selection | n | ΔValObjCE↓ | ΔVal10Way↑ | ΔValCat@1↑ | ΔValVisual10Way↑")
    for key in sorted(by_delta):
        grp = by_delta[key]
        items = [fmt([r.get(k) for r in grp], k, signed=True) for k in (
            "val_object_ce", "val_10way", "val_category_top1", "val_visual_10way")]
        print(" | ".join([*key, str(len(grp)), *items]))

    csv_write(out / "C6_checkpoint_metrics.csv", rows,
              ["run", "seed", "selection", "epoch", *KEYS])
    csv_write(out / "C6_paired_deltas.csv", paired,
              ["run", "seed", "selection", *KEYS])

    print("\n[Notes]")
    print("- Only validation and training diagnostic metrics are read. No official test tuning.")
    print("- Object CE must be compared within the same split, candidate bank and checkpoint rule.")
    print("- Val10Way is representation-level fMRI-to-caption identification, NOT mesh N-way.")
    print("- train-val category gap is diagnostic, not proof of overfitting without split-distribution checks.")
    print("- n=1 has undefined standard deviation; missing seeds are never treated as zeros.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
