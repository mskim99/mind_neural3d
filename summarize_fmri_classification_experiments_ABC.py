#!/usr/bin/env python3
"""Evaluate ONLY A) C4_02 multi-seed, B) within-category/train-bank,
C) C3 MLP vs linear for MinD-3D++ fMRI experiments.

Reads saved summary.json and fusion_sweep.csv files; no GPU needed.
Does NOT mix checkpoint criteria or validation GT-candidate metrics with
inference-feasible train-bank metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = (
    "object_ce", "object_2way_top1", "object_10way_top1",
    "category_top1", "category_top5", "classifier_ce",
    "classifier_top1", "classifier_top5", "classifier_macro_top1",
)
SELECTIONS = (("objce", "best_objce"), ("10way", "best_10way"),
              ("classifier", "best_classifier"))
PERCENT = {"object_2way_top1", "object_10way_top1", "category_top1",
           "category_top5", "classifier_top1", "classifier_top5",
           "classifier_macro_top1", "within_cat_2way", "train_bank_cat_top1",
           "train_bank_cat_top5", "val_10way", "val_2way"}


def numeric(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def mean_sd(values):
    x = [v for v in values if v is not None]
    if not x:
        return None, None
    return statistics.mean(x), statistics.stdev(x) if len(x) >= 2 else None


def fmt(values, key, delta=False):
    m, sd = mean_sd(values)
    if m is None:
        return "n/a"
    scale = 100.0 if key in PERCENT else 1.0
    suffix = "pp" if delta and scale == 100.0 else "%" if scale == 100.0 else ""
    digits = 2 if scale == 100.0 else 4
    if sd is None:
        return f"{m*scale:+.{digits}f}{suffix} (n=1)" if delta else f"{m*scale:.{digits}f}{suffix} (n=1)"
    sign = "+" if delta else ""
    return f"{m*scale:{sign}.{digits}f}±{sd*scale:.{digits}f}{suffix}"


def save_csv(path, rows):
    if not rows:
        print(f"[*] No rows for {path.name}; not writing empty file")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] Saved {path} ({len(rows)} rows)")


def load_training(root, runs, source, seeds):
    result = []
    for run in runs:
        for seed in seeds:
            p = root / f"{run}_seed{seed}" / "summary.json"
            if not p.is_file():
                print(f"[Missing] {source}/{run} seed={seed}: {p}")
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            for select, key in SELECTIONS:
                block = data.get(key)
                if not isinstance(block, dict) or not isinstance(block.get("validation"), dict):
                    continue
                met = block["validation"]
                row = dict(source=source, run=run, seed=seed, selection=select,
                           epoch=block.get("epoch"), checkpoint=block.get("checkpoint", ""),
                           summary=str(p))
                row.update({k: numeric(met.get(k)) for k in FIELDS})
                result.append(row)
    return result


def index_rows(rows):
    index = {}
    for row in rows:
        key = (row["source"], row["run"], row["seed"], row["selection"])
        if key in index:
            raise ValueError(f"Duplicate entry: {key}")
        index[key] = row
    return index


def aggregate_print(rows, metrics, group_fields, title, delta=False):
    print(f"\n=== {title} ===")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_fields)].append(row)
    head = " | ".join(group_fields)
    print(head + " | n | " + " | ".join(metrics))
    for group, members in sorted(groups.items(), key=lambda x: str(x[0])):
        values = []
        for k in metrics:
            values.append(fmt([r.get(k) for r in members], k, delta=delta))
        print(" | ".join(map(str, group)) + f" | {len(members)} | " + " | ".join(values))


def paired(rows, left_source, left_run, right_source, right_run, selections,
           metrics, keys=("selection",)):
    idx = index_rows(rows)
    seeds = sorted({r["seed"] for r in rows})
    result = []
    for seed in seeds:
        for select in selections:
            lhs = idx.get((left_source, left_run, seed, select))
            rhs = idx.get((right_source, right_run, seed, select))
            if lhs is None or rhs is None:
                continue
            row = dict(run=left_run,
                       baseline=right_run, comparison=f"{left_source}-minus-{right_source}", seed=seed,
                       selection=select)
            for k in metrics:
                a, b = lhs.get(k), rhs.get(k)
                row[k] = a-b if a is not None and b is not None else None
            result.append(row)
    return result


def load_fusion(root, runs, seeds, selections, alphas):
    result = []
    for run in runs:
        for seed in seeds:
            for select in selections:
                folder = root / f"{run}_{'best_objce' if select == 'objce' else 'best_10way'}_seed{seed}"
                path = folder / "fusion_sweep.csv"
                if not path.is_file():
                    print(f"[Missing] B fusion: {path}")
                    continue
                with path.open("r", encoding="utf-8-sig", newline="") as f:
                    raw = list(csv.DictReader(f))
                for alpha in alphas:
                    matches = [r for r in raw if r.get("scope") == "holdout" and
                               numeric(r.get("alpha")) is not None and
                               abs(float(r["alpha"]) - alpha) < 1e-7]
                    if len(matches) != 1:
                        print(f"[Missing/Ambiguous] B: {path}: holdout alpha={alpha}")
                        continue
                    m = matches[0]
                    result.append(dict(run=run, seed=seed, selection=select,
                                       alpha=alpha, n_queries=int(m["n_queries"]),
                                       n_within2=int(m["n_within2"]),
                                       val_2way=numeric(m.get("val_2way")),
                                       val_10way=numeric(m.get("val_10way")),
                                       within_cat_2way=numeric(m.get("within_cat_2way")),
                                       train_bank_cat_top1=numeric(m.get("train_bank_cat_top1")),
                                       train_bank_cat_top5=numeric(m.get("train_bank_cat_top5")),
                                       source_file=str(path)))
    return result


def paired_fusion(rows, metrics):
    idx = {(r["run"], r["seed"], r["selection"], r["alpha"]): r for r in rows}
    output = []
    for seed in sorted({r["seed"] for r in rows}):
        for select in sorted({r["selection"] for r in rows}):
            for alpha in sorted({r["alpha"] for r in rows}):
                a = idx.get(("C4_02", seed, select, alpha))
                b = idx.get(("C0", seed, select, alpha))
                if a is None or b is None:
                    continue
                row = dict(run="C4_02", baseline="C0", seed=seed,
                           selection=select, alpha=alpha,
                           n_within2_C4=a["n_within2"], n_within2_C0=b["n_within2"])
                for k in metrics:
                    row[k] = a[k]-b[k] if a[k] is not None and b[k] is not None else None
                output.append(row)
    return output


def parse_seeds(s):
    result = sorted({int(v) for v in re.split(r"[,\s]+", s.strip()) if v})
    if not result or any(v < 0 for v in result):
        raise ValueError("--seeds must include nonnegative integers")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/jionkim/mind_3d_output/fmri_classification_experiments",
                        help="C0/C4_02 + existing linear C3 experiment root")
    parser.add_argument("--mlp_root", default="/data/jionkim/mind_3d_output/fmri_classification_mlp",
                        help="New C3 MLP experiments root")
    parser.add_argument("--diagnostic_root", default="/data/jionkim/mind_3d_output/fmri_classification_experiments/fusion_diagnostics",
                        help="B fusion diagnostics root")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out_dir", default="", help="Default: <root>/ABC_summary")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    mlp_root = Path(args.mlp_root).expanduser().resolve()
    diag_root = Path(args.diagnostic_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "ABC_summary"
    seeds = parse_seeds(args.seeds)
    print(f"[*] Seeds: {seeds}\n[*] A/C: {root}\n[*] MLP: {mlp_root}\n[*] B: {diag_root}")

    original = load_training(root, ("C0", "C4_02", "C3_01", "C3_03"), "linear", seeds)
    mlp = load_training(mlp_root, ("C3_01", "C3_03"), "mlp", seeds)

    # A: Same-seed paired C4 vs C0, never compare different checkpoint selections.
    a_rows = [r for r in original if r["run"] in ("C0", "C4_02") and
              r["selection"] in ("objce", "10way")]
    a_metrics = ("object_ce", "object_10way_top1", "category_top1")
    aggregate_print(a_rows, a_metrics, ("run", "selection"), "A. C4_02 vs C0: validation")
    a_delta = paired(a_rows, "linear", "C4_02", "linear", "C0",
                     ("objce", "10way"), a_metrics)
    aggregate_print(a_delta, a_metrics, ("run", "selection"), "A. Paired C4_02-C0 difference (same seed)", delta=True)
    save_csv(out / "A_checkpoint_metrics.csv", a_rows)
    save_csv(out / "A_paired_deltas.csv", a_delta)

    # B: NOTE validation GT visual candidates are diagnostic ONLY. Train bank is inference-feasible.
    b_metrics = ("within_cat_2way", "train_bank_cat_top1", "train_bank_cat_top5", "val_10way")
    b_rows = load_fusion(diag_root, ("C0", "C4_02"), seeds,
                         ("objce", "10way"), (1.0, 0.5))
    aggregate_print(b_rows, b_metrics, ("run", "selection", "alpha"),
                    "B. Holdout within-category / train-bank retrieval")
    b_delta = paired_fusion(b_rows, b_metrics)
    aggregate_print(b_delta, b_metrics, ("selection", "alpha"),
                    "B. Paired C4_02-C0 difference (same holdout + alpha)", delta=True)
    save_csv(out / "B_fusion_holdout_metrics.csv", b_rows)
    save_csv(out / "B_paired_deltas.csv", b_delta)

    # C: Linear vs MLP C3; compare classifier rows to classifier rows, ObjCE rows to ObjCE rows.
    c_rows = [r for r in original + mlp if r["run"] in ("C3_01", "C3_03")]
    c_metrics = ("classifier_ce", "classifier_top1", "classifier_top5",
                 "classifier_macro_top1", "object_10way_top1", "object_ce")
    aggregate_print(c_rows, c_metrics, ("source", "run", "selection"),
                    "C. C3: linear vs MLP classification")
    c_delta = []
    for run in ("C3_01", "C3_03"):
        c_delta += paired(c_rows, "mlp", run, "linear", run,
                          ("objce", "10way", "classifier"), c_metrics)
    aggregate_print(c_delta, c_metrics, ("run", "selection"),
                    "C. Paired MLP-linear difference (same seed + checkpoint criterion)", delta=True)
    save_csv(out / "C_head_metrics.csv", c_rows)
    save_csv(out / "C_paired_deltas.csv", c_delta)

    print("\n[Notes]")
    print("- n=1: SD is undefined; no ±0.00 shown.")
    print("- A/C are original validation metrics; do not choose models using official test.")
    print("- B alpha=1: semantic only; alpha=0.5: fixed semantic/visual fusion.")
    print("- B within_cat_2way uses validation GT candidate features (diagnostic), not real mesh ID.")
    print("- B train_bank_cat_top1/5 uses TRAIN UID candidates only (inference-feasible).")
    print("- B holdout excludes only fusion alpha calibration queries, not the original checkpoint selection data.")
    print("- C best_classifier selects minimum validation classifier CE, not highest Cat@1.")
    print(f"[*] Done. Files under {out}")


if __name__ == "__main__":
    main()
