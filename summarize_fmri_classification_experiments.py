#!/usr/bin/env python3
"""Aggregate E3-V050 baseline and C3/C4 loss ablations across model seeds."""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

METRICS = (
    "object_ce", "object_2way_top1", "object_10way_top1", "object_top1", "object_top5",
    "category_top1", "category_top5", "classifier_ce", "classifier_top1",
    "classifier_top5", "classifier_macro_top1",
)


def extract(root: Path, prefix: str = ""):
    found = []
    for folder in sorted(root.glob("*_seed*")):
        if not folder.is_dir():
            continue
        match = re.fullmatch(r"(.+?)_(?:ft_)?seed(\d+)", folder.name)
        if not match:
            continue
        run, seed = match.group(1), int(match.group(2))
        summary_path = folder / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for selection, field in (("objce", "best_objce"), ("10way", "best_10way"),
                                 ("classifier", "best_classifier")):
            section = summary.get(field)
            if not section:
                continue
            met = section["validation"]
            row = dict(run=prefix+run, seed=seed, selection=selection,
                       best_epoch=section["epoch"])
            row.update({k: met.get(k) for k in METRICS})
            found.append(row)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline_root", default="")
    ap.add_argument("--baseline_run", default="E3V050")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    rows = extract(root)
    if args.baseline_root:
        rows += [r for r in extract(Path(args.baseline_root).expanduser().resolve(), prefix="baseline_")
                 if r["run"] == "baseline_" + args.baseline_run]
    if not rows:
        raise ValueError(f"No complete experiment summaries in {root}")
    out = Path(args.out_csv) if args.out_csv else root / "classification_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["run"],r["selection"]), []).append(r)
    print("RUN                    SELECT       SEEDS    ObjCE mean±sd    10Way mean±sd     Cat@1 mean±sd   LearnedCls@1 mean±sd")
    for (name, sel), members in sorted(groups.items()):
        def fmt(k, percent=False):
            values=[float(m[k]) for m in members if m[k] is not None]
            if not values:
                return "n/a"
            mean=statistics.mean(values); sd=statistics.stdev(values) if len(values)>1 else 0.0
            return f"{mean*100:.2f}±{sd*100:.2f}%" if percent else f"{mean:.4f}±{sd:.4f}"
        print(f"{name:22} {sel:12} {len(members):2d}       "
              f"{fmt('object_ce'):15} {fmt('object_10way_top1',True):17} "
              f"{fmt('category_top1',True):17} {fmt('classifier_top1',True)}")
    print(f"[*] Saved {out}")


if __name__ == "__main__":
    main()
