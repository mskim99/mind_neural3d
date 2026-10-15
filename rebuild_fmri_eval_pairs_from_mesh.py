#!/usr/bin/env python3
"""Reconstruct MinD-3D++ evaluation pairs from EXISTING TRELLIS GLBs.

This script does not regenerate fMRI predictions or semantic retrieval records.
It only writes a NEW structure/image evaluation CSV containing unambiguous
same-UID predicted meshes, generated 6-views, GT meshes and GT 6-views.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_RE = re.compile(r"^(?P<index>\d{5})__(?P<category>.+)__(?P<uid12>[0-9A-Fa-f]{12})$")
VIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True, help="G2_..._test directory (contains mesh/ and views/)")
    p.add_argument("--data-root", type=Path, required=True, help="MinD-3D++ root containing test_list.txt")
    p.add_argument("--gt-view-root", type=Path, required=True, help="GT views root containing <uid>/00.png .. 05.png")
    p.add_argument("--gt-mesh-root", type=Path, required=True, help="GT GLB root containing 000-019/<uid>.glb")
    p.add_argument("--out-csv", type=Path, default=None, help="New CSV path; must NOT be inference_records.csv")
    return p


def read_test_records(test_list: Path):
    if not test_list.is_file():
        raise FileNotFoundError(f"Test metadata not found: {test_list}")
    indexed = defaultdict(list)
    with test_list.open("r", encoding="utf-8-sig", newline="") as f:
        for lineno, parts in enumerate(csv.reader(f), 1):
            fields = [x.strip() for x in parts if x.strip()]
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 2:
                # Also accept whitespace-separated metadata lines.
                fields = fields[0].split(maxsplit=1)
            if len(fields) < 2:
                raise ValueError(f"Invalid metadata at {test_list}:{lineno}: {parts!r}")
            category = ",".join(fields[:-1]).strip()
            object_path = fields[-1].replace("\\", "/").strip()
            if lineno == 1 and category.lower() == "category" and object_path.lower() in ("uid", "object_path", "path"):
                continue
            uid = Path(object_path).stem
            indexed[uid[:12].lower()].append((uid, category, object_path))
    return indexed


def valid_six_views(directory: Path):
    if not directory.is_dir():
        return False
    for i in range(6):
        if not any((directory / (stem + ext)).is_file()
                   for stem in (f"{i:02d}", str(i)) for ext in VIEW_EXTS):
            return False
    return True


def sanitized(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip()).strip("_") or "sample"


def main():
    args = make_parser().parse_args()
    root = args.run_root.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    gt_views = args.gt_view_root.expanduser().resolve()
    gt_meshes = args.gt_mesh_root.expanduser().resolve()
    pred_meshes = root / "mesh"
    pred_views = root / "views"
    out = (args.out_csv.expanduser().resolve() if args.out_csv else root / "eval_pairs_from_mesh.csv")
    original_csv = root / "inference_records.csv"
    if out == original_csv:
        raise ValueError("Refusing to overwrite inference_records.csv. Choose a separate --out-csv.")
    if not pred_meshes.is_dir():
        raise FileNotFoundError(f"Predicted mesh root missing: {pred_meshes}")
    if not pred_views.is_dir():
        raise FileNotFoundError(f"Predicted view root missing: {pred_views}")
    index = read_test_records(data / "test_list.txt")
    all_glbs = sorted(pred_meshes.rglob("*.glb"))
    if not all_glbs:
        raise RuntimeError(f"No GLBs found beneath {pred_meshes}")

    reasons = Counter()
    excluded = []
    rows = []
    seen = set()
    for mesh in all_glbs:
        sample_id = mesh.parent.name
        m = SAMPLE_RE.fullmatch(sample_id)
        reason = None
        if not m or mesh.stem != sample_id:
            reason = "invalid_sample_dir_or_file_name"
        else:
            candidates = index.get(m.group("uid12").lower(), [])
            if len(candidates) != 1:
                reason = "uid_prefix_missing_or_ambiguous_in_test_list"
            else:
                uid, category, object_path = candidates[0]
                if sanitized(category) != m.group("category"):
                    reason = "category_mismatch"
                elif uid in seen:
                    reason = "duplicate_uid"
                else:
                    pv = pred_views / sample_id
                    gv = gt_views / uid
                    gm = gt_meshes / object_path
                    if not valid_six_views(pv):
                        reason = "predicted_six_views_missing"
                    elif not valid_six_views(gv):
                        reason = "gt_six_views_missing"
                    elif not gm.is_file():
                        reason = "gt_mesh_missing"
                    else:
                        seen.add(uid)
                        rows.append({
                            "sample_index": int(m.group("index")),
                            "sample_id": sample_id,
                            "uid": uid,
                            "true_category": category,
                            "views_dir": str(pv.resolve()),
                            "gt_view_dir": str(gv.resolve()),
                            "mesh_path": str(mesh.resolve()),
                            "gt_mesh_path": str(gm.resolve()),
                        })
        if reason:
            reasons[reason] += 1
            excluded.append((sample_id, reason, str(mesh)))

    rows.sort(key=lambda x: (x["sample_index"], x["uid"]))
    if not rows:
        print(f"Found {len(all_glbs)} GLBs but 0 complete evaluation pairs.")
        for name, count in reasons.most_common():
            print(f"  {name}: {count}")
        for entry in excluded[:8]:
            print("  excluded:", entry)
        raise RuntimeError("No matched mesh + 6-view pairs. Original CSV was not changed.")

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_index", "sample_id", "uid", "true_category", "views_dir", "gt_view_dir", "mesh_path", "gt_mesh_path"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_path = out.with_name(out.stem + "_skipped.csv")
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("sample_id", "reason", "glb_path"))
        writer.writerows(excluded)

    print(f"[INPUT]  Predicted GLBs: {len(all_glbs)}")
    print(f"[OUTPUT] Complete matching pairs: {len(rows)}")
    for name, count in reasons.most_common():
        print(f"[SKIPPED] {name}: {count}")
    print(f"[SAVED]  {out}")
    print(f"[REPORT] {report_path}")
    for row in rows[:3]:
        print("[PAIR]", row["sample_id"], "->", Path(row["gt_mesh_path"]).name)
    if len(rows) < 2:
        print("[NOTICE] CD/EMD can run, but FPD needs at least 2 successfully evaluated meshes.")


if __name__ == "__main__":
    main()
