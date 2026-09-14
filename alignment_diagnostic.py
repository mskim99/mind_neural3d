#!/usr/bin/env python3
"""
Diagnose label / stimulus / EEG alignment in Neuro-3D static EEG arrays.

This script does NOT train a neural network. It checks whether the assumed
index correspondence

    [subject, class, object, trial, channel, time]
    [1,       72,    8,      2,     64,      250]

is internally supported by the EEG itself.

Main diagnostics
----------------
1) For each object, build a 72x72 cosine-similarity matrix between
   trial 0 classes and trial 1 classes.
2) Evaluate the assumed identity alignment (class k <-> class k).
3) Use Hungarian matching to infer the best trial-0 -> trial-1 class mapping.
4) Measure whether the inferred permutation is consistent across objects.
5) Run permutation tests for the identity diagonal.
6) Leave-one-object-out test:
      infer a class mapping from 7 objects,
      then test whether it improves similarity on the held-out object.
   This is important because an arbitrary Hungarian assignment can always
   improve the score on the same matrix.
7) Reverse-axis diagnostic:
   within each class, retrieve object identity (8x8) across trials.
   This helps determine whether the 8-axis is more reproducible than the
   assumed 72-class axis.

The script cannot prove that a stimulus filename/event code is correct because
those metadata are not contained in the .npy tensor. It tests internal
consistency of the tensor indexing. Event/stimulus logs should still be traced
in the original preprocessing code.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "scipy is required. Install it with: pip install scipy"
    ) from exc


CLASS_COUNT = 72
OBJECT_COUNT = 8
TRIAL_COUNT = 2
CHANNELS = 64
SAMPLES = 250


def normalize_eeg(x, mode="global_zscore"):
    """Normalize each EEG example independently, matching the training code."""
    x = np.asarray(x, dtype=np.float32)

    if mode == "none":
        return x

    if mode == "sample_minmax":
        minimum = x.min(axis=(-2, -1), keepdims=True)
        maximum = x.max(axis=(-2, -1), keepdims=True)
        return ((x - minimum) / np.maximum(maximum - minimum, 1e-6)).astype(
            np.float32
        )

    if mode == "channel_zscore":
        mean = x.mean(axis=-1, keepdims=True)
        scale = x.std(axis=-1, keepdims=True)
    elif mode == "global_zscore":
        mean = x.mean(axis=(-2, -1), keepdims=True)
        scale = x.std(axis=(-2, -1), keepdims=True)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return ((x - mean) / np.maximum(scale, 1e-6)).astype(np.float32)


def load_train_array(data_path, sub_id):
    path = (
        Path(data_path).expanduser()
        / "EEGdata"
        / sub_id
        / f"{sub_id}_train_data_1s_250Hz.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = np.load(path, mmap_mode="r")

    # Accept either [1,72,8,2,64,250] or [72,8,2,64,250].
    if raw.shape == (CLASS_COUNT, OBJECT_COUNT, TRIAL_COUNT, CHANNELS, SAMPLES):
        raw = raw[None]

    expected = (1, CLASS_COUNT, OBJECT_COUNT, TRIAL_COUNT, CHANNELS, SAMPLES)
    if tuple(raw.shape) != expected:
        raise ValueError(f"Expected {expected}, got {tuple(raw.shape)}")

    if not np.isfinite(raw).all():
        raise ValueError("Input contains NaN or Inf")

    return np.asarray(raw[0], dtype=np.float32), path


def crop_window(x, start_ms, end_ms, sample_rate=250):
    if start_ms == 0 and end_ms == 1000:
        return x

    start = int(round(start_ms * sample_rate / 1000.0))
    end = int(round(end_ms * sample_rate / 1000.0))
    start = max(0, min(SAMPLES, start))
    end = max(0, min(SAMPLES, end))
    if end <= start:
        raise ValueError(f"Invalid window {start_ms}-{end_ms} ms")
    return x[..., start:end]


def parse_window(value):
    try:
        a, b = value.split("-", 1)
        start_ms, end_ms = int(a), int(b)
    except Exception as exc:
        raise ValueError("--window must look like 0-1000 or 200-500") from exc

    if start_ms < 0 or end_ms > 1000 or start_ms >= end_ms:
        raise ValueError("--window must satisfy 0 <= start < end <= 1000")
    return start_ms, end_ms


def flatten_unit(x):
    z = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    z -= z.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norm, 1e-12)


def cosine_matrix(a, b):
    """Pairwise cosine similarity between example sets a and b."""
    a = flatten_unit(a)
    b = flatten_unit(b)
    return a @ b.T


def retrieval_metrics(sim):
    n = sim.shape[0]
    if sim.shape != (n, n):
        raise ValueError("Expected square similarity matrix")

    target = np.arange(n)
    order = np.argsort(-sim, axis=1)
    top1 = float(np.mean(order[:, 0] == target))
    k = min(5, n)
    top5 = float(np.mean(np.any(order[:, :k] == target[:, None], axis=1)))

    diag = np.diag(sim)
    off_mask = ~np.eye(n, dtype=bool)
    off = sim[off_mask]

    return {
        "top1": top1,
        "top5": top5,
        "diag_mean": float(diag.mean()),
        "offdiag_mean": float(off.mean()),
        "diag_minus_off": float(diag.mean() - off.mean()),
    }


def hungarian_mapping(sim):
    """Return mapping[row] = assigned column maximizing total similarity."""
    rows, cols = linear_sum_assignment(-sim)
    mapping = np.empty(sim.shape[0], dtype=np.int64)
    mapping[rows] = cols
    score = float(sim[rows, cols].mean())
    identity_fraction = float(np.mean(mapping == np.arange(sim.shape[0])))
    return mapping, score, identity_fraction


def permutation_test_score(sim, mapping, permutations, rng):
    """
    Test a fixed mapping against random one-to-one permutations.

    This is valid for:
      * identity mapping on an observed matrix;
      * a mapping learned from OTHER objects and evaluated on held-out data.

    Do NOT use this to claim significance for a Hungarian mapping learned and
    tested on the same matrix.
    """
    n = sim.shape[0]
    rows = np.arange(n)
    mapping = np.asarray(mapping, dtype=np.int64)
    observed = float(sim[rows, mapping].mean())

    null = np.empty(permutations, dtype=np.float64)
    for i in range(permutations):
        p = rng.permutation(n)
        null[i] = sim[rows, p].mean()

    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1)) if permutations > 1 else 0.0
    z = float((observed - null_mean) / max(null_std, 1e-12))

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "p_value": p_value,
    }


def inverse_mapping(mapping):
    inv = np.empty_like(mapping)
    inv[mapping] = np.arange(mapping.size)
    return inv


def permutation_cycles(mapping):
    """Cycle decomposition, useful for spotting a systematic label permutation."""
    mapping = np.asarray(mapping, dtype=np.int64)
    visited = np.zeros(mapping.size, dtype=bool)
    cycles = []

    for start in range(mapping.size):
        if visited[start]:
            continue
        cur = start
        cycle = []
        while not visited[cur]:
            visited[cur] = True
            cycle.append(int(cur))
            cur = int(mapping[cur])
        cycles.append(cycle)

    cycles.sort(key=lambda c: (-len(c), c[0]))
    return cycles


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_matrix_csv(path, matrix):
    np.savetxt(path, matrix, delimiter=",", fmt="%.8f")


def maybe_plot_matrix(matrix, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable; skipping plots")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xlabel("Trial 1 index")
    ax.set_ylabel("Trial 0 index")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def class_axis_diagnostic(x, permutations, rng, out_dir, save_plots=False):
    """
    For each object:
      trial0 classes [72,...] vs trial1 classes [72,...]
    """
    object_rows = []
    matrices = []
    mappings = []

    identity = np.arange(CLASS_COUNT)

    print("\n=== CLASS-AXIS DIAGNOSTIC: 72x72 T0->T1 per object ===")

    for obj in range(OBJECT_COUNT):
        a = x[:, obj, 0]  # [72,C,T]
        b = x[:, obj, 1]

        sim = cosine_matrix(a, b)
        matrices.append(sim)

        metrics = retrieval_metrics(sim)
        mapping, hungarian_score, identity_fraction = hungarian_mapping(sim)
        mappings.append(mapping)

        identity_test = permutation_test_score(
            sim, identity, permutations, rng
        )

        row = {
            "object": obj,
            **metrics,
            "identity_perm_observed": identity_test["observed"],
            "identity_perm_null_mean": identity_test["null_mean"],
            "identity_perm_null_std": identity_test["null_std"],
            "identity_perm_z": identity_test["z"],
            "identity_perm_p": identity_test["p_value"],
            "hungarian_score": hungarian_score,
            "hungarian_identity_fraction": identity_fraction,
        }
        object_rows.append(row)

        save_matrix_csv(out_dir / f"class_similarity_object_{obj:02d}.csv", sim)
        np.save(out_dir / f"class_similarity_object_{obj:02d}.npy", sim)
        np.savetxt(
            out_dir / f"class_hungarian_mapping_object_{obj:02d}.csv",
            np.column_stack([np.arange(CLASS_COUNT), mapping]),
            delimiter=",",
            fmt="%d",
            header="trial0_index,matched_trial1_index",
            comments="",
        )

        if save_plots:
            maybe_plot_matrix(
                sim,
                out_dir / f"class_similarity_object_{obj:02d}.png",
                f"Object {obj:02d}: Trial 0 vs Trial 1 class similarity",
            )

        print(
            f"[object {obj:02d}] "
            f"identity top1={metrics['top1']:.4%} "
            f"top5={metrics['top5']:.4%} "
            f"gap={metrics['diag_minus_off']:+.6f} "
            f"identity-p={identity_test['p_value']:.6f} "
            f"Hungarian identity={identity_fraction:.4%}"
        )

    matrices = np.stack(matrices, axis=0)  # [8,72,72]
    mappings = np.stack(mappings, axis=0)  # [8,72]

    write_csv(out_dir / "class_axis_by_object.csv", object_rows)

    # Pairwise agreement between independently inferred object permutations.
    pairwise = np.eye(OBJECT_COUNT, dtype=np.float64)
    pairwise_rows = []
    for i in range(OBJECT_COUNT):
        for j in range(i + 1, OBJECT_COUNT):
            agreement = float(np.mean(mappings[i] == mappings[j]))
            pairwise[i, j] = pairwise[j, i] = agreement
            pairwise_rows.append(
                {"object_a": i, "object_b": j, "mapping_agreement": agreement}
            )

    save_matrix_csv(out_dir / "hungarian_mapping_pairwise_agreement.csv", pairwise)
    write_csv(out_dir / "hungarian_mapping_pairwise_pairs.csv", pairwise_rows)

    # Aggregate similarity matrix and global mapping.
    aggregate = matrices.mean(axis=0)
    aggregate_metrics = retrieval_metrics(aggregate)
    aggregate_mapping, aggregate_h_score, aggregate_identity_fraction = (
        hungarian_mapping(aggregate)
    )
    aggregate_identity_test = permutation_test_score(
        aggregate, identity, permutations, rng
    )

    save_matrix_csv(out_dir / "class_similarity_aggregate.csv", aggregate)
    np.save(out_dir / "class_similarity_aggregate.npy", aggregate)
    np.savetxt(
        out_dir / "class_hungarian_mapping_aggregate.csv",
        np.column_stack([np.arange(CLASS_COUNT), aggregate_mapping]),
        delimiter=",",
        fmt="%d",
        header="trial0_index,matched_trial1_index",
        comments="",
    )
    if save_plots:
        maybe_plot_matrix(
            aggregate,
            out_dir / "class_similarity_aggregate.png",
            "Mean over 8 objects: Trial 0 vs Trial 1 class similarity",
        )

    # Leave-one-object-out:
    # infer mapping from 7 objects; evaluate FIXED mapping on unseen object.
    loo_rows = []
    loo_mappings = []

    print("\n=== LEAVE-ONE-OBJECT-OUT SYSTEMATIC-PERMUTATION TEST ===")

    for heldout in range(OBJECT_COUNT):
        train_ids = [i for i in range(OBJECT_COUNT) if i != heldout]
        train_mean = matrices[train_ids].mean(axis=0)
        learned_mapping, _, train_identity_fraction = hungarian_mapping(train_mean)
        loo_mappings.append(learned_mapping)

        heldout_sim = matrices[heldout]
        fixed_test = permutation_test_score(
            heldout_sim, learned_mapping, permutations, rng
        )
        identity_test = permutation_test_score(
            heldout_sim, identity, permutations, rng
        )

        row = {
            "heldout_object": heldout,
            "train_mapping_identity_fraction": train_identity_fraction,
            "heldout_learned_mapping_score": fixed_test["observed"],
            "heldout_learned_mapping_null_mean": fixed_test["null_mean"],
            "heldout_learned_mapping_z": fixed_test["z"],
            "heldout_learned_mapping_p": fixed_test["p_value"],
            "heldout_identity_score": identity_test["observed"],
            "heldout_identity_z": identity_test["z"],
            "heldout_identity_p": identity_test["p_value"],
        }
        loo_rows.append(row)

        print(
            f"[held-out object {heldout:02d}] "
            f"learned-map p={fixed_test['p_value']:.6f} "
            f"z={fixed_test['z']:+.3f}; "
            f"identity p={identity_test['p_value']:.6f} "
            f"z={identity_test['z']:+.3f}"
        )

    write_csv(out_dir / "class_axis_leave_one_object_out.csv", loo_rows)

    # Per-class consensus of mappings learned independently on each object.
    # This is descriptive only; a genuine permutation should have high agreement.
    consensus = np.zeros(CLASS_COUNT, dtype=np.int64)
    consensus_fraction = np.zeros(CLASS_COUNT, dtype=np.float64)
    for cls in range(CLASS_COUNT):
        counts = np.bincount(mappings[:, cls], minlength=CLASS_COUNT)
        consensus[cls] = int(np.argmax(counts))
        consensus_fraction[cls] = float(counts[consensus[cls]] / OBJECT_COUNT)

    np.savetxt(
        out_dir / "class_mapping_consensus.csv",
        np.column_stack(
            [np.arange(CLASS_COUNT), consensus, consensus_fraction]
        ),
        delimiter=",",
        fmt=["%d", "%d", "%.6f"],
        header="trial0_index,consensus_trial1_index,object_agreement_fraction",
        comments="",
    )

    summary = {
        "mean_identity_top1": float(np.mean([r["top1"] for r in object_rows])),
        "mean_identity_top5": float(np.mean([r["top5"] for r in object_rows])),
        "mean_diag_minus_off": float(
            np.mean([r["diag_minus_off"] for r in object_rows])
        ),
        "mean_identity_perm_p": float(
            np.mean([r["identity_perm_p"] for r in object_rows])
        ),
        "mean_pairwise_hungarian_mapping_agreement": float(
            np.mean([r["mapping_agreement"] for r in pairwise_rows])
        ),
        "aggregate": {
            **aggregate_metrics,
            "identity_perm_z": aggregate_identity_test["z"],
            "identity_perm_p": aggregate_identity_test["p_value"],
            "hungarian_score": aggregate_h_score,
            "hungarian_identity_fraction": aggregate_identity_fraction,
            "hungarian_cycles": permutation_cycles(aggregate_mapping),
        },
        "loo_mean_learned_mapping_z": float(
            np.mean([r["heldout_learned_mapping_z"] for r in loo_rows])
        ),
        "loo_fraction_significant_p_lt_0_05": float(
            np.mean(
                [r["heldout_learned_mapping_p"] < 0.05 for r in loo_rows]
            )
        ),
        "mean_consensus_fraction": float(consensus_fraction.mean()),
    }

    return summary


def object_axis_diagnostic(x, permutations, rng, out_dir, save_plots=False):
    """
    Reverse-axis diagnostic.

    For each class:
       trial0 objects [8,...] vs trial1 objects [8,...]

    If object identity is much more reproducible than class identity, it can
    indicate that the signal is dominated by object-specific rather than
    class-specific structure, or that the assumed axis semantics need review.
    """
    rows = []
    matrices = []
    identity = np.arange(OBJECT_COUNT)

    print("\n=== OBJECT-AXIS DIAGNOSTIC: 8x8 T0->T1 within each class ===")

    for cls in range(CLASS_COUNT):
        a = x[cls, :, 0]  # [8,C,T]
        b = x[cls, :, 1]
        sim = cosine_matrix(a, b)
        matrices.append(sim)

        metrics = retrieval_metrics(sim)
        identity_test = permutation_test_score(
            sim, identity, permutations, rng
        )

        rows.append(
            {
                "class": cls,
                **metrics,
                "identity_perm_z": identity_test["z"],
                "identity_perm_p": identity_test["p_value"],
            }
        )

    matrices = np.stack(matrices)
    aggregate = matrices.mean(axis=0)
    aggregate_metrics = retrieval_metrics(aggregate)
    aggregate_test = permutation_test_score(
        aggregate, identity, permutations, rng
    )

    write_csv(out_dir / "object_axis_by_class.csv", rows)
    save_matrix_csv(out_dir / "object_similarity_aggregate.csv", aggregate)
    np.save(out_dir / "object_similarity_aggregate.npy", aggregate)

    if save_plots:
        maybe_plot_matrix(
            aggregate,
            out_dir / "object_similarity_aggregate.png",
            "Mean over 72 classes: Trial 0 vs Trial 1 object similarity",
        )

    summary = {
        "mean_top1_over_classes": float(np.mean([r["top1"] for r in rows])),
        "mean_top5_over_classes": float(np.mean([r["top5"] for r in rows])),
        "mean_diag_minus_off_over_classes": float(
            np.mean([r["diag_minus_off"] for r in rows])
        ),
        "aggregate": {
            **aggregate_metrics,
            "identity_perm_z": aggregate_test["z"],
            "identity_perm_p": aggregate_test["p_value"],
        },
    }

    print(
        f"[object-axis aggregate] "
        f"top1={aggregate_metrics['top1']:.4%} "
        f"top5={aggregate_metrics['top5']:.4%} "
        f"gap={aggregate_metrics['diag_minus_off']:+.6f} "
        f"p={aggregate_test['p_value']:.6f}"
    )

    return summary


def interpret(class_summary, object_summary):
    agg = class_summary["aggregate"]
    loo_sig = class_summary["loo_fraction_significant_p_lt_0_05"]
    pair_agree = class_summary["mean_pairwise_hungarian_mapping_agreement"]
    consensus = class_summary["mean_consensus_fraction"]
    object_agg = object_summary["aggregate"]

    notes = []

    if agg["identity_perm_p"] < 0.05 and agg["diag_minus_off"] > 0:
        notes.append(
            "The assumed class identity alignment has significant positive "
            "cross-trial similarity after averaging across objects."
        )
    else:
        notes.append(
            "The assumed class identity alignment is not strongly supported "
            "by aggregate cross-trial similarity."
        )

    if loo_sig >= 0.5 and pair_agree > (1.0 / CLASS_COUNT) * 3:
        notes.append(
            "A non-identity class permutation appears reproducible across "
            "objects; inspect class_hungarian_mapping_aggregate.csv and the "
            "leave-one-object-out results for a possible systematic ordering mismatch."
        )
    else:
        notes.append(
            "No strong evidence of one stable non-identity class permutation "
            "generalizing across objects was detected."
        )

    if object_agg["top1"] > agg["top1"] + 0.10:
        notes.append(
            "Object identity is substantially more reproducible across trials "
            "than the assumed 72-class identity. Review the meaning/order of "
            "the 72 and 8 axes and possible object-dominant EEG structure."
        )

    if consensus < 0.25:
        notes.append(
            "Hungarian mappings vary strongly by object, which argues against "
            "a single fixed class permutation as the sole problem."
        )

    notes.append(
        "This script cannot verify stimulus filenames, event codes, or onset "
        "timestamps because those are not stored in the EEG tensor. Trace the "
        "original preprocessing/event log before concluding that alignment is correct."
    )

    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    parser.add_argument("--sub_id", default="sub01")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--input_norm",
        choices=["none", "sample_minmax", "channel_zscore", "global_zscore"],
        default="global_zscore",
    )
    parser.add_argument(
        "--window",
        default="0-1000",
        help="Temporal crop in milliseconds, e.g. 0-1000 or 300-700.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Number of random permutations for each significance test.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save similarity heatmaps as PNG files.",
    )
    args = parser.parse_args()

    if args.permutations < 100:
        raise ValueError("--permutations should be at least 100")

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    raw, path = load_train_array(args.data_path, args.sub_id)
    start_ms, end_ms = parse_window(args.window)

    # Crop first, then normalize within the selected window.
    # This avoids letting samples outside the requested window affect the
    # normalization statistics.
    x = crop_window(raw, start_ms, end_ms)
    x = normalize_eeg(x, args.input_norm)

    print(f"[data] path={path}")
    print(f"[data] raw_shape={tuple(raw.shape)}")
    print(
        f"[data] assumed_axes=[class=72, object=8, trial=2, channel=64, time=250]"
    )
    print(
        f"[data] window={start_ms}-{end_ms} ms "
        f"cropped_samples={x.shape[-1]} norm={args.input_norm}"
    )
    print(
        f"[chance] class top1={1/CLASS_COUNT:.4%} "
        f"top5={5/CLASS_COUNT:.4%}; "
        f"object top1={1/OBJECT_COUNT:.4%} "
        f"top5={5/OBJECT_COUNT:.4%}"
    )

    class_summary = class_axis_diagnostic(
        x, args.permutations, rng, out, save_plots=args.plots
    )
    object_summary = object_axis_diagnostic(
        x, args.permutations, rng, out, save_plots=args.plots
    )

    notes = interpret(class_summary, object_summary)

    summary = {
        "input": {
            "path": str(path),
            "shape": list(raw.shape),
            "assumed_axes": ["class", "object", "trial", "channel", "time"],
            "normalization": args.input_norm,
            "window_ms": [start_ms, end_ms],
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "class_axis": class_summary,
        "object_axis": object_summary,
        "interpretation": notes,
    }

    (out / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== INTERPRETATION ===")
    for i, note in enumerate(notes, 1):
        print(f"{i}. {note}")

    print("\n[outputs]")
    print(f"  {out / 'alignment_summary.json'}")
    print(f"  {out / 'class_axis_by_object.csv'}")
    print(f"  {out / 'class_hungarian_mapping_aggregate.csv'}")
    print(f"  {out / 'class_axis_leave_one_object_out.csv'}")
    print(f"  {out / 'hungarian_mapping_pairwise_agreement.csv'}")
    print(f"  {out / 'object_axis_by_class.csv'}")
    print(f"  {out / 'object_similarity_aggregate.csv'}")


if __name__ == "__main__":
    main()
