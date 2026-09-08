#!/usr/bin/env python3
"""Auxiliary EEG diagnostics: nuisance IDs, strict splits, and timing controls.

This script does not train the semantic decoder and does not use an Oracle or
the diffusion generator. It reads ``base.eeg_data`` directly, so rendered
images are not loaded after the dataset constructor.

Tasks
-----
1. Trial-ID decoding: can EEG distinguish trial 0 from trial 1?
2. Object-ID decoding: can EEG distinguish the object suffix/column?
   Both use class-grouped OOF Ridge so the same semantic class is not present
   in both train and validation folds.
3. Strict trial-ID decoding with ``subject+class+object`` groups, followed by
   within-pair trial-label permutation nulls.
4. Optional acquisition-time-block decoding. Supply ``--acquisition_csv``
   with an acquisition index/timestamp or an explicit time block; this tests
   whether trial decoding survives chronological block holdout.
5. Semantic temporal-alignment controls: original, reversal, circular shifts,
   valid-crop shifts (no zero-padding), and short temporal windows. A
   stimulus-locked signal should peak near the original condition/window and
   degrade under reversal or sufficiently large shifts. This is evidence, not
   proof of absolute acquisition timing.
6. Optional event metadata audit: if ``--event_csv`` is supplied, verify that
   event rows cover every EEG index and that onset/sample information is within
   the declared window.

Default objects are 00--06; object 07 is not sampled. Use 00--05 to match the
previous cross-object train split exactly.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def object_suffix(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[-1]
    return ""


def selected_columns(base, suffixes):
    wanted = set(suffixes)
    columns = []
    for o in range(int(base.obj_num)):
        suffixes_seen = {
            object_suffix(base.name_list[c, o])
            for c in range(int(base.cls_num))
        }
        if len(suffixes_seen) != 1:
            raise RuntimeError(f"Object column {o} has inconsistent suffixes: {suffixes_seen}")
        if next(iter(suffixes_seen)) in wanted:
            columns.append(o)
    found = {object_suffix(base.name_list[0, o]) for o in columns}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Requested object suffixes are missing: {sorted(missing)}")
    return columns


def make_records(base, suffixes):
    import numpy as np

    eeg = np.asarray(base.eeg_data, dtype=np.float32)
    expected = (
        int(base.eeg_data.shape[0]),
        int(base.cls_num),
        int(base.obj_num),
        int(base.trails_num),
        64,
        600,
    )
    if tuple(eeg.shape) != expected:
        raise RuntimeError(f"Expected EEG shape {expected}, got {tuple(eeg.shape)}")
    columns = selected_columns(base, suffixes)
    suffix_values = [object_suffix(base.name_list[0, o]) for o in columns]
    suffix_to_label = {suffix: i for i, suffix in enumerate(sorted(suffix_values))}
    records = []
    arrays = []
    record_index = 0
    for s in range(eeg.shape[0]):
        for c in range(int(base.cls_num)):
            for o in columns:
                suffix = object_suffix(base.name_list[c, o])
                for r in range(int(base.trails_num)):
                    x = eeg[s, c, o, r]
                    if not np.isfinite(x).all():
                        raise RuntimeError(f"Non-finite EEG at {(s, c, o, r)}")
                    records.append(
                        {
                            "record_index": int(record_index),
                            "subject": int(s),
                            "class": int(c),
                            "object_col": int(o),
                            "object_suffix": suffix,
                            "object_label": int(suffix_to_label[suffix]),
                            "trial": int(r),
                            "name": str(base.name_list[c, o]),
                        }
                    )
                    arrays.append(x.copy())
                    record_index += 1
    return np.stack(arrays), records, columns, suffix_to_label


def normalize_channels(x):
    import numpy as np

    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def pooled_features(x, pool_width):
    import numpy as np

    if x.ndim != 3 or x.shape[1] != 64:
        raise ValueError(f"Expected raw EEG [N,64,T], got {tuple(x.shape)}")
    time_length = int(x.shape[-1])
    if pool_width < 1 or time_length % pool_width:
        raise ValueError(
            f"pool_width must be a positive divisor of the current time length {time_length}"
        )
    z = normalize_channels(x)
    z = z.reshape(len(z), 64, time_length // pool_width, pool_width)
    mean = z.mean(axis=-1)
    std = z.std(axis=-1)
    return np.concatenate([mean, std], axis=1).reshape(len(z), -1).astype(np.float32)


def zero_padded_shift(x, shift):
    import numpy as np

    if shift == 0:
        return x.copy()
    if abs(int(shift)) >= x.shape[-1]:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    if shift > 0:
        out[..., shift:] = x[..., :-shift]
    else:
        out[..., :shift] = x[..., -shift:]
    return out


def circular_shift(x, shift):
    import numpy as np

    return np.roll(x, int(shift), axis=-1)


def valid_crop_shift(x, shift, pool_width):
    """Return the overlap after a temporal shift, without inserting zeros."""
    shift = int(shift)
    if shift == 0:
        out = x.copy()
    elif abs(shift) >= x.shape[-1]:
        raise ValueError("valid crop shift must be smaller than the time length")
    elif shift > 0:
        out = x[..., :-shift]
    else:
        out = x[..., -shift:]
    usable = (out.shape[-1] // pool_width) * pool_width
    if usable < pool_width:
        raise ValueError(
            f"valid crop leaves {out.shape[-1]} samples, incompatible with pool_width={pool_width}"
        )
    return out[..., :usable].copy()


def transform_raw(x, condition):
    if condition == "original":
        return x.copy()
    if condition == "reverse":
        return x[..., ::-1].copy()
    if condition.startswith("zero_shift_"):
        return zero_padded_shift(x, int(condition.split("_", 2)[2]))
    if condition.startswith("circular_shift_"):
        return circular_shift(x, int(condition.split("_", 2)[2]))
    raise ValueError(f"Unknown temporal condition: {condition}")


def group_ids(records, key):
    import numpy as np

    values = [r[key] for r in records]
    mapping = {value: i for i, value in enumerate(sorted(set(values), key=str))}
    return np.asarray([mapping[v] for v in values], dtype=np.int64)


def grouped_ids(records, keys):
    import numpy as np

    values = [tuple(r[key] for key in keys) for r in records]
    mapping = {value: i for i, value in enumerate(sorted(set(values), key=str))}
    return np.asarray([mapping[v] for v in values], dtype=np.int64)


def oof_ridge(x, y, groups, alphas, seed=42, fixed_alpha=None):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("At least two groups are required for OOF evaluation")
    n_splits = min(6, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)

    def make(alpha):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "ridge",
                    RidgeClassifier(
                        alpha=float(alpha),
                        solver="lsqr",
                        tol=1e-5,
                        max_iter=3000,
                    ),
                ),
            ]
        )

    if fixed_alpha is None:
        search = GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", RidgeClassifier(solver="lsqr", tol=1e-5, max_iter=3000)),
                ]
            ),
            {"ridge__alpha": list(alphas)},
            cv=cv,
            scoring="accuracy",
            refit=False,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(x, y, groups=groups)
        alpha = float(search.best_params_["ridge__alpha"])
        selection_score = float(search.best_score_)
    else:
        alpha = float(fixed_alpha)
        selection_score = float("nan")

    model = make(alpha)
    scores = cross_val_predict(
        model,
        x,
        y,
        groups=groups,
        cv=cv,
        method="decision_function",
        n_jobs=1,
    )
    # RidgeClassifier returns a one-dimensional decision function for binary
    # targets. Convert it to two columns so argmax and saved score files have
    # the same semantics for trial-ID and multi-class tasks.
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    predictions = np.asarray(scores).argmax(axis=-1)
    return scores, predictions, alpha, selection_score, n_splits


def classification_metrics(y, pred, num_classes):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    confusion = np.bincount(
        y.astype(np.int64) * int(num_classes) + pred.astype(np.int64),
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "n": int(len(y)),
        "num_classes": int(num_classes),
        "confusion": confusion.tolist(),
    }


def group_cv_coverage(y, groups, max_splits=6):
    """Check whether every grouped fold contains all target classes."""
    import numpy as np
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {
            "valid": False,
            "n_splits": int(len(unique_groups)),
            "folds": [],
            "reason": "fewer_than_two_groups",
        }
    n_splits = min(int(max_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    all_classes = set(np.unique(y).tolist())
    folds = []
    valid = True
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups)
    ):
        train_classes = set(np.unique(y[train_idx]).tolist())
        test_classes = set(np.unique(y[test_idx]).tolist())
        fold_valid = train_classes == all_classes and test_classes == all_classes
        valid = valid and fold_valid
        folds.append(
            {
                "fold": int(fold),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_classes": sorted(int(v) for v in train_classes),
                "test_classes": sorted(int(v) for v in test_classes),
                "valid": bool(fold_valid),
            }
        )
    return {
        "valid": bool(valid),
        "n_splits": int(n_splits),
        "folds": folds,
        "reason": "ok" if valid else "at_least_one_fold_missing_a_class",
    }


def make_result_row(task, condition, alpha, selection_score, n_splits, chance, result,
                    status="ok", grouping=None, note=""):
    row = {
        "task": task,
        "condition": condition,
        "alpha": alpha,
        "selection_cv_accuracy": selection_score,
        "n_splits": n_splits,
        "chance": chance,
        "status": status,
        "grouping": grouping or "",
        "note": note,
    }
    if result is None:
        row.update(
            {
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "n": None,
                "num_classes": None,
            }
        )
    else:
        row.update({k: v for k, v in result.items() if k != "confusion"})
        row["accuracy_delta_chance"] = float(result["accuracy"] - chance)
    return row


def permute_labels_within_groups(y, groups, rng):
    import numpy as np

    permuted = np.asarray(y).copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def run_trial_permutation_null(
    x,
    y,
    pair_groups,
    cv_groups,
    fixed_alpha,
    repeats,
    seed,
):
    import numpy as np

    if repeats < 1:
        return {
            "status": "not_run",
            "reason": "permutation_repeats_less_than_one",
            "repeats": 0,
        }, []
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(repeats)):
        permuted = permute_labels_within_groups(y, pair_groups, rng)
        _, pred, _, _, _ = oof_ridge(
            x,
            permuted,
            cv_groups,
            [fixed_alpha],
            fixed_alpha=fixed_alpha,
        )
        values.append(float((pred == permuted).mean()))
    values = np.asarray(values, dtype=np.float64)
    return (
        {
            "status": "ok",
            "seed": int(seed),
            "repeats": int(repeats),
            "null_mean": float(values.mean()),
            "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "null_min": float(values.min()),
            "null_max": float(values.max()),
            "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
        },
        values.tolist(),
    )


def run_semantic_window_permutation_null(
    window_features,
    window_observed,
    window_alphas,
    y,
    groups,
    repeats,
    seed,
):
    """Permutation null for every temporal window plus a max-window statistic.

    Class labels are permuted within each object column. This preserves the
    class-count distribution and the object-grouped CV structure while
    destroying the EEG/class correspondence. The max statistic controls the
    multiple comparisons introduced by scanning overlapping windows.
    """
    import numpy as np

    if repeats < 1:
        return {
            "status": "not_run",
            "reason": "window_permutation_repeats_less_than_one",
            "repeats": 0,
        }, []
    if not window_features:
        return {
            "status": "not_run",
            "reason": "no_valid_window_features",
            "repeats": 0,
        }, []

    rng = np.random.default_rng(int(seed))
    conditions = list(window_features.keys())
    null_by_condition = {condition: [] for condition in conditions}
    null_rows = []
    for repeat in range(int(repeats)):
        permuted = permute_labels_within_groups(y, groups, rng)
        values = {}
        for condition in conditions:
            alpha = float(window_alphas[condition])
            _, pred, _, _, _ = oof_ridge(
                window_features[condition],
                permuted,
                groups,
                [alpha],
                fixed_alpha=alpha,
            )
            value = float((pred == permuted).mean())
            values[condition] = value
            null_by_condition[condition].append(value)
        max_condition = max(conditions, key=lambda condition: values[condition])
        null_rows.append(
            {
                "repeat": int(repeat),
                **{condition: values[condition] for condition in conditions},
                "max_accuracy": float(values[max_condition]),
                "max_condition": max_condition,
            }
        )

    summary = {
        "status": "ok",
        "seed": int(seed),
        "repeats": int(repeats),
        "grouping": "object column; class labels permuted within object",
        "windows": {},
    }
    for condition in conditions:
        values = np.asarray(null_by_condition[condition], dtype=np.float64)
        observed = float(window_observed[condition])
        summary["windows"][condition] = {
            "observed_accuracy": observed,
            "null_mean": float(values.mean()),
            "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "null_q025": float(np.quantile(values, 0.025)),
            "null_q975": float(np.quantile(values, 0.975)),
            "empirical_p_value": float(
                (1 + int((values >= observed).sum())) / (1 + len(values))
            ),
        }

    observed_condition = max(conditions, key=lambda condition: window_observed[condition])
    observed_max = float(window_observed[observed_condition])
    null_max = np.asarray([row["max_accuracy"] for row in null_rows], dtype=np.float64)
    summary["max_window_statistic"] = {
        "observed_max_accuracy": observed_max,
        "observed_max_condition": observed_condition,
        "null_mean": float(null_max.mean()),
        "null_std": float(null_max.std(ddof=1)) if len(null_max) > 1 else 0.0,
        "null_q025": float(np.quantile(null_max, 0.025)),
        "null_q975": float(np.quantile(null_max, 0.975)),
        "empirical_p_value": float(
            (1 + int((null_max >= observed_max).sum())) / (1 + len(null_max))
        ),
    }
    return summary, null_rows


def summarize_dataset_attributes(base):
    import numpy as np

    keywords = ("time", "timestamp", "onset", "event", "sample", "rate", "freq")
    result = {}
    for name in sorted(vars(base)):
        if not any(token in name.lower() for token in keywords):
            continue
        try:
            value = getattr(base, name)
            item = {"type": type(value).__name__}
            if isinstance(value, (str, int, float, bool)) or value is None:
                item["value"] = value
            else:
                arr = np.asarray(value)
                item["shape"] = list(arr.shape)
                item["dtype"] = str(arr.dtype)
                if arr.size and np.issubdtype(arr.dtype, np.number):
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        item["min"] = float(finite.min())
                        item["max"] = float(finite.max())
                item["repr_prefix"] = repr(value)[:300]
            result[name] = item
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def find_column(fieldnames, aliases, required=True):
    normalized = {str(name).strip().lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    if required:
        raise ValueError(f"Missing one of columns {aliases}; available={list(fieldnames)}")
    return None


def load_acquisition_groups(path, records, block_size):
    """Load chronological blocks for a trial-ID holdout diagnostic.

    The CSV must contain the EEG key columns and either an explicit block
    column (``time_block``/``block``) or an acquisition order/timestamp column.
    When only order is supplied, consecutive rows after sorting by that order
    are grouped into blocks of ``block_size`` records.
    """
    import numpy as np

    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        c_subject = find_column(fields, ["subject", "subject_index", "s"], required=False)
        c_class = find_column(fields, ["class", "cls", "class_index", "c"])
        c_object = find_column(
            fields,
            ["object_suffix", "object_col", "object", "obj", "object_index", "o"],
        )
        c_trial = find_column(fields, ["trial", "trial_index", "r"])
        c_block = find_column(
            fields,
            ["time_block", "acquisition_block", "session_block", "block"],
            required=False,
        )
        c_order = find_column(
            fields,
            [
                "acquisition_index",
                "acq_index",
                "record_order",
                "sample_order",
                "timestamp_sec",
                "timestamp",
                "time_sec",
                "time",
                "time_index",
            ],
            required=False,
        )
        rows = list(reader)

    if c_block is None and c_order is None:
        return {
            "status": "not_usable",
            "path": str(path),
            "reason": "CSV needs an explicit block or acquisition order/timestamp column",
        }, None
    if block_size < 1:
        raise ValueError("acquisition_block_size must be positive")

    suffix_to_column = {
        str(record["object_suffix"]): int(record["object_col"])
        for record in records
    }
    by_key = {}
    invalid_rows = []
    duplicate_keys = []
    for row_number, row in enumerate(rows, start=2):
        try:
            subject = int(row[c_subject]) if c_subject and row[c_subject] != "" else 0
            cls = int(row[c_class])
            object_value = str(row[c_object]).strip()
            object_col = suffix_to_column.get(object_value)
            if object_col is None:
                object_col = int(object_value)
            trial = int(row[c_trial])
            key = (subject, cls, object_col, trial)
            if key in by_key:
                duplicate_keys.append(key)
            by_key[key] = row
        except (TypeError, ValueError) as exc:
            invalid_rows.append({"row": row_number, "error": str(exc), "values": row})

    expected = {
        (record["subject"], record["class"], record["object_col"], record["trial"])
        for record in records
    }
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if invalid_rows or duplicate_keys or missing or extra:
        return {
            "status": "not_usable",
            "path": str(path),
            "rows": len(rows),
            "expected_rows": len(expected),
            "invalid_rows": invalid_rows,
            "duplicate_keys": [list(key) for key in duplicate_keys],
            "missing_keys": [list(key) for key in missing],
            "extra_keys": [list(key) for key in extra],
            "reason": "metadata keys do not exactly cover the selected EEG records",
        }, None

    keys = [
        (record["subject"], record["class"], record["object_col"], record["trial"])
        for record in records
    ]
    if c_block is not None:
        raw_blocks = [str(by_key[key][c_block]).strip() for key in keys]
        labels = {value: i for i, value in enumerate(sorted(set(raw_blocks), key=str))}
        groups = np.asarray([labels[value] for value in raw_blocks], dtype=np.int64)
        source = "explicit_block"
    else:
        order = np.asarray([float(by_key[key][c_order]) for key in keys], dtype=np.float64)
        if not np.isfinite(order).all():
            return {
                "status": "not_usable",
                "path": str(path),
                "reason": "acquisition order contains non-finite values",
            }, None
        sorted_indices = np.argsort(order, kind="stable")
        groups = np.empty(len(order), dtype=np.int64)
        groups[sorted_indices] = np.arange(len(order), dtype=np.int64) // int(block_size)
        source = "sorted_order_chunks"

    report = {
        "status": "ok",
        "path": str(path),
        "source": source,
        "order_column": c_order,
        "block_column": c_block,
        "block_size_records": int(block_size),
        "num_blocks": int(len(np.unique(groups))),
        "block_counts": {
            str(int(block)): int((groups == block).sum())
            for block in np.unique(groups)
        },
    }
    return report, groups


def audit_event_csv(path, records, sample_rate, window_start, window_end):
    import numpy as np

    path = Path(path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        c_subject = find_column(fields, ["subject", "subject_index", "s"], required=False)
        c_class = find_column(fields, ["class", "cls", "class_index", "c"])
        c_object = find_column(
            fields,
            ["object_suffix", "object_col", "object", "obj", "object_index", "o"],
        )
        c_trial = find_column(fields, ["trial", "trial_index", "r"])
        c_onset = find_column(fields, ["onset_sec", "stimulus_onset_sec", "onset"], required=False)
        c_sample = find_column(fields, ["onset_sample", "onset_index", "sample_index"], required=False)
        rows = list(reader)

    suffix_to_column = {}
    for record in records:
        suffix_to_column.setdefault(str(record["object_suffix"]), int(record["object_col"]))

    by_key = {}
    duplicate_keys = []
    invalid_rows = []
    for row_number, row in enumerate(rows, start=2):
        try:
            subject = int(row[c_subject]) if c_subject and row[c_subject] != "" else 0
            cls = int(row[c_class])
            object_value = str(row[c_object]).strip()
            object_col = suffix_to_column.get(object_value)
            if object_col is None:
                object_col = int(object_value)
            trial = int(row[c_trial])
            key = (subject, cls, object_col, trial)
        except (TypeError, ValueError) as exc:
            invalid_rows.append({"row": row_number, "error": str(exc), "values": row})
            continue
        if key in by_key:
            duplicate_keys.append(key)
        by_key[key] = row

    expected = {
        (r["subject"], r["class"], r["object_col"], r["trial"])
        for r in records
    }
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    onset_values = []
    sample_errors = []
    bounds_errors = []
    for key in sorted(expected & set(by_key)):
        row = by_key[key]
        if c_onset and row[c_onset] != "":
            onset = float(row[c_onset])
            onset_values.append(onset)
            if window_start is not None and onset < window_start - 1e-8:
                bounds_errors.append({"key": key, "onset_sec": onset, "reason": "before_window"})
            if window_end is not None and onset > window_end + 1e-8:
                bounds_errors.append({"key": key, "onset_sec": onset, "reason": "after_window"})
            if sample_rate is not None and c_sample and row[c_sample] != "":
                expected_sample = round((onset - float(window_start or 0.0)) * sample_rate)
                actual_sample = int(row[c_sample])
                if expected_sample != actual_sample:
                    sample_errors.append(
                        {"key": key, "expected_sample": expected_sample, "actual_sample": actual_sample}
                    )

    passed = (
        not invalid_rows
        and not duplicate_keys
        and not missing
        and not sample_errors
        and not bounds_errors
    )
    return {
        "path": str(path),
        "passed": passed,
        "rows": len(rows),
        "expected_rows": len(expected),
        "missing_keys": [list(k) for k in missing],
        "extra_keys": [list(k) for k in extra],
        "duplicate_keys": [list(k) for k in duplicate_keys],
        "invalid_rows": invalid_rows,
        "onset_column": c_onset,
        "sample_column": c_sample,
        "onset_min_sec": float(np.min(onset_values)) if onset_values else None,
        "onset_max_sec": float(np.max(onset_values)) if onset_values else None,
        "sample_errors": sample_errors,
        "bounds_errors": bounds_errors,
        "sample_rate_hz": sample_rate,
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "limitation": (
            "This verifies the supplied event table and declared window. It cannot "
            "recover an acquisition trigger that is absent from both inputs."
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument("--rendered_view_path", default="/data/jionkim/neuro_3D/render_grid_v4")
    p.add_argument("--sub_id", default="sub01")
    p.add_argument(
        "--object_suffixes",
        nargs="+",
        default=[f"{i:02d}" for i in range(7)],
        choices=[f"{i:02d}" for i in range(8)],
    )
    p.add_argument("--pool_width", type=int, default=10)
    p.add_argument("--ridge_alphas", nargs="+", type=float, default=[1, 10, 100, 1000])
    p.add_argument("--shift_samples", nargs="+", type=int, default=[-100, -50, 0, 50, 100])
    p.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Temporal window length in samples for semantic window probes",
    )
    p.add_argument(
        "--window_step",
        type=int,
        default=50,
        help="Temporal window stride in samples for semantic window probes",
    )
    p.add_argument(
        "--permutation_repeats",
        type=int,
        default=100,
        help="Within-pair trial-label permutations; use 0 to disable",
    )
    p.add_argument(
        "--window_permutation_repeats",
        type=int,
        default=None,
        help=(
            "Semantic-window permutations; default reuses --permutation_repeats. "
            "The max-window p-value controls the window scan."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--acquisition_csv",
        default=None,
        help=(
            "Optional keyed CSV with time_block or acquisition_index/timestamp "
            "for chronological block holdout"
        ),
    )
    p.add_argument(
        "--acquisition_block_size",
        type=int,
        default=144,
        help="Number of chronologically sorted records per derived time block",
    )
    p.add_argument("--event_csv", default=None)
    p.add_argument("--sample_rate", type=float, default=None)
    p.add_argument("--window_start_sec", type=float, default=None)
    p.add_argument("--window_end_sec", type=float, default=None)
    p.add_argument("--device", default="cpu", help="Only used for metadata; Ridge runs on CPU")
    return p.parse_args()


def main(args):
    import numpy as np
    from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG

    if args.pool_width < 1 or 600 % args.pool_width:
        raise ValueError("pool_width must be a positive divisor of 600")
    if args.window_size < 1 or args.window_size > 600:
        raise ValueError("window_size must be in [1, 600]")
    if args.window_step < 1:
        raise ValueError("window_step must be positive")
    if args.window_size % args.pool_width:
        raise ValueError("window_size must be divisible by pool_width")
    if any(alpha <= 0 for alpha in args.ridge_alphas):
        raise ValueError("All ridge alphas must be positive")
    if args.permutation_repeats < 0:
        raise ValueError("permutation_repeats must be nonnegative")
    if args.window_permutation_repeats is not None and args.window_permutation_repeats < 0:
        raise ValueError("window_permutation_repeats must be nonnegative")
    if args.acquisition_block_size < 1:
        raise ValueError("acquisition_block_size must be positive")
    if args.sample_rate is not None and args.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh diagnostic out_dir")
    out.mkdir(parents=True, exist_ok=True)

    base = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=False,
    )
    raw, records, columns, suffix_to_label = make_records(base, args.object_suffixes)
    attrs = summarize_dataset_attributes(base)
    write_json(out / "dataset_timing_attributes.json", attrs)

    write_json(
        out / "data_manifest.json",
        {
            "raw_shape": list(raw.shape),
            "object_columns": columns,
            "object_suffix_to_label": suffix_to_label,
            "num_records": len(records),
            "num_classes": int(base.cls_num),
            "num_objects": len(columns),
            "num_trials": int(base.trails_num),
            "records_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "object_07_sampled": "07" in suffix_to_label,
        },
    )

    x = pooled_features(raw, args.pool_width)
    y_class = np.asarray([r["class"] for r in records], dtype=np.int64)
    y_trial = np.asarray([r["trial"] for r in records], dtype=np.int64)
    y_object = np.asarray([r["object_label"] for r in records], dtype=np.int64)
    groups_class = grouped_ids(records, ["subject", "class"])
    groups_trial_strict = grouped_ids(records, ["subject", "class", "object_col"])
    groups_object = grouped_ids(records, ["object_col"])

    all_results = []
    all_confusions = {}

    def run_task(
        task,
        condition,
        features,
        labels,
        groups,
        chance,
        grouping,
        fixed_alpha=None,
        selection_score_override=None,
        save_name=None,
    ):
        coverage = group_cv_coverage(labels, groups)
        if not coverage["valid"]:
            row = make_result_row(
                task,
                condition,
                fixed_alpha,
                selection_score_override,
                coverage.get("n_splits"),
                chance,
                None,
                status="skipped",
                grouping=grouping,
                note=coverage["reason"],
            )
            all_results.append(row)
            return {
                "status": "skipped",
                "row": row,
                "coverage": coverage,
                "alpha": fixed_alpha,
                "selection_score": selection_score_override,
            }

        scores, pred, alpha, selection_score, n_splits = oof_ridge(
            features,
            labels,
            groups,
            args.ridge_alphas,
            fixed_alpha=fixed_alpha,
        )
        if selection_score_override is not None:
            selection_score = selection_score_override
        result = classification_metrics(labels, pred, int(labels.max()) + 1)
        row = make_result_row(
            task,
            condition,
            alpha,
            selection_score,
            n_splits,
            chance,
            result,
            grouping=grouping,
        )
        all_results.append(row)
        all_confusions[f"{task}_{condition}"] = result["confusion"]
        if save_name:
            np.savez_compressed(
                out / save_name,
                scores=scores,
                labels=labels,
                groups=groups,
            )
        return {
            "status": "ok",
            "row": row,
            "scores": scores,
            "pred": pred,
            "result": result,
            "coverage": coverage,
            "alpha": alpha,
            "selection_score": selection_score,
            "n_splits": n_splits,
        }

    # Nuisance-ID classifiers. The first two preserve the original class-grouped
    # protocol; strict trial-ID additionally holds out each class/object pair.
    trial_chance = 1.0 / len(np.unique(y_trial))
    object_chance = 1.0 / len(np.unique(y_object))
    trial_class = run_task(
        "trial_id",
        "class_grouped",
        x,
        y_trial,
        groups_class,
        trial_chance,
        "subject+class",
        save_name="scores_trial_id_class_grouped.npz",
    )
    run_task(
        "object_id",
        "class_grouped",
        x,
        y_object,
        groups_class,
        object_chance,
        "subject+class",
        save_name="scores_object_id.npz",
    )
    trial_strict = run_task(
        "trial_id",
        "strict_pair_grouped",
        x,
        y_trial,
        groups_trial_strict,
        trial_chance,
        "subject+class+object",
        save_name="scores_trial_id_strict.npz",
    )

    # Trial-label permutation null. Labels are shuffled only within the same
    # subject/class/object pair, preserving the 0/1 balance of each pair.
    permutation_summary, permutation_values = run_trial_permutation_null(
        x,
        y_trial,
        groups_trial_strict,
        groups_trial_strict,
        trial_strict.get("alpha") if trial_strict["status"] == "ok" else 1.0,
        args.permutation_repeats if trial_strict["status"] == "ok" else 0,
        args.seed,
    )
    if trial_strict["status"] == "ok":
        observed = float(trial_strict["result"]["accuracy"])
        permutation_summary["observed_accuracy"] = observed
        if permutation_values:
            greater_equal = sum(value >= observed for value in permutation_values)
            permutation_summary["empirical_p_value"] = float(
                (1 + greater_equal) / (1 + len(permutation_values))
            )
            permutation_summary["observed_minus_null_mean"] = float(
                observed - permutation_summary["null_mean"]
            )
        permutation_summary["grouping"] = "subject+class+object"
        permutation_summary["label_shuffle"] = "within subject/class/object pair"
    write_json(out / "trial_permutation_summary.json", permutation_summary)
    write_csv(
        out / "trial_permutation_null.csv",
        [
            {"repeat": int(i), "accuracy": float(value)}
            for i, value in enumerate(permutation_values)
        ],
    )

    # Chronological block holdout. This cannot be inferred safely from the
    # tensor index, so the script requires keyed acquisition metadata.
    if args.acquisition_csv:
        time_report, time_groups = load_acquisition_groups(
            args.acquisition_csv,
            records,
            args.acquisition_block_size,
        )
    else:
        time_report, time_groups = (
            {
                "status": "not_requested",
                "reason": "supply --acquisition_csv with time_block or acquisition_index/timestamp",
            },
            None,
        )
    if time_groups is not None and time_report.get("status") == "ok":
        time_coverage = group_cv_coverage(y_trial, time_groups)
        time_report["cv_coverage"] = time_coverage
        if time_coverage["valid"]:
            time_fixed_alpha = (
                trial_strict["alpha"] if trial_strict["status"] == "ok" else None
            )
            time_result = run_task(
                "trial_id_time_block",
                "leave_time_block_out",
                x,
                y_trial,
                time_groups,
                trial_chance,
                "acquisition time block",
                fixed_alpha=time_fixed_alpha,
                selection_score_override=(
                    trial_strict["selection_score"]
                    if time_fixed_alpha is not None
                    else None
                ),
                save_name="scores_trial_id_time_block.npz",
            )
            time_report["evaluation_status"] = time_result["status"]
        else:
            row = make_result_row(
                "trial_id_time_block",
                "leave_time_block_out",
                None,
                None,
                time_coverage.get("n_splits"),
                trial_chance,
                None,
                status="skipped",
                grouping="acquisition time block",
                note=time_coverage["reason"],
            )
            all_results.append(row)
            time_report["evaluation_status"] = "skipped"
    else:
        row = make_result_row(
            "trial_id_time_block",
            "leave_time_block_out",
            None,
            None,
            None,
            trial_chance,
            None,
            status=time_report.get("status", "not_usable"),
            grouping="acquisition time block",
            note=time_report.get("reason", "metadata unavailable"),
        )
        all_results.append(row)
        time_report["evaluation_status"] = "not_run"
    write_json(out / "time_block_report.json", time_report)

    # Class decoding under timing perturbations. Alpha is selected only on the
    # original condition and reused for every temporal control.
    alignment_chance = 1.0 / int(base.cls_num)
    alignment_original = run_task(
        "semantic_alignment_class",
        "original",
        x,
        y_class,
        groups_object,
        alignment_chance,
        "object column",
        save_name="scores_original.npz",
    )
    alignment_alpha = alignment_original.get("alpha")
    alignment_selection = alignment_original.get("selection_score")
    temporal_rows = [alignment_original["row"]]
    temporal_conditions = ["original", "reverse"]
    for shift in args.shift_samples:
        if shift == 0:
            continue
        temporal_conditions.extend(
            [
                f"zero_shift_{shift}",
                f"circular_shift_{shift}",
                f"valid_crop_{shift}",
            ]
        )

    for condition in temporal_conditions[1:]:
        if condition == "reverse":
            transformed = raw[..., ::-1].copy()
        elif condition.startswith("valid_crop_"):
            shift = int(condition.split("_", 2)[2])
            transformed = valid_crop_shift(raw, shift, args.pool_width)
        else:
            transformed = transform_raw(raw, condition)
        transformed_x = pooled_features(transformed, args.pool_width)
        temporal_result = run_task(
            "semantic_alignment_class",
            condition,
            transformed_x,
            y_class,
            groups_object,
            alignment_chance,
            "object column",
            fixed_alpha=alignment_alpha,
            selection_score_override=alignment_selection,
            save_name=f"scores_{condition}.npz",
        )
        temporal_rows.append(temporal_result["row"])

    # Local temporal probes: every window is evaluated with the same object
    # holdout and the alpha chosen on the full original condition.
    window_rows = []
    window_features = {}
    window_alphas = {}
    window_observed = {}
    window_starts = range(0, 600 - args.window_size + 1, args.window_step)
    for start in window_starts:
        end = start + args.window_size
        window_raw = raw[..., start:end]
        window_x = pooled_features(window_raw, args.pool_width)
        window_condition = f"window_{start}_{end}"
        window_result = run_task(
            "semantic_window_class",
            window_condition,
            window_x,
            y_class,
            groups_object,
            alignment_chance,
            "object column",
            fixed_alpha=alignment_alpha,
            selection_score_override=alignment_selection,
            save_name=f"scores_window_{start:04d}_{end:04d}.npz",
        )
        window_result["row"]["window_start_sample"] = int(start)
        window_result["row"]["window_end_sample"] = int(end)
        window_rows.append(window_result["row"])
        if window_result["status"] == "ok":
            window_features[window_condition] = window_x
            window_alphas[window_condition] = float(window_result["alpha"])
            window_observed[window_condition] = float(
                window_result["result"]["accuracy"]
            )

    window_permutation_repeats = (
        args.permutation_repeats
        if args.window_permutation_repeats is None
        else args.window_permutation_repeats
    )
    window_permutation_summary, window_permutation_rows = (
        run_semantic_window_permutation_null(
            window_features,
            window_observed,
            window_alphas,
            y_class,
            groups_object,
            window_permutation_repeats,
            args.seed + 1,
        )
    )
    write_json(
        out / "semantic_window_permutation_summary.json",
        window_permutation_summary,
    )
    write_csv(
        out / "semantic_window_permutation_null.csv",
        window_permutation_rows,
    )

    write_csv(out / "auxiliary_results.csv", all_results)
    write_json(out / "confusions.json", all_confusions)

    timing = {
        "dataset_attributes": attrs,
        "event_csv": None,
        "status": "not_verifiable_from_eeg_array_only",
        "interpretation": (
            "Compare original/reverse/circular/valid-crop conditions and the local "
            "window curve. A reproducible peak near the original timing supports "
            "temporal alignment; it does not establish absolute onset without trigger metadata."
        ),
        "conditions": temporal_rows,
        "windows": window_rows,
        "window_permutation": window_permutation_summary,
        "time_block": time_report,
        "permutation": permutation_summary,
    }
    if args.event_csv:
        timing["event_csv"] = audit_event_csv(
            args.event_csv,
            records,
            args.sample_rate,
            args.window_start_sec,
            args.window_end_sec,
        )
        timing["status"] = "passed" if timing["event_csv"]["passed"] else "failed"
    write_json(out / "alignment_report.json", timing)
    write_json(
        out / "protocol.json",
        {
            "tasks": [
                "trial_id",
                "object_id",
                "trial_id_strict",
                "trial_id_permutation_null",
                "trial_id_time_block",
                "semantic_alignment_class",
                "semantic_window_class",
                "semantic_window_permutation_null",
            ],
            "object_suffixes": args.object_suffixes,
            "grouping": {
                "trial_id": "subject+class",
                "object_id": "subject+class",
                "trial_id_strict": "subject+class+object",
                "trial_id_permutation_null": "subject+class+object; labels permuted within pair",
                "trial_id_time_block": "acquisition time block",
                "semantic_alignment_class": "object column",
                "semantic_window_class": "object column",
                "semantic_window_permutation_null": "object column; labels permuted within object",
            },
            "feature": f"per-channel z-normalization -> mean/std temporal pooling width={args.pool_width}",
            "alignment_alpha_selected_on": "original semantic condition only",
            "alignment_controls": temporal_conditions,
            "window_size_samples": int(args.window_size),
            "window_step_samples": int(args.window_step),
            "shift_samples": [int(v) for v in args.shift_samples],
            "permutation_repeats": int(args.permutation_repeats),
            "window_permutation_repeats": int(window_permutation_repeats),
            "acquisition_csv": args.acquisition_csv,
            "chance": {
                "trial_id": float(1.0 / len(np.unique(y_trial))),
                "object_id": float(1.0 / len(np.unique(y_object))),
                "semantic_class": float(1.0 / int(base.cls_num)),
            },
            "warning": (
                "Best checkpoint selection is not involved. These are fixed-fold OOF "
                "Ridge diagnostics; no causal claim follows from a single shift curve."
            ),
        },
    )
    print(f"[done] {out / 'auxiliary_results.csv'}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    args.repo_dir = str(Path(args.repo_dir).expanduser().resolve())
    sys.path.insert(0, args.repo_dir)
    if not (Path(args.repo_dir) / "src").is_dir():
        raise FileNotFoundError("--repo_dir must contain src/")
    main(args)
