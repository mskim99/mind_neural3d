#!/usr/bin/env python3
"""C1: semantic/visual fusion, C2: frozen category probes for MinD-3D++ E3.

Uses the E3 model and the exact train/val UID split saved during training.
C1 val-GT visual candidates are for diagnostic retrieval ONLY (not inference).
A second C1 metric uses the train candidate bank, which can be built at inference.
The official test list is not used for selection or evaluation.

Dependencies: torch, numpy, Pillow, open_clip_torch, tqdm, original E3 script.
C2 requires no OpenCLIP weights or GT visual targets beyond the checkpoint.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import train_fmri_clip_align_mind3dpp_dual as base


def _json_safe(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sorted_train_val_records(data_root, split_path):
    train_list = base.read_object_list(data_root / "train_list.txt")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    train_set, val_set = set(split["train_uids"]), set(split["val_uids"])
    test_set = set(split.get("test_uids", []))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("UID leakage detected between train/val/test splits")
    train_records = [r for r in train_list if r.uid in train_set]
    val_records = [r for r in train_list if r.uid in val_set]
    if {r.uid for r in train_records} != train_set or {r.uid for r in val_records} != val_set:
        raise RuntimeError("Checkpoint split.json does not match train_list.txt")
    if len({r.uid for r in train_list}) != len(train_list):
        raise RuntimeError("train_list.txt contains duplicated UIDs; check split integrity")
    return train_records, val_records, split


def setup(args):
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = base._load_cache(checkpoint_path)
    if "dual_model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Requires E3 checkpoint containing dual_model and config")
    cfg = checkpoint["config"]
    root = Path(args.data_path or cfg.get("resolved_data_path", cfg.get("data_path", ""))).expanduser().resolve()
    split_path = Path(args.split_json).expanduser().resolve() if args.split_json else checkpoint_path.parent / "split.json"
    train_records, val_records, split = sorted_train_val_records(root, split_path)
    if "split_seed" in cfg and int(cfg["split_seed"]) != int(split["split_seed"]):
        raise RuntimeError("split_seed in checkpoint and split.json differ")
    subjects = (args.subjects or cfg.get("subjects", "sub-0001/npy_data")).split(",")
    subjects = [s.strip() for s in subjects if s.strip()]
    categories = sorted({r.category for r in train_records + val_records})
    category_to_idx = {c: i for i,c in enumerate(categories)}
    sd = checkpoint["dual_model"]
    geometry_dim = int(sd["geometry_head.3.weight"].shape[0]) if "geometry_head.3.weight" in sd else 0
    clip_dim = int(sd["net.4.weight"].shape[0])
    # C3 checkpoints also include a classifier_head. Load them without stripping it.
    has_cls = any(k.startswith("classifier_head.") for k in sd)
    if has_cls:
        from train_fmri_clip_align_mind3dpp_cls_hard import ClassificationDualFMRIClipAligner
        train_names = cfg.get("train_category_names")
        if not train_names:
            raise ValueError("C3 checkpoint missing train_category_names")
        model = ClassificationDualFMRIClipAligner(
            clip_dim=clip_dim, num_frames=int(cfg["num_frames"]),
            frame_dim=int(cfg["frame_dim"]), hidden_dim=int(cfg["hidden_dim"]),
            temporal_layers=int(cfg["temporal_layers"]), temporal_heads=int(cfg["temporal_heads"]),
            dropout=float(cfg["dropout"]), visual_hidden_dim=int(cfg.get("visual_hidden_dim", 512)),
            geometry_dim=geometry_dim, num_train_categories=len(train_names),
            cls_head_type=cfg.get("cls_head_type", "linear"),
            cls_hidden_dim=int(cfg.get("cls_hidden_dim", 256)),
            cls_dropout=float(cfg.get("cls_dropout", 0.3)),
        )
    else:
        model = base.DualFMRIClipAligner(
            clip_dim=clip_dim, num_frames=int(cfg["num_frames"]),
            frame_dim=int(cfg["frame_dim"]), hidden_dim=int(cfg["hidden_dim"]),
            temporal_layers=int(cfg["temporal_layers"]), temporal_heads=int(cfg["temporal_heads"]),
            dropout=float(cfg["dropout"]), visual_hidden_dim=int(cfg.get("visual_hidden_dim", 512)),
            geometry_dim=geometry_dim,
        )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval().requires_grad_(False)
    # Text features are not needed to extract fMRI representations.
    blank = {r.uid: torch.zeros(clip_dim, dtype=torch.float32) for r in train_records + val_records}
    dkwargs = dict(data_root=root, subjects=subjects, text_embeddings=blank,
                   category_to_index=category_to_idx, num_frames=int(cfg["num_frames"]),
                   fmri_norm=cfg.get("fmri_norm", "none"), strict_files=True)
    train_ds = base.FMRIObjaverseDataset(records=train_records, train_mode=False, **dkwargs)
    val_ds = base.FMRIObjaverseDataset(records=val_records, train_mode=False, **dkwargs)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }
    print(f"[*] Loaded E3 checkpoint epoch={checkpoint.get('epoch','?')} "
          f"trainUID={len(train_records)} valUID={len(val_records)} "
          f"trainSamples={len(train_ds)} valSamples={len(val_ds)} "
          f"categories={len(categories)} subjects={subjects}", flush=True)
    return dict(model=model, ckpt=checkpoint, cfg=cfg, device=device, root=root,
                train=train_records, val=val_records, split=split, loaders=loaders,
                path=checkpoint_path, subjects=subjects)


@torch.no_grad()
def extract_features(ctx, which):
    model, device = ctx["model"], ctx["device"]
    pooled, semantics, visuals, uids, cats = [], [], [], [], []
    for batch in ctx["loaders"][which]:
        x = batch["fmri"].to(device, non_blocking=True).float()
        h = model.encode_pooled(x)
        pooled.append(h.cpu())
        semantics.append(F.normalize(model.net(h), dim=-1).cpu())
        visuals.append(F.normalize(model.visual_net(h), dim=-1).cpu())
        uids.extend(batch["uid"])
        cats.extend(batch["category"])
    return dict(pooled=torch.cat(pooled), semantic=torch.cat(semantics),
                visual=torch.cat(visuals), uids=uids, categories=cats)


def get_features(ctx, args, which):
    if not args.feature_cache:
        return extract_features(ctx, which)
    cache_dir = Path(args.feature_cache).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{which}.pt"
    metadata = {
        "checkpoint": str(ctx["path"]),
        "ckpt_stat_mtime_ns": ctx["path"].stat().st_mtime_ns,
        "ckpt_stat_size": ctx["path"].stat().st_size,
        "split_train": ctx["split"]["train_uids"],
        "split_val": ctx["split"]["val_uids"],
        "subjects": ctx["subjects"],
        "fmri_norm": ctx["cfg"].get("fmri_norm", "none"),
        "num_frames": int(ctx["cfg"]["num_frames"]),
        "scope": which,
    }
    if path.is_file():
        cached = base._load_cache(path)
        if cached.get("metadata") == metadata:
            print(f"[*] Reusing {which} embeddings from {path}", flush=True)
            return cached["features"]
    data = extract_features(ctx, which)
    torch.save(dict(metadata=metadata, features=data), path)
    print(f"[*] Saved {which} feature cache to {path}", flush=True)
    return data


def normalize_scores(scores, kind):
    if kind == "none":
        return scores
    if kind == "zscore":
        # Normalize each modality across the SAME candidate bank.
        return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    raise ValueError(kind)


def within_category_nway(scores, targets, categories, way):
    if len(categories) != scores.shape[1]:
        raise ValueError("Candidate category count must match score matrix")
    cache = {}
    wins = []
    for i, target in enumerate(targets.tolist()):
        cat = categories[target]
        if cat not in cache:
            cache[cat] = [j for j,c in enumerate(categories) if c == cat]
        columns = cache[cat]
        if len(columns) < way:
            continue
        local_target = columns.index(target)
        p = base.expected_nway_top1(scores[i:i+1, columns], torch.tensor([local_target]), way)
        wins.append(p)
    return (float(np.mean(wins)) if wins else float("nan"), len(wins))


def evaluate_object_fusion(scores, subset, targets, cat_candidates):
    sub_scores = scores[subset]
    sub_y = targets[subset]
    result = dict(
        n_queries=len(subset),
        val_2way=base.expected_nway_top1(sub_scores, sub_y, 2),
        val_10way=base.expected_nway_top1(sub_scores, sub_y, min(10, len(cat_candidates))),
    )
    a = base.topk_accuracy(sub_scores, sub_y, ks=(1,5))
    result["val_obj_top1"] = a[1]
    result["val_obj_top5"] = a[5]
    p2, n2 = within_category_nway(sub_scores, sub_y, cat_candidates, 2)
    p5, n5 = within_category_nway(sub_scores, sub_y, cat_candidates, 5)
    result.update(within_cat_2way=p2, within_cat_5way=p5,
                  n_within2=n2, n_within5=n5)
    return result


def train_bank_category_metrics(scores, sample_cats, candidate_cats):
    # Inference-feasible: all candidates are TRAIN objects only.
    # Top-5 is the five distinct categories of highest-ranked train objects.
    order = torch.argsort(scores, dim=1, descending=True)
    good1 = good5 = 0
    for row, gt_cat in zip(order.tolist(), sample_cats):
        top_categories = []
        for j in row:
            cat = candidate_cats[j]
            if cat not in top_categories:
                top_categories.append(cat)
            if len(top_categories) == 5:
                break
        good1 += bool(top_categories and top_categories[0] == gt_cat)
        good5 += bool(gt_cat in top_categories)
    return dict(train_bank_cat_top1=good1/len(sample_cats),
                train_bank_cat_top5=good5/len(sample_cats))


def visual_bank(ctx, args):
    cfg = ctx["cfg"]
    records = ctx["train"] + ctx["val"]
    required = {r.uid for r in records}
    cache_path = Path(args.visual_cache).expanduser().resolve() if args.visual_cache else None
    if cache_path and cache_path.is_file():
        stored = base._load_cache(cache_path)
        meta = stored.get("metadata", {})
        features = stored.get("embeddings", {})
        if meta.get("clip_model") != cfg.get("clip_model", "ViT-B-32") or meta.get("clip_pretrained") != cfg.get("clip_pretrained", "openai"):
            raise ValueError("GT image teacher cache CLIP model/pretrained differs from the E3 checkpoint")
        if int(meta.get("view_count", -1)) != int(cfg.get("visual_view_count", 6)):
            raise ValueError("GT image teacher view count mismatch")
        if not required.issubset(features):
            raise ValueError(f"GT visual cache is missing {len(required-set(features))} train/val UIDs")
        if required.intersection(ctx["split"].get("test_uids", [])):
            raise RuntimeError("Official test UID leakage")
        print(f"[*] Loaded {len(required)} frozen GT multi-view features from {cache_path}", flush=True)
        return {u: F.normalize(features[u].float(), dim=-1) for u in required}
    if not args.visual_views_root:
        raise FileNotFoundError("Provide --visual_cache (E3's gt_multiview_clip_vitb32_openai.pt) or --visual_views_root")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(cfg.get("clip_model", "ViT-B-32"), pretrained=cfg.get("clip_pretrained", "openai"))
    model.to(ctx["device"]).eval().requires_grad_(False)
    features = base.prepare_multiview_clip_targets(
        records, args.visual_views_root, model, preprocess, ctx["device"],
        view_count=int(cfg.get("visual_view_count", 6)), batch_size=args.visual_batch_size,
        cache_path=args.visual_cache or None,
        clip_name=cfg.get("clip_model", "ViT-B-32"),
        clip_weights=cfg.get("clip_pretrained", "openai"),
    )
    del model
    return features


def caption_bank(ctx):
    import open_clip
    cfg = ctx["cfg"]
    records = ctx["train"] + ctx["val"]
    caption_file = Path(cfg.get("caption_csv") or (ctx["root"] / "Object_caption_Cap3D.csv"))
    cap = base.load_caption_map(caption_file)
    texts = base.build_texts(records, cap, cfg.get("text_source", "caption"))
    model, _, _ = open_clip.create_model_and_transforms(cfg.get("clip_model", "ViT-B-32"), pretrained=cfg.get("clip_pretrained", "openai"))
    model.to(ctx["device"]).eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(cfg.get("clip_model", "ViT-B-32"))
    embeddings = base.encode_text_table(texts, model, tokenizer, ctx["device"], batch_size=256)
    del model
    return embeddings


def run_fusion(args, ctx):
    val = get_features(ctx, args, "val")
    visual = visual_bank(ctx, args)
    text = caption_bank(ctx)
    val_recs, train_recs = ctx["val"], ctx["train"]
    val_uids = [r.uid for r in val_recs]
    val_cats = [r.category for r in val_recs]
    train_uids = [r.uid for r in train_recs]
    train_cats = [r.category for r in train_recs]
    target_index = {uid: i for i,uid in enumerate(val_uids)}
    targets = torch.tensor([target_index[u] for u in val["uids"]], dtype=torch.long)
    val_cat_metrics = [val_cats[target_index[u]] for u in val["uids"]]
    zsem, zvis = F.normalize(val["semantic"], dim=-1), F.normalize(val["visual"], dim=-1)
    s_val_sem = zsem @ F.normalize(torch.stack([text[u] for u in val_uids]).float(), dim=-1).T
    s_val_vis = zvis @ F.normalize(torch.stack([visual[u] for u in val_uids]).float(), dim=-1).T
    s_train_sem = zsem @ F.normalize(torch.stack([text[u] for u in train_uids]).float(), dim=-1).T
    s_train_vis = zvis @ F.normalize(torch.stack([visual[u] for u in train_uids]).float(), dim=-1).T
    if s_val_sem.shape != s_val_vis.shape:
        raise RuntimeError("Semantic and visual candidate banks differ")
    s_val_sem, s_val_vis = [normalize_scores(s, args.score_normalize) for s in (s_val_sem, s_val_vis)]
    s_train_sem, s_train_vis = [normalize_scores(s, args.score_normalize) for s in (s_train_sem, s_train_vis)]
    # Select alpha on a disjoint subset of validation queries, report remaining.
    unique = list(dict.fromkeys(val["uids"]))
    rng = random.Random(args.calibration_seed)
    rng.shuffle(unique)
    ncal = max(1, min(len(unique)-1, round(len(unique)*args.calibration_fraction)))
    cal_uids = set(unique[:ncal])
    calibration = [i for i,u in enumerate(val["uids"]) if u in cal_uids]
    evaluation = [i for i,u in enumerate(val["uids"]) if u not in cal_uids]
    if not calibration or not evaluation:
        raise ValueError("Calibration/holdout query partition is empty")
    alphas = [float(v.strip()) for v in args.alphas.split(",") if v.strip()]
    if not alphas or any(not 0 <= a <= 1 for a in alphas):
        raise ValueError("--alphas must contain comma-separated values in [0,1]")
    rows = []
    for alpha in alphas:
        scores = alpha*s_val_sem + (1.0-alpha)*s_val_vis
        train_scores = alpha*s_train_sem + (1.0-alpha)*s_train_vis
        for name, ids in (("calibration", calibration), ("holdout", evaluation),
                          ("all_diagnostic", list(range(len(val["uids"]))))):
            metrics = evaluate_object_fusion(scores, ids, targets, val_cats)
            cats = train_bank_category_metrics(train_scores[ids], [val_cat_metrics[i] for i in ids], train_cats)
            rows.append(dict(alpha=alpha, scope=name, **metrics, **cats))
    best_alpha = max(alphas, key=lambda a: next(r["val_10way"] for r in rows if r["scope"] == "calibration" and r["alpha"] == a))
    print(f"[*] Fusion alpha selected from calibration queries only: {best_alpha}", flush=True)
    print("alpha   cal10Way   holdout10Way  holdout2Way  withinCat2Way  trainBankCat1", flush=True)
    for a in alphas:
        cal = next(r for r in rows if r["scope"] == "calibration" and r["alpha"] == a)
        hold = next(r for r in rows if r["scope"] == "holdout" and r["alpha"] == a)
        print(f"{a:5.2f}   {cal['val_10way']:8.3%}  {hold['val_10way']:11.3%}  "
              f"{hold['val_2way']:11.3%}  {hold['within_cat_2way']:13.3%} "
              f"{hold['train_bank_cat_top1']:13.3%}", flush=True)
    save_csv(Path(args.out_dir)/"fusion_sweep.csv", rows)
    # Error complementarity uses same val GT candidate bank (no inference claim).
    psem, pvis = s_val_sem.argmax(dim=1).eq(targets), s_val_vis.argmax(dim=1).eq(targets)
    summary = dict(task="fusion", checkpoint=str(ctx["path"]), checkpoint_epoch=ctx["ckpt"].get("epoch"),
                   selected_alpha=best_alpha, selection="calibration val 10-way",
                   score_normalize=args.score_normalize, n_cal_queries=len(calibration),
                   n_holdout_queries=len(evaluation), n_val_candidates=len(val_uids),
                   n_train_candidates=len(train_uids),
                   complementary=dict(semantic_only=int((psem&~pvis).sum()), visual_only=int((pvis&~psem).sum()),
                                      both=int((psem&pvis).sum()), neither=int((~psem&~pvis).sum())),
                   selected_holdout=next(r for r in rows if r["scope"]=="holdout" and r["alpha"] == best_alpha),
                   caveat="Validation GT visual candidate bank is for diagnostic retrieval only; train-bank category metrics use train UID GT features.")
    save_json(Path(args.out_dir)/"fusion_summary.json", summary)


def metrics_classification(logits, y):
    with torch.no_grad():
        yhat = logits.argmax(dim=1)
        valid = y.unique()
        macro = torch.stack([yhat[y==c].eq(c).float().mean() for c in valid]).mean()
        top = base.topk_accuracy(logits, y, (1,5))
        return dict(cat_ce=float(F.cross_entropy(logits.float(), y).item()),
                    cat_top1=top[1], cat_top5=top[5], macro_cat_top1=float(macro.item()))


def train_probe(args, ctx, features_train, features_val, head_type):
    train_names = sorted({r.category for r in ctx["train"]})
    index = {c:i for i,c in enumerate(train_names)}
    x_train = features_train["pooled"].float()
    y_train = torch.tensor([index[c] for c in features_train["categories"]], dtype=torch.long)
    covered = [i for i,c in enumerate(features_val["categories"]) if c in index]
    unseen = len(features_val["categories"]) - len(covered)
    if not covered:
        raise ValueError("No validation category belongs to the training category vocabulary")
    x_val = features_val["pooled"][covered].float()
    y_val = torch.tensor([index[features_val["categories"][i]] for i in covered], dtype=torch.long)
    dim = x_train.shape[-1]
    if head_type=="linear":
        net=nn.Linear(dim, len(train_names))
    elif head_type=="mlp":
        net=nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, args.probe_hidden_dim), nn.GELU(),
                          nn.Dropout(args.probe_dropout), nn.Linear(args.probe_hidden_dim, len(train_names)))
    else:
        raise ValueError(head_type)
    device = ctx["device"]
    net=net.to(device)
    optimizer=torch.optim.AdamW(net.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    loader=DataLoader(TensorDataset(x_train, y_train),batch_size=args.probe_batch_size,shuffle=True,drop_last=False)
    xv=x_val.to(device)
    best=float("inf"); best_epoch=0; stale=0; best_metrics=None; log=[]
    for epoch in range(1,args.probe_epochs+1):
        net.train()
        total_loss=total_n=0
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=net(x)
            loss=F.cross_entropy(logits,y,label_smoothing=args.probe_label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*len(y); total_n+=len(y)
        net.eval()
        with torch.no_grad():
            met=metrics_classification(net(xv).cpu(), y_val)
        row=dict(epoch=epoch, train_loss=total_loss/max(total_n,1),**met)
        log.append(row)
        if met["cat_ce"] < best-1e-5:
            best=met["cat_ce"]; stale=0; best_epoch=epoch; best_metrics=met
            torch.save(dict(classifier=net.state_dict(), head_type=head_type,
                            train_category_names=train_names, epoch=epoch,
                            base_checkpoint=str(ctx["path"]), val_metrics=met,
                            frame_dim=dim, args=vars(args)),
                       Path(args.out_dir)/f"probe_{head_type}_best.pt")
        else:
            stale+=1
        if args.probe_patience>0 and stale>=args.probe_patience:
            break
    save_csv(Path(args.out_dir)/f"probe_{head_type}_training_log.csv",log)
    print(f"[*] Probe {head_type:6} epoch={best_epoch:3d} CE={best_metrics['cat_ce']:.4f} "
          f"Cat@1={best_metrics['cat_top1']:.3%} Cat@5={best_metrics['cat_top5']:.3%} "
          f"MacroCat@1={best_metrics['macro_cat_top1']:.3%} "
          f"known_val={len(covered)}, unseen_val={unseen}", flush=True)
    return dict(head=head_type, best_epoch=best_epoch, **best_metrics,
                train_samples=len(x_train), valid_samples=len(covered), unseen_val_samples=unseen)


def run_probe(args, ctx):
    train = get_features(ctx, args, "train")
    val = get_features(ctx, args, "val")
    base.seed_everything(args.probe_seed)
    heads = [h.strip() for h in args.probe_heads.split(",") if h.strip()]
    if len(heads)!=len(set(heads)) or any(h not in ("linear","mlp") for h in heads):
        raise ValueError("--probe_heads accepts linear,mlp only")
    rows = [train_probe(args,ctx,train,val,h) for h in heads]
    save_csv(Path(args.out_dir)/"probe_comparison.csv",rows)
    save_json(Path(args.out_dir)/"probe_summary.json",
              dict(checkpoint=str(ctx["path"]), checkpoint_epoch=ctx["ckpt"].get("epoch"),
                   results=rows, caveat="Category probe is trained only on train UIDs; checkpoint selection uses same validation as original E3 and needs independent final test."))


def parse_args():
    p=argparse.ArgumentParser(description="C1/C2 E3 frozen-representation classification experiments")
    p.add_argument("--task", required=True, choices=["fusion","probe"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_path", default="", help="fMRI Objaverse root; default from checkpoint")
    p.add_argument("--split_json", default="", help="default <checkpoint dir>/split.json")
    p.add_argument("--subjects", default="", help="default checkpoint's subjects")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int,default=16)
    p.add_argument("--num_workers", type=int,default=4)
    p.add_argument("--feature_cache", default="", help="folder for checkpoint-matched fMRI features (train/val)")
    p.add_argument("--visual_cache", default="",help="E3 per UID frozen CLIP image cache .pt")
    p.add_argument("--visual_views_root",default="")
    p.add_argument("--visual_batch_size",type=int,default=48)
    p.add_argument("--score_normalize",choices=["none","zscore"],default="zscore")
    p.add_argument("--alphas",default="0,0.25,0.5,0.75,1")
    p.add_argument("--calibration_fraction",type=float,default=0.3)
    p.add_argument("--calibration_seed",type=int,default=7)
    p.add_argument("--probe_heads",default="linear,mlp")
    p.add_argument("--probe_epochs",type=int,default=50)
    p.add_argument("--probe_patience",type=int,default=10)
    p.add_argument("--probe_lr",type=float,default=1e-3)
    p.add_argument("--probe_weight_decay",type=float,default=1e-2)
    p.add_argument("--probe_batch_size",type=int,default=128)
    p.add_argument("--probe_hidden_dim",type=int,default=256)
    p.add_argument("--probe_dropout",type=float,default=0.3)
    p.add_argument("--probe_label_smoothing",type=float,default=0.05)
    p.add_argument("--probe_seed",type=int,default=0)
    return p.parse_args()


def main():
    args=parse_args()
    if not (0 < args.calibration_fraction < 1):
        raise ValueError("--calibration_fraction must be in (0,1)")
    if args.probe_epochs<1 or args.probe_batch_size<1:
        raise ValueError("Probe epochs and batch size must be positive")
    if not (0<=args.probe_label_smoothing<1 and 0<=args.probe_dropout<1):
        raise ValueError("Invalid probe smoothing/dropout")
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    ctx=setup(args)
    if args.task=="fusion":
        run_fusion(args,ctx)
    else:
        run_probe(args,ctx)
    print(f"[*] Completed {args.task}: {args.out_dir}",flush=True)


if __name__=="__main__":
    main()
