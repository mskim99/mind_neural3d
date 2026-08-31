#!/usr/bin/env python3
"""
EEG Semantic Distillation
=========================

Teacher
-------
True category prototype
 -> FROZEN oracle generator
 -> v_teacher

Student
-------
EEG
 -> frozen TemporalStem
 -> STRICT train-only PCA
 -> small semantic classifier
 -> soft category distribution
 -> prototype mixture
 -> SAME FROZEN generator
 -> v_student

Loss
----
L = MSE(v_student, stopgrad(v_teacher))
  + 0.1 * KL(q_text || p_eeg)

Trainable
---------
semantic_prior_classifier_norm
semantic_prior_classifier

Everything else is frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from train_oracle_semantic_generator import (
    seed_everything,
    EEGOnlyDataset,
    AllDataFeatureTwoEEG,
    build_fixed_category_text_prototypes,
    full_dataset_indices,
    build_model,
    build_oracle_condition,
    prepare_noisy_latents,
)


# =========================================================
# Frozen TemporalStem
# =========================================================

def load_only_temporal_stem(
    model,
    checkpoint_path,
):

    path = Path(
        checkpoint_path
    ).expanduser().resolve()

    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    state = (
        ckpt["model"]
        if isinstance(ckpt, dict)
        and "model" in ckpt
        else ckpt
    )

    target = (
        model.fmri_encoder
        .temporal_stem
        .state_dict()
    )

    extracted = {}

    prefix = (
        "fmri_encoder."
        "temporal_stem."
    )

    for key, value in state.items():

        k = str(key)

        if k.startswith("module."):
            k = k[len("module."):]

        if k.startswith(prefix):

            extracted[
                k[len(prefix):]
            ] = value

    if set(extracted) != set(target):

        raise RuntimeError(
            "TemporalStem mismatch\n"
            f"expected={sorted(target)}\n"
            f"got={sorted(extracted)}"
        )

    (
        model.fmri_encoder
        .temporal_stem
        .load_state_dict(
            extracted,
            strict=True,
        )
    )

    model.fmri_encoder.temporal_stem.eval()

    for p in (
        model.fmri_encoder
        .temporal_stem
        .parameters()
    ):
        p.requires_grad_(False)

    print(
        f"[TemporalStem] loaded {path}"
    )


# =========================================================
# Strict train-only PCA
# =========================================================

@torch.no_grad()
def collect_temporal_features(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    rows = []

    for batch in loader:

        eeg = batch[
            "eeg_data"
        ].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        feat = (
            model
            .extract_semantic_prior_temporal(
                eeg
            )
        )

        rows.append(
            feat.cpu()
        )

    return torch.cat(
        rows,
        dim=0,
    ).numpy().astype(
        np.float32
    )


def fit_train_only_pca(
    model,
    dataset,
    args,
    device,
):

    x = collect_temporal_features(
        model,
        dataset,
        device,
        args.feature_batch_size,
        args.num_workers,
    )

    pca = PCA(
        n_components=args.pca_dim,
        whiten=False,
        svd_solver="randomized",
        random_state=args.seed,
    )

    pca.fit(x)

    model.set_semantic_prior_pca(
        pca.mean_,
        pca.components_,
    )

    return pca


# =========================================================
# Student semantic prior
# =========================================================

def student_prior(
    model,
    eeg,
    temperature,
):

    feat = model._semantic_prior_feature(
        eeg
    )

    with torch.autocast(
        "cuda",
        enabled=False,
    ):

        feat = (
            model
            .semantic_prior_classifier_norm(
                feat.float()
            )
        )

        logits = (
            model
            .semantic_prior_classifier(
                feat
            )
        )

        probs = F.softmax(
            logits / temperature,
            dim=-1,
        )

        prior = F.normalize(
            probs
            @ model.fixed_text_prototypes.float(),
            dim=-1,
        )

    return (
        logits,
        probs,
        prior,
    )


def condition_from_prior(
    model,
    prior,
):

    B = prior.shape[0]

    dtype = next(
        model.pipeline.unet.parameters()
    ).dtype

    empty_prompt = (
        model.get_empty_text_embeds(B)
    )

    # semantic_to_cross is FROZEN,
    # but gradient still flows to prior.
    semantic_cond = (
        model.semantic_to_cross(
            prior.float()
        )
        .unsqueeze(1)
    )

    semantic_cond = semantic_cond.to(
        dtype
    )

    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(
        1,
        -1,
        1,
    )

    prompt = (
        empty_prompt
        + semantic_cond * ramp
    )

    spatial = torch.zeros(
        B,
        4,
        64,
        64,
        device=prior.device,
        dtype=dtype,
    )

    return (
        prompt,
        spatial,
    )


# =========================================================
# Soft semantic geometry
# =========================================================

def build_soft_semantic_table(
    prototypes,
    temperature,
):

    prototypes = F.normalize(
        prototypes.float(),
        dim=-1,
    )

    similarity = (
        prototypes
        @ prototypes.t()
    )

    q = F.softmax(
        similarity / temperature,
        dim=-1,
    )

    return q


def soft_semantic_kl(
    logits,
    labels,
    q_table,
    student_temperature,
):

    target_q = q_table[
        labels
    ]

    log_p = F.log_softmax(
        logits / student_temperature,
        dim=-1,
    )

    return F.kl_div(
        log_p,
        target_q,
        reduction="batchmean",
    )


# =========================================================
# Freeze oracle generator
# =========================================================

def freeze_generator(
    model,
):

    model.requires_grad_(False)

    model.semantic_prior_classifier_norm.requires_grad_(
        True
    )

    model.semantic_prior_classifier.requires_grad_(
        True
    )

    model.fmri_encoder.eval()
    model.semantic_to_cross.eval()
    model.unet.eval()
    model.pipeline.vae.eval()
    model.pipeline.text_encoder.eval()

    model.semantic_prior_classifier_norm.train()
    model.semantic_prior_classifier.train()

    n = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[student trainable] "
        f"{n / 1e6:.4f}M"
    )


# =========================================================
# Semantic validation
# =========================================================

@torch.no_grad()
def evaluate_semantic(
    model,
    dataset,
    args,
    device,
):

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    total = 0
    correct = 0
    entropy_sum = 0.0

    for batch in loader:

        eeg = batch[
            "eeg_data"
        ].to(
            device,
            dtype=torch.float32,
        )

        labels = batch[
            "cls_index"
        ].to(
            device,
            dtype=torch.long,
        )

        logits, probs, _ = student_prior(
            model,
            eeg,
            args.student_temperature,
        )

        correct += int(
            (
                logits.argmax(dim=-1)
                == labels
            )
            .sum()
        )

        entropy = -(
            probs.clamp_min(1e-8)
            * probs.clamp_min(1e-8).log()
        ).sum(
            dim=-1
        )

        entropy_sum += float(
            entropy.sum()
        )

        total += labels.shape[0]

    return {
        "top1": correct / total,
        "entropy": entropy_sum / total,
    }


# =========================================================
# Generator-effect validation
# =========================================================

@torch.no_grad()
def evaluate_generation_effect(
    model,
    val_subset,
    args,
    device,
):

    loader = DataLoader(
        val_subset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
    )

    distills = []
    student_losses = []
    teacher_losses = []
    uncond_losses = []

    with torch.random.fork_rng(
        devices=[0]
        if device.type == "cuda"
        else []
    ):

        torch.manual_seed(
            args.seed + 8119
        )

        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                args.seed + 8119
            )

        for i, batch in enumerate(loader):

            if i >= args.val_diff_batches:
                break

            eeg = batch[
                "eeg_data"
            ].to(
                device,
                dtype=torch.float32,
            )

            (
                labels,
                t,
                noisy,
                v_target,
            ) = prepare_noisy_latents(
                model,
                batch,
                device,
            )

            _, _, s_prior = student_prior(
                model,
                eeg,
                args.student_temperature,
            )

            (
                s_prompt,
                s_spatial,
            ) = condition_from_prior(
                model,
                s_prior,
            )

            (
                t_prompt,
                t_spatial,
            ) = build_oracle_condition(
                model,
                labels,
                unconditional=False,
            )

            (
                u_prompt,
                u_spatial,
            ) = build_oracle_condition(
                model,
                labels,
                unconditional=True,
            )

            v_student = model.forward_unet(
                noisy,
                t,
                s_prompt,
                s_spatial,
            )

            v_teacher = model.forward_unet(
                noisy,
                t,
                t_prompt,
                t_spatial,
            )

            v_uncond = model.forward_unet(
                noisy,
                t,
                u_prompt,
                u_spatial,
            )

            distills.append(
                float(
                    F.mse_loss(
                        v_student.float(),
                        v_teacher.float(),
                    )
                )
            )

            s_loss, _ = model.compute_loss(
                v_student,
                v_target,
            )

            t_loss, _ = model.compute_loss(
                v_teacher,
                v_target,
            )

            u_loss, _ = model.compute_loss(
                v_uncond,
                v_target,
            )

            student_losses.append(
                float(s_loss)
            )

            teacher_losses.append(
                float(t_loss)
            )

            uncond_losses.append(
                float(u_loss)
            )

    student = float(
        np.mean(student_losses)
    )

    teacher = float(
        np.mean(teacher_losses)
    )

    uncond = float(
        np.mean(uncond_losses)
    )

    return {
        "distill": float(
            np.mean(distills)
        ),
        "student_diff": student,
        "oracle_diff": teacher,
        "uncond_diff": uncond,
        "student_gain": (
            uncond - student
        ),
        "oracle_gain": (
            uncond - teacher
        ),
    }


# =========================================================
# Args
# =========================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        default="./configs/mind3d_pp.yaml",
    )

    p.add_argument(
        "--data_path",
        default="/data/jionkim/neuro_3D/",
    )

    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )

    p.add_argument(
        "--sub_id",
        default="sub01",
    )

    p.add_argument(
        "--out_dir",
        default=(
            "/data/jionkim/mind_3d_output/"
            "eeg_semantic_distillation_h06"
        ),
    )

    p.add_argument(
        "--oracle_ckpt",
        required=True,
    )

    p.add_argument(
        "--temporal_ckpt",
        default=(
            "/data/jionkim/mind_3d_output/"
            "holdout_06/checkpoints/"
            "best_semantic.pt"
        ),
    )

    p.add_argument(
        "--holdout_suffix",
        default="06",
    )

    p.add_argument(
        "--pca_dim",
        type=int,
        default=512,
    )

    p.add_argument(
        "--feature_batch_size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--teacher_semantic_temperature",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--student_temperature",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--lambda_distill",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--lambda_soft_sem",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--classifier_lr",
        type=float,
        default=2e-4,
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--max_steps",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--batchsize",
        type=int,
        default=1,
    )

    p.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--validate_every",
        type=int,
        default=250,
    )

    p.add_argument(
        "--val_diff_batches",
        type=int,
        default=32,
    )

    p.add_argument(
        "--print_every",
        type=int,
        default=20,
    )

    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    universe = tuple(
        f"{i:02d}"
        for i in range(7)
    )

    train_suffixes = tuple(
        x
        for x in universe
        if x != args.holdout_suffix
    )

    val_suffixes = (
        args.holdout_suffix,
    )

    print(
        f"[strict] train={train_suffixes}; "
        f"val={val_suffixes}; "
        "object07=NOT_USED; final08_09=NOT_USED"
    )

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    ckpt_dir = (
        out_dir
        / "checkpoints"
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )

    train_subset = Subset(
        base,
        full_dataset_indices(
            base,
            train_suffixes,
        ),
    )

    val_subset = Subset(
        base,
        full_dataset_indices(
            base,
            val_suffixes,
        ),
    )

    train_eeg = EEGOnlyDataset(
        base,
        train_suffixes,
        mode="individual",
    )

    val_ind = EEGOnlyDataset(
        base,
        val_suffixes,
        mode="individual",
    )

    val_avg = EEGOnlyDataset(
        base,
        val_suffixes,
        mode="averaged",
    )

    (
        prototypes,
        categories,
        source_objects,
    ) = build_fixed_category_text_prototypes(
        base,
        train_suffixes,
    )

    model, _ = build_model(
        args,
        out_dir,
    )

    model = model.to(device)

    # -----------------------------------------------
    # Load Oracle generator
    # -----------------------------------------------

    oracle = torch.load(
        args.oracle_ckpt,
        map_location="cpu",
    )

    model.load_state_dict(
        oracle["model"],
        strict=False,
    )

    model.set_fixed_prototypes(
        prototypes.to(device)
    )

    print(
        f"[oracle] loaded "
        f"{args.oracle_ckpt} "
        f"step={oracle.get('step')}"
    )

    # -----------------------------------------------
    # Install chosen frozen TemporalStem
    # -----------------------------------------------

    load_only_temporal_stem(
        model,
        args.temporal_ckpt,
    )

    # -----------------------------------------------
    # Strict train-only PCA
    # -----------------------------------------------

    pca = fit_train_only_pca(
        model,
        train_eeg,
        args,
        device,
    )

    np.savez_compressed(
        out_dir
        / "semantic_prior_pca_train_only.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(
            np.float32
        ),
        train_suffixes=np.asarray(
            train_suffixes
        ),
    )

    print(
        f"[PCA] explained_var="
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    # Oracle checkpoint's classifier was unused.
    # Start EEG semantic student from scratch.
    model.semantic_prior_classifier_norm.reset_parameters()
    model.semantic_prior_classifier.reset_parameters()

    freeze_generator(
        model
    )

    q_table = build_soft_semantic_table(
        prototypes.to(device),
        args.teacher_semantic_temperature,
    ).detach()

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )

    loader = DataLoader(
        train_subset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
        drop_last=True,
    )

    step = 0
    micro = 0
    best = None
    best_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    while step < args.max_steps:

        for batch in loader:

            eeg = batch[
                "eeg_data"
            ].to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            (
                labels,
                t,
                noisy,
                _,
            ) = prepare_noisy_latents(
                model,
                batch,
                device,
            )

            # ---------------------------------------
            # Frozen Oracle teacher
            # ---------------------------------------

            with torch.no_grad():

                (
                    teacher_prompt,
                    teacher_spatial,
                ) = build_oracle_condition(
                    model,
                    labels,
                    unconditional=False,
                )

                v_teacher = model.forward_unet(
                    noisy,
                    t,
                    teacher_prompt,
                    teacher_spatial,
                ).detach()

            # ---------------------------------------
            # EEG student
            # ---------------------------------------

            (
                logits,
                probs,
                prior,
            ) = student_prior(
                model,
                eeg,
                args.student_temperature,
            )

            (
                student_prompt,
                student_spatial,
            ) = condition_from_prior(
                model,
                prior,
            )

            v_student = model.forward_unet(
                noisy,
                t,
                student_prompt,
                student_spatial,
            )

            loss_distill = F.mse_loss(
                v_student.float(),
                v_teacher.float(),
            )

            loss_soft = soft_semantic_kl(
                logits,
                labels,
                q_table,
                args.student_temperature,
            )

            loss = (
                args.lambda_distill
                * loss_distill
                + args.lambda_soft_sem
                * loss_soft
            )

            (
                loss
                / args.accumulation_steps
            ).backward()

            micro += 1

            if (
                micro
                % args.accumulation_steps
                != 0
            ):
                continue

            clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                args.grad_clip,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            step += 1

            if (
                step == 1
                or step
                % args.print_every
                == 0
            ):

                top1 = float(
                    (
                        logits.argmax(dim=-1)
                        == labels
                    )
                    .float()
                    .mean()
                )

                print(
                    f"[TRAIN] step={step:06d} "
                    f"L={loss.item():.6f} "
                    f"Distill={loss_distill.item():.6f} "
                    f"SoftKL={loss_soft.item():.4f} "
                    f"Top1={top1:.3f}"
                )

            if (
                step
                % args.validate_every
                == 0
            ):

                sem_ind = evaluate_semantic(
                    model,
                    val_ind,
                    args,
                    device,
                )

                sem_avg = evaluate_semantic(
                    model,
                    val_avg,
                    args,
                    device,
                )

                gen = evaluate_generation_effect(
                    model,
                    val_subset,
                    args,
                    device,
                )

                print(
                    f"[VAL] step={step:06d} "
                    f"Distill={gen['distill']:.6f} "
                    f"StudentDiff={gen['student_diff']:.5f} "
                    f"OracleDiff={gen['oracle_diff']:.5f} "
                    f"UncondDiff={gen['uncond_diff']:.5f} "
                    f"StudentGain={gen['student_gain']:+.5f} | "
                    f"IndTop1={sem_ind['top1']:.4f} "
                    f"AvgTop1={sem_avg['top1']:.4f} "
                    f"AvgH={sem_avg['entropy']:.3f}"
                )

                # Primary:
                # student generator effect ≈ oracle effect.
                #
                # Secondary:
                # student should outperform unconditional.
                key = (
                    -gen["distill"],
                    gen["student_gain"],
                )

                payload = {
                    "model": model.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "train_suffixes": train_suffixes,
                    "holdout_suffix": args.holdout_suffix,
                    "oracle_ckpt": args.oracle_ckpt,
                    "val_generation": gen,
                    "val_ind": sem_ind,
                    "val_avg": sem_avg,
                }

                torch.save(
                    payload,
                    ckpt_dir / "last.pt",
                )

                if (
                    best is None
                    or key > best
                ):

                    best = key
                    best_step = step

                    torch.save(
                        payload,
                        ckpt_dir / "best.pt",
                    )

                    print(
                        f"[BEST] step={step}"
                    )

            if step >= args.max_steps:
                break

    print(
        f"[done] best_step={best_step}"
    )


if __name__ == "__main__":
    main()