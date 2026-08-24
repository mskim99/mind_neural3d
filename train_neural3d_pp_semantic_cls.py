#!/usr/bin/env python3
"""
Two-stage Neuro-3D training with explicit 72-way semantic supervision.

Stage 1: semantic warm-up
    EEG -> semantic / variation / bias
          |
          +-> CLIP image/text alignment
          +-> 72-way semantic classification
          +-> orthogonality
    Zero123++ LoRA and generation heads are frozen.
    No VAE/UNet diffusion forward is executed.

Stage 2: joint generation training
    semantic  -> Zero123++ cross-attention
    variation -> Zero123++ spatial cond_lat
    + diffusion loss
    + CLIP loss
    + 72-way classification loss
    + orthogonality loss

Recommended first run on a 24-GB GPU:
    CUDA_VISIBLE_DEVICES=1 python train_neural3d_pp_semantic_cls.py \
        --config ./configs/mind3d_pp.yaml \
        --sub_id 0001 \
        --rendered_view_path /data/jionkim/neuro_3D/eeg3d_training \
        --batchsize 1 \
        --accumulation_steps 2 \
        --semantic_stage_steps 5000 \
        --max_steps 60000 \
        --out_dir stage2_semantic_cls

This script is intentionally single-GPU focused. Physical GPU 1 is selected
by default before torch import.
"""

# -------------------------------------------------------------------------
# GPU selection MUST happen before importing torch.
# -------------------------------------------------------------------------
import os
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from omegaconf import OmegaConf

from src.mvdiffusion_var_semantic_cls import MVDiffusion, unscale_image
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG
from src.utils import set_random_seed


# =========================================================================
# Utilities
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage EEG->Zero123++ training with 72-way semantic supervision"
    )

    p.add_argument("--config", default="./configs/mind3d_pp.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument("--rendered_view_path",
                   default="/data/jionkim/neuro_3D/eeg3d_training")
    p.add_argument("--sub_id", default="0001")
    p.add_argument("--out_dir", default="stage2_semantic_cls")

    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help=(
            "LoRA/diffusion learning rate. If omitted, the script searches "
            "the YAML for learning_rate / training.learning_rate / "
            "model.learning_rate / model.base_learning_rate. "
            "Falls back to 1e-5 only if none is present."
        ),
    )

    # Optimizer-step counts, NOT micro-batch iterations.
    p.add_argument("--semantic_stage_steps", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=60000)

    # Loss weights.
    p.add_argument("--lambda_diff", type=float, default=1.0)
    p.add_argument("--lambda_clip", type=float, default=0.5)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_ortho", type=float, default=0.05)
    p.add_argument("--cls_label_smoothing", type=float, default=0.05)

    p.add_argument("--grad_clip", type=float, default=1.0)

    # Logging / validation.
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--validate_every", type=int, default=200)
    p.add_argument("--visualize_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--vis_steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=4.0)

    p.add_argument("--resume", default="",
                   help="Resume only from a checkpoint created by this script.")
    p.add_argument("--aug_data", action="store_true")
    p.add_argument("--audit_only", action="store_true")

    return p.parse_args()


def unwrap(model):
    return model



def _find_key_recursive(cfg, target_key):
    """
    Recursively find every occurrence of `target_key` in an OmegaConf tree.

    Returns:
        list[(path, value)]
    """
    container = OmegaConf.to_container(cfg, resolve=False)
    matches = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == target_key:
                    matches.append((child_path, value))
                walk(value, child_path)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                child_path = f"{path}[{idx}]"
                walk(value, child_path)

    walk(container)
    return matches


def resolve_model_configs(cfg, config_path, cli_learning_rate=None):
    """
    Resolve the MinD-3D++ / Zero123++ training config.

    IMPORTANT:
      configs/mind3d.yaml    -> original MinD-3D config (WRONG for this trainer)
      configs/mind3d_pp.yaml -> MinD-3D++ Zero123++ config (CORRECT)

    The expected structure is:
        learning_rate: ...
        model:
          params:
            stable_diffusion_config: ...
            fmri_encoder_config: ...

    Fail explicitly instead of trying to reinterpret the original MinD-3D YAML.
    """
    top_keys = list(cfg.keys()) if OmegaConf.is_dict(cfg) else []
    print(f"[config] loaded: {config_path}")
    print(f"[config] top-level keys: {top_keys}")

    stable_cfg = OmegaConf.select(
        cfg,
        "model.params.stable_diffusion_config",
        default=None,
    )
    fmri_cfg = OmegaConf.select(
        cfg,
        "model.params.fmri_encoder_config",
        default=None,
    )

    if stable_cfg is None:
        # Detect the original MinD-3D YAML from its characteristic keys.
        original_mind3d_keys = {
            "diff_prior_config_path",
            "fmri_model",
            "3d_model",
        }
        if original_mind3d_keys.intersection(set(top_keys)):
            raise KeyError(
                "\nWrong config file for EEG -> Zero123++ training.\n\n"
                f"Loaded: {config_path}\n"
                f"Top-level keys: {top_keys}\n\n"
                "This is the original MinD-3D config (`mind3d.yaml`). "
                "The current MVDiffusion/Zero123++ trainer requires "
                "`configs/mind3d_pp.yaml`, which contains:\n"
                "  model.params.stable_diffusion_config\n"
                "  model.params.fmri_encoder_config\n\n"
                "Run with:\n"
                "  --config ./configs/mind3d_pp.yaml"
            )

        raise KeyError(
            "\n`model.params.stable_diffusion_config` is missing.\n"
            f"Loaded config: {config_path}\n"
            f"Top-level keys: {top_keys}\n"
            "Use the MinD-3D++ config: ./configs/mind3d_pp.yaml"
        )

    if fmri_cfg is None:
        # Current EEG disentangling encoder does not consume this config directly,
        # but preserve the official interface and warn rather than crash.
        print(
            "[config warning] model.params.fmri_encoder_config not found; "
            "passing None because the current EEG encoder is constructed explicitly."
        )

    # Preserve the original project's behavior:
    # MVDiffusion reads `args.learning_rate` from the first argument.
    if cli_learning_rate is not None:
        cfg.learning_rate = float(cli_learning_rate)
        print(
            f"[config] overriding learning_rate with CLI: "
            f"{cfg.learning_rate}"
        )
    elif OmegaConf.select(cfg, "learning_rate", default=None) is None:
        # mind3d_pp.yaml should contain this. Fail rather than silently inventing it.
        raise KeyError(
            "`learning_rate` is missing from the MinD-3D++ config. "
            "Pass --learning_rate explicitly."
        )

    print(
        "[config] stable_diffusion_config: "
        "model.params.stable_diffusion_config"
    )
    print(
        "[config] fmri_encoder_config    : "
        "model.params.fmri_encoder_config"
    )
    print(f"[config] learning_rate          : {cfg.learning_rate}")

    return stable_cfg, fmri_cfg, cfg


def dump_dataset_mapping(dataset, path):
    """
    Save the exact EEG class-row -> rendered object mapping before training.
    This is a preflight audit artifact, not a learned prediction.
    """
    rows = []
    for cls_index in range(dataset.name_list.shape[0]):
        for obj_index in range(dataset.name_list.shape[1]):
            name = str(dataset.name_list[cls_index, obj_index])
            rows.append({
                "cls_index": cls_index,
                "obj_index": obj_index,
                "class_prefix": name[:3],
                "dataset_name": name,
                "label": name[3:],
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cls_index", "obj_index", "class_prefix",
                "dataset_name", "label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer_step, micro_step, stage, args):
    payload = {
        "model": model.state_dict(),
        "optimizer": model.opt.state_dict(),
        "scheduler": model.sche.state_dict(),
        "optimizer_step": int(optimizer_step),
        "micro_step": int(micro_step),
        "stage": str(stage),
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location="cpu")

    if "model" not in ckpt:
        raise RuntimeError(
            "--resume expects a checkpoint produced by this training script "
            "with a top-level `model` entry."
        )

    incompatible = model.load_state_dict(ckpt["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[RESUME MODEL MISMATCH]")
        print("missing:", incompatible.missing_keys[:30])
        print("unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError("Resume checkpoint architecture mismatch.")

    if "optimizer" in ckpt:
        model.opt.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        model.sche.load_state_dict(ckpt["scheduler"])

    return (
        int(ckpt.get("optimizer_step", 0)),
        int(ckpt.get("micro_step", 0)),
    )


@torch.no_grad()
def validate_semantics(model, batch):
    metrics = model.semantic_validation(batch)
    return {
        k: float(v.item())
        for k, v in metrics.items()
    }


@torch.no_grad()
def generate_validation_grid(
    model,
    batch,
    save_path,
    num_steps=50,
    cfg_scale=4.0,
):
    """
    Generate the 3x2 grid with the SAME semantic/variation conditioning used
    in training and save GT above prediction.
    """
    was_training = model.training
    model.eval()

    cond_eeg, target_imgs = model.prepare_batch_data(batch)
    B = cond_eeg.shape[0]

    _, prompt_cond, latent_cond = model.encode_embed_fmri_condition_fmri(
        cond_eeg,
        drop_condition=False,
    )
    _, prompt_uncond, latent_uncond = model.encode_embed_fmri_condition_fmri(
        cond_eeg,
        drop_condition=True,
    )

    prompt_embeds = torch.cat([prompt_uncond, prompt_cond], dim=0)
    cond_latents = torch.cat([latent_uncond, latent_cond], dim=0)

    scheduler = model.pipeline.scheduler
    scheduler.set_timesteps(num_steps, device=cond_eeg.device)

    dtype = next(model.pipeline.unet.parameters()).dtype
    latents = torch.randn(
        (B, 4, 120, 80),
        device=cond_eeg.device,
        dtype=dtype,
    )
    latents = latents * scheduler.init_noise_sigma

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in scheduler.timesteps:
            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = scheduler.scale_model_input(latent_in, t)

            noise_pred = model.forward_unet(
                latent_in,
                t,
                prompt_embeds,
                cond_latents,
            )
            pred_u, pred_c = noise_pred.chunk(2)
            noise_pred = pred_u + cfg_scale * (pred_c - pred_u)

            latents = scheduler.step(
                noise_pred,
                t,
                latents,
            ).prev_sample

        images_pred = model.pipeline.vae.decode(
            latents / model.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images_pred = unscale_image(images_pred)
        images_pred = (images_pred * 0.5 + 0.5).clamp(0, 1)

    # Each item is already one [3,960,640] 3x2 view grid.
    comparison = make_grid(
        torch.cat(
            [
                target_imgs.detach().float().cpu(),
                images_pred.detach().float().cpu(),
            ],
            dim=0,
        ),
        nrow=B,
        normalize=True,
        value_range=(0, 1),
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(comparison, save_path)

    if was_training:
        model.train()


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    if args.semantic_stage_steps < 0:
        raise ValueError("--semantic_stage_steps must be >= 0")
    if args.max_steps <= args.semantic_stage_steps:
        raise ValueError(
            "--max_steps must be larger than --semantic_stage_steps "
            "so Stage-2 joint training actually runs."
        )
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be >= 1")

    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda:0")
    print(
        "[device]",
        device,
        "CUDA_VISIBLE_DEVICES=",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )

    cfg = OmegaConf.load(config_path)
    stable_cfg, fmri_cfg, model_args = resolve_model_configs(
        cfg,
        config_path,
        cli_learning_rate=args.learning_rate,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    shutil.copyfile(
        config_path,
        out_dir / "config.yaml",
    )
    with open(out_dir / "train_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------
    # Dataset: explicit class labels are mandatory.
    # ---------------------------------------------------------------------
    sub_list = [args.sub_id]

    train_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=sub_list,
        train=True,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=args.aug_data,
        strict_rendered_views=True,
    )
    val_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=sub_list,
        train=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )

    dump_dataset_mapping(
        train_dataset,
        out_dir / "mapping_train.csv",
    )
    dump_dataset_mapping(
        val_dataset,
        out_dir / "mapping_test.csv",
    )

    print(
        f"[mapping] train name_list={train_dataset.name_list.shape}, "
        f"test name_list={val_dataset.name_list.shape}"
    )
    print(
        f"[mapping] audit CSVs: {out_dir / 'mapping_train.csv'}, "
        f"{out_dir / 'mapping_test.csv'}"
    )

    if args.audit_only:
        print("[audit-only] Dataset mapping validated. No training started.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=False,
    )

    # ---------------------------------------------------------------------
    # Model.
    # ---------------------------------------------------------------------
    model = MVDiffusion(
        model_args,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(out_dir),
        num_classes=72,
        cls_label_smoothing=args.cls_label_smoothing,
    ).to(device)

    optimizer_step = 0
    micro_step = 0

    if args.resume:
        optimizer_step, micro_step = load_checkpoint(
            args.resume,
            model,
        )
        print(
            f"[resume] optimizer_step={optimizer_step}, "
            f"micro_step={micro_step}"
        )

    # Determine current stage from the optimizer step, not from the checkpoint
    # text field, so resuming is deterministic.
    stage = (
        "semantic"
        if optimizer_step < args.semantic_stage_steps
        else "joint"
    )
    model.set_training_stage(stage)
    model.train()

    # Stage-1 does NOT step the diffusion scheduler. Thus when Stage-2 starts,
    # the original 300-step LR warm-up begins at its first joint optimizer step.
    model.opt.zero_grad(set_to_none=True)

    writer = SummaryWriter(str(out_dir / "logs"))
    val_iter = iter(val_loader)

    start_time = time.time()
    epoch = 0
    accum_counter = 0

    print(
        f"[training] Stage-1 semantic: 0 -> {args.semantic_stage_steps} optimizer steps"
    )
    print(
        f"[training] Stage-2 joint   : {args.semantic_stage_steps} -> "
        f"{args.max_steps} optimizer steps"
    )
    print(
        "[weights] "
        f"diff={args.lambda_diff}, clip={args.lambda_clip}, "
        f"cls={args.lambda_cls}, ortho={args.lambda_ortho}"
    )

    while optimizer_step < args.max_steps:
        epoch += 1

        for batch in train_loader:
            if optimizer_step >= args.max_steps:
                break

            wanted_stage = (
                "semantic"
                if optimizer_step < args.semantic_stage_steps
                else "joint"
            )
            if wanted_stage != stage:
                stage = wanted_stage
                model.set_training_stage(stage)
                model.opt.zero_grad(set_to_none=True)
                accum_counter = 0
                print(
                    f"\n[stage transition] optimizer_step={optimizer_step} "
                    f"-> {stage}\n"
                )

            micro_step += 1
            accum_counter += 1

            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(
                    batch,
                    stage=stage,
                )

                total_raw = (
                    args.lambda_clip * losses["clip_loss"]
                    + args.lambda_cls * losses["cls_loss"]
                    + args.lambda_ortho * losses["ortho_loss"]
                )

                if stage == "joint":
                    total_raw = (
                        total_raw
                        + args.lambda_diff * losses["diff_loss"]
                    )

                loss_for_backward = (
                    total_raw / args.accumulation_steps
                )

            loss_for_backward.backward()

            if accum_counter < args.accumulation_steps:
                continue

            # One optimizer update.
            if args.grad_clip > 0:
                grad_norm = clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            else:
                grad_norm = torch.tensor(0.0, device=device)

            model.opt.step()
            model.opt.zero_grad(set_to_none=True)

            # Diffusion/LoRA scheduler begins ONLY in joint stage.
            if stage == "joint":
                model.sche.step()

            accum_counter = 0
            optimizer_step += 1

            # -------------------------------------------------------------
            # Logging.
            # -------------------------------------------------------------
            writer.add_scalar(
                "train/total_loss",
                float(total_raw.detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/diff_loss",
                float(losses["diff_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/clip_loss",
                float(losses["clip_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/cls_loss",
                float(losses["cls_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/cls_acc",
                float(losses["cls_acc"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/ortho_loss",
                float(losses["ortho_loss"].detach().item()),
                optimizer_step,
            )
            writer.add_scalar(
                "train/grad_norm",
                float(grad_norm),
                optimizer_step,
            )
            writer.add_scalar(
                "train/stage",
                0 if stage == "semantic" else 1,
                optimizer_step,
            )

            if stage == "joint":
                writer.add_scalar(
                    "train/lr_lora",
                    model.opt.param_groups[0]["lr"],
                    optimizer_step,
                )

            if (
                args.print_every > 0
                and optimizer_step % args.print_every == 0
            ):
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch:03d}] "
                    f"step={optimizer_step:06d} "
                    f"stage={stage:<8s} "
                    f"Total={total_raw.item():.4f} "
                    f"Diff={losses['diff_loss'].item():.4f} "
                    f"CLIP={losses['clip_loss'].item():.4f} "
                    f"Cls={losses['cls_loss'].item():.4f} "
                    f"Acc={losses['cls_acc'].item():.3f} "
                    f"Ortho={losses['ortho_loss'].item():.4f} "
                    f"time={elapsed / 60.0:.1f}m"
                )

            # -------------------------------------------------------------
            # Semantic validation.
            # -------------------------------------------------------------
            if (
                args.validate_every > 0
                and optimizer_step % args.validate_every == 0
            ):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                val_metrics = validate_semantics(
                    model,
                    val_batch,
                )

                for key, value in val_metrics.items():
                    writer.add_scalar(
                        f"val/{key}",
                        value,
                        optimizer_step,
                    )

                print(
                    f"[VAL semantic] step={optimizer_step:06d} "
                    f"CLIP={val_metrics['clip_loss']:.4f} "
                    f"Cls={val_metrics['cls_loss']:.4f} "
                    f"Acc={val_metrics['cls_acc']:.3f} "
                    f"Ortho={val_metrics['ortho_loss']:.4f}"
                )

            # -------------------------------------------------------------
            # Expensive generation visualization only after Stage-2 starts.
            # -------------------------------------------------------------
            if (
                stage == "joint"
                and args.visualize_every > 0
                and optimizer_step % args.visualize_every == 0
            ):
                try:
                    vis_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    vis_batch = next(val_iter)

                generate_validation_grid(
                    model,
                    vis_batch,
                    str(
                        out_dir
                        / "images"
                        / f"val_{optimizer_step:06d}.png"
                    ),
                    num_steps=args.vis_steps,
                    cfg_scale=args.cfg_scale,
                )
                model.train()

            # -------------------------------------------------------------
            # Checkpoint.
            # -------------------------------------------------------------
            if (
                args.save_every > 0
                and optimizer_step % args.save_every == 0
            ):
                ckpt_path = (
                    out_dir
                    / "checkpoints"
                    / f"model_{optimizer_step:06d}.pt"
                )
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer_step,
                    micro_step,
                    stage,
                    args,
                )
                print(f"[checkpoint] {ckpt_path}")

            # Release references to large Stage-2 tensors.
            del losses, total_raw, loss_for_backward, batch

    final_path = out_dir / "checkpoints" / "model_final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer_step,
        micro_step,
        stage,
        args,
    )

    writer.close()
    print(f"[done] final checkpoint: {final_path}")


if __name__ == "__main__":
    main()