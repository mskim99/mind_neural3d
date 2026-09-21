from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from huggingface_hub import hf_hub_download

# Static EEG semantic decoder trained by train_static_eeg_mlp.py.
from train_static_eeg_mlp import (
    StaticMLP,
    apply_scaler,
    normalize_eeg,
    preprocess_eeg,
)

# ============================================================
# Optional dependencies
# ============================================================
try:
    import rembg
except ImportError:
    rembg = None

try:
    from diffusers import (
        StableDiffusionXLControlNetPipeline, # 일반 Pipeline 대신 사용
        ControlNetModel,                     # ControlNet 가중치 로드용
        DiffusionPipeline,
        EulerAncestralDiscreteScheduler,
    )
except ImportError:
    StableDiffusionXLControlNetPipeline = None
    ControlNetModel = None
    DiffusionPipeline = None

# ============================================================
# MinD-3D Dataset Loader
# ============================================================
from src.data.egg_dataset_ext_el import (
    AllDataFeatureTwoEEG,
)

# ============================================================
# Constants
# ============================================================
DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_ZERO123_MODEL = "sudo-ai/zero123plus-v1.1"
DEFAULT_SDXL_NEGATIVE_PROMPT = (
    "multiple objects, duplicate object, extra object, group, collection, cluster, "
    "cluttered background, complex background, perspective distortion, "
    "text, watermark, logo, cropped object, partial object, "
    "deformed geometry, distorted geometry, blurry, noisy, low quality"
)

TRIAL_SUFFIX_PATTERN = re.compile(r"__trial\d+$", flags=re.IGNORECASE)


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG -> MLP Decoder -> SDXL + Auto-Downloaded LoRA -> Zero123++ Pipeline"
    )
    # [PATCH] 원하는 횟수만큼만 돌기 위한 파라미터 추가
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of samples to process. -1 means all.")
    parser.add_argument("--semantic_ckpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/data/jionkim/neuro_3D/")
    parser.add_argument("--out_dir", type=str, default="./inference_results_sdxl_lora_download")
    parser.add_argument("--sub_id", type=str, default="sub01")
    parser.add_argument("--rendered_view_path", type=str, default="/data/jionkim/neuro_3D/eeg3d_training")
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--gt_render_root", type=str, default="/data/jionkim/neuro_3D/render_grid_v4")
    parser.add_argument("--allow_missing_gt", action="store_true")

    parser.add_argument(
        "--eval_split",
        choices=["test", "train"],
        default="test",
    )

    parser.add_argument("--bg_clean_alpha_threshold", type=int, default=8)
    parser.add_argument("--bg_clean_binary_alpha", action="store_true")

    # SDXL & LoRA Hub Download Options
    parser.add_argument("--sdxl_model", type=str, default=DEFAULT_SDXL_MODEL)
    parser.add_argument("--sdxl_lora_path", type=str, default="", help="Local path to LoRA weights (optional)")
    parser.add_argument("--sdxl_lora_repo_id", type=str, default="",
                        help="Hugging Face Hub Repo ID for LoRA (e.g., username/repo-name)")
    parser.add_argument("--sdxl_lora_filename", type=str, default="", help="Filename of the LoRA weights on HF Hub")
    parser.add_argument("--sdxl_lora_scale", type=float, default=0.7, help="Scale for SDXL LoRA")
    parser.add_argument("--sdxl_steps", type=int, default=30)
    parser.add_argument("--sdxl_cfg", type=float, default=6.0)
    parser.add_argument("--sdxl_negative_prompt", type=str, default=DEFAULT_SDXL_NEGATIVE_PROMPT)

    # Zero123++ (Multi-view Generation)
    parser.add_argument("--zero123_model", type=str, default=DEFAULT_ZERO123_MODEL)
    parser.add_argument("--zero123_steps", type=int, default=75)

    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================
# Static EEG semantic decoder
# ============================================================
def load_static_semantic_decoder(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Invalid semantic checkpoint.")
    config = checkpoint["config"]
    model = StaticMLP(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        dropout=float(config["dropout"]),
        use_transformer=bool(config.get("use_transformer", False)),
        token_dim=int(config.get("token_dim", 50)),
        transformer_heads=int(config.get("transformer_heads", 4)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_ff_dim=int(config.get("transformer_ff_dim", 256)),
        input_representation=str(config.get("input_representation", "temporal_stats")),
        conv_channels=int(config.get("conv_channels", 32)),
        conv_pooled_steps=int(config.get("conv_pooled_steps", 4)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    input_norm = str(config.get("input_norm", "none"))
    input_representation = str(config.get("input_representation", "temporal_stats"))
    return model, scaler_mean, scaler_scale, input_norm, input_representation


def static_eeg_batch_features(batch, static_test, scaler_mean, scaler_scale, input_norm="none",
                              input_representation="temporal_stats"):
    raws = []
    for index in range(len(batch["cls_index"])):
        subject = int(batch["subject_index"][index])
        cls_index = int(batch["cls_index"][index])
        obj_index = int(batch["obj_index"][index])
        raw = np.asarray(static_test[subject, cls_index, obj_index], dtype=np.float32)
        raws.append(raw)
    stacked = np.stack(raws, axis=0)
    if input_representation == "raw_waveform":
        features = normalize_eeg(stacked, mode=input_norm)
    elif input_representation == "temporal_stats":
        features = preprocess_eeg(stacked, input_norm=input_norm)
    return apply_scaler(features, scaler_mean, scaler_scale)


@torch.no_grad()
def predict_static_semantics(model, features, device):
    logits, _ = model(torch.from_numpy(features).to(device))
    top5 = logits.topk(5, dim=1).indices.cpu().tolist()
    predicted = logits.argmax(dim=1).cpu().tolist()
    confidence = logits.softmax(dim=1).max(dim=1).values.cpu().tolist()
    return predicted, top5, confidence


# ============================================================
# Tensor <-> PIL
# ============================================================
def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    x = image_tensor.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_tensor_image(image_tensor: torch.Tensor, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image_tensor).save(path)


# ============================================================
# Output Helpers
# ============================================================
def make_sample_id(label: str, trial_index: int):
    return f"{label}__trial{int(trial_index):02d}"


def derive_prompt_from_sample_id(sample_id: str):
    return TRIAL_SUFFIX_PATTERN.sub("", str(sample_id)).strip()


def prepare_output_layout(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "render"
    views_dir = out_dir / "views"
    base_dir = out_dir / "base_images"
    for d in [render_dir, views_dir, base_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return str(render_dir), str(views_dir), str(base_dir)


def write_evaluation_pairs(out_dir, records):
    csv_path = os.path.join(out_dir, "evaluation_pairs.csv")
    fieldnames = [
        "sample_id", "dataset_name", "label", "class_prefix", "cls_index",
        "obj_index", "trial_index", "subject_index", "semantic_pred_cls",
        "semantic_pred_label", "semantic_top5", "semantic_confidence",
        "semantic_correct", "semantic_top5_correct", "base_image", "render_grid", "sdxl_prompt"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    return csv_path


# ============================================================
# Background Removal & Slicing
# ============================================================
def clean_generated_grid_background(generated_grid, rembg_session, alpha_threshold=8, binary_alpha=False):
    _, height, width = generated_grid.shape
    tile_h = height // 3
    tile_w = width // 2
    clean = torch.ones((3, height, width), dtype=torch.float32)
    masks = torch.zeros((1, height, width), dtype=torch.float32)
    threshold = float(alpha_threshold) / 255.0

    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w
        tile = generated_grid[:, y0:y1, x0:x1]
        tile_pil = tensor_to_pil(tile).convert("RGBA")

        rgba = rembg.remove(tile_pil, session=rembg_session, alpha_matting=False)
        rgba = rgba.convert("RGBA") if isinstance(rgba, Image.Image) else Image.open(rgba).convert("RGBA")
        rgba_np = np.asarray(rgba, dtype=np.uint8)

        rgb = rgba_np[..., :3].astype(np.float32) / 255.0
        alpha = rgba_np[..., 3].astype(np.float32) / 255.0
        alpha[alpha < threshold] = 0.0
        if binary_alpha:
            alpha = (alpha >= 0.5).astype(np.float32)

        clean_rgb = rgb * alpha[..., None] + (1.0 - alpha[..., None])
        clean_tile = torch.from_numpy(clean_rgb).permute(2, 0, 1).contiguous()
        mask_tile = torch.from_numpy(alpha).unsqueeze(0).contiguous()

        clean[:, y0:y1, x0:x1] = clean_tile
        masks[:, y0:y1, x0:x1] = mask_tile

    return clean.clamp(0, 1), masks.clamp(0, 1)


def split_grid_to_six_views(grid_tensor: torch.Tensor):
    _, height, width = grid_tensor.shape
    tile_h = height // 3
    tile_w = width // 2
    views = []
    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w
        views.append(grid_tensor[:, y0:y1, x0:x1].contiguous())
    return views


def save_six_individual_views(grid_tensor, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = split_grid_to_six_views(grid_tensor)
    saved = []
    for view_idx, view in enumerate(views):
        path = output_dir / f"{view_idx:02d}.png"
        save_tensor_image(view, path)
        saved.append(str(path))
    return saved


# ============================================================
# SDXL (with Auto-downloaded LoRA) & Zero123++ Pipelines
# ============================================================
def load_pipelines(args, device):
    print("\n" + "=" * 70)
    print("[SDXL + ControlNet(Depth) + Zero123++]")
    print("SDXL Model     :", args.sdxl_model)
    print("=" * 70)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1. ControlNet 로드 (SDXL용 Depth 모델)
    print("[*] Loading SDXL Depth ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        torch_dtype=dtype,
        use_safetensors=True
    )

    # 2. ControlNet이 결합된 SDXL 파이프라인 로드
    sdxl_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.sdxl_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True
    )

    if args.cpu_offload and device.type == "cuda":
        sdxl_pipe.enable_model_cpu_offload()
    else:
        sdxl_pipe = sdxl_pipe.to(device)
    sdxl_pipe.set_progress_bar_config(disable=True)

    # 3. Zero123++ Pipeline
    zero123_pipe = DiffusionPipeline.from_pretrained(
        args.zero123_model,
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=dtype,
    )
    zero123_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        zero123_pipe.scheduler.config, timestep_spacing='trailing'
    )
    if args.cpu_offload and device.type == "cuda":
        zero123_pipe.enable_model_cpu_offload()
    else:
        zero123_pipe = zero123_pipe.to(device)
    zero123_pipe.set_progress_bar_config(disable=True)

    return sdxl_pipe, zero123_pipe


@torch.inference_mode()
def run_sdxl_base_image(pipe, prompt, args, seed):
    effective_prompt = (
        f"A standalone solo object, a single isolated {prompt}, only one object, "
        "orthographic side profile view, flat shading, strict symmetry, "
        "zero perspective distortion, perfectly centered, "
        "isolated on a pure solid white background, high quality 3D asset"
    )

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(int(seed))

    # 생성된 가상의 Depth Map 로드
    depth_condition = create_generic_depth_condition(size=1024, radius=350)

    result = pipe(
        prompt=effective_prompt,
        negative_prompt=args.sdxl_negative_prompt,
        image=depth_condition,  # ControlNet 가이드 이미지
        controlnet_conditioning_scale=0.6,  # 제어 강도 (0.0 ~ 1.0, 0.6 추천)
        height=1024,
        width=1024,
        num_inference_steps=args.sdxl_steps,
        guidance_scale=args.sdxl_cfg,
        generator=generator,
    )
    return result.images[0].convert("RGB")


def prepare_zero123_input(image: Image.Image, rembg_session) -> Image.Image:
    rgba = rembg.remove(image, session=rembg_session, alpha_matting=False)
    if not isinstance(rgba, Image.Image):
        rgba = Image.open(rgba)
    rgba = rgba.convert("RGBA")

    gray_bg = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
    gray_bg.paste(rgba, (0, 0), rgba)
    return gray_bg.convert("RGB")


@torch.inference_mode()
def run_zero123plus(pipe, image: Image.Image, args, seed):
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(int(seed))

    result = pipe(
        image,
        num_inference_steps=args.zero123_steps,
        generator=generator,
    )
    return result.images[0].convert("RGB")


def create_generic_depth_condition(size=1024, radius=350):
    """
    SDXL ControlNet에 주입할 가상의 Depth Map을 생성합니다.
    검은색 배경(먼 곳)에 중앙으로 갈수록 하얘지는(가까운 곳) 완벽한 정면 구(Sphere) 형태를 만듭니다.
    """
    y, x = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]
    mask = x ** 2 + y ** 2 <= radius ** 2

    # 구의 둥근 입체감을 나타내는 깊이 값 계산
    depth = np.zeros((size, size), dtype=np.float32)
    depth[mask] = np.sqrt(radius ** 2 - (x ** 2 + y ** 2)[mask])

    # 0 ~ 255 스케일로 정규화
    depth = (depth / radius * 255).astype(np.uint8)

    # 약간의 블러를 주어 경계를 부드럽게 (선택적)
    img = Image.fromarray(depth).convert("RGB")
    return img


# ============================================================
# Main
# ============================================================
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_root, views_root, base_root = prepare_output_layout(args.out_dir)

    print("[*] Creating rembg session...")
    rembg_session = rembg.new_session()

    print("[*] Loading static EEG semantic checkpoint...")
    (
        semantic_model,
        semantic_scaler_mean,
        semantic_scaler_scale,
        semantic_input_norm,
        semantic_input_representation,
    ) = load_static_semantic_decoder(args.semantic_ckpt, device)

    # ========================================================
    # [PATCH] Bypass GT Image Validation and Loading
    # ========================================================
    if hasattr(AllDataFeatureTwoEEG, "_validate_rendered_dataset"):
        AllDataFeatureTwoEEG._validate_rendered_dataset = lambda self: None
    if hasattr(AllDataFeatureTwoEEG, "load_rotation_images"):
        AllDataFeatureTwoEEG.load_rotation_images = lambda self, name: torch.zeros((6, 3, 256, 256))

    is_train_split = (args.eval_split == "train")
    test_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=is_train_split,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
    )

    split_name = "train" if is_train_split else "test"
    static_test_path = Path(
        args.data_path).expanduser() / "EEGdata" / args.sub_id / f"{args.sub_id}_{split_name}_data_1s_250Hz.npy"
    static_test = np.load(static_test_path, mmap_mode="r")

    expected_shape = (1, 72, 8, 2, 64, 250) if is_train_split else (1, 72, 2, 4, 64, 250)
    if tuple(static_test.shape) == expected_shape[1:]:
        static_test = static_test[None]
    elif tuple(static_test.shape) != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tuple(static_test.shape)}")

    static_test = np.asarray(static_test).mean(axis=3)

    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=4, drop_last=False)

    sdxl_pipe, zero123_pipe = load_pipelines(args, device)

    img_idx = 0
    semantic_top1_correct = 0
    semantic_top5_correct = 0
    semantic_total = 0
    evaluation_records = []
    seen_sample_ids = set()

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="EEG -> SDXL+LoRA(Hub) -> Zero123++")):
        # [PATCH] 바깥쪽 루프: max_samples에 도달했으면 즉시 종료
        if args.max_samples > 0 and img_idx >= args.max_samples:
            print(f"\n[*] Reached max_samples limit ({args.max_samples}). Stopping generation.")
            break

        batch_size = len(batch["cls_index"])

        semantic_features = static_eeg_batch_features(
            batch, static_test, semantic_scaler_mean, semantic_scaler_scale,
            semantic_input_norm, semantic_input_representation,
        )
        semantic_pred_cls, semantic_top5, semantic_confidence = predict_static_semantics(
            semantic_model, semantic_features, device,
        )

        for i in range(batch_size):
            # [PATCH] 안쪽 루프: 배치(Batch) 처리 중간이더라도 max_samples에 도달하면 즉시 중단
            if args.max_samples > 0 and img_idx >= args.max_samples:
                break

            dataset_name = str(batch["name"][i])
            label = str(batch["label"][i])
            class_prefix = str(batch["class_prefix"][i])
            cls_index = int(batch["cls_index"][i])
            obj_index = int(batch["obj_index"][i])
            trial_index = int(batch["trial_index"][i])
            subject_index = int(batch["subject_index"][i])

            base_sample_id = make_sample_id(label=label, trial_index=trial_index)
            sample_id = base_sample_id
            if sample_id in seen_sample_ids:
                sample_id = f"{base_sample_id}__{img_idx:05d}"
            seen_sample_ids.add(sample_id)

            semantic_cls = int(semantic_pred_cls[i])
            semantic_name = str(test_dataset.name_list[semantic_cls, obj_index])
            semantic_label = semantic_name[3:]
            semantic_prompt_id = make_sample_id(label=semantic_label, trial_index=trial_index)

            semantic_is_correct = int(semantic_cls == cls_index)
            semantic_top1_correct += semantic_is_correct
            semantic_top5_correct += int(cls_index in semantic_top5[i])
            semantic_total += 1

            sdxl_prompt = derive_prompt_from_sample_id(semantic_prompt_id)
            sample_seed = args.seed + img_idx

            # 1. SDXL + 다운로드된 LoRA 기반 단일 이미지 생성
            base_image_pil = run_sdxl_base_image(sdxl_pipe, sdxl_prompt, args, sample_seed)
            base_image_path = os.path.join(base_root, f"{sample_id}_base.png")
            base_image_pil.save(base_image_path)

            # 2. 배경 제거 및 Zero123++ 입력용 회색 배경 합성
            cond_image_pil = prepare_zero123_input(base_image_pil, rembg_session)

            # 3. Zero123++ 구동
            grid_pil = run_zero123plus(zero123_pipe, cond_image_pil, args, sample_seed)

            # 4. 그리드 분할 및 최종 배경 제거
            raw_grid_tensor = pil_to_tensor(grid_pil).clamp(0, 1)
            clean_pred, mask_grid = clean_generated_grid_background(
                raw_grid_tensor, rembg_session, args.bg_clean_alpha_threshold, args.bg_clean_binary_alpha
            )

            render_grid_path = os.path.join(render_root, f"{sample_id}.png")
            save_tensor_image(clean_pred, render_grid_path)

            pred_view_dir = os.path.join(views_root, sample_id)
            save_six_individual_views(clean_pred, pred_view_dir)

            evaluation_records.append({
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "label": label,
                "class_prefix": class_prefix,
                "cls_index": cls_index,
                "obj_index": obj_index,
                "trial_index": trial_index,
                "subject_index": subject_index,
                "semantic_pred_cls": semantic_cls,
                "semantic_pred_label": semantic_label,
                "semantic_top5": " ".join(str(int(x)) for x in semantic_top5[i]),
                "semantic_confidence": float(semantic_confidence[i]),
                "semantic_correct": semantic_is_correct,
                "semantic_top5_correct": int(cls_index in semantic_top5[i]),
                "base_image": base_image_path,
                "render_grid": render_grid_path,
                "sdxl_prompt": sdxl_prompt,
            })

            img_idx += 1
            del base_image_pil, cond_image_pil, grid_pil, raw_grid_tensor, clean_pred, mask_grid
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_evaluation_pairs(args.out_dir, evaluation_records)

    print("\n" + "=" * 70)
    print("[INFERENCE COMPLETE]")
    print(f"Total Samples  : {img_idx}")
    print(f"Top-1 Accuracy : {semantic_top1_correct / semantic_total:.4%}")
    print(f"Top-5 Accuracy : {semantic_top5_correct / semantic_total:.4%}")
    print("=" * 70)


if __name__ == "__main__":
    main()