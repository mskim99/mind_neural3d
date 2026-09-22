from __future__ import annotations

import argparse
import csv
import gc
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from huggingface_hub import hf_hub_download

# ============================================================
# Optional dependencies
# ============================================================
try:
    import rembg
except ImportError:
    rembg = None

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    from diffusers import (
        StableDiffusionXLPipeline,
        DiffusionPipeline,
        EulerAncestralDiscreteScheduler,
    )
except ImportError:
    StableDiffusionXLPipeline = None
    DiffusionPipeline = None

# ============================================================
# MinD-3D Dataset Loader
# ============================================================
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG

# ============================================================
# Constants & Global Variables
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

# 전역 임베딩 텐서 보관용
CLIP_TEXT_EMBEDS_72 = None
SDXL_PROMPT_EMBEDS_72 = None
SDXL_POOLED_EMBEDS_72 = None
SDXL_NEG_PROMPT_EMBEDS = None
SDXL_NEG_POOLED_EMBEDS = None


# ============================================================
# EEG -> CLIP Aligner Model Structure
# ============================================================
class EEGCLIPAligner(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, clip_dim=512, dropout=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, x):
        features = self.net(x)
        return F.normalize(features, dim=-1)


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="EEG -> CLIP Aligner -> Direct SDXL Tensor Injection -> Zero123++")

    parser.add_argument("--aligner_ckpt", type=str, required=True, help="Path to eeg_clip_aligner_best.pt")
    parser.add_argument("--data_path", type=str, default="/data/jionkim/neuro_3D/")
    parser.add_argument("--out_dir", type=str, default="./inference_results_direct_embeds")
    parser.add_argument("--sub_id", type=str, default="sub01")
    parser.add_argument("--rendered_view_path", type=str, default="/data/jionkim/neuro_3D/render_grid_v4")
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to generate. -1 for all.")

    parser.add_argument("--eval_split", choices=["test", "train"], default="test")
    parser.add_argument("--temperature", type=float, default=15.0, help="Softmax temperature for prompt interpolation")

    # SDXL & LoRA Options
    parser.add_argument("--sdxl_model", type=str, default=DEFAULT_SDXL_MODEL)
    parser.add_argument("--sdxl_lora_path", type=str, default="")
    parser.add_argument("--sdxl_lora_repo_id", type=str, default="")
    parser.add_argument("--sdxl_lora_filename", type=str, default="")
    parser.add_argument("--sdxl_lora_scale", type=float, default=0.7)
    parser.add_argument("--sdxl_steps", type=int, default=30)
    parser.add_argument("--sdxl_cfg", type=float, default=6.0)
    parser.add_argument("--sdxl_negative_prompt", type=str, default=DEFAULT_SDXL_NEGATIVE_PROMPT)

    # Zero123++ Options
    parser.add_argument("--zero123_model", type=str, default=DEFAULT_ZERO123_MODEL)
    parser.add_argument("--zero123_steps", type=int, default=75)
    parser.add_argument("--bg_clean_alpha_threshold", type=int, default=8)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================
# Pre-compute Global Embeddings (OpenCLIP & SDXL)
# ============================================================
@torch.no_grad()
def init_global_embeddings(test_dataset, sdxl_pipe, args, device):
    global CLIP_TEXT_EMBEDS_72, SDXL_PROMPT_EMBEDS_72, SDXL_POOLED_EMBEDS_72
    global SDXL_NEG_PROMPT_EMBEDS, SDXL_NEG_POOLED_EMBEDS

    print("\n[*] Initializing Global Text & Tensor Embeddings...")

    # 1. OpenCLIP (ViT-B/32) 72-Class Embeddings (512-dim)
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    categories = []
    sdxl_prompts = []

    for c in range(72):
        name = str(test_dataset.name_list[c, 0])[3:].replace("_", " ")
        categories.append(f"a photo of a {name}")

        effective_prompt = (
            f"A standalone solo object, a single isolated {name}, only one object, "
            "orthographic side profile view, flat shading, strict symmetry, "
            "zero perspective distortion, perfectly centered, "
            "isolated on a pure solid white background, high quality 3D asset"
        )
        sdxl_prompts.append(effective_prompt)

    tokens = clip_tokenizer(categories).to(device)
    CLIP_TEXT_EMBEDS_72 = F.normalize(clip_model.encode_text(tokens), dim=-1)
    del clip_model
    torch.cuda.empty_cache()

    # 2. SDXL Native Embeddings (2048-dim & 1280-dim) for Positive Prompts
    print("[*] Pre-computing SDXL native tensors for 72 classes (This may take a moment)...")
    prompt_embeds_list = []
    pooled_embeds_list = []

    for prompt in sdxl_prompts:
        pe, _, ppe, _ = sdxl_pipe.encode_prompt(
            prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        prompt_embeds_list.append(pe)
        pooled_embeds_list.append(ppe)

    SDXL_PROMPT_EMBEDS_72 = torch.cat(prompt_embeds_list, dim=0)  # [72, 77, 2048]
    SDXL_POOLED_EMBEDS_72 = torch.cat(pooled_embeds_list, dim=0)  # [72, 1280]

    # 3. SDXL Negative Prompts Embeddings
    _, neg_pe, _, neg_ppe = sdxl_pipe.encode_prompt(
        prompt=sdxl_prompts[0],
        negative_prompt=args.sdxl_negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True
    )
    SDXL_NEG_PROMPT_EMBEDS = neg_pe
    SDXL_NEG_POOLED_EMBEDS = neg_ppe
    print("[*] Global Embeddings Initialized Successfully.")


# ============================================================
# Utilities
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


def make_sample_id(label: str, trial_index: int):
    return f"{label}__trial{int(trial_index):02d}"


def prepare_output_layout(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_dir = out_dir / "render"
    views_dir = out_dir / "views"
    base_dir = out_dir / "base_images"
    for d in [render_dir, views_dir, base_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return str(render_dir), str(views_dir), str(base_dir)


def clean_generated_grid_background(generated_grid, rembg_session, alpha_threshold=8):
    _, height, width = generated_grid.shape
    tile_h, tile_w = height // 3, width // 2
    clean = torch.ones((3, height, width), dtype=torch.float32)
    threshold = float(alpha_threshold) / 255.0

    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1, x0, x1 = row * tile_h, (row + 1) * tile_h, col * tile_w, (col + 1) * tile_w
        tile_pil = tensor_to_pil(generated_grid[:, y0:y1, x0:x1]).convert("RGBA")

        rgba = rembg.remove(tile_pil, session=rembg_session, alpha_matting=False).convert("RGBA")
        rgba_np = np.asarray(rgba, dtype=np.uint8)

        rgb = rgba_np[..., :3].astype(np.float32) / 255.0
        alpha = rgba_np[..., 3].astype(np.float32) / 255.0
        alpha[alpha < threshold] = 0.0

        clean_rgb = rgb * alpha[..., None] + (1.0 - alpha[..., None])
        clean[:, y0:y1, x0:x1] = torch.from_numpy(clean_rgb).permute(2, 0, 1).contiguous()
    return clean.clamp(0, 1)


def save_six_individual_views(grid_tensor, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, h, w = grid_tensor.shape
    th, tw = h // 3, w // 2
    for view_idx in range(6):
        r, c = divmod(view_idx, 2)
        save_tensor_image(grid_tensor[:, r * th:(r + 1) * th, c * tw:(c + 1) * tw], output_dir / f"{view_idx:02d}.png")


# ============================================================
# Pipelines Load & Execution
# ============================================================
def load_pipelines(args, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("\n[*] Loading SDXL Pipeline...")
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(args.sdxl_model, torch_dtype=dtype, use_safetensors=True)

    # LoRA Auto-Download & Load
    lora_path = args.sdxl_lora_path
    if args.sdxl_lora_repo_id and args.sdxl_lora_filename:
        try:
            print(f"[*] Downloading LoRA from Hub: {args.sdxl_lora_repo_id}...")
            lora_path = hf_hub_download(repo_id=args.sdxl_lora_repo_id, filename=args.sdxl_lora_filename)
        except Exception as e:
            print(f"[Warning] LoRA download failed: {e}")
            lora_path = ""

    if lora_path and os.path.isfile(lora_path):
        sdxl_pipe.load_lora_weights(lora_path)
        print(f"[*] Loaded SDXL LoRA weights (scale: {args.sdxl_lora_scale})")

    sdxl_pipe = sdxl_pipe.to(device) if not args.cpu_offload else sdxl_pipe.enable_model_cpu_offload()
    sdxl_pipe.set_progress_bar_config(disable=True)

    print("[*] Loading Zero123++ Pipeline...")
    zero123_pipe = DiffusionPipeline.from_pretrained(args.zero123_model, custom_pipeline="sudo-ai/zero123plus-pipeline",
                                                     torch_dtype=dtype)
    zero123_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(zero123_pipe.scheduler.config,
                                                                         timestep_spacing='trailing')
    zero123_pipe = zero123_pipe.to(device) if not args.cpu_offload else zero123_pipe.enable_model_cpu_offload()
    zero123_pipe.set_progress_bar_config(disable=True)

    return sdxl_pipe, zero123_pipe


@torch.inference_mode()
def run_sdxl_base_image_direct(pipe, prompt_embeds, pooled_prompt_embeds, args, seed):
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    has_lora = bool(args.sdxl_lora_path or (args.sdxl_lora_repo_id and args.sdxl_lora_filename))

    result = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=SDXL_NEG_PROMPT_EMBEDS,
        negative_pooled_prompt_embeds=SDXL_NEG_POOLED_EMBEDS,
        height=1024, width=1024,
        num_inference_steps=args.sdxl_steps,
        guidance_scale=args.sdxl_cfg,
        cross_attention_kwargs={"scale": args.sdxl_lora_scale} if has_lora else None,
        generator=generator,
    )
    return result.images[0].convert("RGB")


@torch.inference_mode()
def run_zero123plus(pipe, image: Image.Image, args, seed):
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    return pipe(image, num_inference_steps=args.zero123_steps, generator=generator).images[0].convert("RGB")


def prepare_zero123_input(image: Image.Image, rembg_session) -> Image.Image:
    rgba = rembg.remove(image, session=rembg_session, alpha_matting=False)
    rgba = (Image.open(rgba) if not isinstance(rgba, Image.Image) else rgba).convert("RGBA")
    gray_bg = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
    gray_bg.paste(rgba, (0, 0), rgba)
    return gray_bg.convert("RGB")


# ============================================================
# Main Execution
# ============================================================
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    render_root, views_root, base_root = prepare_output_layout(args.out_dir)

    print("[*] Creating rembg session...")
    rembg_session = rembg.new_session()

    # Dataset & Preprocessing Loader
    if hasattr(AllDataFeatureTwoEEG, "_validate_rendered_dataset"):
        AllDataFeatureTwoEEG._validate_rendered_dataset = lambda self: None
    if hasattr(AllDataFeatureTwoEEG, "load_rotation_images"):
        AllDataFeatureTwoEEG.load_rotation_images = lambda self, name: torch.zeros((6, 3, 256, 256))

    is_train_split = (args.eval_split == "train")
    test_dataset = AllDataFeatureTwoEEG(
        data_path=args.data_path, sub_list=[args.sub_id], train=is_train_split,
        num_frames=6, rendered_view_path=args.rendered_view_path
    )

    # Static Raw EEG Loader
    split_name = "train" if is_train_split else "test"
    static_test_path = Path(
        args.data_path).expanduser() / "EEGdata" / args.sub_id / f"{args.sub_id}_{split_name}_data_1s_250Hz.npy"
    static_test = np.load(static_test_path, mmap_mode="r")

    expected_shape = (1, 72, 8, 2, 64, 250) if is_train_split else (1, 72, 2, 4, 64, 250)
    if tuple(static_test.shape) == expected_shape[1:]:
        static_test = static_test[None]
    static_test = np.asarray(static_test).mean(axis=3)  # Trial Axis Mean -> (1, 72, N, 64, 250)

    # Load Pipelines & Precompute Embeddings
    sdxl_pipe, zero123_pipe = load_pipelines(args, device)
    init_global_embeddings(test_dataset, sdxl_pipe, args, device)

    # Load EEG-CLIP Aligner Model
    aligner_model = EEGCLIPAligner(input_dim=128, hidden_dim=1024, clip_dim=512).to(device)
    aligner_model.load_state_dict(torch.load(args.aligner_ckpt, map_location=device), strict=False)
    aligner_model.eval()

    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=4, drop_last=False)

    img_idx = 0
    evaluation_records = []
    seen_sample_ids = set()

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="EEG(Direct Tensor) -> SDXL -> Zero123++")):
        if 0 < args.max_samples <= img_idx:
            print(f"\n[*] Reached max_samples limit ({args.max_samples}). Stopping.")
            break

        batch_size = len(batch["cls_index"])

        # 1. Extract 128-dim temporal stats from raw EEG
        eeg_raw = []
        for i in range(batch_size):
            s, c, o = int(batch["subject_index"][i]), int(batch["cls_index"][i]), int(batch["obj_index"][i])
            eeg_raw.append(static_test[s, c, o])
        eeg_raw = np.stack(eeg_raw)  # [Batch, 64, 250]

        eeg_mean = torch.from_numpy(eeg_raw.mean(axis=-1)).float().to(device)
        eeg_std = torch.from_numpy(eeg_raw.std(axis=-1)).float().to(device)
        features = torch.cat([eeg_mean, eeg_std], dim=-1)  # [Batch, 128]

        # 2. Get 512-dim EEG-CLIP Embeddings
        eeg_embeds = aligner_model(features)  # [Batch, 512]

        # 3. Soft Prompt Interpolation (Cosine Similarity -> Softmax -> Weighted Sum of Tensors)
        logits = eeg_embeds @ CLIP_TEXT_EMBEDS_72.t()  # [Batch, 72]
        weights = F.softmax(logits * args.temperature, dim=-1)  # [Batch, 72]

        batch_prompt_embeds = torch.einsum("bc,csl->bsl", weights.to(SDXL_PROMPT_EMBEDS_72.dtype), SDXL_PROMPT_EMBEDS_72)
        batch_pooled_embeds = torch.einsum("bc,cl->bl", weights.to(SDXL_POOLED_EMBEDS_72.dtype), SDXL_POOLED_EMBEDS_72)

        # Evaluations (Argmax for reference)
        semantic_pred_cls = logits.argmax(dim=1).cpu().tolist()
        semantic_top5 = logits.topk(5, dim=1).indices.cpu().tolist()
        semantic_confidence = weights.max(dim=1).values.cpu().tolist()

        for i in range(batch_size):
            if 0 < args.max_samples <= img_idx:
                break

            dataset_name = str(batch["name"][i])
            label = str(batch["label"][i])
            cls_index, obj_index = int(batch["cls_index"][i]), int(batch["obj_index"][i])
            trial_index, subject_index = int(batch["trial_index"][i]), int(batch["subject_index"][i])

            base_sample_id = make_sample_id(label, trial_index)
            sample_id = base_sample_id if base_sample_id not in seen_sample_ids else f"{base_sample_id}__{img_idx:05d}"
            seen_sample_ids.add(sample_id)

            pred_label = str(test_dataset.name_list[semantic_pred_cls[i], obj_index])[3:]
            sample_seed = args.seed + img_idx

            # 4. SDXL Direct Tensor Generation
            single_pe = batch_prompt_embeds[i].unsqueeze(0)
            single_ppe = batch_pooled_embeds[i].unsqueeze(0)

            base_image_pil = run_sdxl_base_image_direct(
                sdxl_pipe, prompt_embeds=single_pe, pooled_prompt_embeds=single_ppe, args=args, seed=sample_seed
            )
            base_image_path = os.path.join(base_root, f"{sample_id}_base.png")
            base_image_pil.save(base_image_path)

            # 5. Zero123++ 6-View Generation
            cond_image_pil = prepare_zero123_input(base_image_pil, rembg_session)
            grid_pil = run_zero123plus(zero123_pipe, cond_image_pil, args, sample_seed)

            # 6. BG Remove & Slicing
            clean_pred = clean_generated_grid_background(pil_to_tensor(grid_pil), rembg_session,
                                                         args.bg_clean_alpha_threshold)

            render_grid_path = os.path.join(render_root, f"{sample_id}.png")
            save_tensor_image(clean_pred, render_grid_path)
            save_six_individual_views(clean_pred, Path(views_root) / sample_id)

            evaluation_records.append({
                "sample_id": sample_id, "dataset_name": dataset_name, "label": label,
                "class_prefix": str(batch["class_prefix"][i]), "cls_index": cls_index,
                "obj_index": obj_index, "trial_index": trial_index, "subject_index": subject_index,
                "semantic_pred_cls": semantic_pred_cls[i], "semantic_pred_label": pred_label,
                "semantic_top5": " ".join(map(str, semantic_top5[i])), "semantic_confidence": semantic_confidence[i],
                "semantic_correct": int(semantic_pred_cls[i] == cls_index),
                "semantic_top5_correct": int(cls_index in semantic_top5[i]),
                "base_image": base_image_path, "render_grid": render_grid_path
            })

            img_idx += 1
            del base_image_pil, cond_image_pil, grid_pil, clean_pred
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save CSV
    if evaluation_records:
        csv_path = os.path.join(args.out_dir, "evaluation_pairs.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=evaluation_records[0].keys())
            writer.writeheader()
            writer.writerows(evaluation_records)

    print(f"\n[*] INFERENCE COMPLETE. Total generated: {img_idx}")


if __name__ == "__main__":
    main()