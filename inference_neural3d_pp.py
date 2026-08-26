import os
import torch
import argparse
import csv
import json
import shutil
import tempfile
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import rembg
except ImportError:
    rembg = None

from src.mvdiffusion_var_semantic_cls import MVDiffusion, unscale_image
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for End-to-End EEG-to-3D')
    parser.add_argument('--config', type=str, default="./configs/mind3d.yaml", help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the trained model checkpoint (e.g., model_3000.pt)')
    parser.add_argument('--out_dir', type=str, default="./inference_results", help='Directory to save generated images')
    parser.add_argument('--sub_id', type=str, default="0001", help='Subject ID for testing')
    parser.add_argument(
        '--rendered_view_path',
        type=str,
        default='/data/jionkim/neuro_3D/eeg3d_training',
        help='Same canonical 6-view dataset root used during training.'
    )
    parser.add_argument('--batchsize', type=int, default=2, help='Inference batch size')
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                        help='Classifier-Free Guidance scale (1.0 = no guidance)')
    parser.add_argument('--num_steps', type=int, default=75, help='Number of DDIM/Euler denoising steps')

    # ---------------------------------------------------------------------
    # Evaluation-aligned output
    # ---------------------------------------------------------------------
    parser.add_argument(
        '--gt_render_root',
        type=str,
        default='/data/jionkim/neuro_3D/render_grid_v4',
        help='GT six-view root. Expected: <root>/<label>/00.png ... 05.png'
    )
    parser.add_argument(
        '--allow_missing_gt',
        action='store_true',
        help='Do not stop if a predicted label has no matching GT six-view directory.'
    )

    # ---------------------------------------------------------------------
    # Experiment B: generated-view background cleanup before InstantMesh
    # ---------------------------------------------------------------------
    parser.add_argument(
        '--bg_clean',
        action='store_true',
        help='Deprecated compatibility flag. Background cleanup is always enabled '
             'because only cleaned outputs are saved.'
    )
    parser.add_argument(
        '--bg_clean_alpha_threshold',
        type=int,
        default=8,
        help='Set rembg alpha values below this 0-255 threshold to zero. Default: 8.'
    )
    parser.add_argument(
        '--bg_clean_binary_alpha',
        action='store_true',
        help='Use a hard binary foreground mask after rembg. '
             'Default keeps soft alpha edges.'
    )
    return parser.parse_args()



def _tensor_grid_to_pil(image_tensor):
    """
    [3,H,W] float tensor in [0,1] -> RGB PIL image.
    No normalization is applied.
    """
    x = image_tensor.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def _pil_to_tensor(image):
    """
    RGB PIL -> [3,H,W] float32 in [0,1].
    """
    arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _save_exact_tensor_image(image_tensor, path):
    """
    Save a [3,H,W] tensor directly as RGB PNG without make_grid/normalize.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _tensor_grid_to_pil(image_tensor).save(path)



def _canonical_label_from_dataset_name(name):
    """
    Original Neuro-3D dataset uses name[3:] as the object key for CLIP/point-cloud
    lookup. The GT renderer directory follows the same key, e.g.
        dataset name -> ...airplane_00
        label        -> airplane_00
    """
    name = str(name)
    if len(name) <= 3:
        raise ValueError(f"Dataset name is too short to derive label: {name!r}")
    return name[3:]


def split_grid_to_six_views(grid_tensor):
    """
    [3, H, W] packed grid -> list of six [3, H/3, W/2] tensors.

    Fixed canonical order:
        0 | 1
        2 | 3
        4 | 5

    NO resize / crop-to-object / recenter / flip / reorder.
    """
    if grid_tensor.ndim != 3 or grid_tensor.shape[0] != 3:
        raise ValueError(
            f"Expected RGB grid [3,H,W], got {tuple(grid_tensor.shape)}"
        )

    _, H, W = grid_tensor.shape
    if H % 3 != 0 or W % 2 != 0:
        raise ValueError(
            f"Grid must be divisible into 3x2 tiles, got H={H}, W={W}"
        )

    tile_h, tile_w = H // 3, W // 2
    views = []
    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w
        views.append(
            grid_tensor[:, y0:y1, x0:x1].contiguous()
        )
    return views


def save_six_individual_views(grid_tensor, output_dir):
    """
    Save a packed prediction in the SAME representation as GT renderer output:
        <output_dir>/00.png
        ...
        <output_dir>/05.png
    """
    os.makedirs(output_dir, exist_ok=True)
    views = split_grid_to_six_views(grid_tensor)

    saved = []
    for view_idx, view in enumerate(views):
        path = os.path.join(output_dir, f"{view_idx:02d}.png")
        _save_exact_tensor_image(view, path)
        saved.append(path)
    return saved


def validate_gt_view_dir(gt_root, label):
    """
    Return (<gt_dir>, <ok>, <missing_files>).
    """
    gt_dir = os.path.join(gt_root, label)
    missing = [
        f"{i:02d}.png"
        for i in range(6)
        if not os.path.isfile(os.path.join(gt_dir, f"{i:02d}.png"))
    ]
    return gt_dir, len(missing) == 0, missing


def save_sample_metadata(
    out_dir,
    sample_id,
    dataset_name,
    label,
    gt_dir,
    raw_view_dir,
    clean_view_dir,
    raw_grid_path,
    clean_grid_path,
):
    meta_dir = os.path.join(out_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    payload = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "label": label,
        "view_order": [
            ["00", "01"],
            ["02", "03"],
            ["04", "05"],
        ],
        "gt_dir": gt_dir,
        "raw_view_dir": raw_view_dir,
        "clean_view_dir": clean_view_dir,
        "raw_grid": raw_grid_path,
        "clean_grid": clean_grid_path,
    }

    path = os.path.join(meta_dir, f"{sample_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def write_evaluation_manifests(out_dir, records):
    """
    Write:
      inference_index.csv   : full label/path mapping
      eval_pairs_raw.csv    : GT six views <-> raw predicted six views
      eval_pairs_clean.csv  : GT six views <-> cleaned predicted six views

    These CSVs enforce one-to-one matching by `label`.
    """
    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(out_dir, "inference_index.csv")
    fieldnames = [
        "sample_id",
        "dataset_name",
        "label",
        "gt_dir",
        "pred_raw_dir",
        "pred_clean_dir",
        "raw_grid",
        "clean_grid",
    ]

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    raw_path = os.path.join(out_dir, "eval_pairs_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "label", "gt_dir", "pred_dir"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sample_id": r["sample_id"],
                "label": r["label"],
                "gt_dir": r["gt_dir"],
                "pred_dir": r["pred_raw_dir"],
            })

    clean_path = os.path.join(out_dir, "eval_pairs_clean.csv")
    with open(clean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "label", "gt_dir", "pred_dir"]
        )
        writer.writeheader()
        for r in records:
            if r.get("pred_clean_dir"):
                writer.writerow({
                    "sample_id": r["sample_id"],
                    "label": r["label"],
                    "gt_dir": r["gt_dir"],
                    "pred_dir": r["pred_clean_dir"],
                })

    return index_path, raw_path, clean_path




def write_evaluation_pairs(out_dir, records):
    """
    Root-level 1:1 GT/prediction mapping for evaluation.
    """
    csv_path = os.path.join(out_dir, "evaluation_pairs.csv")
    fieldnames = [
        "sample_id",
        "dataset_name",
        "label",
        "class_prefix",
        "cls_index",
        "obj_index",
        "trial_index",
        "subject_index",
        "gt_dir",
        "pred_dir",
        "render_grid",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return csv_path


def prepare_output_layout(out_dir):
    """
    Keep only two output directories:
        render/ : cleaned packed 3x2 grids for InstantMesh
        views/  : cleaned 00.png ... 05.png folders for evaluation
    """
    os.makedirs(out_dir, exist_ok=True)

    legacy_dirs = [
        "raw_for_instantmesh",
        "raw_views",
        "clean_for_instantmesh",
        "clean_views",
        "gt_pred",
        "bg_masks",
        "bg_compare",
        "metadata",
        "images",
        "images_val",
    ]
    for name in legacy_dirs:
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)

    render_dir = os.path.join(out_dir, "render")
    views_dir = os.path.join(out_dir, "views")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    return render_dir, views_dir



def clean_generated_grid_background(
    generated_grid,
    rembg_session,
    alpha_threshold=8,
    binary_alpha=False,
):
    """
    Experiment B.

    Input:
        generated_grid: [3, 960, 640] (3 rows x 2 columns of 320x320 views)

    Processing:
        1. Split into the canonical six fixed tiles.
        2. Run rembg independently on each tile.
        3. Composite foreground onto exact RGB=(255,255,255).
        4. Reassemble in the SAME locations/order.

    IMPORTANT:
        - NO crop
        - NO recenter
        - NO resize
        - NO view reordering

    Returns:
        clean_grid: [3,H,W], float32 [0,1]
        mask_grid:  [1,H,W], float32 [0,1]
    """
    if rembg is None:
        raise ImportError(
            "Experiment B requires `rembg`. Install it in this environment, e.g. "
            "`pip install rembg onnxruntime-gpu` (or onnxruntime for CPU)."
        )

    if generated_grid.ndim != 3 or generated_grid.shape[0] != 3:
        raise ValueError(
            f"Expected generated grid [3,H,W], got {tuple(generated_grid.shape)}"
        )

    _, H, W = generated_grid.shape
    if H % 3 != 0 or W % 2 != 0:
        raise ValueError(
            f"Generated grid must be divisible into 3x2 tiles, got H={H}, W={W}"
        )

    tile_h = H // 3
    tile_w = W // 2

    clean = torch.ones((3, H, W), dtype=torch.float32)
    masks = torch.zeros((1, H, W), dtype=torch.float32)

    threshold = float(alpha_threshold) / 255.0

    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w

        tile = generated_grid[:, y0:y1, x0:x1]
        tile_pil = _tensor_grid_to_pil(tile).convert('RGBA')

        rgba = rembg.remove(
            tile_pil,
            session=rembg_session,
            alpha_matting=False,
        )
        if not isinstance(rgba, Image.Image):
            rgba = Image.open(rgba).convert('RGBA')
        else:
            rgba = rgba.convert('RGBA')

        rgba_np = np.asarray(rgba, dtype=np.uint8)
        rgb = rgba_np[..., :3].astype(np.float32) / 255.0
        alpha = rgba_np[..., 3].astype(np.float32) / 255.0

        # Suppress tiny residual alpha values that can otherwise become
        # large faint shells after 3D reconstruction.
        alpha[alpha < threshold] = 0.0

        if binary_alpha:
            alpha = (alpha >= 0.5).astype(np.float32)

        # Exact white background. Preserve original tile coordinates.
        clean_rgb = rgb * alpha[..., None] + (1.0 - alpha[..., None])

        clean_tile = torch.from_numpy(clean_rgb).permute(2, 0, 1).contiguous()
        mask_tile = torch.from_numpy(alpha).unsqueeze(0).contiguous()

        clean[:, y0:y1, x0:x1] = clean_tile
        masks[:, y0:y1, x0:x1] = mask_tile

    return clean.clamp(0, 1), masks.clamp(0, 1)


def save_experiment_b_outputs(
    out_dir,
    sample_idx,
    raw_grid,
    clean_grid,
    mask_grid=None,
):
    """
    Save files in paths that can be passed directly to InstantMesh.

    raw_for_instantmesh/infer_XXXXX.png
    clean_for_instantmesh/infer_XXXXX.png
    bg_masks/infer_XXXXX.png
    bg_compare/infer_XXXXX.png
    """
    stem = f"infer_{sample_idx:05d}.png"

    raw_path = os.path.join(out_dir, "raw_for_instantmesh", stem)
    clean_path = os.path.join(out_dir, "clean_for_instantmesh", stem)

    _save_exact_tensor_image(raw_grid, raw_path)
    _save_exact_tensor_image(clean_grid, clean_path)

    if mask_grid is not None:
        mask_rgb = mask_grid.repeat(3, 1, 1)
        mask_path = os.path.join(out_dir, "bg_masks", stem)
        _save_exact_tensor_image(mask_rgb, mask_path)

    # Diagnostic only: left/raw and right/clean.
    # Do NOT feed this comparison image to InstantMesh.
    compare = torch.cat([raw_grid, clean_grid], dim=2)
    compare_path = os.path.join(out_dir, "bg_compare", stem)
    _save_exact_tensor_image(compare, compare_path)

    return raw_path, clean_path


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_root, views_root = prepare_output_layout(args.out_dir)

    # Background cleanup is always enabled in this output-focused version.
    args.bg_clean = True
    if rembg is None:
        raise ImportError(
            "Background cleanup is required. Install `rembg` and "
            "`onnxruntime-gpu` (or `onnxruntime`)."
        )
    print("[*] Background cleanup enabled: per-view rembg -> exact white composite")
    print("[*] IMPORTANT: no crop / no recenter / no resize / no view reorder")
    rembg_session = rembg.new_session()

    # Keep model runtime folders outside args.out_dir.
    runtime_logdir = tempfile.mkdtemp(prefix="neural3d_infer_")

    # =========================================================================
    # 🚀 [수정] 체크포인트가 위치한 폴더의 백업 config.yaml을 강제로 불러오도록 로직 변경
    # =========================================================================
    ckpt_dir = os.path.dirname(args.ckpt_path)
    backup_config_path = os.path.join(ckpt_dir, "config.yaml")

    if os.path.exists(backup_config_path):
        print(f"[*] Loading matched training config from {backup_config_path}")
        cfg = OmegaConf.load(backup_config_path)
    else:
        print(f"[*] Backup config not found. Falling back to {args.config}")
        cfg = OmegaConf.load(args.config)
    # =========================================================================

    # 2. 데이터셋 및 데이터로더 준비 (Validation/Test Set)
    print(f"[*] Preparing Test Dataset for subject {args.sub_id}")
    data_path = "/data/jionkim/neuro_3D/"
    test_dataset = AllDataFeatureTwoEEG(
        data_path=data_path,
        sub_list=[args.sub_id],
        train=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=4, drop_last=False)

    # 3. 모델 초기화
    print("[*] Initializing MVDiffusion Model...")
    model = MVDiffusion(
        cfg,
        cfg.model.params.stable_diffusion_config,
        fmri_encoder_config=cfg.model.params.fmri_encoder_config,
        logdir=runtime_logdir
    ).to(device)

    # 4. 체크포인트 로드 (DDP의 'module.' 접두어 자동 처리)
    print(f"[*] Loading weights from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # IMPORTANT:
    # This inference must use the exact same architecture as training.
    # strict=False previously allowed an incompatible checkpoint to load silently,
    # which can produce semantically unrelated objects while filenames/labels look correct.
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if missing or unexpected:
        print("\n[CHECKPOINT MISMATCH]")
        print(f"  missing keys    : {len(missing)}")
        for k in missing[:30]:
            print(f"    MISSING    {k}")
        print(f"  unexpected keys : {len(unexpected)}")
        for k in unexpected[:30]:
            print(f"    UNEXPECTED {k}")
        raise RuntimeError(
            "Checkpoint/model architecture mismatch. "
            "Use the checkpoint trained with "
            "src.mvdiffusion_var_semvar_connected.MVDiffusion."
        )

    print("[*] Checkpoint load verified: 0 missing / 0 unexpected keys")
    model.eval()
    scheduler = model.pipeline.scheduler
    dtype = next(model.pipeline.unet.parameters()).dtype
    
    print(f"[*] Starting Inference (CFG Scale: {args.guidance_scale}, Steps: {args.num_steps})")

    img_idx = 0
    eval_records = []
    seen_labels = set()

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating")):
        cond_eeg, target_imgs = model.prepare_batch_data(batch)
        B = cond_eeg.shape[0]

        # ----------------------------------------------------------------
        # [조건 추출] EEG 기반 조건과 Unconditional(빈) 조건 추출 (CFG용)
        # ----------------------------------------------------------------
        # 1) Conditional: real EEG -> semantic cross-attn + variation spatial latent
        _, prompt_embeds_cond, latents_cond = model.encode_embed_fmri_condition_fmri(
            cond_eeg,
            drop_condition=False,
        )

        # 2) Unconditional: use the SAME real EEG encoder path but explicitly
        # drop generation conditioning. Do NOT pass zero EEG through the encoder.
        _, prompt_embeds_uncond, latents_uncond = model.encode_embed_fmri_condition_fmri(
            cond_eeg,
            drop_condition=True,
        )

        # 배치(Batch) 축으로 결합 [2*B, ...]
        prompt_embeds = torch.cat([prompt_embeds_uncond, prompt_embeds_cond], dim=0)
        eeg_cond_latents = torch.cat([latents_uncond, latents_cond], dim=0)

        # ----------------------------------------------------------------
        # [초기 노이즈 생성] Zero123++의 출력 해상도는 3x2 그리드 형태
        # (전체 이미지 960x640 -> 잠재 공간 120x80)
        # ----------------------------------------------------------------
        scheduler.set_timesteps(args.num_steps, device=device)
        latents = torch.randn((B, 4, 960 // 8, 640 // 8), device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma

        # ----------------------------------------------------------------
        # [디노이징 루프]
        # ----------------------------------------------------------------
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in scheduler.timesteps:
                # CFG 연산을 위해 잠재 벡터 2배 확장
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # UNet 예측
                noise_pred = model.forward_unet(latent_model_input, t, prompt_embeds, eeg_cond_latents)

                # CFG 스케일 적용 (Uncond + scale * (Cond - Uncond))
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # 스케줄러 스텝 진행
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        # ----------------------------------------------------------------
        # [VAE 디코딩 및 저장]
        # ----------------------------------------------------------------
        with torch.autocast("cuda", dtype=torch.bfloat16):
            images_pred = \
            model.pipeline.vae.decode(latents / model.pipeline.vae.config.scaling_factor, return_dict=False)[0]
            images_pred = unscale_image(images_pred)
            images_pred = (images_pred * 0.5 + 0.5).clamp(0, 1)

        # ----------------------------------------------------------------
        # Save raw prediction + Experiment B background-clean prediction
        # ----------------------------------------------------------------
        for i in range(B):
            raw_pred = images_pred[i].detach().float().cpu().clamp(0, 1)

            # --------------------------------------------------------------
            # 1) Stable one-to-one object label.
            # --------------------------------------------------------------
            if 'name' not in batch:
                raise KeyError(
                    "Dataset batch has no `name`. A stable label is required "
                    "for one-to-one GT/prediction evaluation."
                )

            dataset_name = str(batch['name'][i])

            if 'label' not in batch:
                raise KeyError(
                    "Explicit `label` missing from dataset batch. "
                    "Use src.data.egg_dataset_ext_explicit_label."
                )

            # IMPORTANT: do not infer the label from filename slicing here.
            # The dataset item that supplied cond_eeg also supplies its label.
            label = str(batch['label'][i])
            class_prefix = str(batch['class_prefix'][i])
            cls_index = int(batch['cls_index'][i])
            obj_index = int(batch['obj_index'][i])
            trial_index = int(batch['trial_index'][i])
            subject_index = int(batch['subject_index'][i])

            print(
                f"[MAP] batch={batch_idx} item={i} "
                f"sub={subject_index} cls={cls_index} obj={obj_index} "
                f"trial={trial_index} prefix={class_prefix} "
                f"name={dataset_name} -> label={label}"
            )

            # Internal consistency check: the explicit label must still equal
            # the object key used by the dataset for CLIP / rendered-view lookup.
            expected_label = dataset_name[3:]
            if label != expected_label:
                raise RuntimeError(
                    f"Dataset mapping inconsistency: explicit label={label!r}, "
                    f"name[3:]={expected_label!r}, name={dataset_name!r}"
                )

            if label not in seen_labels:
                sample_id = label
                seen_labels.add(label)
            else:
                sample_id = f"{label}__{img_idx:05d}"

            gt_dir, gt_ok, missing_gt = validate_gt_view_dir(
                args.gt_render_root, label
            )
            if not gt_ok and not args.allow_missing_gt:
                raise FileNotFoundError(
                    f"No 1:1 GT match for EEG sample label '{label}'.\n"
                    f"Expected: {gt_dir}/00.png ... 05.png\n"
                    f"Missing: {missing_gt}\n"
                    "Use --allow_missing_gt only if intentional."
                )

            # --------------------------------------------------------------
            # 2) Background cleanup.
            # --------------------------------------------------------------
            clean_pred, mask_grid = clean_generated_grid_background(
                raw_pred,
                rembg_session=rembg_session,
                alpha_threshold=args.bg_clean_alpha_threshold,
                binary_alpha=args.bg_clean_binary_alpha,
            )

            # --------------------------------------------------------------
            # 3) render/: packed cleaned 3x2 grid for InstantMesh.
            # --------------------------------------------------------------
            render_grid_path = os.path.join(
                render_root, f"{sample_id}.png"
            )
            _save_exact_tensor_image(clean_pred, render_grid_path)

            # --------------------------------------------------------------
            # 4) views/: individual cleaned six views for evaluation.
            #    views/<label>/00.png ... 05.png
            # --------------------------------------------------------------
            pred_view_dir = os.path.join(views_root, label)
            save_six_individual_views(clean_pred, pred_view_dir)

            if img_idx < 5 or img_idx % 50 == 0:
                fg_ratio = float(mask_grid.mean().item())
                bg_pixels = clean_pred[:, mask_grid[0] < 1e-6]
                if bg_pixels.numel() > 0:
                    max_bg_error = float(
                        (bg_pixels - 1.0).abs().max().item()
                    )
                else:
                    max_bg_error = 0.0

                print(
                    f"[SAVE] {sample_id}: label={label}, "
                    f"fg_ratio={fg_ratio:.4f}, "
                    f"pure_white_bg_max_error={max_bg_error:.6f}\n"
                    f"       GT     : {gt_dir}\n"
                    f"       render : {render_grid_path}\n"
                    f"       views  : {pred_view_dir}"
                )

            eval_records.append({
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "label": label,
                "class_prefix": class_prefix,
                "cls_index": cls_index,
                "obj_index": obj_index,
                "trial_index": trial_index,
                "subject_index": subject_index,
                "gt_dir": gt_dir,
                "pred_dir": pred_view_dir,
                "render_grid": render_grid_path,
            })

            img_idx += 1

    evaluation_csv = write_evaluation_pairs(
        args.out_dir, eval_records
    )

    shutil.rmtree(runtime_logdir, ignore_errors=True)

    print(f"[*] Inference Complete! Results saved to {args.out_dir}")
    print(f"    InstantMesh grids : {render_root}")
    print(f"    Six-view folders  : {views_root}")
    print(f"    Evaluation pairs  : {evaluation_csv}")


if __name__ == '__main__':
    main()