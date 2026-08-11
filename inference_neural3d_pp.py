import os
import torch
import argparse
from omegaconf import OmegaConf
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mvdiffusion_egg import MVDiffusion, unscale_image
from src.data.egg_dataset import AllDataFeatureTwoEEG


def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for End-to-End EEG-to-3D')
    parser.add_argument('--config', type=str, default="./configs/mind3d.yaml", help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the trained model checkpoint (e.g., model_3000.pt)')
    parser.add_argument('--out_dir', type=str, default="./inference_results", help='Directory to save generated images')
    parser.add_argument('--sub_id', type=str, default="0001", help='Subject ID for testing')
    parser.add_argument('--batchsize', type=int, default=2, help='Inference batch size')
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                        help='Classifier-Free Guidance scale (1.0 = no guidance)')
    parser.add_argument('--num_steps', type=int, default=75, help='Number of DDIM/Euler denoising steps')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

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
        num_frames=16
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=4, drop_last=False)

    # 3. 모델 초기화
    print("[*] Initializing MVDiffusion Model...")
    model = MVDiffusion(
        cfg,
        cfg.model.params.stable_diffusion_config,
        fmri_encoder_config=cfg.model.params.fmri_encoder_config,
        logdir=args.out_dir
    ).to(device)

    # 4. 체크포인트 로드 (DDP의 'module.' 접두어 자동 처리)
    print(f"[*] Loading weights from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    scheduler = model.pipeline.scheduler
    dtype = next(model.pipeline.unet.parameters()).dtype
    
    print(f"[*] Starting Inference (CFG Scale: {args.guidance_scale}, Steps: {args.num_steps})")

    img_idx = 0
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating")):
        cond_eeg, target_imgs = model.prepare_batch_data(batch)
        B = cond_eeg.shape[0]

        # ----------------------------------------------------------------
        # [조건 추출] EEG 기반 조건과 Unconditional(빈) 조건 추출 (CFG용)
        # ----------------------------------------------------------------
        # 1) 조건부 (Conditional)
        _, prompt_embeds_cond, latents_cond = model.encode_embed_fmri_condition_fmri(cond_eeg)

        # 2) 비조건부 (Unconditional) - 영점 텐서 주입
        _, prompt_embeds_uncond, latents_uncond = model.encode_embed_fmri_condition_fmri(torch.zeros_like(cond_eeg))

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

        # 정답(Target) 이미지와 예측(Pred) 이미지를 위아래로 이어 붙여 저장
        for i in range(B):
            grid = make_grid(torch.stack([target_imgs[i], images_pred[i]], dim=0), nrow=1, normalize=True,
                             value_range=(0, 1))
            save_path = os.path.join(args.out_dir, f"infer_{img_idx:05d}.png")
            save_image(grid, save_path)
            img_idx += 1

    print(f"[*] Inference Complete! Results saved to {args.out_dir}")


if __name__ == '__main__':
    main()