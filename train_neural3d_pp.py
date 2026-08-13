import os
import math
import torch
import numpy as np
import argparse
import shutil
import time
# from src.mvdiffusion_egg import MVDiffusion
from src.mvdiffusion_var import MVDiffusion
from src.utils import set_random_seed, CheckpointIO

# [수정] egg_dataset에서 새로운 데이터셋 클래스 임포트
from src.data.egg_dataset import AllDataFeatureTwoEEG

from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from contextlib import nullcontext
from omegaconf import OmegaConf

if __name__ == '__main__':

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser(description='Generate 3D mesh based on image input')
    parser.add_argument('--config', type=str, help='Path to config file.', default="./configs/mind3d.yaml")
    # Training
    parser.add_argument('--sub_id', type=str, default="0001")
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default="stage2_model")
    parser.add_argument('--check_point_path', type=str, default="mind3d_30k.pt")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")

    args = parser.parse_args()

    # [수정] DDP 모드가 아닐 때를 대비하여 기본 rank를 -1로 설정
    rank = -1
    local_rank = -1

    if args.ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()

    cfg = OmegaConf.load(args.config)

    # initialize random seed
    seed = 42
    set_random_seed(seed)

    # Output directory and copy the config file
    # out_dir = os.path.join("./output", args.out_dir)
    out_dir = os.path.join(args.out_dir)
    if (args.ddp and rank == 0) or not args.ddp:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

    # Setup device
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

    # [수정] 새로운 데이터셋 로드 부분 적용
    data_path = "/data/jionkim/neuro_3D/"  # 사용자 환경에 맞는 경로 설정
    sub_list = [args.sub_id]

    train_dataset = AllDataFeatureTwoEEG(
        data_path=data_path,
        sub_list=sub_list,
        train=True,
        num_frames=16
    )

    train_sampler = None
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
        sampler=train_sampler,
    )

    test_dataset = AllDataFeatureTwoEEG(
        data_path=data_path,
        sub_list=sub_list,
        train=False,
        num_frames=16
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsize,
        num_workers=4,
        drop_last=False,
    )

    # load model
    model_full = MVDiffusion(
        cfg,
        cfg.model.params.stable_diffusion_config,
        fmri_encoder_config=cfg.model.params.fmri_encoder_config,
        logdir=out_dir
    ).cuda()

    opt_fmri = torch.optim.AdamW(model_full.fmri_encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    # [수정] DDP 미사용 시 model_full_ddp를 None으로 초기화하여 참조 오류 방지
    model_full_ddp = None
    if args.ddp:
        model_full_ddp = torch.nn.parallel.DistributedDataParallel(model_full, device_ids=[local_rank],
                                                                   output_device=local_rank)
        model_full = model_full_ddp.module

    checkpoint_io = CheckpointIO(out_dir, model=model_full)

    # [수정] 로거 초기화 로직 단순화
    if (args.ddp and rank == 0) or not args.ddp:
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    print_every = cfg.training.print_every
    backup_every = cfg.training.backup_every
    validate_every = cfg.training.validate_every
    visualize_every = cfg.training.visualize_every

    test_iterator = iter(test_loader)

    epoch_it = 0
    it = 0
    t0 = time.time()

    while True:
        epoch_it += 1
        if args.ddp:
            train_loader.sampler.set_epoch(epoch_it)

        for train_item in train_loader:
            model_full.train()
            it += 1

            # [수정] args.ddp가 True일 때만 no_sync 컨텍스트를 사용하도록 안전하게 수정
            mcontext = model_full_ddp.no_sync if (args.ddp and it % args.accumulation_steps != 0) else nullcontext

            # mvdiffusion_egg
            '''
            with mcontext():
                diff_loss, clip_loss = model_full.get_loss(train_item)
                # loss = clip_loss
                # loss = diff_loss + clip_loss
                loss = diff_loss + (0.2 * clip_loss)
                loss = loss / args.accumulation_steps
                loss.backward()
            '''

            # mvdiffusion_var
            with mcontext():
                # 🌟 수정: get_loss가 이제 3개의 값을 반환합니다.
                diff_loss, clip_loss, ortho_loss = model_full.get_loss(train_item)

                # 🌟 각 손실(Loss)에 대한 스케일링 가중치 설정 (하이퍼파라미터)
                lambda_diff = 1.0  # MSE (픽셀/잠재 공간 생성 손실)
                lambda_clip = 1.0  # Semantic 정렬 손실
                lambda_ortho = 0.1  # 직교 제약 손실 (너무 크면 의미 공간이 붕괴될 수 있으므로 0.1 권장)

                # Total Loss 합산
                loss = (lambda_diff * diff_loss) + (lambda_clip * clip_loss) + (lambda_ortho * ortho_loss)

                # Gradient Accumulation 적용 및 역전파
                loss = loss / args.accumulation_steps
                loss.backward()

            if it % args.accumulation_steps == 0:
                model_full.opt.step()
                model_full.opt.zero_grad()
                model_full.sche.step()

                opt_fmri.step()
                opt_fmri.zero_grad()

            # 단일 GPU 및 DDP 환경 로그 분기 처리
            if (args.ddp and rank == 0) or not args.ddp:
                logger.add_scalar('train/diff_loss', diff_loss.item(), it)
                logger.add_scalar('train/clip_loss', clip_loss.item(), it)
                logger.add_scalar('lr', model_full.sche.get_lr()[0], it)

                # mvdiffusion_egg.py
                '''
                if print_every > 0 and (it % print_every) == 0:
                    t = time.time() - t0
                    print('[Epoch %02d] it=%03d, Train: diff_loss=%.4f, clip_loss=%.4f, lr=%.8f, time: %.0fm %0.2fs'
                          % (epoch_it, it, diff_loss.item(), clip_loss.item(), model_full.sche.get_lr()[0], t // 60, t % 60))
                '''

                # mvdiffusion_var.py
                # 로그 출력 부분 수정
                if (args.ddp and rank == 0) or not args.ddp:
                    if (print_every > 0 and (it % print_every) == 0):
                        print(
                            f"[Epoch {epoch_it}] it={it:07d}, "
                            f"Loss: {loss.item():.4f}, "
                            f"Diff: {diff_loss.item():.4f}, "
                            f"CLIP: {clip_loss.item():.4f}, "
                            f"Ortho: {ortho_loss.item():.4f}"
                        )

                # save model
                if (backup_every > 0 and (it % backup_every) == 0):
                    checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it)

            if args.ddp:
                dist.barrier()

            # =========================================================
            # 🚀 [수정] 시각화 (Visualization) 모듈 (에러 수정본)
            # =========================================================
            if (visualize_every > 0 and (it % visualize_every) == 0):
                if (args.ddp and rank == 0) or not args.ddp:
                    model_full.eval()

                    # 1. Validation 데이터 가져오기
                    try:
                        val_items = next(test_iterator)
                    except StopIteration:
                        test_iterator = iter(test_loader)
                        val_items = next(test_iterator)

                    from torchvision.utils import make_grid, save_image

                    # [내부 헬퍼 함수] 수동 디노이징 루프를 통한 이미지 생성
                    def generate_and_save(batch, save_name):
                        with torch.no_grad():
                            cond_eeg, target_imgs = model_full.prepare_batch_data(batch)
                            B = cond_eeg.shape[0]

                            # A. CFG를 위한 조건(Cond) / 비조건(Uncond) 피처 추출
                            _, prompt_embeds_cond, latents_cond = model_full.encode_embed_fmri_condition_fmri(cond_eeg)
                            _, prompt_embeds_uncond, latents_uncond = model_full.encode_embed_fmri_condition_fmri(torch.zeros_like(cond_eeg))

                            prompt_embeds = torch.cat([prompt_embeds_uncond, prompt_embeds_cond], dim=0)
                            eeg_cond_latents = torch.cat([latents_uncond, latents_cond], dim=0)

                            # B. 스케줄러 초기화 및 초기 노이즈(Latent) 생성
                            scheduler = model_full.pipeline.scheduler
                            scheduler.set_timesteps(50, device=cond_eeg.device)
                            dtype = next(model_full.unet.parameters()).dtype

                            # 960x640 해상도에 맞춘 120x80 잠재 텐서
                            latents = torch.randn((B, 4, 120, 80), device=cond_eeg.device, dtype=dtype)
                            latents = latents * scheduler.init_noise_sigma

                            # C. 수동 디노이징(Denoising) 루프 실행
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                for t in scheduler.timesteps:
                                    latent_input = torch.cat([latents] * 2)
                                    latent_input = scheduler.scale_model_input(latent_input, t)

                                    noise_pred = model_full.forward_unet(latent_input, t, prompt_embeds, eeg_cond_latents)
                                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                                    # CFG 가이던스 스케일 적용 (4.0)
                                    noise_pred = noise_pred_uncond + 4.0 * (noise_pred_cond - noise_pred_uncond)
                                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                            # D. VAE 디코딩 및 정규화 해제
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                images_pred = model_full.pipeline.vae.decode(
                                    latents / model_full.pipeline.vae.config.scaling_factor, return_dict=False
                                )[0]
                                images_pred = (images_pred / 0.5 * 0.8)
                                images_pred = (images_pred * 0.5 + 0.5).clamp(0, 1)

                            # E. 원본 타겟(위)과 생성된 이미지(아래)를 하나의 파일로 결합 후 저장
                            grid = make_grid(torch.cat([target_imgs, images_pred], dim=0), nrow=B, normalize=True, value_range=(0, 1))
                            save_image(grid, save_name)

                    # Validation 셋 (images_val 폴더)
                    generate_and_save(val_items, os.path.join(out_dir, 'images_val', f'val_{it:07d}.png'))

                    # Train 셋 (images 폴더)
                    generate_and_save(train_item, os.path.join(out_dir, 'images', f'train_{it:07d}.png'))

                    model_full.train()
            # =========================================================

            del loss, diff_loss, clip_loss, train_item
            torch.cuda.empty_cache()