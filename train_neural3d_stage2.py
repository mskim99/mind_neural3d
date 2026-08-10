import os
import math
import torch
import numpy as np
import argparse
import shutil
import time
from src.utils import load_config
from src.stage2_model import MinD3D
from src.utils import torch_init_model, set_random_seed, CheckpointIO
from src.data.fmri_shape import fmri_shape_object
from einops import rearrange
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from contextlib import nullcontext


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D mesh based on image input')
    parser.add_argument('--config', type=str, help='Path to config file.', default="./configs/mind3d.yaml")
    # 3D decoder
    parser.add_argument('--transformer_embed_dim', type=int, default=3072)
    parser.add_argument('--transformer_n_head', type=int, default=24)
    parser.add_argument('--transformer_layer', type=int, default=32)
    parser.add_argument('--topk', type=int, default=250)

    # Training
    parser.add_argument('--sub_id', type=str, default="0001")
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default="stage2_model")
    parser.add_argument('--check_point_path', type=str, default="mind3d_30k.pt")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")

    args = parser.parse_args()
    if args.ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        rank = dist.get_rank()

    cfg = load_config(args.config, 'configs/default.yaml')

    # initialize random seed
    seed=42
    set_random_seed(seed)

    # Output directory and copy the config file
    out_dir = os.path.join("./output", args.out_dir)
    if args.ddp and rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))
    elif not args.ddp:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

    # Setup device
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

    # load dataset
    train_dataset = fmri_shape_object(sub_id=args.sub_id, mode="train")
    train_sampler = None
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        num_workers=12,
        drop_last=True,
        sampler=train_sampler,
    )

    test_dataset = fmri_shape_object(sub_id=args.sub_id, mode="test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsize,
        num_workers=4,
        drop_last=False,
    )

    # load model
    model_full = MinD3D(cfg, device=device).cuda().to(torch.float16) # fp16 is option
    if args.ddp:
        model_full_ddp = torch.nn.parallel.DistributedDataParallel(model_full, device_ids=[args.local_rank], output_device=args.local_rank)
        model_full = model_full_ddp.module

    # define checkpoint IO
    # state_dict = torch.load(args.check_point_path, map_location='cpu')["model"]
    # torch_init_model(model_full, state_dict)
    checkpoint_io = CheckpointIO(out_dir, model=model_full)
    
    if args.ddp:
        if rank==0:
            logger = SummaryWriter(os.path.join(out_dir, 'logs'))
    else:
        logger = SummaryWriter(os.path.join(out_dir, 'logs'))


    print_every = cfg['training']['print_every']
    backup_every = cfg['training']['backup_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    epoch_it=0
    it=0
    vis_count = 0
    t0 = time.time()

    # 🚀 [추가] Validation 배치를 순차적으로 가져오기 위한 이터레이터
    test_iterator = iter(test_loader)

    while True:
        epoch_it += 1
        model_full.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch_it)
        for train_item in train_loader:
            it+=1
            mcontext = model_full_ddp.no_sync if rank != -1 and it % args.accumulation_steps != 0 else nullcontext

            with mcontext():
                z_loss, diff_loss = model_full.get_loss(train_item)
                # model_full.backward(clip_loss)
                loss = z_loss + diff_loss
                loss = loss / args.accumulation_steps
                loss.backward()

            if it % args.accumulation_steps == 0:
                model_full.opt.step() 
                model_full.opt.zero_grad()
                model_full.sche.step()

            if rank == 0:
                logger.add_scalar('train/z_loss', z_loss, it)
                logger.add_scalar('train/diff_loss', diff_loss, it)
                logger.add_scalar('lr', model_full.sche.get_lr()[0], it)

                if print_every > 0 and (it % print_every) == 0:
                    t = time.time() - t0
                    print('[Epoch %02d] it=%03d, Train: z_loss=%.4f, d_loss=%.4f, lr=%.8f, time: %.0fm %0.2fs'
                    % (epoch_it, it, z_loss, diff_loss, model_full.sche.get_lr()[0], t // 60, t % 60))

                # validate model   
                if validate_every > 0 and (it % validate_every) == 0:
                    model_full.eval()
                    test_z_loss = 0
                    for test_item in test_loader:
                        test_z_loss += model_full.get_test_metric(test_item)
                    print('[Epoch %02d] it=%03d, Test: z_loss=%.4f' % (
                        epoch_it, it, test_z_loss/104,
                    ))
                    logger.add_scalar('test/z_loss', test_z_loss/104, it)

                # save model
                if (backup_every > 0 and (it % backup_every) == 0):
                    checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it)

            # visualize
            if (visualize_every > 0 and (it % visualize_every) == 0):
                model_full.eval()
                vis_count+=1
                vis_items = next(sample_iterator)
                uid = vis_items["uid"][0]
                class_name = vis_items["class_name"][0]
                img_path = vis_items["img_path"][0]
                fmri_embedding = model_full.get_fmri_embedding_batch(vis_items, diffusion_bs=16, decoder_bs=1)
                mesh_list = model_full.generate_mesh_from_img_embedding(fmri_embedding, bs=1)
                
                # Get statistics
                save_dir = os.path.join(out_dir, "val_obj_rank{}".format(rank))
                os.makedirs(save_dir,exist_ok=True)
                for j in range(len(mesh_list)):
                    mesh = mesh_list[j]
                    if not mesh.vertices.shape[0]:
                        continue
                    mesh.export(os.path.join(save_dir, '{}_{}_{}_{}.obj'.format(vis_count, class_name, uid, j)))
                os.system("cp {} {}".format(
                    img_path,
                    os.path.join(save_dir, "{}_{}_{}.png".format(vis_count, class_name, uid))
                ))
            dist.barrier()

            # =========================================================
            # 🚀 [추가] 시각화 (Visualization) 모듈
            # image_620b2d.png의 구조에 맞추어 images와 images_val 폴더에 저장
            # =========================================================
            if (visualize_every > 0 and (it % visualize_every) == 0):
                if (args.ddp and rank == 0) or not args.ddp:
                    model_full.eval()

                    # 1. Validation 데이터 가져오기 (끝에 도달하면 초기화)
                    try:
                        val_items = next(test_iterator)
                    except StopIteration:
                        test_iterator = iter(test_loader)
                        val_items = next(test_iterator)

                    with torch.no_grad():
                        # 2. Validation 셋 시각화 -> 'images_val' 폴더에 저장
                        # (mvdiffusion_egg.py 내부에 정의된 validation_step 활용)
                        model_full.validation_step(val_items)

                        # 3. Train 셋 시각화 -> 'images' 폴더에 저장
                        # 현재 훈련 중인 배치를 사용하여 이미지를 생성하고 학습 진척도를 직접 비교
                        from torchvision.utils import make_grid, save_image

                        cond_eeg, target_imgs = model_full.prepare_batch_data(train_item)
                        # 추론 스텝을 50으로 주어 시각화 속도 확보
                        latent = model_full.pipeline(cond_eeg, num_inference_steps=50, output_type='latent').images

                        # VAE 디코딩 및 정규화 해제
                        images_pred = model_full.pipeline.vae.decode(
                            latent / model_full.pipeline.vae.config.scaling_factor, return_dict=False
                        )[0]
                        images_pred = (images_pred / 0.5 * 0.8)
                        images_pred = (images_pred * 0.5 + 0.5).clamp(0, 1)

                        # 원본(위)과 생성 결과(아래)를 하나의 그리드로 묶어서 저장
                        grid = make_grid(torch.cat([target_imgs, images_pred], dim=0), nrow=8, normalize=True,
                                         value_range=(0, 1))
                        save_image(grid, os.path.join(out_dir, 'images', f'train_{it:07d}.png'))

                    model_full.train()
            # =========================================================

            # (기존) 캐시 정리 코드
            # del loss, diff_loss, clip_loss, train_item
            del loss, diff_loss, train_item
            torch.cuda.empty_cache()