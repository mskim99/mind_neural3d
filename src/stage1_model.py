import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
# fmri encoder
from src.encoder.autoencoder import fMRI_Encoder
from src.encoder.feature_aggregation import Feature_Aggregation, Clip_Header
# diffusion prior
from src.diffusion.fbdm import DiffusionNetwork, FBDM
# # 3d decoder
# import clip
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from torch import distributed as dist


class EEG_Encoder(nn.Module):
    def __init__(self, clip_dim=512):
        super().__init__()
        # 입력: [B, 64, 600]
        self.conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, clip_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(clip_dim),
            nn.GELU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, 64, 600]
        features = self.conv(x)  # [B, 512, 75] (길이가 75로 압축됨)

        # 1. 확산 모델(Diffusion Prior)을 위한 시퀀스 특징 [B, Seq_Len, Dim]
        eeg_embed = features.transpose(1, 2)  # [B, 75, 512]

        # 2. CLIP Contrastive Loss를 위한 글로벌 특징 [B, 1, Dim]
        eeg_feature = self.global_pool(features).transpose(1, 2)  # [B, 1, 512]

        return eeg_embed, eeg_feature


class MinD3D(nn.Module):
    def __init__(self, config, device=None, dataset=None):
        super(MinD3D, self).__init__()
        self.device = device

        # ---------------------------------------------------------
        # 기존 fMRI 인코더 및 어그리게이션 모듈 삭제 후 EEG 인코더 도입
        # ---------------------------------------------------------
        self.eeg_encoder = EEG_Encoder(clip_dim=512).to(device)

        # ---------------------------------------------------------
        # 1024차원의 Neuro-3D 비디오 피처를 512차원으로 투영
        # ---------------------------------------------------------
        self.video_proj = nn.Linear(1024, 512).to(device)

        # Diffusion prior
        d_prior_path = config['diff_prior_config_path']
        # ... (이하 Diffusion prior 및 CLIP 로드 코드는 기존과 동일) ...
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # fMRI encoder
        # self.fmri_encoder = fMRI_Encoder(config).to(device)
        self.fa_module = Feature_Aggregation(2048).to(device)
        self.clip_header = Clip_Header(2048).to(device)
        
        # Diffusion prior
        d_prior_path = config['diff_prior_config_path']
        self.d_prior = OmegaConf.load(d_prior_path)
        self.prior_network = DiffusionNetwork(**self.d_prior["diffusion_prior"])
        self.diffusion_prior = FBDM(
            net = self.prior_network,
            timesteps = self.d_prior["timesteps"],
            sample_timesteps = self.d_prior["sample_timesteps"],
            cond_drop_prob = self.d_prior["cond_drop_prob"],
        ).to(device)
        
        # Code for training
        # self.clip_path = config['3d_model']['clip']['model_name']
        # print("Loading CLIP Model from", self.clip_path)
        # self.CLIP_Model, self.img_transform = clip.load(self.clip_path, device=device)
        # self.CLIP_Model.float()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        '''
        optimizer_parameters = [
            {'params': [self.logit_scale]},
            {'params': [p for n, p in self.clip_header.named_parameters()]},
            {'params': [p for n, p in self.fmri_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.fmri_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.fa_module.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.fa_module.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        '''
        optimizer_parameters = [
            {'params': [self.logit_scale]},

            {'params': [p for n, p in self.eeg_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.eeg_encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.video_proj.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.video_proj.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.diffusion_prior.net.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in self.diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        '''
        for name, param in self.CLIP_Model.named_parameters():
            param.requires_grad = False
        '''
        self.opt = AdamW(params=optimizer_parameters,
            lr=float(config["training"]["lr"]),
            betas=(0.9, 0.95)
        )

        self.sche = get_linear_schedule_with_warmup(
            self.opt, 
            num_warmup_steps=config["training"]['warmup_iters'],
            num_training_steps=config["training"]["max_iters"]
        )

        # Generation Parameter
        self.points_batch_size = 100000
        self.threshold=config['test']['threshold']
        self.resolution0=config['generation']['resolution_0']
        self.upsampling_steps=config['generation']['upsampling_steps']
        self.sample=config['generation']['use_sampling']
        self.refinement_step=config['generation']['refinement_step']
        self.simplify_nfaces=config['generation']['simplify_nfaces']
        self.input_type=config['data']['input_type']
        self.padding=config['data']['padding']
        self.vol_bound = None
        self.with_normals = False
        # self.world_size = dist.get_world_size()
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    '''
    @torch.autocast("cuda", dtype=torch.float16)  # fp16 is option
    def get_loss(self, meta):
        x = meta["fmri"].cuda()
        b,f,w,h = x.size()

        # Encode the fMRI frames
        x = rearrange(x, 'b f w h -> (b f) 1 w h')
        x = self.fmri_encoder.forward_encoder_w_pred_all(x)
        x_cls,x = x[:,:1,:], x[:,1:,:]
        fmri_embed = self.fa_module(x, b, f)         # [B, 768, 512]
        fmri_feature = self.clip_header(x_cls, b, f) # [B,   1, 512]
        
        # calculate diffusion loss
        img = meta["img"].cuda()
        b,f_i,c,w,h = img.size()
        self.CLIP_Model.eval()
        with torch.no_grad(): 
            img = img.view(b*f_i, 3, 224, 224).contiguous()
            clip_img_feature = self.CLIP_Model.encode_image(img)
            clip_img_feature = clip_img_feature.view(b,f_i,512).contiguous()
            image_features = torch.mean(clip_img_feature, dim=1).view(b,512).contiguous()

        diff_loss, pred = self.diffusion_prior(fmri_embed = fmri_embed, image_embed = image_features)
        # contrastive loss
        clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.logit_scale.exp())
        return diff_loss, clip_loss, self.logit_scale
    '''


    @torch.autocast("cuda", dtype=torch.float16)
    def get_loss(self, meta):
        # 1. EEG 신호 로드 및 인코딩
        x = meta["eeg_data"].cuda().float()
        eeg_embed, eeg_feature = self.eeg_encoder(x)

        # 2. 실시간 인코딩 대신 Neuro-3D의 사전 추출 피처 활용
        # meta["color_video_fea"]는 [B, 1024] 형태의 텐서입니다.
        raw_video_features = meta["color_video_fea"].cuda().float()

        # Diffusion Prior의 입력 규격에 맞게 1024차원을 512차원으로 투영
        image_features = self.video_proj(raw_video_features)

        # 3. 확산 모델 및 대조 학습(Contrastive) Loss 계산
        diff_loss, pred = self.diffusion_prior(fmri_embed=eeg_embed, image_embed=image_features)
        clip_loss = self.cal_clip_loss(image_features, eeg_feature, self.logit_scale.exp())

        return diff_loss, clip_loss, self.logit_scale


    '''
    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.float16)  # fp16 is option
    def get_test_metric(self, meta):
        x = meta["fmri"].cuda()
        b,f,w,h = x.size()

        # Encode the fMRI frames
        x = rearrange(x, 'b f w h -> (b f) 1 w h')
        x = self.fmri_encoder.forward_encoder_w_pred_all(x)
        x_cls,x = x[:,:1,:], x[:,1:,:]
        fmri_embed = self.fa_module(x, b, f)         # [B, 768, 512]
        fmri_feature = self.clip_header(x_cls, b, f) # [B,   1, 512]
        
        # calculate diffusion loss
        img = meta["img"].cuda()
        b,f_i,c,w,h = img.size()
        img = img.view(b*f_i, 3, 224, 224).contiguous()
        clip_img_feature = self.CLIP_Model.encode_image(img)
        clip_img_feature = clip_img_feature.view(b, f_i, 512).contiguous()
        image_features = torch.mean(clip_img_feature,dim=1).view(b,512).contiguous()

        diff_loss, pred = self.diffusion_prior(fmri_embed = fmri_feature, image_embed = image_features)
        # contrastive loss
        clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.logit_scale.exp())

        return diff_loss, clip_loss
    '''


    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.float16)
    def get_test_metric(self, meta):
        x = meta["eeg_data"].cuda().float()
        eeg_embed, eeg_feature = self.eeg_encoder(x)

        raw_video_features = meta["color_video_fea"].cuda().float()
        image_features = self.video_proj(raw_video_features)

        diff_loss, pred = self.diffusion_prior(fmri_embed=eeg_embed, image_embed=image_features)
        clip_loss = self.cal_clip_loss(image_features, eeg_feature, self.logit_scale.exp())

        return diff_loss, clip_loss


    def cal_clip_loss(self, image_features, fmri_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_fmri = self.get_logits(image_features, fmri_features.squeeze(1), logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_fmri, labels)
        ) / 2
        return total_loss

    def get_logits(self, image_features, fmri_features, logit_scale):
        logits_per_image = logit_scale * image_features @ fmri_features.T
        logits_per_fmri = logit_scale * fmri_features @ image_features.T
        return logits_per_image, logits_per_fmri

    def backward(self, loss=None):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sche.step()
