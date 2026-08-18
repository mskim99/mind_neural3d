import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, UNet2DConditionModel
from src.zero123plus.pipeline import RefOnlyNoisedUNet
from peft import LoraConfig, get_peft_model
from torch import distributed as dist
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
import open_clip
from safetensors.torch import load_file


# =========================================================================
# 1. Temporal Stem (시간적 특징 추출)
# =========================================================================
class TemporalStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, kernel_size=8, stride=4):
        super().__init__()
        # 각 EEG 채널을 독립적으로 처리하여 시간 축의 동태적 특징 추출
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x shape: [B * num_electrodes, 1, seq_len]
        x = self.conv(x)
        x = self.bn(x)
        return self.gelu(x)


# =========================================================================
# 2. Spatio-Temporal EEG GCN (시공간 그래프 합성곱)
# =========================================================================
class SpatioTemporalGCN(nn.Module):
    def __init__(self, num_electrodes=64, in_features=149, out_features=128):
        super().__init__()
        # 학습 가능한 전극 간 인접 행렬 (Learnable Adjacency Matrix)
        self.adj = nn.Parameter(torch.randn(num_electrodes, num_electrodes))
        self.fc = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x shape: [B, num_electrodes, in_features]
        adj_norm = F.softmax(self.adj, dim=-1)

        # Spatial Mixing: 전극 간 정보 교환
        out = torch.matmul(adj_norm, x)

        # Feature Transformation
        out = self.fc(out)
        return self.gelu(out)


# =========================================================================
# 3. Semantic-Variation Disentanglement Module (의미-변형 분리 모듈)
# =========================================================================
class DisentanglementModule(nn.Module):
    def __init__(self, combined_dim, embed_dim):
        super().__init__()

        # 🚨 기존의 nn.LayerNorm을 제거하고,
        # CLIP 공간으로 이동할 수 있도록 GELU와 Linear(Projection Head)를 덧붙입니다.
        self.semantic_routing = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.variation_routing = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.bias_filtering = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, mixed_features):
        # 3차원 축으로의 분리
        semantic_feature = self.semantic_routing(mixed_features)
        variational_features = self.variation_routing(mixed_features)
        subject_bias = self.bias_filtering(mixed_features)

        return semantic_feature, variational_features, subject_bias


# =========================================================================
# 4. 전체 통합 모델 (EEG Detangling Encoder + Disentanglement)
# =========================================================================
class EEG_Detangling_Disentanglement_Model(nn.Module):
    def __init__(self, num_electrodes=64, seq_len=600, embed_dim=1024):
        super().__init__()
        self.num_electrodes = num_electrodes

        # 1. Temporal Stem (채널 독립적 1D Conv)
        self.temporal_stem = TemporalStem(in_channels=1, out_channels=32)

        # 2. Spatio-Temporal EEG GCN
        # (seq_len 600, kernel 8, stride 4 적용 시 길이 149로 가정)
        temporal_out_len = ((seq_len - 8) // 4) + 1
        gcn_in_features = 32 * temporal_out_len
        self.gcn = SpatioTemporalGCN(num_electrodes=num_electrodes, in_features=gcn_in_features, out_features=128)

        # 3. Semantic-Variation Encoder (Combined Feature Embedding 생성)
        self.sv_encoder = nn.Sequential(
            nn.Linear(num_electrodes * 128, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 4. Disentanglement Module
        self.disentanglement = DisentanglementModule(combined_dim=embed_dim, embed_dim=embed_dim)

    def forward(self, eeg_sequence):
        B = eeg_sequence.shape[0]

        # eeg_sequence shape: [B, 64, 600]
        # 채널을 독립적인 배치처럼 처리하기 위해 변환
        x = eeg_sequence.view(B * self.num_electrodes, 1, -1)

        # 1. Temporal Stem
        x = self.temporal_stem(x)  # [B*64, 32, 149]
        x = x.view(B, self.num_electrodes, -1)  # [B, 64, 32 * 149]

        # 2. Spatio-Temporal EEG GCN
        x = self.gcn(x)  # [B, 64, 128]

        # 3. Semantic-Variation Encoder
        x = x.view(B, -1)  # Flatten: [B, 64 * 128]
        combined_feature = self.sv_encoder(x)  # [B, embed_dim]

        # 4. Disentanglement Module
        sem_feat, var_feat, bias_feat = self.disentanglement(combined_feature)

        return sem_feat, var_feat, bias_feat

    def get_orthogonal_loss(self, sem_feat, var_feat, bias_feat):
        """
        직교 투영(Orthogonal Projection)을 강제하기 위한 코사인 유사도 페널티
        세 특징 벡터 간의 내적(Dot Product)이 0에 가까워지도록 유도합니다.
        """
        sem_norm = F.normalize(sem_feat, dim=-1)
        var_norm = F.normalize(var_feat, dim=-1)
        bias_norm = F.normalize(bias_feat, dim=-1)

        loss_sv = torch.mean(torch.abs(torch.sum(sem_norm * var_norm, dim=-1)))
        loss_sb = torch.mean(torch.abs(torch.sum(sem_norm * bias_norm, dim=-1)))
        loss_vb = torch.mean(torch.abs(torch.sum(var_norm * bias_norm, dim=-1)))

        return loss_sv + loss_sb + loss_vb


# =========================================================================
# 1. 1D EEG Transformer 인코더 (End-to-End의 핵심)
# =========================================================================
class EEG_Transformer_Encoder(nn.Module):
    def __init__(self, in_channels=64, seq_len=600, patch_size=12, embed_dim=512, num_heads=8, depth=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = seq_len // patch_size

        # 🚀 1-A. 0이 아닌 정규분포(Random Normal)로 초기화하여 초기 기울기 확보
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # 🚀 1-B. 채널별 독립적인 정규화 (Instance-level Standardization) 추가
        # 극단적인 원시 뇌파 값을 평균 0, 분산 1로 안정화하여 모델에 주입
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

        x = self.patch_embed(x)
        x = x.transpose(1, 2)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)

        eeg_embed = x[:, 1:, :]  # Sequence (Spatial Mapping용)
        eeg_feature = x[:, 0:1, :]  # Global (Cross Attention용)
        return eeg_embed, eeg_feature


class SpatialLatentDecoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 4 * 4)  # 초기 저해상도 맵

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1)  # 64x64 (최종 4채널)
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        return self.decoder(x)


# =========================================================================
# 2. 유틸리티 함수
# =========================================================================
def scale_latents(latents): return (latents - 0.22) * 0.75


def unscale_latents(latents): return latents / 0.75 + 0.22


def scale_image(image): return image * 0.5 / 0.8


def unscale_image(image): return image / 0.5 * 0.8


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# =========================================================================
# 3. Main MVDiffusion Pipeline
# =========================================================================
class MVDiffusion(nn.Module):
    def __init__(self, args, stable_diffusion_config, drop_cond_prob=0.1, fmri_encoder_config=None, logdir=None):
        super(MVDiffusion, self).__init__()
        self.drop_cond_prob = drop_cond_prob
        self.register_schedule()

        if "pretrained_model_name_or_path" in stable_diffusion_config:
            stable_diffusion_config["pretrained_model_name_or_path"] = os.path.expanduser(
                stable_diffusion_config["pretrained_model_name_or_path"]
            )

        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        self.pipeline = pipeline

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(self.pipeline.unet, train_sched, self.pipeline.scheduler)

        self.train_scheduler = train_sched
        self.unet = pipeline.unet

        # ---------------------------------------------------------
        # 🚀 VAE 메모리 최적화 강제 적용
        # ---------------------------------------------------------
        self.pipeline.vae.requires_grad_(False)  # VAE 가중치 동결 재확인
        self.pipeline.vae.enable_slicing()  # 배치(Batch)를 쪼개서 연산
        self.pipeline.vae.enable_tiling()  # 거대한 이미지를 타일(Tile) 단위로 쪼개서 연산
        # ---------------------------------------------------------

        # 🚀 [추가] 우리가 새로 만든 EEG 분리 모듈을 초기화하여 장착합니다.
        # 앞서 작성했던 EEG_Detangling_Disentanglement_Model 클래스가
        # 같은 파일 내에 있거나 import 되어 있어야 합니다.
        self.fmri_encoder = EEG_Detangling_Disentanglement_Model(
            num_electrodes=64,
            seq_len=600,
            embed_dim=1024
        )

        print('Loading custom white-background unet ...')
        state_dict = load_file("/home/jionkim/.cache/huggingface/hub/models--sudo-ai--zero123plus-v1.2"
                               "/snapshots/2da07e89919e1a130c9b5add1584c70c7aa065fd/unet/"
                               "diffusion_pytorch_model.safetensors", device='cpu')
        new_ckpt = {"unet." + key: value for key, value in state_dict.items()}
        self.unet.load_state_dict(new_ckpt, strict=True)

        # LoRA Config 적용
        lora_config = LoraConfig(
            r=64, lora_alpha=64, lora_dropout=0.1,
            target_modules=["attn1.to_q", "attn1.to_v", "attn2.to_q", "attn2.to_v"],
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        # End-to-End 인코더 및 프로젝터
        self.eeg_encoder = EEG_Transformer_Encoder(in_channels=64, seq_len=600, patch_size=12, embed_dim=512,
                                                   num_heads=8, depth=4)
        # 🚀 2. 단순 Linear에서 LayerNorm을 포함한 Sequential로 변경 (붕괴 방지)
        self.clip_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        '''
        self.latent_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 4 * 64 * 64)
        )
        '''

        self.latent_proj = SpatialLatentDecoder(embed_dim=512)

        self.i_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.t_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # src/mvdiffusion_egg.py 의 __init__ 내부 옵티마이저 설정 부분
        lr = args.learning_rate  # (예: 1e-5)
        eeg_lr = 1e-4  # Transformer 전용 높은 학습률

        optimizer_parameters = [
            # 1. 확산 모델 및 LoRA (낮은 학습률)
            {'params': self.unet.parameters(), 'lr': lr},

            # 2. EEG 인코더 및 프로젝터 (높은 학습률)
            {'params': self.eeg_encoder.parameters(), 'lr': eeg_lr},
            {'params': self.clip_proj.parameters(), 'lr': eeg_lr},
            {'params': self.latent_proj.parameters(), 'lr': eeg_lr},

            # 3. CLIP Logit Scale
            {'params': [self.i_logit_scale, self.t_logit_scale], 'lr': eeg_lr}
        ]

        self.opt = AdamW(optimizer_parameters, betas=(0.9, 0.95))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=300, num_training_steps=100000)

        # DDP 호환 초기화
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.global_rank = self.rank
            self.device = f"cuda:{self.rank}"
        else:
            self.rank, self.world_size, self.global_rank = 0, 1, 0
            self.device = "cuda:0"

        self.global_step = 0
        self.logdir = logdir
        self.on_fit_start()

        self.clip_model, _, transform = open_clip.create_model_and_transforms(model_name="ViT-H-14",
                                                                              pretrained="laion2b_s32b_b79k")
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')

        # [추가] 대조 학습용 메모리 큐 (VRAM 소모 거의 없음)
        '''
        self.queue_size = 16
        self.register_buffer("eeg_queue", torch.randn(self.queue_size, 1024))
        self.register_buffer("img_queue", torch.randn(self.queue_size, 1024))
        self.eeg_queue = F.normalize(self.eeg_queue, dim=-1)
        self.img_queue = F.normalize(self.img_queue, dim=-1)
        self.queue_ptr = 0
        '''

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def get_clip_feature(self, color_video_fea, txt_fea):
        image_features = color_video_fea.cuda().to(torch.bfloat16)
        text_features = txt_fea.cuda().to(torch.bfloat16)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return image_features, text_features

    # [수정] 버전 충돌을 방지하는 안전한 Empty Text Embedding 생성
    @torch.no_grad()
    def get_empty_text_embeds(self, batch_size=1):
        text_inputs = self.pipeline.tokenizer(
            [""] * batch_size, padding="max_length", max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        prompt_embeds = self.pipeline.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return prompt_embeds.to(next(self.pipeline.unet.parameters()).dtype)

    def register_schedule(self):
        self.num_timesteps = 1000
        betas = torch.linspace(0.00085, 0.0120, 1000, dtype=torch.float32)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())

    def on_fit_start(self):
        self.pipeline.to(torch.device(f'cuda:{self.global_rank}'))
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)

    def prepare_batch_data(self, batch):
        dtype = next(self.eeg_encoder.parameters()).dtype
        cond_eeg = batch['eeg_data'].to(self.device).to(dtype)

        # [수정] 원본 이미지 훼손 없이 회색 배경 그대로 로드
        target_imgs = batch['rotation_images'].to(self.device).to(dtype)
        target_imgs = v2.functional.resize(target_imgs, 320, interpolation=3, antialias=True).clamp(0, 1)
        target_imgs = target_imgs[:, :6]
        target_imgs = rearrange(target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2)
        return cond_eeg, target_imgs

    def encode_embed_fmri_condition_fmri(self, eeg):
        dtype = next(self.eeg_encoder.parameters()).dtype
        B = eeg.shape[0]

        # 🚀 [수정] Autocast를 일시적으로 끄고 FP32로 인코더 및 프로젝터 연산 수행
        with torch.autocast("cuda", enabled=False):
            eeg_fp32 = eeg.to(torch.float32)

            # 1. 1D Transformer 기반 피처 추출
            eeg_embed, eeg_feature = self.eeg_encoder(eeg_fp32)

            # 2. Global Embedding (Cross-Attention)
            eeg_feature_1024 = self.clip_proj(eeg_feature)

            # 3. Spatial Latent Mapping (Zero123++ 6-view condition)
            pooled_embed = eeg_embed.mean(dim=1)
            latents = self.latent_proj(pooled_embed).view(B, 4, 64, 64)

        # 🚀 [추가] UNet(Zero123++)과 연동하기 위해 다시 모델의 원래 데이터 타입(bfloat16)으로 복구
        dtype = next(self.pipeline.unet.parameters()).dtype
        eeg_feature_1024 = eeg_feature_1024.to(dtype)
        latents = latents.to(dtype)

        # [수정] 안전한 Empty prompt 주입
        encoder_hidden_states = self.get_empty_text_embeds(B)
        ramp = eeg_feature_1024.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + eeg_feature_1024 * ramp

        return eeg_feature_1024, encoder_hidden_states, latents

    def forward_encoder_wo_pred_clip(self, eeg):
        # 🚀 [수정] 동일하게 FP32 연산 강제 적용
        with torch.autocast("cuda", enabled=False):
            eeg_fp32 = eeg.to(torch.float32)
            _, eeg_feature = self.eeg_encoder(eeg_fp32)
            eeg_feature_1024 = self.clip_proj(eeg_feature)

        # _, eeg_feature = self.eeg_encoder(eeg.to(dtype))
        # return self.clip_proj(eeg_feature)

        dtype = next(self.eeg_encoder.parameters()).dtype
        return eeg_feature_1024.to(dtype)

    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        images = (images - 0.5) / 0.8
        latents = self.pipeline.vae.encode(
            images.to(dtype)).latent_dist.sample() * self.pipeline.vae.config.scaling_factor
        return scale_latents(latents)

    def forward_unet(self, latents, t, prompt_embeds, cond_latents):
        return self.pipeline.unet(
            latents.to(next(self.pipeline.unet.parameters()).dtype), t,
            encoder_hidden_states=prompt_embeds.to(next(self.pipeline.unet.parameters()).dtype),
            cross_attention_kwargs=dict(cond_lat=cond_latents.to(next(self.pipeline.unet.parameters()).dtype)),
            return_dict=False,
        )[0]

    def get_v(self, x, noise, t):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_loss(self, batch, **kwargs):
        # ==========================================================
        # 1. 데이터 및 기본 세팅 준비
        # ==========================================================
        cond_eeg, target_imgs = self.prepare_batch_data(batch)
        B = cond_eeg.shape[0]
        color_video_fea, txt_fea = batch['color_video_fea'], batch['txt_fea']
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # ==========================================================
        # 2. 🌟 새로운 EEG 분리 모듈 적용 (Red Box)
        # ==========================================================
        # 원본 뇌파를 넣어 3가지 특징 분리 및 직교 손실 계산
        sem_feat, var_feat, bias_feat = self.fmri_encoder(cond_eeg)
        ortho_loss = self.fmri_encoder.get_orthogonal_loss(sem_feat, var_feat, bias_feat)

        # 향후 노란색 박스(Feature-to-Geometry)가 완성되면 사용될 결합 벡터
        combined_cond = sem_feat + var_feat

        # 🌟 [추가] CLIP Loss 계산을 위한 L2 정규화 (Gradient Vanishing 방지)
        combined_cond_norm = F.normalize(combined_cond, p=2, dim=-1)

        # ==========================================================
        # 3. 기존 인코더를 통한 프롬프트 추출 및 CLIP Loss 계산
        # ==========================================================
        self.pipeline.unet.training = True

        if np.random.rand() < self.drop_cond_prob:
            # [조건 제외 상황] 빈(Empty) 뇌파 텐서를 넣습니다.
            _, prompt_embeds, eeg_cond_latents = self.encode_embed_fmri_condition_fmri(torch.zeros_like(cond_eeg))

            # CLIP Loss 계산 시 기존 eeg_feature 대신, 빈 뇌파로 생성된 combined_cond(영벡터)를 흉내냅니다.
            empty_cond = torch.zeros_like(combined_cond)
            image_features, text_features = self.get_clip_feature(color_video_fea, txt_fea)
            # 🌟 [수정] 정규화된 벡터(combined_cond_norm)를 사용하여 CLIP Loss 계산
            clip_loss = self.cal_clip_loss(image_features, combined_cond_norm, self.i_logit_scale.exp()) + \
                        self.cal_clip_loss(text_features, combined_cond_norm, self.t_logit_scale.exp())

        else:
            _, prompt_embeds, eeg_cond_latents = self.encode_embed_fmri_condition_fmri(cond_eeg)
            image_features, text_features = self.get_clip_feature(color_video_fea, txt_fea)

            # cal_clip_loss (InfoNCE) 제거 -> Cosine Loss로 직접 교체
            # 두 벡터가 같은 방향을 가리키면 1, 반대면 -1을 반환합니다.
            # 1.0에서 이 값을 빼주면, 벡터가 정렬될수록 Loss가 0으로 떨어집니다.
            cos_sim_img = F.cosine_similarity(image_features, combined_cond, dim=-1)
            cos_sim_txt = F.cosine_similarity(text_features, combined_cond, dim=-1)

            # 평균을 내어 직접적인 스칼라 Loss 생성 (초기값 약 2.0에서 시작)
            clip_loss = (1.0 - cos_sim_img.mean()) + (1.0 - cos_sim_txt.mean())

        # ==========================================================
        # 4. UNet 노이즈 예측 및 Diffusion MSE Loss 계산
        # ==========================================================
        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)

        v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, eeg_cond_latents)
        v_target = self.get_v(latents, noise, t)

        diff_loss, loss_dict = self.compute_loss(v_pred, v_target)
        self.global_step += 1

        # ==========================================================
        # 5. 최종 산출된 3개의 Loss 반환
        # ==========================================================
        return diff_loss, clip_loss, ortho_loss


    def cal_clip_loss(self, target_features, eeg_features, logit_scale=None):
        # logit_scale은 기존 get_loss와의 인자 호환성을 위해 남겨둠 (사용하지 않음)
        eeg_features = eeg_features.squeeze(1)

        # 벡터 크기 정규화 (방향성만 비교하기 위함)
        eeg_features = F.normalize(eeg_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)

        # 🚀 [수정] 대조 학습 대신 코사인 유사도를 이용한 직접 회귀 방식
        # 방향이 완벽히 일치하면 1, 정반대면 -1을 반환함
        cos_sim = F.cosine_similarity(eeg_features, target_features, dim=-1)

        # Loss는 0에 가까울수록 좋으므로 (1 - 유사도)로 설정
        # 0 ~ 2 사이의 값을 가지며, 스케일을 키워(x 5.0) 기울기 전파를 돕습니다.
        loss = (1.0 - cos_sim.mean()) * 5.0

        return loss

    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)
        return loss, {'train/loss': loss}

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def validation_step(self, batch):
        self.pipeline.unet.training = False
        cond_eeg, target_imgs = self.prepare_batch_data(batch)

        latent = self.pipeline(cond_eeg, num_inference_steps=75, output_type='latent').images
        images = unscale_image(
            self.pipeline.vae.decode(latent / self.pipeline.vae.config.scaling_factor, return_dict=False)[0])
        images = (images * 0.5 + 0.5).clamp(0, 1)

        grid = make_grid(torch.cat([target_imgs, images], dim=0), nrow=8, normalize=True, value_range=(0, 1))
        save_image(grid, os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png'))