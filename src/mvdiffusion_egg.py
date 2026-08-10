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

        self.latent_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 4 * 64 * 64)
        )

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
    def get_loss(self, batch):
        self.pipeline.unet.training = True
        cond_eeg, target_imgs = self.prepare_batch_data(batch)
        B = cond_eeg.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        color_video_fea, txt_fea = batch['color_video_fea'], batch['txt_fea']

        # Classifier-Free Guidance 적용 (조건부 드롭아웃)
        if np.random.rand() < self.drop_cond_prob:
            eeg_feature = self.forward_encoder_wo_pred_clip(cond_eeg)
            image_features, text_features = self.get_clip_feature(color_video_fea, txt_fea)
            clip_loss = self.cal_clip_loss(image_features, eeg_feature, self.i_logit_scale.exp()) + \
                        self.cal_clip_loss(text_features, eeg_feature, self.t_logit_scale.exp())

            prompt_embeds = self.get_empty_text_embeds(B)
            _, _, eeg_cond_latents = self.encode_embed_fmri_condition_fmri(torch.zeros_like(cond_eeg))
        else:
            eeg_feature, prompt_embeds, eeg_cond_latents = self.encode_embed_fmri_condition_fmri(cond_eeg)
            image_features, text_features = self.get_clip_feature(color_video_fea, txt_fea)
            clip_loss = self.cal_clip_loss(image_features, eeg_feature, self.i_logit_scale.exp()) + \
                        self.cal_clip_loss(text_features, eeg_feature, self.t_logit_scale.exp())

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)

        v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, eeg_cond_latents)
        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target)
        self.global_step += 1

        return loss, clip_loss

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