"""
Final EEG semantic architecture:
- fixed category prototype supervision only (plus optional Ortho in trainer)
- small EEG-semantic -> CLIP bottleneck projection
- Stage-2 shared EEG trunk frozen
- diffusion semantic gradient stopped at z_clip
- no learned class CE, no SupCon, no AvgCons, no CLIP cosine objective
"""
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

    def encode_shared(self, eeg_sequence):
        """Shared EEG representation before semantic/variation/bias routing."""
        B = eeg_sequence.shape[0]
        x = eeg_sequence.view(B * self.num_electrodes, 1, -1)
        x = self.temporal_stem(x)
        x = x.view(B, self.num_electrodes, -1)
        x = self.gcn(x)
        x = x.view(B, -1)
        return self.sv_encoder(x)

    def route_features(self, combined_feature):
        return self.disentanglement(combined_feature)

    def forward(self, eeg_sequence):
        combined_feature = self.encode_shared(eeg_sequence)
        return self.route_features(combined_feature)

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
    def __init__(self, args, stable_diffusion_config, drop_cond_prob=0.1, fmri_encoder_config=None, logdir=None, num_classes=72, cls_label_smoothing=0.05):
        super(MVDiffusion, self).__init__()
        self.drop_cond_prob = drop_cond_prob
        self.num_classes = int(num_classes)
        self.cls_label_smoothing = float(cls_label_smoothing)
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

        # 24-GB class GPUs: trade compute for activation memory during training.
        base_unet = getattr(self.unet, "base_model", self.unet)
        if hasattr(base_unet, "enable_gradient_checkpointing"):
            base_unet.enable_gradient_checkpointing()
        elif hasattr(self.pipeline.unet, "enable_gradient_checkpointing"):
            self.pipeline.unet.enable_gradient_checkpointing()

        # =========================================================
        # Final semantic structure
        # =========================================================
        # EEG-specific semantic representation is NOT assumed to already live
        # in CLIP coordinates. A deliberately small bottleneck maps it into the
        # frozen 72-category CLIP prototype space.
        self.semantic_to_clip = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.Linear(128, 1024),
        )

        # CLIP-aligned semantic feature -> Zero123++ cross-attention.
        self.semantic_to_cross = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
        )
        nn.init.eye_(self.semantic_to_cross[1].weight)
        nn.init.zeros_(self.semantic_to_cross[1].bias)

        # variation -> Zero123++ spatial latent conditioning.
        self.variation_to_latent = SpatialLatentDecoder(embed_dim=1024)

        # Frozen category geometry. Trainer overwrites this with train-only
        # prototypes before optimization; fixed shape keeps checkpoint loading
        # simple for inference.
        self.register_buffer(
            "fixed_text_prototypes",
            torch.zeros(self.num_classes, 1024, dtype=torch.float32),
            persistent=True,
        )

        # Keep exact PEFT/LoRA names so only LoRA is restored in Stage-2.
        self._unet_lora_trainable_names = {
            name for name, p in self.unet.named_parameters() if p.requires_grad
        }

        # Optimizer is owned by the new trainer. These aliases remain None so
        # older utilities fail visibly rather than silently stepping stale state.
        self.opt = None
        self.sche = None

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

        # Stage-2 always blocks diffusion gradients from the semantic branch.
        # The shared EEG trunk is frozen in Stage-2; only the routing heads may
        # adapt, with semantic routing updated by ProtoCE and variation routing
        # updated by diffusion.
        self._stage2_debug_printed = False

        self.on_fit_start()

        # [추가] 대조 학습용 메모리 큐 (VRAM 소모 거의 없음)
        '''
        self.queue_size = 16
        self.register_buffer("eeg_queue", torch.randn(self.queue_size, 1024))
        self.register_buffer("img_queue", torch.randn(self.queue_size, 1024))
        self.eeg_queue = F.normalize(self.eeg_queue, dim=-1)
        self.img_queue = F.normalize(self.img_queue, dim=-1)
        self.queue_ptr = 0
        '''

    def train(self, mode=True):
        """
        Keep the Stage-2 shared EEG encoder in eval mode so BatchNorm running
        statistics cannot drift after the object-invariant Stage-1 checkpoint
        has been selected. Routing Linear layers still receive gradients in
        eval mode.
        """
        super().train(mode)
        if mode and getattr(self, "training_stage", None) == "joint":
            self.fmri_encoder.eval()
        return self

    def set_fixed_prototypes(self, prototypes):
        prototypes = torch.as_tensor(
            prototypes, dtype=torch.float32, device=self.fixed_text_prototypes.device
        )
        if tuple(prototypes.shape) != (self.num_classes, 1024):
            raise ValueError(
                f"Expected fixed prototypes [{self.num_classes},1024], "
                f"got {tuple(prototypes.shape)}"
            )
        prototypes = F.normalize(prototypes, dim=-1)
        self.fixed_text_prototypes.copy_(prototypes)
        self.fixed_text_prototypes.requires_grad_(False)

    def set_training_stage(self, stage):
        """
        semantic:
          train EEG shared encoder + semantic routing + semantic_to_clip.
          generation modules are frozen; only ProtoCE (+ optional Ortho in
          trainer) supplies semantic supervision.

        joint:
          freeze the shared EEG trunk learned in Stage-1.
          semantic_routing + semantic_to_clip remain trainable by ProtoCE only.
          variation_routing is trainable by diffusion from the frozen shared
          representation. Bias routing remains frozen.
          diffusion trains semantic_to_cross, variation_to_latent and LoRA.
        """
        if stage not in {"semantic", "joint"}:
            raise ValueError(f"Unknown training stage: {stage}")

        self.training_stage = stage

        # Start fully frozen, then selectively enable exactly what each stage uses.
        self.requires_grad_(False)
        self.fixed_text_prototypes.requires_grad_(False)

        if stage == "semantic":
            self.fmri_encoder.requires_grad_(True)
            self.semantic_to_clip.requires_grad_(True)

        else:
            # Shared trunk frozen to preserve Stage-1 object-invariant features.
            self.fmri_encoder.requires_grad_(False)
            self.fmri_encoder.disentanglement.semantic_routing.requires_grad_(True)
            self.fmri_encoder.disentanglement.variation_routing.requires_grad_(True)
            # bias_filtering intentionally stays frozen.

            self.semantic_to_clip.requires_grad_(True)
            self.semantic_to_cross.requires_grad_(True)
            self.variation_to_latent.requires_grad_(True)

            for name, p in self.unet.named_parameters():
                p.requires_grad_(name in self._unet_lora_trainable_names)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[stage={stage}] trainable params: "
            f"{trainable / 1e6:.2f}M / {total / 1e6:.2f}M"
        )

    def _prototype_loss(self, clip_sem, cls_target, temperature):
        if temperature <= 0:
            raise ValueError("prototype_temperature must be > 0")
        z = F.normalize(clip_sem.float(), dim=-1)
        p = F.normalize(self.fixed_text_prototypes.float(), dim=-1)
        cosine = z @ p.t()
        logits = cosine / float(temperature)
        loss = F.cross_entropy(logits, cls_target)

        with torch.no_grad():
            pred = cosine.argmax(dim=1)
            rows = torch.arange(len(cls_target), device=cls_target.device)
            correct = cosine[rows, cls_target]
            wrong = cosine.clone()
            wrong[rows, cls_target] = -torch.inf
            best_wrong = wrong.max(dim=1).values
            margin = correct - best_wrong

        return loss, {
            "top1": (pred == cls_target).float().mean(),
            "margin": margin.mean(),
            "correct_cosine": correct.mean(),
        }

    def encode_semantic_features(self, eeg):
        """EEG -> raw semantic/variation/bias + CLIP-aligned semantic."""
        with torch.autocast("cuda", enabled=False):
            eeg = eeg.to(torch.float32)
            combined = self.fmri_encoder.encode_shared(eeg)
            sem_feat, var_feat, bias_feat = self.fmri_encoder.route_features(combined)
            clip_sem = self.semantic_to_clip(sem_feat.float())
        return sem_feat, var_feat, bias_feat, clip_sem

    def _semantic_forward(self, batch, prototype_temperature=0.07):
        dtype = next(self.fmri_encoder.parameters()).dtype
        cond_eeg = batch["eeg_data"].to(
            self.device, dtype=dtype, non_blocking=True
        )
        if "cls_index" not in batch:
            raise KeyError("Batch is missing explicit `cls_index`.")
        cls_target = batch["cls_index"].to(
            self.device, dtype=torch.long, non_blocking=True
        )

        sem_feat, var_feat, bias_feat, clip_sem = (
            self.encode_semantic_features(cond_eeg)
        )
        proto_loss, proto_metrics = self._prototype_loss(
            clip_sem, cls_target, prototype_temperature
        )
        ortho_loss = self.fmri_encoder.get_orthogonal_loss(
            sem_feat, var_feat, bias_feat
        )

        return {
            "cond_eeg": cond_eeg,
            "cls_target": cls_target,
            "sem_feat": sem_feat,
            "var_feat": var_feat,
            "bias_feat": bias_feat,
            "clip_sem": clip_sem,
            "proto_loss": proto_loss,
            "proto_top1": proto_metrics["top1"],
            "proto_margin": proto_metrics["margin"],
            "proto_correct_cosine": proto_metrics["correct_cosine"],
            "ortho_loss": ortho_loss,
        }

    def forward(self, batch, stage="joint", prototype_temperature=0.07):
        sem = self._semantic_forward(
            batch, prototype_temperature=prototype_temperature
        )

        if stage == "semantic":
            zero = sem["proto_loss"].new_zeros(())
            return {
                "diff_loss": zero,
                "proto_loss": sem["proto_loss"],
                "proto_top1": sem["proto_top1"],
                "proto_margin": sem["proto_margin"],
                "proto_correct_cosine": sem["proto_correct_cosine"],
                "ortho_loss": sem["ortho_loss"],
            }

        if stage != "joint":
            raise ValueError(f"Unknown stage: {stage}")

        _, target_imgs = self.prepare_batch_data(batch)
        B = sem["cond_eeg"].shape[0]
        t = torch.randint(
            0, self.num_timesteps, size=(B,), device=self.device
        ).long()
        drop_condition = bool(
            torch.rand((), device=sem["cond_eeg"].device) < self.drop_cond_prob
        )

        # Critical isolation:
        # - diffusion NEVER updates semantic_routing / semantic_to_clip.
        # - shared EEG trunk is frozen by set_training_stage("joint").
        # - variation_routing is allowed to learn from diffusion, but its input
        #   originates from the frozen shared trunk.
        clip_for_diff = sem["clip_sem"].detach()
        var_for_diff = sem["var_feat"]

        if not self._stage2_debug_printed:
            print(
                "[joint-grad-boundary] "
                f"clip_source_grad={sem['clip_sem'].requires_grad}, "
                f"clip_for_diff_grad={clip_for_diff.requires_grad}, "
                f"var_for_diff_grad={var_for_diff.requires_grad}"
            )
            self._stage2_debug_printed = True

        _, prompt_embeds, eeg_cond_latents = (
            self.build_zero123_condition_from_features(
                clip_for_diff,
                var_for_diff,
                drop_condition=drop_condition,
            )
        )

        self.pipeline.unet.training = True
        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)

        v_pred = self.forward_unet(
            latents_noisy, t, prompt_embeds, eeg_cond_latents
        )
        v_target = self.get_v(latents, noise, t)
        diff_loss, _ = self.compute_loss(v_pred, v_target)

        self.global_step += 1
        return {
            "diff_loss": diff_loss,
            "proto_loss": sem["proto_loss"],
            "proto_top1": sem["proto_top1"],
            "proto_margin": sem["proto_margin"],
            "proto_correct_cosine": sem["proto_correct_cosine"],
            "ortho_loss": sem["ortho_loss"],
        }

    @torch.no_grad()
    def semantic_validation(self, batch, prototype_temperature=0.07):
        was_training = self.training
        self.eval()
        out = self._semantic_forward(
            batch, prototype_temperature=prototype_temperature
        )
        result = {
            "proto_loss": out["proto_loss"].detach(),
            "proto_top1": out["proto_top1"].detach(),
            "proto_margin": out["proto_margin"].detach(),
            "proto_correct_cosine": out["proto_correct_cosine"].detach(),
            "ortho_loss": out["ortho_loss"].detach(),
        }
        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def get_clip_feature(self, color_video_fea, txt_fea):
        # Features are already precomputed by the dataset. No OpenCLIP model is
        # needed in memory during training. Keep this small alignment loss in FP32.
        image_features = color_video_fea.to(self.device, dtype=torch.float32, non_blocking=True)
        text_features = txt_fea.to(self.device, dtype=torch.float32, non_blocking=True)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return image_features, text_features

    @torch.no_grad()
    def get_empty_text_embeds(self, batch_size=1):
        text_inputs = self.pipeline.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
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
        dtype = next(self.fmri_encoder.parameters()).dtype
        cond_eeg = batch['eeg_data'].to(self.device, dtype=dtype, non_blocking=True)

        target_imgs = batch['rotation_images'].to(self.device, dtype=dtype, non_blocking=True)
        if target_imgs.ndim != 5 or target_imgs.shape[1] != 6 or target_imgs.shape[2] != 3:
            raise ValueError(
                f"rotation_images must be [B,6,3,H,W], got {tuple(target_imgs.shape)}"
            )
        if target_imgs.shape[-2:] != (320, 320):
            target_imgs = v2.functional.resize(
                target_imgs, [320, 320],
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            )
        target_imgs = target_imgs.clamp(0, 1)
        target_imgs = rearrange(
            target_imgs, 'b (x y) c h w -> b c (x h) (y w)', x=3, y=2
        )
        return cond_eeg, target_imgs

    def build_zero123_condition_from_features(self, clip_sem, var_feat, drop_condition=False):
        """
        CLIP-aligned semantic -> cross-attention encoder_hidden_states
        variation            -> spatial cond_lat
        subject bias         -> excluded from generation
        """
        B = clip_sem.shape[0]
        unet_dtype = next(self.pipeline.unet.parameters()).dtype

        empty_prompt = self.get_empty_text_embeds(B)

        if drop_condition:
            semantic_cond = torch.zeros(
                (B, 1, 1024), device=clip_sem.device, dtype=unet_dtype
            )
            cond_latents = torch.zeros(
                (B, 4, 64, 64), device=clip_sem.device, dtype=unet_dtype
            )
            return semantic_cond, empty_prompt, cond_latents

        # Keep the learned conditioning heads in FP32 for stability, then cast to UNet dtype.
        with torch.autocast("cuda", enabled=False):
            semantic_cond = self.semantic_to_cross(clip_sem.float()).unsqueeze(1)
            cond_latents = self.variation_to_latent(var_feat.float())

        semantic_cond = semantic_cond.to(unet_dtype)
        cond_latents = cond_latents.to(unet_dtype)

        ramp = semantic_cond.new_tensor(
            self.pipeline.config.ramping_coefficients
        ).view(1, -1, 1)
        encoder_hidden_states = empty_prompt + semantic_cond * ramp

        return semantic_cond, encoder_hidden_states, cond_latents

    def encode_embed_fmri_condition_fmri(self, eeg, drop_condition=False):
        """Compatibility wrapper used by visualization/inference code."""
        _, var_feat, _, clip_sem = self.encode_semantic_features(eeg)
        return self.build_zero123_condition_from_features(
            clip_sem, var_feat, drop_condition=drop_condition
        )

    def forward_encoder_wo_pred_clip(self, eeg):
        # Return the CLIP-aligned semantic condition, not raw EEG semantics.
        _, _, _, clip_sem = self.encode_semantic_features(eeg)
        return clip_sem.unsqueeze(1).to(
            next(self.pipeline.unet.parameters()).dtype
        )

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
        # Use the exact training scheduler coefficients used by add_noise().
        if hasattr(self.train_scheduler, "get_velocity"):
            return self.train_scheduler.get_velocity(x, noise, t)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x)

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_loss(self, batch, stage="joint", **kwargs):
        losses = self.forward(
            batch,
            stage=stage,
            prototype_temperature=kwargs.get("prototype_temperature", 0.07),
        )
        return (
            losses["diff_loss"],
            losses["proto_loss"],
            losses["ortho_loss"],
        )

    def cal_clip_loss(self, target_features, eeg_features, logit_scale=None):
        # Unified cosine-regression scale for both CFG/non-CFG batches.
        if eeg_features.ndim == 3 and eeg_features.shape[1] == 1:
            eeg_features = eeg_features.squeeze(1)
        eeg_features = F.normalize(eeg_features.float(), dim=-1)
        target_features = F.normalize(target_features.float(), dim=-1)
        cos_sim = F.cosine_similarity(eeg_features, target_features, dim=-1)
        return 1.0 - cos_sim.mean()

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