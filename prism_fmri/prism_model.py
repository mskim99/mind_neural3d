# PRISM fMRI판 학습 모델 — 공식 MinD-3D++ MVDiffusion(src/mvdiffusion.py, MinD-3D 커밋 4b74ebe)을 상속.
# 바꾼 것만:
#  ① fMRI 인코더 → PrismFMRIEncoder (의미/변형/피험자 헤드, prism_encoder.py)
#  ② 의미 성분 감독: 범주 프로토타입 분류(ProtoCE, EEG판 mvdiffusion_var_protofinal.py와 같은 방식)
#     의미(6프레임 평균) → semantic_to_clip(1024→128→1024) → 범주 이름 OpenCLIP ViT-H 텍스트 프로토타입과 코사인/temperature → CE
#  ③ 의미 → cross-attention 경로에서 확산 손실 gradient 차단(sem_detach, EEG판과 같음): 의미는 범주 손실로만, 변형은 확산 손실로만 학습
#  ④ 공식 CLIP 대조 손실(12층 CLS, OpenCLIP 이미지·텍스트)은 그대로 둠
#  ⑤ 직교 손실은 넣지 않음 (10-21 지시). 피험자 헤드는 자리만 있고 학습 신호 없음
# 실행 위치: 공식 코드처럼 cwd에 pretrained_models/ 와 src(→ MinD-3D/src 링크)가 있어야 함.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mvdiffusion import MVDiffusion
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from prism_encoder import PrismFMRIEncoder


class PrismMVDiffusion(MVDiffusion):
    def __init__(self, args, stable_diffusion_config, fmri_encoder_config, logdir, class_names,
                 lr=1e-4, warmup=300, total_steps=100000, sem_detach=True, proto_temp=0.07, head_hidden=1024):
        super().__init__(args, stable_diffusion_config, fmri_encoder_config=fmri_encoder_config, logdir=logdir)
        # ① 인코더 교체: 공식 __init__이 LEA 사전학습(checkpoint_120000)을 넣은 인코더의 가중치를 그대로 옮김
        enc = PrismFMRIEncoder(fmri_encoder_config, head_hidden=head_hidden)
        r = enc.load_state_dict(self.fmri_encoder.state_dict(), strict=False)
        assert not r.unexpected_keys, r.unexpected_keys
        self.fmri_encoder = enc.to(self.device)
        self.pipeline.fmri_encoder = self.fmri_encoder

        # ② 의미 → CLIP 공간 병목 (EEG판과 같은 모양)
        self.semantic_to_clip = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1024)).to(self.device)
        self.class_names = list(class_names)
        self.register_buffer('cls_prototypes', torch.zeros(len(self.class_names), 1024, device=self.device))
        self.sem_detach = sem_detach
        self.proto_temp = proto_temp

        self._opt_cfg = (lr, warmup, total_steps)
        self.build_optimizer()

    def build_optimizer(self):
        """공식과 같은 AdamW β(0.9, 0.95)·warmup·linear decay. requires_grad인 파라미터만 (인코더 고정 시 본체 제외)."""
        lr, warmup, total_steps = self._opt_cfg
        params = list(self.unet.parameters()) + list(self.fmri_encoder.parameters()) + \
            [self.i_logit_scale, self.t_logit_scale] + list(self.semantic_to_clip.parameters())
        self.opt = AdamW([p for p in params if p.requires_grad], lr=lr, betas=(0.9, 0.95))
        self.sche = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    def set_freeze_encoder(self, flag=True):
        """fMRI 인코더 ViT 본체 고정 (공개 가중치 그대로) → 헤드·aggregation·LoRA·semantic_to_clip만 학습. 메모리·시간 절약."""
        self.fmri_encoder.set_freeze_backbone(flag)
        self.build_optimizer()

    @torch.no_grad()
    def build_prototypes(self, template='a 3D object of {}'):
        """범주 이름 → OpenCLIP ViT-H 텍스트 임베딩(정규화), 학습 중 고정."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = self.clip_model.encode_text(self.tokenizer([template.format(n) for n in self.class_names]).cuda())
        self.cls_prototypes.copy_(F.normalize(f.float(), dim=-1))

    def load_init(self, path):
        """MinD-3D++ 공개 가중치(또는 우리 재학습 model_final.pt)에서 인코더·LoRA·logit scale 불러오기.
        peft 버전에 따라 '…to_q.weight' ↔ '…to_q.base_layer.weight'가 다를 수 있어 양쪽으로 맞춰 봄."""
        sd = torch.load(path, map_location='cpu')
        sd = sd.get('model', sd)
        own = self.state_dict()
        new, skipped = {}, []
        for k, v in sd.items():
            cands = [k, k.replace('.weight', '.base_layer.weight').replace('.bias', '.base_layer.bias'), k.replace('.base_layer.', '.')]
            hit = next((c for c in cands if c in own and own[c].shape == v.shape), None)
            if hit is None:
                skipped.append(k)
            else:
                new[hit] = v
        r = self.load_state_dict(new, strict=False)
        n_lora = sum('lora_' in k for k in new); n_enc = sum(k.startswith('fmri_encoder.') for k in new)
        missing_core = [k for k in r.missing_keys if k.startswith('fmri_encoder.') and not any(h in k for h in ('sem_head', 'var_head', 'subj_head'))]
        print(f'[init] {path}: loaded {len(new)} (fmri_encoder {n_enc}, lora {n_lora}), skipped {len(skipped)}, '
              f'missing fmri_encoder core {len(missing_core)}', flush=True)
        assert n_lora > 0 and n_enc > 0 and not missing_core, 'init 가중치에서 인코더·LoRA를 못 찾음'
        return r

    def proto_loss(self, sem, labels):
        z = F.normalize(self.semantic_to_clip(sem.float()), dim=-1)
        logits = z @ self.cls_prototypes.t() / self.proto_temp
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        return loss, acc

    def cond_latents_from_tokens(self, tokens, B):
        """변형 패치 토큰 → cond_lat [B, 4, 64, 64] (공식 encode_embed_fmri_condition_fmri와 같은 재배열)."""
        lat = tokens[:, 1:, :]
        lat = lat.view(B, 6, 16, 16, 4, 4, 64).permute(0, 1, 2, 4, 3, 5, 6)
        lat = lat.reshape(B, 6, 16 * 4, 16 * 4, 64)
        lat = lat.permute(0, 2, 3, 1, 4).reshape(B, 64, 64, 64 * 6)
        return self.fmri_encoder.aggregation(lat).permute(0, 3, 1, 2)

    def prompt_from_semantic(self, sem):
        """의미 [B, 1024] → encoder_hidden_states (빈 프롬프트 + 의미 × ramp, 공식과 같음)."""
        ehs = self.pipeline._encode_prompt("", self.device, 1, False)[0]
        ramp = sem.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1)
        return ehs + sem.unsqueeze(1) * ramp

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_loss(self, batch):
        self.pipeline.unet.training = True
        cond_fmri, target_imgs = self.prepare_batch_data(batch)
        B = cond_fmri.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)
        dtype = next(self.fmri_encoder.parameters()).dtype

        tokens, fmri_feature = self.fmri_encoder.forward_encoder_wo_pred_clip(cond_fmri.to(dtype).view(-1, 1, 256, 256))
        fmri_feature = fmri_feature.view(B, 6, 1, 1024).mean(dim=1)
        image_features, text_features = self.get_clip_feature(batch["img"], batch["text"])
        clip_loss = self.cal_clip_loss(image_features, fmri_feature, self.i_logit_scale.exp()) + \
            self.cal_clip_loss(text_features, fmri_feature, self.t_logit_scale.exp())

        sem = tokens[:, 0, :].view(B, 6, 1024).mean(dim=1)               # 의미 성분 (6프레임 평균)
        p_loss, p_acc = self.proto_loss(sem, batch['cls'].to(self.device))

        if np.random.rand() < self.drop_cond_prob:                       # classifier-free guidance (공식과 같음)
            prompt_embeds = self.pipeline._encode_prompt([""] * B, self.device, 1, False)
            zt = self.fmri_encoder.forward_encoder_wo_pred(torch.zeros_like(cond_fmri).to(dtype).view(-1, 1, 256, 256))
            cond_lat = self.cond_latents_from_tokens(zt, B)
        else:
            prompt_embeds = self.prompt_from_semantic(sem.detach() if self.sem_detach else sem)
            cond_lat = self.cond_latents_from_tokens(tokens, B)

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, cond_lat)
        diff_loss, _ = self.compute_loss(v_pred, self.get_v(latents, noise, t))
        self.global_step += 1
        return diff_loss, clip_loss, p_loss, p_acc
