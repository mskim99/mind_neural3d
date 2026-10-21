# PRISM fMRI판 인코더 — MinD-3D++ fMRI 인코더(fMRI_Encoder)를 상속해 성분 분리 헤드 3개를 붙임.
#   의미   (semantic)  : 프레임별 CLS 토큰 → sem_head → (6프레임 평균) → Zero123++ cross-attention
#   변형   (variation) : 패치 토큰 256개   → var_head → (MinD-3D++ 재배열 + aggregation) → Zero123++ cond_lat
#   피험자 (perceptive): CLS 토큰          → subj_head → 생성에 안 씀 (지금은 손실 없음, 자리만 둠)
# 의미·변형 헤드는 잔차 MLP이고 마지막 층을 0으로 시작 → 처음에는 입력을 그대로 통과(항등)시켜
# MinD-3D++ 공개 가중치를 불러오면 MinD-3D++와 같은 조건(같은 6뷰)에서 학습이 시작됨.
# 기존 fMRI_Encoder의 키(patch_embed·blocks·norm·clip_norm·aggregation)는 이름이 그대로라 공개 가중치가 바로 들어감.
# forward_encoder_wo_pred(_clip)을 덮어써서 공식 학습 코드(src/mvdiffusion.py)·추론 파이프라인(src/zero123plus/pipeline.py)을
# 고치지 않고 그대로 쓸 수 있게 함.
import torch
import torch.nn as nn
from src.encoder.autoencoder2 import fMRI_Encoder


class ResidualMLP(nn.Module):
    """x + MLP(LN(x)). 마지막 Linear를 0으로 초기화 → 처음엔 항등."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class PrismFMRIEncoder(fMRI_Encoder):
    def __init__(self, config, head_hidden=1024):
        super().__init__(config)
        dim = config.embed_dim
        self.sem_head = ResidualMLP(dim, head_hidden)
        self.var_head = ResidualMLP(dim, head_hidden)
        self.subj_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, dim))

    def route(self, tokens):
        """tokens [N, 257, D] (마지막 층 출력) → [의미 CLS ; 변형 패치] 같은 모양."""
        cls, patch = tokens[:, :1], tokens[:, 1:]
        return torch.cat([self.sem_head(cls), self.var_head(patch)], dim=1)

    # 공식 코드가 부르는 이름 그대로 덮어씀 (학습: _clip, 추론 파이프라인: wo_pred)
    def forward_encoder_wo_pred(self, x):
        return self.route(super().forward_encoder_wo_pred(x))

    def forward_encoder_wo_pred_clip(self, x):
        tokens, fmri_feature = super().forward_encoder_wo_pred_clip(x)
        return self.route(tokens), fmri_feature

    def components(self, x):
        """분석용: 프레임별 (의미 [N, D], 변형 패치 [N, 256, D], 피험자 [N, D])."""
        tokens = super().forward_encoder_wo_pred(x)
        cls, patch = tokens[:, 0], tokens[:, 1:]
        return self.sem_head(cls.unsqueeze(1)).squeeze(1), self.var_head(patch), self.subj_head(cls)
