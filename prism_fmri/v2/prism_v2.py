# PRISM v2 모델 — 박사님 10-27 구조도를 ViT·TRELLIS로 옮긴 1차 구현 (설계: .보고 자료/261029_7차보고/PRISM_설계논의.md)
#  fMRI 6프레임 → MinD-3D++ fMRI ViT(고정) → 공통 인코더(학습, + 프레임 임베딩) → Mixed Features
#    ├ [학습 때만] 형태 감독 head: Mixed 평균 → 3D 업샘플 → 64³ 점유 (3D GT, BCE + Dice)
#    └ 직교 투영 P (정확한 직교 행렬, 토큰마다) → [Semantic | Variational | Perceptive]  (P가 직교라 Perceptive = 나머지 칸)
#        Semantic    : CLS 부분 6프레임 평균 → CLIP projection → 범주 텍스트 프로토타입 CE
#        Variational : 프레임마다 (CLS + 패치 평균) → Center(tanh, 2) / Scale(softplus, 1) / Rotate(6D) — 정답은 자극 영상
#        Perceptive  : 토큰 pooling → Subject Projector(Linear-BN-GELU) → FiLM(γ, β) + 피험자 분류(보조)
#    → 조건 토큰 디코더: [Semantic ; Variational] 토큰 (FiLM 보정) → 학습 query 1374개 cross-attention → TRELLIS 조건 토큰 [1374, 1024]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal


class ResMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
    def forward(self, x):
        return x + self.f2(F.gelu(self.f1(self.n(x))))


def rot6d_to_matrix(x):
    """[..., 6] → 회전 행렬 [..., 3, 3] (Zhou et al. 2019, 앞 두 열을 Gram-Schmidt)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


class GeoDecoder(nn.Module):
    """[B, D] → 64³ 점유 logit (4³ → 8 → 16 → 32 → 64)."""
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 256 * 4 ** 3)
        ch = [256, 128, 64, 32, 16]
        self.up = nn.Sequential(*[m for i in range(4) for m in (nn.ConvTranspose3d(ch[i], ch[i + 1], 4, 2, 1), nn.GroupNorm(8, ch[i + 1]), nn.GELU())])
        self.out = nn.Conv3d(16, 1, 3, padding=1)
    def forward(self, z):
        return self.out(self.up(self.fc(z).view(-1, 256, 4, 4, 4))).squeeze(1)


class PrismV2(nn.Module):
    def __init__(self, encoder, n_classes, n_subjects, d=1024, splits=(384, 384, 256), dc=768, n_query=1374, dec_layers=2, n_frames=10):
        super().__init__()
        assert sum(splits) == d
        self.encoder = encoder.requires_grad_(False).eval()            # MinD-3D++ fMRI ViT, 고정
        self.splits = splits
        self.frame_emb = nn.Embedding(n_frames, d)
        self.shared = nn.Sequential(ResMLP(d, 2 * d), ResMLP(d, 2 * d))   # 공통 인코더 ("Semantic-Variation Encoder")
        self.geo = GeoDecoder(d)
        self.P = orthogonal(nn.Linear(d, d, bias=False))               # 직교 투영
        ds, dv, dp = splits
        self.sem_proj = nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, 1024), nn.GELU(), nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1024))
        self.register_buffer('prototypes', torch.zeros(n_classes, 1024))
        self.var_head = nn.Sequential(nn.LayerNorm(2 * dv), nn.Linear(2 * dv, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU())
        self.center = nn.Linear(512, 2); self.scale = nn.Linear(512, 1); self.rotate = nn.Linear(512, 6)
        self.subj_proj = nn.Sequential(nn.Linear(dp, 256), nn.BatchNorm1d(256), nn.GELU())
        self.subj_cls = nn.Linear(256, max(n_subjects, 1))
        self.film = nn.Linear(256, 2 * dc); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # 처음엔 보정 없음
        self.mem_in = nn.Linear(ds + dv, dc)
        self.query = nn.Parameter(torch.randn(1, n_query, dc) * 0.02)
        layer = nn.TransformerDecoderLayer(dc, 12, 4 * dc, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, dec_layers)
        self.tok_out = nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, 1024))

    def train(self, mode=True):
        super().train(mode); self.encoder.eval(); return self

    @torch.no_grad()
    def encode_frames(self, fmri):
        """fmri [B, T, 256, 256] → ViT 토큰 [B, T, 257, 1024]."""
        B, T = fmri.shape[:2]
        return self.encoder.forward_encoder_wo_pred(fmri.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)

    def forward(self, fmri, frame_idx, perceptive_override=None):
        """fmri [B, T, 256, 256], frame_idx [B, T] (fMRI 프레임 번호, 0~9).
        perceptive_override: 평가 때 Perceptive 벡터를 바꿔 끼우기 ('mean' → 배치 평균 = 사실상 버리기, 텐서 → 그 값)."""
        tok = self.encode_frames(fmri).float()
        B, T, N, D = tok.shape
        mixed = self.shared(tok + self.frame_emb(frame_idx)[:, :, None, :])           # [B, T, 257, D]
        out = {'geo_logit': self.geo(mixed.mean((1, 2)))}
        z = self.P(mixed)
        ds, dv, dp = self.splits
        s, v, p = z.split(self.splits, dim=-1)
        # Semantic
        sem = s[:, :, 0].mean(1)
        out['sem_clip'] = F.normalize(self.sem_proj(sem), dim=-1)
        out['sem_vec'] = sem
        # Variational (프레임마다)
        vf = self.var_head(torch.cat([v[:, :, 0], v[:, :, 1:].mean(2)], -1))           # [B, T, 512]
        out['center'] = torch.tanh(self.center(vf)); out['scale'] = F.softplus(self.scale(vf)).squeeze(-1)
        out['rot6d'] = self.rotate(vf); out['var_vec'] = v[:, :, 0].mean(1)
        # Perceptive
        pv = p.mean((1, 2))
        if perceptive_override is not None:
            pv = pv.mean(0, keepdim=True).expand_as(pv) if isinstance(perceptive_override, str) else perceptive_override
        h = self.subj_proj(pv)
        out['subj_logit'] = self.subj_cls(h); out['perc_vec'] = pv
        gamma, beta = self.film(h).chunk(2, -1)
        # 조건 토큰 디코더 (Semantic·Variational 토큰을 FiLM으로 보정)
        mem = self.mem_in(torch.cat([s, v], -1)).reshape(B, T * N, -1)
        mem = mem * (1 + gamma[:, None]) + beta[:, None]
        q = self.query.expand(B, -1, -1)
        out['cond_tokens'] = self.tok_out(self.dec(q, mem))                              # [B, 1374, 1024]
        return out


def dice_loss(logit, target, eps=1.0):
    p = torch.sigmoid(logit).flatten(1); t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def prism_v2_losses(out, batch, proto_temp=0.07):
    L = {}
    tgt = batch['cond_tokens'].float()
    L['tok_mse'] = F.mse_loss(out['cond_tokens'], tgt)
    L['tok_cos'] = (1 - F.cosine_similarity(out['cond_tokens'], tgt, dim=-1)).mean()
    logits = out['sem_clip'] @ batch['prototypes'].t() / proto_temp
    L['proto'] = F.cross_entropy(logits, batch['cls']); L['proto_acc'] = (logits.argmax(-1) == batch['cls']).float().mean()
    occ = batch['voxels'].float()
    pos_w = ((occ.numel() - occ.sum()) / occ.sum().clamp(min=1)).clamp(max=100)
    L['geo'] = F.binary_cross_entropy_with_logits(out['geo_logit'], occ, pos_weight=pos_w) + dice_loss(out['geo_logit'], occ)
    L['center'] = F.mse_loss(out['center'], batch['center']); L['scale'] = F.mse_loss(out['scale'], batch['scale'])
    L['rot'] = F.mse_loss(rot6d_to_matrix(out['rot6d']), batch['rot'])
    if out['subj_logit'].shape[-1] > 1:
        L['subj'] = F.cross_entropy(out['subj_logit'], batch['subj']); L['subj_acc'] = (out['subj_logit'].argmax(-1) == batch['subj']).float().mean()
    else:
        L['subj'] = out['subj_logit'].sum() * 0; L['subj_acc'] = torch.zeros(())
    return L
