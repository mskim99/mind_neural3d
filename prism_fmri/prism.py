# PRISM 모델 — 박사님 10-27 구조도(.참고자료/261027_구조도.png)의 분리부를 그림대로 구현 (11-02)
#  형태 감독(Geometry head) 없음. 각 성분 head의 출력(Semantic·Variational Features)이 그대로 재구성 입력
#  fMRI 6프레임 → [Spatio-Temporal] MinD-3D++ fMRI ViT(고정, 프레임마다 공간) + Temporal Stem(학습, 패치 위치마다 프레임 축 1D conv)
#    → Semantic-Variation Encoder(학습, + 프레임 임베딩) → Mixed Features
#    → Orthogonal Projection P (직교 행렬, 토큰마다) → [Semantic Routing | Variation Routing | Perceptive Filtering]
#      Semantic    : 패치 16×16 격자에서 ConvNeXt식 residual encoder (Down Stage → ConvNeXt Block×N → Residual Encoder) → Semantic Features
#                    (+ 학습 감독: 풀링 → CLIP projection → 범주 텍스트 프로토타입 CE)
#      Variational : 프레임마다 Center / Scale / Rotate head (MLP×3 → LayerNorm → Linear → tanh / softplus / 6D) → Aggregate → Variational Features
#      Perceptive  : Token Pooling → Subject Projector (Linear-BN-GELU) → Perceptive Features (+ 피험자 분류 보조)
#    → 재구성: 구조도의 Tri-plane 대신 TRELLIS — [Semantic ; Variational] Features를 Perceptive(FiLM)로 보정 → 조건 토큰 디코더 → TRELLIS 조건 토큰 [1374, 1024]
#  구조도에 없어 정한 것: ConvNeXt 블록 수(2)·채널, head 은닉 256, Aggregate = 세 head 특징(LayerNorm 출력) 이어 붙여 Linear, Perceptive 입력 방식 FiLM,
#    Temporal Stem — 구조도엔 이름만 있어 박사님 EEG 코드(src/mvdiffusion_var_semantic_cls_sg.py TemporalStem: 채널마다 시간축 1D conv + BN + GELU)를
#    fMRI에 옮김: 채널마다(depthwise) 프레임 축 conv(커널 3, 6프레임 유지) + GELU + Linear, 잔차. 마지막 Linear 0 초기화(처음엔 ViT 특징 그대로).
#    EEG는 시간 → 공간 순서지만, 여기선 공간(ViT)이 고정·사전학습이라 ViT 뒤에 둠. temporal=False면 이전 체크포인트와 같은 구조
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


class LN2d(nn.Module):
    """채널 LayerNorm ([B, C, H, W])."""
    def __init__(self, c):
        super().__init__(); self.n = nn.LayerNorm(c)
    def forward(self, x):
        return self.n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """구조도 ConvNeXt Block: SimConv(깊이별 7×7) → LayerNorm → FFN(4배, GELU) + 잔차."""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c); self.n = nn.LayerNorm(c)
        self.f1 = nn.Linear(c, 4 * c); self.f2 = nn.Linear(4 * c, c)
    def forward(self, x):
        h = self.n(self.dw(x).permute(0, 2, 3, 1))
        return x + self.f2(F.gelu(self.f1(h))).permute(0, 3, 1, 2)


class SemanticEncoder(nn.Module):
    """구조도 Semantic: Down Stage(16×16 → 8×8) → ConvNeXt Block × N → Residual Encoder(8×8 → 16×16, 입력 잔차)."""
    def __init__(self, cin, c=512, n_blocks=2, cout=768):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(cin, c, 2, stride=2), LN2d(c))
        self.blocks = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(n_blocks)])
        self.up = nn.Sequential(nn.ConvTranspose2d(c, cout, 2, stride=2), LN2d(cout))
        self.skip = nn.Conv2d(cin, cout, 1)
    def forward(self, x):                                   # [B, cin, 16, 16] → [B, cout, 16, 16]
        return self.up(self.blocks(self.down(x))) + self.skip(x)


class PoseHead(nn.Module):
    """구조도 Center/Scale/Rotate head: MLP×3 → LayerNorm → Linear (+ 활성은 밖에서). 특징(LayerNorm 출력)도 돌려줌."""
    def __init__(self, d, k, h=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.n = nn.LayerNorm(h); self.out = nn.Linear(h, k)
    def forward(self, x):
        f = self.n(self.mlp(x)); return self.out(f), f


class TemporalStem(nn.Module):
    """[B, T, N, D] → 같은 크기. 패치 위치(N)·채널(D)마다 따로 프레임 축(T) 1D conv (박사님 EEG TemporalStem의 fMRI판), 잔차."""
    def __init__(self, d, k=3):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.out = nn.Linear(d, d); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x):
        B, T, N, D = x.shape
        h = self.n(x).permute(0, 2, 3, 1).reshape(B * N, D, T)                  # 패치 위치마다 [D, T]
        h = F.gelu(self.conv(h)).reshape(B, N, D, T).permute(0, 3, 1, 2)        # [B, T, N, D]
        return x + self.out(h)


class Prism(nn.Module):
    def __init__(self, encoder, n_classes, n_subjects, d=1024, splits=(384, 384, 256), dc=768, n_query=1374, dec_layers=2, n_frames=10,
                 sem_ch=512, sem_blocks=2, head_dim=256, temporal=True):
        super().__init__()
        assert sum(splits) == d
        self.encoder = encoder.requires_grad_(False).eval()            # MinD-3D++ fMRI ViT (패치 16, 256×256 → 16×16 + CLS), 고정
        self.splits = splits
        ds, dv, dp = splits
        self.frame_emb = nn.Embedding(n_frames, d)
        self.temporal = TemporalStem(d) if temporal else None              # Spatio-Temporal의 시간 부분
        self.shared = nn.Sequential(ResMLP(d, 2 * d), ResMLP(d, 2 * d))   # Semantic-Variation Encoder
        self.P = orthogonal(nn.Linear(d, d, bias=False))               # Orthogonal Projection
        # Semantic
        self.sem_enc = SemanticEncoder(ds, sem_ch, sem_blocks, dc)
        self.sem_proj = nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, 1024), nn.GELU(), nn.Linear(1024, 1024))   # 학습 감독용
        self.register_buffer('prototypes', torch.zeros(n_classes, 1024))
        # Variational
        self.center = PoseHead(2 * dv, 2, head_dim); self.scale = PoseHead(2 * dv, 1, head_dim); self.rotate = PoseHead(2 * dv, 6, head_dim)
        self.aggregate = nn.Sequential(nn.Linear(3 * head_dim, dc), nn.GELU(), nn.LayerNorm(dc))
        # Perceptive
        self.subj_proj = nn.Sequential(nn.Linear(dp, 256), nn.BatchNorm1d(256), nn.GELU())
        self.subj_cls = nn.Linear(256, max(n_subjects, 1))
        self.film = nn.Linear(256, 2 * dc); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # 처음엔 보정 없음
        # 재구성 (TRELLIS 조건 토큰)
        self.type_emb = nn.Parameter(torch.zeros(2, dc))                 # Semantic / Variational 토큰 구분
        self.time_emb = nn.Embedding(n_frames, dc)
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
        x = tok + self.frame_emb(frame_idx)[:, :, None, :]
        if self.temporal is not None:
            x = self.temporal(x)                                                     # 프레임 사이 정보 교환
        mixed = self.shared(x)                                                       # Mixed Features [B, T, 257, D]
        s, v, p = self.P(mixed).split(self.splits, dim=-1)
        out = {}
        # Semantic: 프레임 평균(정적 성분) → 패치 16×16 격자 → ConvNeXt식 encoder → Semantic Features [B, 256, dc]
        sg = s[:, :, 1:].mean(1).transpose(1, 2).reshape(B, -1, 16, 16)
        sem_feat = self.sem_enc(sg).flatten(2).transpose(1, 2)
        out['sem_vec'] = sem_feat.mean(1)
        out['sem_clip'] = F.normalize(self.sem_proj(out['sem_vec']), dim=-1)
        # Variational: 프레임마다 [CLS; 패치 평균] → Center / Scale / Rotate → Aggregate → Variational Features [B, T, dc]
        vin = torch.cat([v[:, :, 0], v[:, :, 1:].mean(2)], -1)
        c, fc = self.center(vin); sc, fs = self.scale(vin); r, fr = self.rotate(vin)
        out['center'] = torch.tanh(c); out['scale'] = F.softplus(sc).squeeze(-1); out['rot6d'] = r
        var_feat = self.aggregate(torch.cat([fc, fs, fr], -1))
        out['var_vec'] = var_feat.mean(1)
        # Perceptive: Token Pooling → Subject Projector
        pv = p.mean((1, 2))
        if perceptive_override is not None:
            pv = pv.mean(0, keepdim=True).expand_as(pv) if isinstance(perceptive_override, str) else perceptive_override
        h = self.subj_proj(pv)
        out['subj_logit'] = self.subj_cls(h); out['perc_vec'] = pv
        # 재구성: [Semantic ; Variational] Features (Perceptive FiLM 보정) → 조건 토큰
        mem = torch.cat([sem_feat + self.type_emb[0], var_feat + self.type_emb[1] + self.time_emb(frame_idx)], 1)   # [B, 256 + T, dc]
        gamma, beta = self.film(h).chunk(2, -1)
        mem = mem * (1 + gamma[:, None]) + beta[:, None]
        out['cond_tokens'] = self.tok_out(self.dec(self.query.expand(B, -1, -1), mem))   # [B, 1374, 1024]
        return out


def prism_losses(out, batch, proto_temp=0.07):
    L = {}
    out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}   # bf16 → fp32
    tgt = batch['cond_tokens'].float()
    L['tok_mse'] = F.mse_loss(out['cond_tokens'], tgt)
    L['tok_cos'] = (1 - F.cosine_similarity(out['cond_tokens'], tgt, dim=-1)).mean()
    logits = out['sem_clip'] @ batch['prototypes'].t() / proto_temp
    L['proto'] = F.cross_entropy(logits, batch['cls']); L['proto_acc'] = (logits.argmax(-1) == batch['cls']).float().mean()
    L['center'] = F.mse_loss(out['center'], batch['center']); L['scale'] = F.mse_loss(out['scale'], batch['scale'])
    L['rot'] = F.mse_loss(rot6d_to_matrix(out['rot6d']), batch['rot'])
    if out['subj_logit'].shape[-1] > 1:
        L['subj'] = F.cross_entropy(out['subj_logit'], batch['subj']); L['subj_acc'] = (out['subj_logit'].argmax(-1) == batch['subj']).float().mean()
    else:
        L['subj'] = out['subj_logit'].sum() * 0; L['subj_acc'] = torch.zeros(())
    return L
