# PRISM v2 모델 — 박사님 10-27 구조도를 ViT·TRELLIS로 옮긴 1차 구현 (설계: .보고 자료/261029_7차보고/PRISM_설계논의.md)
#  fMRI 6프레임 → MinD-3D++ fMRI ViT(고정) → 공통 인코더(학습, + 프레임 임베딩) → Mixed Features
#    ├ [학습 때만] 형태 감독 head: Mixed 평균 → 3D 업샘플 → 64³ 점유 (3D GT, BCE + Dice)
#    └ 직교 투영 P (정확한 직교 행렬, 토큰마다) → [Semantic | Variational | Perceptive]  (P가 직교라 Perceptive = 나머지 칸)
#        Semantic    : CLS 부분 6프레임 평균 → CLIP projection → 범주 텍스트 프로토타입 CE
#        Variational : 프레임마다 (CLS + 패치 평균) → Center(tanh, 2) / Scale(softplus, 1) / Rotate(6D) — 정답은 자극 영상
#        Perceptive  : 토큰 pooling → Subject Projector(Linear-BN-GELU) → FiLM(γ, β) + 피험자 분류(보조)
#    → 조건 토큰 디코더: [Semantic ; Variational] 토큰 (FiLM 보정) → 학습 query 1374개 cross-attention → TRELLIS 조건 토큰 [1374, 1024]
# 10-30 개선 옵션 (설계: .보고 자료/261105_8차보고/PRISM_v2_개선_설계안.md, 끄면 기존과 같은 구조·파라미터 이름)
#  sil    : Variational 정답 = 프레임별 실루엣 32×32 (+ center·scale, 회전 제외)
#  adv    : 적대 분류기 (gradient reversal) — S→피험자, V→범주·피험자, P→범주를 못 맞히게
#  xrecon : 교차 재구성 — [다른 피험자의 S·V + 자기 P]로 자기 fMRI ViT 토큰 요약을 재구성
#  (일관성 loss는 모델이 아니라 학습 쪽: 같은 물체 두 피험자 쌍 배치)
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


class GradReverse(torch.autograd.Function):
    """순전파는 그대로, 역전파 gradient에 -lam을 곱함 (적대 분류기 → 앞단이 그 정보를 지우게)."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def small_cls(d, n):
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, n))


def model_opts(args):
    """체크포인트 args(dict) → PrismV2 옵션. 옛 체크포인트는 전부 False."""
    args = args or {}
    return dict(sil=bool(args.get('sil', False)), adv=bool(args.get('adv', False)), xrecon=bool(args.get('xrecon', False)),
                adv_grl=float(args.get('adv_grl', 1.0)))


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
    def __init__(self, encoder, n_classes, n_subjects, d=1024, splits=(384, 384, 256), dc=768, n_query=1374, dec_layers=2, n_frames=10,
                 sil=False, adv=False, xrecon=False, adv_grl=1.0):
        super().__init__()
        self.opt_sil, self.opt_adv, self.opt_xrecon, self.adv_grl = sil, adv, xrecon, adv_grl
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
        ns = max(n_subjects, 1)
        if sil:      # Variational: 프레임별 실루엣 32×32 logit
            self.sil_head = nn.Linear(512, 32 * 32)
        if adv:      # 적대 분류기 (입력은 GRL을 거침)
            self.adv_s_subj = small_cls(ds, ns); self.adv_v_cls = small_cls(2 * dv, n_classes)
            self.adv_v_subj = small_cls(2 * dv, ns); self.adv_p_cls = small_cls(dp, n_classes)
        if xrecon:   # 프레임 요약 [S (CLS;평균) | V (CLS;평균) | P (CLS;평균)] → ViT 토큰 요약 [CLS; 패치 평균] (2 × 1024)
            self.recon = nn.Sequential(nn.LayerNorm(2 * (ds + dv + dp)), nn.Linear(2 * (ds + dv + dp), 1024), nn.GELU(), nn.Linear(1024, 2 * d))

    def train(self, mode=True):
        super().train(mode); self.encoder.eval(); return self

    @torch.no_grad()
    def encode_frames(self, fmri):
        """fmri [B, T, 256, 256] → ViT 토큰 [B, T, 257, 1024]."""
        B, T = fmri.shape[:2]
        return self.encoder.forward_encoder_wo_pred(fmri.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)

    def recon_from(self, s_sum, v_sum, p_sum):
        return self.recon(torch.cat([s_sum, v_sum, p_sum], -1))

    def forward(self, fmri, frame_idx, perceptive_override=None, pair=False):
        """fmri [B, T, 256, 256], frame_idx [B, T] (fMRI 프레임 번호, 0~9).
        perceptive_override: 평가 때 Perceptive 벡터를 바꿔 끼우기 ('mean' → 배치 평균 = 사실상 버리기, 텐서 → 그 값).
        pair: 배치가 [물체0·피험자a, 물체0·피험자b, 물체1·a, 물체1·b, …] 순서일 때 True (교차 재구성의 짝 = 인덱스 ^ 1)."""
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
        if self.opt_sil:
            out['sil_logit'] = self.sil_head(vf).view(B, T, 32, 32)
        s_sum = torch.cat([s[:, :, 0], s[:, :, 1:].mean(2)], -1)                         # [B, T, 2·ds] 프레임 요약
        v_sum = torch.cat([v[:, :, 0], v[:, :, 1:].mean(2)], -1)                         # [B, T, 2·dv]
        p_sum = torch.cat([p[:, :, 0], p[:, :, 1:].mean(2)], -1)                         # [B, T, 2·dp]
        out['v_sum'] = v_sum
        # Perceptive
        pv = p.mean((1, 2))
        if perceptive_override is not None:
            pv = pv.mean(0, keepdim=True).expand_as(pv) if isinstance(perceptive_override, str) else perceptive_override
        h = self.subj_proj(pv)
        out['subj_logit'] = self.subj_cls(h); out['perc_vec'] = pv
        if self.opt_adv:
            r = lambda x: GradReverse.apply(x, self.adv_grl)
            vp = v_sum.mean(1)
            out['adv_s_subj'] = self.adv_s_subj(r(sem)); out['adv_v_cls'] = self.adv_v_cls(r(vp))
            out['adv_v_subj'] = self.adv_v_subj(r(vp)); out['adv_p_cls'] = self.adv_p_cls(r(p.mean((1, 2))))
        if self.opt_xrecon:
            tgt = torch.cat([tok[:, :, 0], tok[:, :, 1:].mean(2)], -1)                   # 고정 ViT 토큰 요약 (정답)
            out['recon_tgt'] = F.layer_norm(tgt, tgt.shape[-1:]).detach()
            out['recon_self'] = self.recon_from(s_sum, v_sum, p_sum)
            if pair:
                j = torch.arange(B, device=tok.device) ^ 1                               # 같은 물체, 다른 피험자
                out['recon_cross'] = self.recon_from(s_sum[j], v_sum[j], p_sum)          # 남의 S·V + 자기 P → 자기 토큰
            with torch.no_grad():                                                        # 경계 판단용 (학습 안 씀): P 없이
                out['recon_noP'] = self.recon_from(s_sum, v_sum, torch.zeros_like(p_sum))
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


def prism_v2_losses(out, batch, proto_temp=0.07, pair=False):
    L = {}
    # autocast(bf16) 출력을 fp32로 맞춤 (CUDA에서 bf16 출력 vs fp32 정답 MSE는 backward dtype 오류)
    out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
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
    if 'sil_logit' in out and 'sil' in batch:                       # 실루엣: 부드러운 정답(0~1) BCE + Dice
        sl, st = out['sil_logit'].flatten(0, 1), batch['sil'].float().flatten(0, 1)
        L['sil'] = F.binary_cross_entropy_with_logits(sl, st) + dice_loss(sl, st)
    if 'adv_s_subj' in out:                                          # 적대 분류기 CE (정확도 = 새어 든 정도)
        prs = (('adv_s_subj', 'subj'), ('adv_v_cls', 'cls'), ('adv_v_subj', 'subj'), ('adv_p_cls', 'cls'))
        L['adv'] = sum(F.cross_entropy(out[k], batch[t]) for k, t in prs)
        for k, t in prs:
            L[k + '_acc'] = (out[k].argmax(-1) == batch[t]).float().mean()
    if pair:                                                         # 일관성: 같은 물체 두 피험자의 S·V가 같게
        j = torch.arange(out['sem_vec'].shape[0], device=out['sem_vec'].device) ^ 1
        L['cons'] = (1 - F.cosine_similarity(out['sem_vec'], out['sem_vec'][j], dim=-1)).mean() + \
            (1 - F.cosine_similarity(out['v_sum'], out['v_sum'][j], dim=-1)).mean()
    if 'recon_self' in out:
        tg = out['recon_tgt']
        L['recon_withP'] = F.mse_loss(out['recon_self'], tg); L['recon_noP'] = F.mse_loss(out['recon_noP'], tg)
        L['xrecon'] = L['recon_withP'] + (F.mse_loss(out['recon_cross'], tg) if 'recon_cross' in out else 0)
    if out['subj_logit'].shape[-1] > 1:
        L['subj'] = F.cross_entropy(out['subj_logit'], batch['subj']); L['subj_acc'] = (out['subj_logit'].argmax(-1) == batch['subj']).float().mean()
    else:
        L['subj'] = out['subj_logit'].sum() * 0; L['subj_acc'] = torch.zeros(())
    return L
