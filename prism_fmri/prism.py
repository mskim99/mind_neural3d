# PRISM 모델 — 박사님 구조도의 분리부를 구현 (11-02 첫 구현, 11-06 박사님 새 구조도 반영). 재구성은 Tri-plane 대신 TRELLIS
#  fMRI 6프레임 → MinD-3D++ fMRI ViT(고정, 피험자별, 프레임마다 공간) → Temporal Stem(학습, 프레임 축) → SV Encoder(학습) → Mixed Features
#    → 채널 분할 S 384 / V 384 / P 256 (학습 파라미터 없음, 11-05 학습 직교 P 뺌)
#      Semantic    : 패치 토큰 → CLIP Projection(→ 1024, L_sem) → 16×16 격자 → Down Stage → ConvNeXt Block×2 → Res Encoder → Semantic Features
#      Variational : 프레임마다 [CLS; 패치 평균] → Scale · Ratio(3D 크기 비율) head (MLP×3 → LayerNorm → Linear) → Aggregate → Variational Features
#      Perceptive  : Token Pooling → Subject Projector (Linear-BN-GELU) → FiLM γ, β (+ 피험자 분류 보조)
#    → 재구성: Ⓒ[Semantic ; Variational] (+ type · frame emb.) → FiLM ⊙(1+γ)+β → Transformer Decoder(learned query 1374, Cross-Attention K·V) → TRELLIS 조건 토큰 [1374, 1024]
#  옵션 (기본값 = 11-06 구조도, train_prism.py 기본값과 같음). 이전 값은 기준선 prism_nop_feat · prism_nop_value를 읽고 다시 학습하려고 남김
#    arch      'fig' Temporal Stem = LN → + Frame Emb. → DWConv1D → GELU → Linear (잔차 없음), SV Encoder = ResMLP → LN → Linear → GELU → Linear
#              'res' 이전: Frame Emb. 먼저 더함 → Stem(잔차, 0 초기화) → ResMLP×2
#    sem_clip  'front' CLIP Projection이 Down Stage 앞 경로 안 / 'back' 이전: Res Encoder 뒤 곁가지(토큰 평균 → CLIP Projection)
#    var_heads 'sr' Scale(영상 scale, 프레임마다) + Ratio(64³ 복셀의 축별 범위, 물체마다, sigmoid) / 'csr' 이전: Center · Scale · Rotate
#    var_mode  'feat' head의 LayerNorm 특징 → Aggregate / 'value' head가 낸 값 → MLP 임베딩 (11-06 비교 결과 feat 권고)
#  피험자: ViT · Semantic · Variational은 피험자별(자기 데이터로만 학습), Temporal Stem · SV Encoder · Perceptive(FiLM) · 디코더는 공유
#  읽는 체크포인트: per_subject 1 · temporal 1 · P 없음 (c7ae1f8 이후). P가 있던 이전 체크포인트(prism_shape_ps7 등)는 못 읽음 (지표는 이미 있음)
#  구조도에 없어 정한 것: ConvNeXt 블록 수 2 · 채널 512, head 은닉 256, SV Encoder 뒤 MLP 은닉 1024, Perceptive 입력 방식 FiLM
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GradReverse(torch.autograd.Function):
    """GRL (Ganin & Lempitsky 2015): 앞으로는 그대로, 뒤로는 기울기 × (−λ). 붙은 분류기는 맞히려 하고 앞단은 못 맞히게 학습됨."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class PoseHead(nn.Module):
    """구조도 Variational head (Scale · Ratio, 이전 Center · Rotate): MLP×3 → LayerNorm → Linear (+ 활성은 밖에서). 특징(LayerNorm 출력)도 돌려줌."""
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


class TemporalStemFig(nn.Module):
    """11-06 구조도 Temporal Stem: LayerNorm → + Frame Emb. → DWConv1D(프레임 축, 패치 위치·채널마다) → GELU → Linear. 잔차 없음."""
    def __init__(self, d, k=3):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.out = nn.Linear(d, d)
    def forward(self, x, femb):                                                 # x [B, T, N, D], femb [B, T, D]
        B, T, N, D = x.shape
        h = (self.n(x) + femb[:, :, None, :]).permute(0, 2, 3, 1).reshape(B * N, D, T)
        h = F.gelu(self.conv(h)).reshape(B, N, D, T).permute(0, 3, 1, 2)
        return self.out(h)


def _route(mods, idx, x):
    """샘플마다 자기 피험자 모듈로 보냄. mods: 모듈 리스트, idx: [B] 피험자 번호. 출력이 튜플이어도 됨."""
    if len(mods) == 1:
        return mods[0](x)
    outs, order = [], []
    for s in idx.unique().tolist():
        m = (idx == s).nonzero(as_tuple=True)[0]; order.append(m); outs.append(mods[s](x[m]))
    order = torch.cat(order); inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[t] for o in outs])[inv] for t in range(len(outs[0])))
    return torch.cat(outs)[inv]


class Prism(nn.Module):
    """encoder: 피험자별 fMRI ViT 리스트 (n_subjects개, 고정). ViT · Semantic · Variational은 피험자마다 따로 (각자 자기 피험자 데이터로만 학습).
    공유: Temporal Stem · SV Encoder · Perceptive(Subject Projector · 피험자 분류 · FiLM) · Transformer Decoder."""
    def __init__(self, encoder, n_classes, n_subjects, d=1024, splits=(384, 384, 256), dc=768, n_query=1374, dec_layers=2, n_frames=10,
                 sem_ch=512, sem_blocks=2, head_dim=256, var_mode='feat', arch='res', sem_clip='back', var_heads='csr', vdim=0, grl_v=False, img_head=False, film=True):
        super().__init__()
        self.use_film = film                                                # False: ablation — Perceptive FiLM 보정 끔 (모듈은 남기되 안 씀)
        assert sum(splits) == d
        encs = list(encoder) if isinstance(encoder, (list, tuple, nn.ModuleList)) else [encoder]
        self.encoders = nn.ModuleList([e.requires_grad_(False).eval() for e in encs])   # MinD-3D++ fMRI ViT (패치 16, 256×256 → 16×16 + CLS), 고정
        self.enc_grad = False                                          # True: 입력까지 기울기 (성분 지도용)
        K = n_subjects
        assert len(encs) == K, f'피험자 {K}명인데 ViT {len(encs)}개'
        self.splits = splits
        ds, dv, dp = splits
        assert arch in ('res', 'fig'); self.arch = arch
        self.frame_emb = nn.Embedding(n_frames, d)
        if arch == 'fig':                                                   # 11-06 구조도
            self.temporal = TemporalStemFig(d)
            self.shared = nn.Sequential(ResMLP(d, 2 * d), nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        else:                                                               # 이전 구현
            self.temporal = TemporalStem(d)                                 # Spatio-Temporal의 시간 부분
            self.shared = nn.Sequential(ResMLP(d, 2 * d), ResMLP(d, 2 * d))   # Semantic-Variation Encoder
        # Semantic (피험자별 K벌)
        assert sem_clip in ('back', 'front'); self.sem_clip = sem_clip
        if sem_clip == 'front':                                             # CLIP Projection이 경로 안 (Down Stage 앞)
            self.sem_clip_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, 1024), nn.GELU(), nn.Linear(1024, 1024)) for _ in range(K)])
            self.sem_enc = nn.ModuleList([SemanticEncoder(1024, sem_ch, sem_blocks, dc) for _ in range(K)])
        else:
            self.sem_enc = nn.ModuleList([SemanticEncoder(ds, sem_ch, sem_blocks, dc) for _ in range(K)])
            self.sem_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, 1024), nn.GELU(), nn.Linear(1024, 1024)) for _ in range(K)])   # 학습 감독용
        self.register_buffer('prototypes', torch.zeros(n_classes, 1024))
        if img_head:                                                        # 11-10 A안: Semantic 안 이미지 임베딩 head (CLIP Projection 출력 평균 → OpenCLIP ViT-H 이미지 임베딩 1024, unCLIP 입력)
            assert sem_clip == 'front', 'img_head는 sem_clip front에서만'
            self.sem_img = nn.ModuleList([nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 1024)) for _ in range(K)])
        # Variational (피험자별 K벌)
        assert var_heads in ('csr', 'sr', 'sil', 'view', 'shape'); self.var_heads = var_heads
        if var_heads in ('sil', 'view', 'shape'):           # shape (11-09): 범주 안 개체 형상 차이 (물체마다 고정, 프레임마다 같은 정답)                                    # 11-09: V = 변하는 정보 (물체 평균 · 프레임 평균을 뺀 실루엣 / 시점 특징), head 하나
            assert vdim > 0, 'sil · view · shape는 vdim 필요'
            self.vhead = nn.ModuleList([PoseHead(2 * dv, vdim, head_dim) for _ in range(K)])
        else:
            self.scale = nn.ModuleList([PoseHead(2 * dv, 1, head_dim) for _ in range(K)])
        if var_heads == 'csr':
            self.center = nn.ModuleList([PoseHead(2 * dv, 2, head_dim) for _ in range(K)])
            self.rotate = nn.ModuleList([PoseHead(2 * dv, 6, head_dim) for _ in range(K)])
        elif var_heads == 'sr':
            self.ratio = nn.ModuleList([PoseHead(2 * dv, 3, head_dim) for _ in range(K)])
        assert var_mode in ('feat', 'value'); self.var_mode = var_mode
        nh, nv = {'csr': (3, 12), 'sr': (2, 4)}.get(var_heads, (1, vdim))
        agg_in = nh * head_dim if var_mode == 'feat' else nv
        self.aggregate = nn.ModuleList([nn.Sequential(nn.Linear(agg_in, dc), nn.GELU(), nn.LayerNorm(dc)) if var_mode == 'feat' else
                                        nn.Sequential(nn.Linear(agg_in, head_dim), nn.GELU(), nn.Linear(head_dim, dc), nn.GELU(), nn.LayerNorm(dc)) for _ in range(K)])
        # Perceptive (공유)
        self.grl_lam = 0.0                                                  # GRL 세기 (학습 스크립트가 스텝마다 정함)
        if grl_v:                                                           # 11-09: V에서 범주 정보 빼기 — Variational Features로 범주를 맞히는 분류기를 GRL 뒤에 붙임 (공유)
            self.v_adv = nn.Sequential(nn.Linear(dc, 256), nn.GELU(), nn.Linear(256, n_classes))
        self.subj_proj = nn.Sequential(nn.Linear(dp, 256), nn.BatchNorm1d(256), nn.GELU())
        self.subj_cls = nn.Linear(256, max(n_subjects, 1))
        self.film = nn.Linear(256, 2 * dc); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # 처음엔 보정 없음
        # 재구성 (TRELLIS 조건 토큰, 공유)
        self.type_emb = nn.Parameter(torch.zeros(2, dc))                 # Semantic / Variational 토큰 구분
        self.time_emb = nn.Embedding(n_frames, dc)
        self.query = nn.Parameter(torch.randn(1, n_query, dc) * 0.02)
        layer = nn.TransformerDecoderLayer(dc, 12, 4 * dc, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, dec_layers)
        self.tok_out = nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, 1024))

    def train(self, mode=True):
        super().train(mode); self.encoders.eval(); return self

    def trainable_state_dict(self):
        """저장용: 고정 ViT(피험자마다 약 1 GB)는 빼고 저장. 불러올 때 ViT는 enc_paths에서 다시 읽음."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith('encoders.')}

    def _vit(self, e, x):
        B, T = x.shape[:2]
        return e.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)

    def encode_frames(self, fmri, subj):
        """fmri [B, T, 256, 256] → ViT 토큰 [B, T, 257, 1024] (피험자마다 자기 ViT)."""
        with torch.set_grad_enabled(self.enc_grad and torch.is_grad_enabled()):
            if len(self.encoders) == 1:
                return self._vit(self.encoders[0], fmri)
            outs, order = [], []
            for s in subj.unique().tolist():
                m = (subj == s).nonzero(as_tuple=True)[0]; order.append(m); outs.append(self._vit(self.encoders[s], fmri[m]))
            order = torch.cat(order); inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
            return torch.cat(outs)[inv]

    def forward(self, fmri, frame_idx, subj=None, perceptive_override=None):
        """fmri [B, T, 256, 256], frame_idx [B, T] (fMRI 프레임 번호, 0~9), subj [B] 피험자 번호 (없으면 0).
        perceptive_override: 평가 때 Perceptive 벡터를 바꿔 끼우기 ('mean' → 배치 평균 = 사실상 버리기, 텐서 → 그 값)."""
        if subj is None:
            subj = torch.zeros(fmri.shape[0], dtype=torch.long, device=fmri.device)
        tok = self.encode_frames(fmri, subj).float()
        B, T, N, D = tok.shape
        if self.arch == 'fig':
            x = self.temporal(tok, self.frame_emb(frame_idx))                        # LN → + Frame Emb. → 프레임 축 conv
        else:
            x = self.temporal(tok + self.frame_emb(frame_idx)[:, :, None, :])        # 이전: Frame Emb. 더한 뒤 Stem (잔차)
        mixed = self.shared(x)                                                       # Mixed Features [B, T, 257, D]
        s, v, p = mixed.split(self.splits, dim=-1)                                  # 채널 분할 (학습 파라미터 없음)
        out = {}
        # Semantic (피험자별): 프레임 평균(정적 성분) → (front) CLIP Projection → 패치 16×16 격자 → ConvNeXt식 encoder → Semantic Features [B, 256, dc]
        st = s[:, :, 1:].mean(1)                                                     # [B, 256, ds]
        if self.sem_clip == 'front':
            st = _route(self.sem_clip_proj, subj, st)                                # CLIP 공간 [B, 256, 1024]
            out['sem_clip'] = F.normalize(st.mean(1), dim=-1)
            if hasattr(self, 'sem_img'): out['sem_img'] = _route(self.sem_img, subj, st.mean(1))   # [B, 1024] (정규화 전)
        sem_feat = _route(self.sem_enc, subj, st.transpose(1, 2).reshape(B, -1, 16, 16)).flatten(2).transpose(1, 2)
        out['sem_vec'] = sem_feat.mean(1)
        if self.sem_clip == 'back':
            out['sem_clip'] = F.normalize(_route(self.sem_proj, subj, out['sem_vec']), dim=-1)
        # Variational (피험자별): 프레임마다 [CLS; 패치 평균] → Scale · Ratio (csr: Center · Scale · Rotate) → Aggregate → Variational Features [B, T, dc]
        vin = torch.cat([v[:, :, 0], v[:, :, 1:].mean(2)], -1)
        if self.var_heads in ('sil', 'view', 'shape'):
            q, fq = _route(self.vhead, subj, vin); out['vt'] = q                    # [B, T, vdim] 표준화된 변화량
            feats = [fq]; vals = [q]
        else:
            sc, fs = _route(self.scale, subj, vin); out['scale'] = F.softplus(sc).squeeze(-1)
        if self.var_heads == 'csr':
            c, fc = _route(self.center, subj, vin); r, fr = _route(self.rotate, subj, vin)
            out['center'] = torch.tanh(c); out['rot6d'] = r
            feats = [fc, fs, fr]
            vals = [out['center'], out['scale'][..., None], rot6d_to_matrix(r).flatten(-2)]          # 실제 값 2+1+9
        elif self.var_heads == 'sr':
            q, fq = _route(self.ratio, subj, vin); out['ratio'] = torch.sigmoid(q)  # [B, T, 3] 크기 비율 (가장 긴 변 = 1)
            feats = [fs, fq]; vals = [out['scale'][..., None], out['ratio']]                         # 실제 값 1+3
        agg_in = torch.cat(feats if self.var_mode == 'feat' else vals, -1)          # feat: LayerNorm 특징 head 수×256
        var_feat = _route(self.aggregate, subj, agg_in)
        out['var_vec'] = var_feat.mean(1)
        if hasattr(self, 'v_adv'):
            out['v_adv_logit'] = self.v_adv(GradReverse.apply(out['var_vec'], float(self.grl_lam)))
        # Perceptive (공유): Token Pooling → Subject Projector
        pv = p.mean((1, 2))
        if perceptive_override is not None:
            pv = pv.mean(0, keepdim=True).expand_as(pv) if isinstance(perceptive_override, str) else perceptive_override
        h = self.subj_proj(pv)
        out['subj_logit'] = self.subj_cls(h); out['perc_vec'] = pv
        # 재구성 (공유): [Semantic ; Variational] Features (Perceptive FiLM 보정) → 조건 토큰
        mem = torch.cat([sem_feat + self.type_emb[0], var_feat + self.type_emb[1] + self.time_emb(frame_idx)], 1)   # [B, 256 + T, dc]
        if self.use_film:
            gamma, beta = self.film(h).chunk(2, -1)
            mem = mem * (1 + gamma[:, None]) + beta[:, None]
        out['cond_tokens'] = self.tok_out(self.dec(self.query.expand(B, -1, -1), mem))   # [B, 1374, 1024]
        return out


def load_encoder(path, cfg):
    """MinD-3D++ 가중치(.pt, 전체 또는 인코더만) → fMRI_Encoder (fmri_encoder.* 키). 실행 위치 $W/run (src 필요)."""
    from src.encoder.autoencoder2 import fMRI_Encoder
    e = fMRI_Encoder(cfg.model.params.fmri_encoder_config); sd = torch.load(path, map_location='cpu'); sd = sd.get('model', sd)
    r = e.load_state_dict({k[len('fmri_encoder.'):]: v for k, v in sd.items() if k.startswith('fmri_encoder.')}, strict=False)
    assert not r.missing_keys, f'{path}: 인코더 키 {len(r.missing_keys)}개 없음'
    return e


def build_from_ckpt(ck, repo):
    """체크포인트 → Prism. ViT는 enc_paths(피험자별 공개 가중치)에서 다시 읽음. args에 없는 옵션은 이전 값(res · back · csr · feat)으로."""
    import os
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(repo, 'configs', 'mind3d_pp.yaml'))
    args = ck.get('args') or {}
    assert args.get('per_subject', 0) and args.get('temporal', 1), '피험자별 ViT · Temporal Stem 체크포인트만 읽음 (11-04 이전 구조는 지원 안 함)'
    encs = [load_encoder(pth, cfg) for pth in ck['enc_paths']]
    model = Prism(encs, n_classes=len(ck['classes']), n_subjects=len(ck['subs']),
                  var_mode=args.get('var_mode', 'feat'), arch=args.get('arch', 'res'),
                  sem_clip=args.get('sem_clip', 'back'), var_heads=args.get('var_heads', 'csr'),
                  vdim=int(args.get('vdim', 0) or 0), grl_v=float(args.get('lam_vadv', 0) or 0) > 0,
                  img_head=float(args.get('lam_img', 0) or 0) > 0 or float(args.get('lam_img_nce', 0) or 0) > 0,
                  film=bool(int(args.get('film', 1))))
    check_load(model, ck['model'])
    return model


def check_load(model, sd):
    """고정 ViT를 뺀 체크포인트 불러오기: ViT(encoders.*) 말고 빠지거나 남는 키가 있으면 멈춤 (구조 옵션이 안 맞을 때)."""
    r = model.load_state_dict(sd, strict=False)
    bad = [k for k in r.missing_keys if not k.startswith('encoders.')]
    assert not r.unexpected_keys and not bad, f'체크포인트와 구조가 안 맞음: 남는 키 {r.unexpected_keys[:5]} · 빠진 키 {bad[:5]}'
    return r


def prism_losses(out, batch, proto_temp=0.07):
    """L_tok (MSE · 1 − cos) · L_sem (CLIP 텍스트 프로토타입 CE) · L_var (scale · ratio | center · scale · rot) · L_sub (피험자 CE). 없는 항은 0."""
    L = {}
    out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}   # bf16 → fp32
    tgt = batch['cond_tokens'].float()
    L['tok_mse'] = F.mse_loss(out['cond_tokens'], tgt)
    L['tok_cos'] = (1 - F.cosine_similarity(out['cond_tokens'], tgt, dim=-1)).mean()
    logits = out['sem_clip'] @ batch['prototypes'].t() / proto_temp
    L['proto'] = F.cross_entropy(logits, batch['cls']); L['proto_acc'] = (logits.argmax(-1) == batch['cls']).float().mean()
    z = torch.zeros((), device=tgt.device)
    L['scale'] = F.mse_loss(out['scale'], batch['scale']) if 'scale' in out else z
    L['center'] = F.mse_loss(out['center'], batch['center']) if 'center' in out else z
    L['rot'] = F.mse_loss(rot6d_to_matrix(out['rot6d']), batch['rot']) if 'rot6d' in out else z
    if 'ratio' in out:                                                               # 물체 정답을 프레임마다 맞춤, 복셀 없는 물체는 뺌
        m = batch['ratio_ok'].float()[:, None, None]
        L['ratio'] = ((out['ratio'] - batch['ratio'][:, None]) ** 2 * m).sum() / (m.sum() * out['ratio'][0].numel()).clamp_min(1)
    else:
        L['ratio'] = z
    if 'vt' in out:                                                                  # 변하는 정보 (표준화), 정답 없는 물체는 뺌
        m = batch['vt_ok'].float()[:, None, None]
        L['vt'] = ((out['vt'] - batch['vt']) ** 2 * m).sum() / (m.sum() * out['vt'][0].numel()).clamp_min(1)
    else:
        L['vt'] = z
    if 'v_adv_logit' in out:                                                         # GRL 뒤 분류기 (V로 범주 맞히기) — 정확도가 우연(1/13)에 가까울수록 V에 범주가 적음
        L['v_adv'] = F.cross_entropy(out['v_adv_logit'], batch['cls']); L['v_adv_acc'] = (out['v_adv_logit'].argmax(-1) == batch['cls']).float().mean()
    else:
        L['v_adv'] = z; L['v_adv_acc'] = z
    if 'sem_img' in out and 'img_emb' in batch:                                     # 11-10 A안: 이미지 임베딩 (1 − cos) + 배치 InfoNCE (양방향, 온도 0.07)
        e = F.normalize(out['sem_img'], dim=-1); t = F.normalize(batch['img_emb'].float(), dim=-1)
        L['img_cos'] = (1 - (e * t).sum(-1)).mean()
        lg = e @ t.t() / 0.07; y = torch.arange(len(e), device=e.device)
        L['img_nce'] = (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y)) / 2; L['img_r1'] = (lg.argmax(-1) == y).float().mean()
    else:
        L['img_cos'] = z; L['img_nce'] = z; L['img_r1'] = z
    if out['subj_logit'].shape[-1] > 1:
        L['subj'] = F.cross_entropy(out['subj_logit'], batch['subj']); L['subj_acc'] = (out['subj_logit'].argmax(-1) == batch['subj']).float().mean()
    else:
        L['subj'] = out['subj_logit'].sum() * 0; L['subj_acc'] = torch.zeros(())
    return L
