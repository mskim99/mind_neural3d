# S · V · P 분리 손실 (11-10, 사용자 승인 — 설계 `.보고 자료/261112_9차보고/PRISM_분리/분리·텍스트_설계.md` 2절)
#  A. 결합-섞기 판별기 (FactorVAE, Kim & Mnih 2018의 총상관 판별기를 갈래 단위로):
#     진짜 = 같은 물체의 (S_i, V_i, P_i), 가짜 = 갈래마다 다른 물체로 섞은 (S_i, V_π(i), P_σ(i)) (배치 안, 제자리 없는 순열).
#     판별기 D는 둘을 구분하도록 따로 학습(BCE), 인코더 손실 = λ · mean(D logit(진짜)) = 총상관 추정(log D/(1−D))을 줄임.
#     각 갈래는 크기 정보 없이 비교하려고 affine 없는 LayerNorm을 먼저 거침 (크기를 키우거나 줄여 D를 속이는 길 막기)
#  C. HSIC 대조: 갈래 쌍마다 RBF 커널 CKA(= 정규화 HSIC, 폭은 거리 중앙값) 합을 줄임
#  입력은 components.npz와 같은 벡터: sem_vec [B, 768] · var_vec [B, 768] · perc_vec [B, 256]
import torch
import torch.nn as nn
import torch.nn.functional as F


def derange(n, device):
    """제자리(i → i) 없는 무작위 순열 (n ≥ 2)."""
    while True:
        p = torch.randperm(n, device=device)
        if not (p == torch.arange(n, device=device)).any():
            return p


class TCDisc(nn.Module):
    """(S, V, P) 묶음 → 진짜(같은 물체) 로짓. 갈래마다 LayerNorm(affine 없음) → Linear 256 → 합쳐 MLP."""
    def __init__(self, dims=(768, 768, 256), h=512, d=256):
        super().__init__()
        self.inp = nn.ModuleList([nn.Sequential(nn.LayerNorm(k, elementwise_affine=False), nn.Linear(k, d)) for k in dims])
        self.mlp = nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(d * len(dims), h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2),
                                 nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, 1))
    def forward(self, parts):
        return self.mlp(torch.cat([f(x) for f, x in zip(self.inp, parts)], -1)).squeeze(-1)


def shuffle_parts(parts):
    """첫 갈래(S)는 그대로, 나머지는 갈래마다 다른 제자리 없는 순열."""
    n = parts[0].shape[0]
    return [parts[0]] + [x[derange(n, x.device)] for x in parts[1:]]


def tc_enc(disc, parts):
    """인코더가 줄일 총상관 추정 = mean(D 로짓(진짜)). parts: [S, V, P] (기울기 있음).
    주의: 이 값으로 본 backward를 한 뒤에 tc_disc_update를 부를 것 (판별기를 먼저 바꾸면 그래프가 깨짐)."""
    return disc([x.float() for x in parts]).mean()


def tc_disc_update(disc, d_opt, parts, train=True):
    """판별기 한 번 갱신 (진짜 vs 섞은 것, BCE). 본 backward가 판별기에 남긴 기울기는 여기서 지움. 반환: (판별기 손실, 정확도)."""
    real = [x.detach().float() for x in parts]; fake = shuffle_parts(real)
    lr_, lf = disc(real), disc(fake)
    d_loss = F.binary_cross_entropy_with_logits(lr_, torch.ones_like(lr_)) + F.binary_cross_entropy_with_logits(lf, torch.zeros_like(lf))
    acc = float(((lr_ > 0).float().mean() + (lf < 0).float().mean()) / 2)
    if train:
        d_opt.zero_grad(set_to_none=True); d_loss.backward(); d_opt.step()
    return float(d_loss.detach()), acc


def rbf_cka(x, y):
    """RBF 커널 CKA (Kornblith 2019), 폭 σ² = 쌍 거리² 중앙값 / 2 (기울기 끊음). x [B, d1], y [B, d2] → 0~1."""
    def gram(z):
        d2 = torch.cdist(z, z).pow(2)
        med = d2.detach()[~torch.eye(len(z), dtype=torch.bool, device=z.device)].median().clamp_min(1e-6)
        return torch.exp(-d2 / med)
    def center(k):
        n = k.shape[0]; h = torch.eye(n, device=k.device) - 1.0 / n
        return h @ k @ h
    kx, ky = center(gram(x.float())), center(gram(y.float()))
    return (kx * ky).sum() / ((kx * kx).sum().sqrt() * (ky * ky).sum().sqrt()).clamp_min(1e-8)


def hsic_loss(parts):
    """갈래 쌍 RBF CKA 합 (S-V, S-P, V-P)."""
    s, v, p = parts
    return rbf_cka(s, v) + rbf_cka(s, p) + rbf_cka(v, p)
