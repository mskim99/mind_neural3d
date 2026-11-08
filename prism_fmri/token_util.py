# TRELLIS 조건 토큰 [1374, 1024] 다루기 (11-09, 토큰 prior용)
#  배치: [CLS, 레지스터 4, 패치 37×37] (DINOv2 ViT-L/14-reg x_prenorm + layer_norm, TRELLIS get_cond)
#  pool  : 패치를 7×7 평균 풀링 → [5 + 49 = 54, 1024] (prior 조건용 요약, 모델이 달라도 같은 DINOv2 공간)
#  merge : 패치 37×37을 38×38로 0 채움 → 2×2 묶음 → [361, 4096] (손실 없는 재배열, prior가 생성하는 형태), 앞 5토큰은 따로
#  unmerge: merge의 역 (채운 줄 · 칸은 버림)
import torch
import torch.nn.functional as F

NP, G, GP = 5, 37, 38


def pool(tok, k=7):                                    # tok [B, 1374, C] → [B, 5 + k², C]
    B, _, C = tok.shape
    g = tok[:, NP:].reshape(B, G, G, C).permute(0, 3, 1, 2)
    g = F.adaptive_avg_pool2d(g.float(), k).flatten(2).transpose(1, 2).to(tok.dtype)
    return torch.cat([tok[:, :NP], g], 1)


def merge(tok):                                        # [B, 1374, C] → 앞 [B, 5, C], 패치 [B, 361, 4C]
    B, _, C = tok.shape
    g = tok[:, NP:].reshape(B, G, G, C)
    g = F.pad(g, (0, 0, 0, GP - G, 0, GP - G))         # 아래 · 오른쪽에 0 한 줄
    g = g.reshape(B, GP // 2, 2, GP // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, (GP // 2) ** 2, 4 * C)
    return tok[:, :NP], g


def unmerge(pre, g):                                   # merge의 역 → [B, 1374, C]
    B, _, C4 = g.shape; C = C4 // 4
    g = g.reshape(B, GP // 2, GP // 2, 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, GP, GP, C)[:, :G, :G]
    return torch.cat([pre, g.reshape(B, G * G, C)], 1)
