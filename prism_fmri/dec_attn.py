# Transformer Decoder의 Cross-Attention 가중치 진단 (11-12, 사용자 요청) — MinD-3D++ venv, 실행 위치 $W/run
#  디코더: learned query 1374개가 memory = [Semantic 256 ; Variational T(=6)] 토큰을 Cross-Attention으로 봄 (층마다 12 head)
#  피험자마다 test 104에서 층별로:
#    v_share   query가 Variational 토큰에 주는 가중치 합 (균등이면 T/(256+T) = 0.023). v_ratio = v_share ÷ 균등. head별 최대 · 최소도
#    entropy   가중치 분포의 엔트로피 ÷ log(256+T) (1 = 균등하게 퍼짐, 0 = 한 토큰만 봄)
#    fixed     물체마다의 가중치 지도와 104개 평균 지도의 cos (query마다, 평균). 1에 가까우면 입력과 무관하게 같은 곳을 봄
#    same_obj  같은 물체를 본 서로 다른 지도가 없으므로 대신: 물체끼리 지도의 평균 cos (pair) — fixed와 같이 읽음
#    pos_r     패치 query(37×37 격자)의 위치와, 그 query가 보는 Semantic 토큰(16×16 격자)의 가중 평균 위치의 상관 (x, y). 높으면 위치 대응이 있음
#              (가정: 조건 토큰 1374 = 앞 5개(cls · register) + 37×37 패치, Semantic 256 = 16×16 순서 그대로)
#    top_tok   Semantic 토큰별로 받은 가중치 (query · 물체 평균)의 상위 5개 몫 — 몇 개 토큰에 몰리는지
#  활성도 (가중치만으로는 실제로 얼마나 전달됐는지 모름 → 값 벡터 크기까지 본 것, ACT| 줄):
#    cv_share  가중치 × 값 벡터 크기(head마다 out_proj까지 지난 ‖f_h(x_j)‖)로 본 Variational 토큰 몫 — v_share와 비교. vn_s · vn_v = Semantic · Variational 토큰의 값 벡터 평균 크기
#    blk       층 안 세 블록 출력의 평균 크기: self-attention / cross-attention / feed-forward — cross-attention이 query를 얼마나 바꾸나
#    dep       cross-attention 출력 중 입력(물체)에 따라 달라지는 비율 = 1 − ‖104개 평균‖² ÷ 평균 ‖출력‖² (query마다, 평균). 0이면 어떤 물체든 같은 값을 더함
#    dep_out   같은 식을 최종 조건 토큰에 (디코더 전체 출력이 물체에 따라 달라지는 비율)
#  출력 <out>/dec_attn.json, <out>/attn_<sub>.npz (층별 · head 평균 지도 [L, 1374, 256+T] fp16, 104개 평균), 표 줄 ATT|
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument('--repo', default=''); p.add_argument('--data_root', default=''); p.add_argument('--ckpt', default=''); p.add_argument('--out', required=True)
p.add_argument('--subs', default=''); p.add_argument('--bs', type=int, default=8); p.add_argument('--selftest', type=int, default=0)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
NS, GQ, GS, NPRE = 256, 37, 16, 5


def hook(dec):
    """층마다 multihead_attn이 head별 가중치를 돌려주게 바꿈. 반환: 층별 리스트(마지막 호출의 [B, H, Q, M])"""
    store = [None] * len(dec.layers); ext = [dict() for _ in dec.layers]; hook.ext = ext
    for li, layer in enumerate(dec.layers):
        m = layer.multihead_attn; orig = m.forward

        def f(*args, _o=orig, _i=li, _m=m, **kw):
            kw['need_weights'] = True; kw['average_attn_weights'] = False
            out = _o(*args, **kw); w = out[1].float(); store[_i] = w
            E = _m.embed_dim; Hh = _m.num_heads; val = args[2].float()                                   # 값 벡터 크기: v = W_v x + b_v → head별 out_proj
            v = (val @ _m.in_proj_weight[2 * E:].float().t() + _m.in_proj_bias[2 * E:].float()).view(val.shape[0], val.shape[1], Hh, E // Hh)
            Wo = _m.out_proj.weight.float().view(E, Hh, E // Hh)
            vn = torch.einsum('bmhd,ehd->bhme', v, Wo).norm(dim=-1)                                      # [B, H, M]
            ext[_i]['vn'] = vn; ext[_i]['c'] = torch.einsum('bhqm,bhm->bqm', w, vn); ext[_i]['ca'] = out[0].float()
            return out
        m.forward = f
        so = layer.self_attn.forward

        def g(*args, _o=so, _i=li, **kw):
            out = _o(*args, **kw); ext[_i]['sa'] = out[0].float().norm(dim=-1).mean(); return out
        layer.self_attn.forward = g
        layer.linear2.register_forward_hook(lambda mod, inp, out, _i=li: ext[_i].__setitem__('ff', out.float().norm(dim=-1).mean()))
    return store


class Acc:
    def __init__(self, L, H, Q, M):
        self.L, self.H, self.Q, self.M = L, H, Q, M; self.n = 0
        self.sum_map = torch.zeros(L, Q, M, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.v_head = torch.zeros(L, H, dtype=torch.float64); self.v_tok = torch.zeros(L, M - NS, dtype=torch.float64); self.ent = torch.zeros(L, dtype=torch.float64)
        self.maps = [[] for _ in range(L)]                       # 물체별 head 평균 지도 (fp16, CPU)
        self.x = [dict(cv=0.0, vn_s=0.0, vn_v=0.0, sa=0.0, ca=0.0, ff=0.0, sum=None, sq=0.0) for _ in range(L)]; self.o_sum = None; self.o_sq = 0.0

    def add(self, store):
        B = store[0].shape[0]
        for li, w in enumerate(store):                           # [B, H, Q, M]
            self.v_head[li] += w[..., NS:].sum(-1).mean(-1).sum(0).double().cpu()
            wm = w.mean(1)                                       # head 평균 [B, Q, M]
            self.v_tok[li] += wm[..., NS:].mean(1).sum(0).double().cpu()
            self.ent[li] += float((-(w.clamp_min(1e-12).log() * w).sum(-1) / np.log(self.M)).mean((1, 2)).sum())
            self.sum_map[li] += wm.sum(0).double(); self.maps[li].append(wm.half().cpu())
        for li, e in enumerate(getattr(hook, 'ext', [])):
            X = self.x[li]; c = e['c']; X['cv'] += float((c[..., NS:].sum(-1) / c.sum(-1).clamp_min(1e-12)).mean(1).sum())
            X['vn_s'] += float(e['vn'][..., :NS].mean((1, 2)).sum()); X['vn_v'] += float(e['vn'][..., NS:].mean((1, 2)).sum())
            X['sa'] += float(e['sa']) * B; X['ff'] += float(e['ff']) * B; ca = e['ca']; X['ca'] += float(ca.norm(dim=-1).mean()) * B
            X['sum'] = ca.sum(0).double() if X['sum'] is None else X['sum'] + ca.sum(0).double(); X['sq'] = X['sq'] + (ca ** 2).sum(-1).sum(0).double()
        self.n += B

    def add_out(self, tok):                                      # 최종 조건 토큰 [B, Q, 1024]
        tok = tok.float(); self.o_sum = tok.sum(0).double() if self.o_sum is None else self.o_sum + tok.sum(0).double(); self.o_sq = self.o_sq + (tok ** 2).sum(-1).sum(0).double()

    def result(self):
        n = self.n; T = self.M - NS; uni = T / self.M; R = {}
        qy, qx = np.divmod(np.arange(GQ * GQ), GQ); sy, sx = np.divmod(np.arange(NS), GS)
        for li in range(self.L):
            mean = (self.sum_map[li] / n).float(); A = torch.cat(self.maps[li]).to(mean.device).float()      # [n, Q, M]
            cosm = torch.nn.functional.cosine_similarity(A, mean[None], dim=-1).mean()                      # 평균 지도와
            An = torch.nn.functional.normalize(A, dim=-1); S = An.sum(0)                                    # 물체끼리 평균 cos = (|ΣAn|² − n) / (n(n−1))
            pair = (((S * S).sum(-1) - n) / (n * (n - 1))).mean() if n > 1 else torch.tensor(float('nan'))
            ws = mean[NPRE:NPRE + GQ * GQ, :NS]; ws = (ws / ws.sum(-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            cx, cy = ws @ sx, ws @ sy
            rx = float(np.corrcoef(qx, cx)[0, 1]) if cx.std() > 0 else float('nan'); ry = float(np.corrcoef(qy, cy)[0, 1]) if cy.std() > 0 else float('nan')
            recv = mean[:, :NS].mean(0); recv = (recv / recv.sum()).cpu().numpy(); top = np.sort(recv)[::-1]
            vh = (self.v_head[li] / n).numpy()
            R[f'layer{li}'] = dict(v_share=float(vh.mean()), v_ratio=float(vh.mean() / uni), v_head_max=float(vh.max()), v_head_min=float(vh.min()),
                                   v_tok=[float(x) for x in (self.v_tok[li] / n).numpy()], entropy=float(self.ent[li] / n), fixed=float(cosm), pair=float(pair),
                                   pos_rx=rx, pos_ry=ry, top5_share=float(top[:5].sum()), top1_share=float(top[0]), uniform_v=uni)
            X = self.x[li]
            if X['sum'] is not None:
                dep = float((1 - ((X['sum'] / n) ** 2).sum(-1) / (X['sq'] / n).clamp_min(1e-12)).mean())
                R[f'layer{li}'].update(cv_share=X['cv'] / n, cv_ratio=X['cv'] / n / uni, vn_s=X['vn_s'] / n, vn_v=X['vn_v'] / n, blk_sa=X['sa'] / n, blk_ca=X['ca'] / n, blk_ff=X['ff'] / n, dep=dep)
        if self.o_sum is not None: R['out'] = dict(dep_out=float((1 - ((self.o_sum / n) ** 2).sum(-1) / (self.o_sq / n).clamp_min(1e-12)).mean()))
        return R, (self.sum_map / n).half().cpu().numpy()


def line(tag, R):
    for k, v in R.items():
        if k == 'out': print(f"ACT| {tag} 최종 조건 토큰 | 물체에 따라 달라지는 비율 {v['dep_out']:.3f}", flush=True); continue
        if 'cv_share' in v:
            print(f"ACT| {tag} {k} | 값 크기까지 본 V 몫 {v['cv_share']:.4f} (×{v['cv_ratio']:.2f}) | 값 벡터 크기 S {v['vn_s']:.3f} · V {v['vn_v']:.3f} | "
                  f"블록 출력 크기 self {v['blk_sa']:.2f} · cross {v['blk_ca']:.2f} · ff {v['blk_ff']:.2f} | cross 출력이 물체에 따라 달라지는 비율 {v['dep']:.3f}", flush=True)
        print(f"ATT| {tag} {k} | V 몫 {v['v_share']:.4f} (균등 {v['uniform_v']:.4f}, ×{v['v_ratio']:.2f}, head {v['v_head_min']:.4f}~{v['v_head_max']:.4f}) | 엔트로피 {v['entropy']:.3f} | "
              f"평균 지도와 cos {v['fixed']:.3f} · 물체끼리 {v['pair']:.3f} | 위치 상관 x {v['pos_rx']:.2f} y {v['pos_ry']:.2f} | Semantic 상위 5개 몫 {v['top5_share']:.3f} (균등 {5 / NS:.3f})", flush=True)


if a.selftest:                                                   # 가짜 디코더로 hook · 계산 확인
    torch.manual_seed(0); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dec = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(64, 4, 128, batch_first=True, norm_first=True), 2).to(dev).eval(); st = hook(dec)
    acc = Acc(2, 4, NPRE + GQ * GQ, NS + 6)
    for _ in range(3):
        with torch.no_grad(): o = dec(torch.randn(4, NPRE + GQ * GQ, 64, device=dev), torch.randn(4, NS + 6, 64, device=dev))
        assert st[0].shape == (4, 4, NPRE + GQ * GQ, NS + 6) and abs(float(st[0].sum(-1).mean()) - 1) < 1e-4; acc.add(st); acc.add_out(o)
    R, M = acc.result(); line('selftest', R); print('selftest ok', M.shape); sys.exit(0)

from prism import build_from_ckpt
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval(); st = hook(model.dec)
subs = a.subs.split(',') if a.subs else SUBS
TE = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_test_list.txt')) if l.strip()]
L = len(model.dec.layers); H = model.dec.layers[0].multihead_attn.num_heads; Q = model.query.shape[1]; OUT = {'subs': subs, 'ckpt': a.ckpt, 'n': len(TE)}
for sub in subs:
    acc = None
    for i in range(0, len(TE), a.bs):
        ks = TE[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda(); sj = torch.full((len(ks),), SUBS.index(sub), dtype=torch.long, device='cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): o = model(x, fi, sj)
        if acc is None: acc = Acc(L, H, Q, st[0].shape[-1])
        acc.add(st); acc.add_out(o['cond_tokens'])
    R, M = acc.result(); OUT[sub] = R; line(sub, R); np.savez_compressed(f'{a.out}/attn_{sub}.npz', mean=M)
json.dump(OUT, open(f'{a.out}/dec_attn.json', 'w'), indent=1, ensure_ascii=False)
print('DEC_ATTN_DONE', a.out, flush=True)
