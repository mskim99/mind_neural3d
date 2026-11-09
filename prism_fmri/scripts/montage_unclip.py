# A안 1단계 몽타주 (11-09): 물체마다 한 줄 = 자극 프레임 0 | unCLIP 이미지 | pred | text | unclip_gt | unclip_full | gt (gscheck 흰 배경 렌더 첫 뷰)
# 사용: python montage_unclip.py --root <prism_unclip> --frames <tgt>/frames --names <uid 목록> --prompt <json> --out <png>
import os, json, argparse
from PIL import Image, ImageDraw
p = argparse.ArgumentParser()
p.add_argument('--root', required=True); p.add_argument('--frames', required=True); p.add_argument('--names', required=True)
p.add_argument('--prompt', default=''); p.add_argument('--out', required=True); p.add_argument('--size', type=int, default=200)
a = p.parse_args()
S = a.size; names = [l.strip() for l in open(a.names) if l.strip()]
P = json.load(open(a.prompt, encoding='utf-8')) if a.prompt else {}
V = ['pred', 'text', 'unclip_gt', 'unclip_full', 'gt']
H = ['stimulus f0', 'unCLIP img', 'pred', 'text (now)', 'unCLIP->SLat', 'unCLIP full', 'GT token']
top, left = 24, 0
canvas = Image.new('RGB', (S * len(H), top + S * len(names)), 'white'); d = ImageDraw.Draw(canvas)
for j, h in enumerate(H): d.text((j * S + 4, 6), h, fill='black')
def put(img, r, c): canvas.paste(img.convert('RGB').resize((S, S)), (c * S, top + r * S))
for r, n in enumerate(names):
    put(Image.open(f'{a.frames}/{n}/0.png'), r, 0)
    f = f'{a.root}/img_gt/{n}/0.png'
    if os.path.exists(f): put(Image.open(f), r, 1)
    for c, v in enumerate(V):
        f = f'{a.root}/quick/{v}/{n}_gs.png'
        if os.path.exists(f):
            g = Image.open(f); w = g.height                       # 가로로 이어 붙인 뷰 중 첫 뷰
            put(g.crop((0, 0, w, w)), r, c + 2)
    d.text((3 * S + 4, top + r * S + S - 14), P.get(n, ''), fill='blue')
canvas.save(a.out); print('MONTAGE', a.out, canvas.size, flush=True)
