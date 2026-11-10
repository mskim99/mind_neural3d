# 자극 영상 프레임 → 캡션 (11-10, TRELLIS 텍스트판의 풍부한 문장용) — venv_unclip (transformers 4.46.3), GPU
#  모델 Qwen/Qwen2-VL-7B-Instruct (4.46.3에서 되는 Qwen2-VL, 2.5는 4.49+ 필요). HF 캐시는 $W/cache/hf
#  입력: <frames>/<uid>/{0,2,4}.png (prep_shape_targets.py가 만든 영상 프레임 0 · 64 · 128 = 방위 0° · 120° · 240°, 피험자가 본 자극 그대로)
#  정답 범주 이름을 같이 줌 (train 캡션 = 학습 정답, test 캡션은 상한 확인에만 씀 — 방법에 쓰지 않음)
#  출력 jsonl 한 줄 = {uid, syn, raw, shape, color, material} (모양 · 색 · 재질을 나눠 받아 색 넣은 판 · 뺀 판을 만듦), 이어 쓰기 됨
# 사용: python caption_vlm.py --frames <tgt>/frames --list <core_train_list> <core_test_list> --out <captions.jsonl> [--bs 8]
import os, re, json, argparse, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

NAME = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'monitor',
        '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone',
        '04530566': 'boat'}   # text_prompts.py와 같음
PROMPT = ('These are three views (0, 120 and 240 degrees around it) of the same 3D {cat}, rendered on a white background. '
          'Describe this object for a text-to-3D generator. Answer with JSON only, no other text: '
          '{{"shape": "<geometry, parts and proportions; no colors or materials; at most 20 words>", '
          '"color": "<main colors of its parts; at most 10 words>", "material": "<at most 6 words>"}}')
p = argparse.ArgumentParser()
p.add_argument('--frames', required=True); p.add_argument('--list', nargs='+', required=True); p.add_argument('--out', required=True)
p.add_argument('--model', default='Qwen/Qwen2-VL-7B-Instruct'); p.add_argument('--bs', type=int, default=8)
p.add_argument('--views', default='0,2,4'); p.add_argument('--max_px', type=int, default=448 * 448); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
V = [int(v) for v in a.views.split(',')]
ids = [l.strip() for f in a.list for l in open(f) if l.strip()]
done = set()
if os.path.exists(a.out):
    for l in open(a.out, encoding='utf-8'):
        try:
            d = json.loads(l)
            if d.get('shape'): done.add(d['uid'])   # 파싱 실패한 줄은 다시 실행 때 재시도 (caption_prompts.py는 같은 uid의 마지막 성공 줄을 씀)
        except Exception: pass
todo = [k for k in ids if k.split('/')[1] not in done and all(os.path.exists(f'{a.frames}/{k.split("/")[1]}/{v}.png') for v in V)]
miss = [k for k in ids if not all(os.path.exists(f'{a.frames}/{k.split("/")[1]}/{v}.png') for v in V)]
todo = todo[:a.limit] if a.limit else todo
print(f'[cap] 목록 {len(ids)} · 이미 {len(done)} · 할 것 {len(todo)} · 프레임 없음 {len(miss)} {miss[:3]}', flush=True)
proc = AutoProcessor.from_pretrained(a.model, min_pixels=128 * 128, max_pixels=a.max_px); proc.tokenizer.padding_side = 'left'
model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()


def parse(t):
    m = re.search(r'\{.*\}', t, re.S)
    try: d = json.loads(m.group(0)) if m else {}
    except Exception: d = {}
    g = lambda k: str(d.get(k, '')).strip().rstrip('.')
    return g('shape'), g('color'), g('material')


fo = open(a.out, 'a', encoding='utf-8'); n_bad = 0
for i in range(0, len(todo), a.bs):
    ks = todo[i:i + a.bs]; texts, imgs = [], []
    for k in ks:
        syn, uid = k.split('/')
        msg = [{'role': 'user', 'content': [{'type': 'image'} for _ in V] + [{'type': 'text', 'text': PROMPT.format(cat=NAME[syn])}]}]
        texts.append(proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        imgs += [Image.open(f'{a.frames}/{uid}/{v}.png').convert('RGB') for v in V]
    x = proc(text=texts, images=imgs, padding=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=128, do_sample=False)
    outs = proc.batch_decode(y[:, x['input_ids'].shape[1]:], skip_special_tokens=True)
    for k, t in zip(ks, outs):
        syn, uid = k.split('/'); sh, co, ma = parse(t); n_bad += int(not sh)
        fo.write(json.dumps({'uid': uid, 'syn': syn, 'raw': t, 'shape': sh, 'color': co, 'material': ma}, ensure_ascii=False) + '\n')
    fo.flush()
    if i // a.bs % 10 == 0: print(f'[cap] {i + len(ks)}/{len(todo)} 파싱 실패 {n_bad} | {ks[0]}: {outs[0][:160]!r}', flush=True)
print(f'CAPTION_DONE {len(todo)}개 (파싱 실패 {n_bad}) → {a.out}', flush=True)
