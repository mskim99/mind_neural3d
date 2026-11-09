# TRELLIS 텍스트판 외형용 문장 (11-09, CPU): components.npz의 예측 범주(Semantic top-1) → {uid: "a chair"} json
#  --gt 1: 정답 범주로 (상한 확인용, 방법 아님)
# 사용: python text_prompts.py --comp <components.npz> --out <json> [--gt 0]
import json, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--comp', required=True); p.add_argument('--out', required=True); p.add_argument('--gt', type=int, default=0)
a = p.parse_args()
NAME = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'monitor',
        '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone',
        '04530566': 'boat'}   # ShapeNet core 13 범주 (synset → 영어 이름)
z = np.load(a.comp); classes = [str(c) for c in z['classes']]; lab = z['cls'] if a.gt else z['cls_pred']
art = lambda w: ('an ' if w[0] in 'aeiou' else 'a ') + w
P = {str(k).split('/')[-1]: art(NAME.get(classes[int(c)], classes[int(c)])) for k, c in zip(z['name'], lab)}
json.dump(P, open(a.out, 'w', encoding='utf-8'), indent=1)
print(f'문장 {len(P)}개 → {a.out} | 예: {list(P.items())[:3]}', flush=True)
