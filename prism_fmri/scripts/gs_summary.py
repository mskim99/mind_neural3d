# gscheck.jsonl 한 줄 요약 (평균 Gaussian · 텍스처 밝기 · 채도). 사용: python gs_summary.py <gscheck.jsonl>
import sys, json
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
m = lambda k, i: sum(x[k][i] for x in r) / max(len(r), 1)
print('Gaussian 밝기 %.1f 채도 %.1f | 텍스처 밝기 %.1f 채도 %.1f (n %d)' % (m('gauss_mean_chroma', 0), m('gauss_mean_chroma', 1), m('tex_mean_chroma', 0), m('tex_mean_chroma', 1), len(r)))
