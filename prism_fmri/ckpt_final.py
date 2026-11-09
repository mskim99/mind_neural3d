# 학습 중간 체크포인트(model_latest.pt 사본) → infer_prism.py가 읽는 형식(model_final.pt와 같은 키) (11-10)
#  추가 키: subs · classes(train_prism.py SYN과 같음) · enc_paths. 옵티마이저 상태는 뺌
# 사용: python ckpt_final.py <입력 .pt> <출력 .pt>
import sys, torch
CLASSES = ['02691156', '02828884', '02933112', '02958343', '03001627', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04401088', '04530566']
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False); a = ck['args']; subs = a['subs'].split(',')
torch.save({'model': ck['model'], 'step': ck['step'], 'args': a, 'subs': subs, 'classes': sorted(CLASSES),
            'enc_paths': [f"{a['enc_dir']}/sub{s[-2:]}_encoder.pt" for s in subs]}, sys.argv[2])
print('CKPT', ck['step'], '→', sys.argv[2])
