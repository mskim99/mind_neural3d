#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--prototype_ckpt', required=True)
    p.add_argument('--data_path', default='/data/jionkim/neuro_3D/')
    p.add_argument('--rendered_view_path', default='/data/jionkim/neuro_3D/render_grid_v4')
    p.add_argument('--sub_id', default='sub01')
    p.add_argument('--out_dir', default='./fixed_text_audit')
    p.add_argument('--top_pairs', type=int, default=30)
    return p.parse_args()

def category_from_dataset_name(name):
    key=str(name)[3:]
    if '_' in key and key.rsplit('_',1)[-1].isdigit():
        return key.rsplit('_',1)[0]
    return key

def to_flat_feature(x):
    if torch.is_tensor(x): t=x.detach().float().cpu()
    else: t=torch.as_tensor(np.asarray(x), dtype=torch.float32)
    return t.reshape(-1)

def retrieval_metrics(features, labels, prototypes):
    x=F.normalize(features.float(), dim=-1)
    p=F.normalize(prototypes.float(), dim=-1)
    sim=x@p.T
    pred=sim.argmax(1)
    top1=(pred==labels).float().mean().item()
    top5_idx=sim.topk(k=min(5,sim.shape[1]), dim=1).indices
    top5=(top5_idx==labels[:,None]).any(1).float().mean().item()
    rows=torch.arange(len(labels))
    correct=sim[rows,labels]
    wrong=sim.clone(); wrong[rows,labels]=-torch.inf
    best_wrong,best_wrong_idx=wrong.max(1)
    margin=correct-best_wrong
    return dict(sim=sim,pred=pred,top1=float(top1),top5=float(top5),correct=correct,
                best_wrong=best_wrong,best_wrong_idx=best_wrong_idx,margin=margin,
                margin_mean=float(margin.mean()),margin_median=float(margin.median()),
                margin_min=float(margin.min()))

def build_text_tensor(ds, expected_dim):
    C,O=int(ds.cls_num),int(ds.obj_num)
    feats=[]; names=[]; cats=[]
    for c in range(C):
        rf=[]; rn=[]; rc=[]
        for o in range(O):
            dataset_name=str(ds.name_list[c,o]); key=dataset_name[3:]
            cat=category_from_dataset_name(dataset_name)
            feat=to_flat_feature(ds.clip_features[key]['text'])
            if feat.numel()!=expected_dim:
                raise RuntimeError(f'{key}: text dim={feat.numel()} != {expected_dim}')
            rf.append(F.normalize(feat,dim=0)); rn.append(key); rc.append(cat)
        if len(set(rc))!=1: raise RuntimeError(f'cls_index={c} categories={sorted(set(rc))}')
        feats.append(torch.stack(rf)); names.append(rn); cats.append(rc[0])
    return torch.stack(feats),names,cats

def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args(); out=Path(a.out_dir).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    pack=torch.load(Path(a.prototype_ckpt).expanduser().resolve(), map_location='cpu')
    P=F.normalize(pack['prototypes'].detach().float().cpu(),dim=-1)
    saved_categories=list(pack.get('categories',[])); saved_sources=list(pack.get('source_objects',[]))
    if P.ndim!=2 or P.shape[0]!=72: raise RuntimeError(f'Expected [72,D], got {tuple(P.shape)}')
    C,D=P.shape
    ds=AllDataFeatureTwoEEG(data_path=a.data_path,sub_list=[a.sub_id],train=True,test_mean=False,
                            num_frames=6,rendered_view_path=a.rendered_view_path,aug_data=False,
                            strict_rendered_views=True)
    text,names,cats=build_text_tensor(ds,D); O=text.shape[1]
    mapping_ok=True
    if saved_categories:
        bad=[(c,cats[c],saved_categories[c]) for c in range(C) if str(cats[c])!=str(saved_categories[c])]
        if bad: mapping_ok=False; print('[MAPPING] FAIL',bad[:10])
        else: print('[MAPPING] PASS')
    if saved_sources:
        bad=[c for c in range(C) if list(saved_sources[c])!=list(names[c])]
        if bad: mapping_ok=False; print('[SOURCE OBJECTS] FAIL classes=',bad[:10])
        else: print('[SOURCE OBJECTS] PASS')

    # Standard self-included retrieval
    X=text.reshape(C*O,D); y=torch.arange(C).repeat_interleave(O)
    std=retrieval_metrics(X,y,P); rows=[]
    for c in range(C):
        for o in range(O):
            i=c*O+o; pred=int(std['pred'][i]); wrong=int(std['best_wrong_idx'][i])
            rows.append(dict(cls_index=c,obj_index=o,object_name=names[c][o],category=cats[c],
                             pred_cls=pred,pred_category=cats[pred],correct=int(pred==c),
                             correct_cos=float(std['correct'][i]),best_wrong_cls=wrong,
                             best_wrong_category=cats[wrong],best_wrong_cos=float(std['best_wrong'][i]),
                             margin=float(std['margin'][i])))
    save_csv(out/'text_to_saved_prototype.csv',rows,list(rows[0].keys()))

    # Leave-one-object-column-out retrieval
    loo_rows=[]; all_top1=[]; all_top5=[]; all_m=[]
    for o in range(O):
        keep=[j for j in range(O) if j!=o]
        Ploo=F.normalize(text[:,keep,:].mean(1),dim=-1)
        met=retrieval_metrics(text[:,o,:],torch.arange(C),Ploo)
        top5_idx=met['sim'].topk(k=5,dim=1).indices
        for c in range(C):
            pred=int(met['pred'][c]); wrong=int(met['best_wrong_idx'][c]); t5=int((top5_idx[c]==c).any())
            all_top1.append(int(pred==c)); all_top5.append(t5); all_m.append(float(met['margin'][c]))
            loo_rows.append(dict(heldout_obj_index=o,cls_index=c,object_name=names[c][o],category=cats[c],
                                 pred_cls=pred,pred_category=cats[pred],top1_correct=int(pred==c),
                                 top5_correct=t5,correct_cos=float(met['correct'][c]),best_wrong_cls=wrong,
                                 best_wrong_category=cats[wrong],best_wrong_cos=float(met['best_wrong'][c]),
                                 margin=float(met['margin'][c])))
    save_csv(out/'text_to_loo_prototype.csv',loo_rows,list(loo_rows[0].keys()))
    bycol=[]
    for o in range(O):
        r=[x for x in loo_rows if x['heldout_obj_index']==o]
        bycol.append(dict(heldout_obj_index=o,suffix_example=names[0][o].rsplit('_',1)[-1],
                          top1=float(np.mean([x['top1_correct'] for x in r])),
                          top5=float(np.mean([x['top5_correct'] for x in r])),
                          margin_mean=float(np.mean([x['margin'] for x in r])),
                          margin_min=float(np.min([x['margin'] for x in r]))))
    save_csv(out/'loo_by_object_column.csv',bycol,list(bycol[0].keys()))

    # Prototype similarity audit
    S=(P@P.T).numpy(); mask=~np.eye(C,dtype=bool); off=S[mask]
    stats={k:float(v) for k,v in {
        'mean':np.mean(off),'std':np.std(off),'min':np.min(off),'median':np.median(off),
        'p90':np.quantile(off,.90),'p95':np.quantile(off,.95),'p99':np.quantile(off,.99),'max':np.max(off)}.items()}
    with (out/'prototype_cosine_matrix.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['category']+cats)
        for c in range(C): w.writerow([cats[c]]+[f'{v:.8f}' for v in S[c]])
    pairs=[]
    for i in range(C):
        for j in range(i+1,C): pairs.append(dict(cls_i=i,category_i=cats[i],cls_j=j,category_j=cats[j],cosine=float(S[i,j])))
    pairs.sort(key=lambda x:x['cosine'],reverse=True)
    save_csv(out/'prototype_closest_pairs.csv',pairs,list(pairs[0].keys()))
    try:
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(12,10)); ax=fig.add_subplot(111); im=ax.imshow(S,aspect='auto')
        ax.set_title('Fixed text prototype cosine similarity (72x72)'); ax.set_xlabel('class'); ax.set_ylabel('class')
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/'prototype_cosine_heatmap.png',dpi=180); plt.close(fig)
    except Exception as e: print('[heatmap warning]',e)

    summary={
      'mapping_ok':mapping_ok,'num_classes':C,'objects_per_class':O,'feature_dim':D,
      'standard_self_included':{'top1':std['top1'],'top5':std['top5'],'margin_mean':std['margin_mean'],'margin_median':std['margin_median'],'margin_min':std['margin_min']},
      'leave_one_object_column_out':{'top1':float(np.mean(all_top1)),'top5':float(np.mean(all_top5)),'margin_mean':float(np.mean(all_m)),'margin_median':float(np.median(all_m)),'margin_min':float(np.min(all_m))},
      'prototype_off_diagonal_cosine':stats,'top_confusable_pairs':pairs[:a.top_pairs]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')

    print('\n'+'='*72+'\nFIXED TEXT SPACE AUDIT\n'+'='*72)
    print('mapping/order:', 'PASS' if mapping_ok else 'FAIL')
    print(f'classes/objects/dim: {C}/{O}/{D}')
    print('\n[1] text -> SAVED prototype (self-included; optimistic)')
    print(f" Top1={std['top1']:.4f} Top5={std['top5']:.4f} margin_mean={std['margin_mean']:+.6f} margin_min={std['margin_min']:+.6f}")
    print('\n[2] text -> LOO prototype (PRIMARY)')
    print(f" Top1={np.mean(all_top1):.4f} Top5={np.mean(all_top5):.4f} margin_mean={np.mean(all_m):+.6f} margin_min={np.min(all_m):+.6f}")
    print('\n[3] prototype off-diagonal cosine')
    for k in ['mean','std','median','p90','p95','p99','max']: print(f' {k:>6s}={stats[k]:.6f}')
    print(f'\n[4] top-{a.top_pairs} closest pairs')
    for r in pairs[:a.top_pairs]: print(f" {r['cls_i']:02d}:{r['category_i']:<18s} <-> {r['cls_j']:02d}:{r['category_j']:<18s} cos={r['cosine']:.6f}")
    print('\nSaved to',out)

if __name__=='__main__': main()
