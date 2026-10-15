#!/usr/bin/env python3
"""Aggregate C6 fusion_sweep.csv results using paired seeds and fixed holdout queries.

Does not retrain/evaluate checkpoints. Validation GT candidates are used only
for diagnostic N-way; train-bank metrics rank TRAIN-UID captions/images.
"""
from __future__ import annotations
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = ("val_10way", "val_2way", "within_cat_2way", "train_bank_cat_top1", "train_bank_cat_top5")


def fmt(v, paired=False):
    if not v: return "n/a"
    m = statistics.mean(v)*100
    suffix = 'pp' if paired else '%'
    if len(v)==1: return f"{m:+.2f}{suffix} (n=1)" if paired else f"{m:.2f}{suffix} (n=1)"
    sd = statistics.stdev(v)*100
    return f"{m:+.2f}±{sd:.2f}{suffix}" if paired else f"{m:.2f}±{sd:.2f}{suffix}"


def write_csv(path, rows, cols):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    print('[*] Saved',path,'rows=',len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runs',default='C6B0,C6R_DROP04,C6N_SAMPLE,C6RN_COMBINED')
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--selection',choices=['best_objce','best_10way'],default='best_10way')
    p.add_argument('--base',default='C6R_DROP04')
    p.add_argument('--select_by',choices=['train_bank_cat_top1','train_bank_cat_top5','val_10way'],default='train_bank_cat_top1')
    a=p.parse_args()
    runs=[v.strip() for v in a.runs.replace(' ', ',').split(',') if v.strip()]
    seeds=[int(v) for v in a.seeds.replace(' ', ',').split(',') if v.strip()]
    all_rows=[]
    for run in runs:
        for seed in seeds:
            path=a.root/'fusion_diagnostics'/f'{run}_{a.selection}_seed{seed}'/'fusion_sweep.csv'
            if not path.is_file():
                print('[Missing]',path)
                continue
            with path.open(newline='',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            expected=set(FIELDS+('alpha','scope'))
            if not rows or not expected.issubset(rows[0]):
                raise RuntimeError(f'Missing columns {expected-set(rows[0]) if rows else expected} in {path}')
            keys=set()
            for v in rows:
                item={'run':run,'seed':seed,'selection':a.selection,'alpha':float(v['alpha']),'scope':v['scope']}
                item.update({k:float(v[k]) if v[k] else float('nan') for k in FIELDS})
                key=(item['alpha'],item['scope'])
                if key in keys: raise ValueError(f'Duplicate alpha/scope={key} in {path}')
                keys.add(key)
                all_rows.append(item)
    if not all_rows:
        print('[ERROR] no fusion_sweep.csv found'); return 1
    out=a.root/'C6_summary'
    cols=['run','seed','selection','alpha','scope',*FIELDS]
    write_csv(out/f'C6_retrieval_all_{a.selection}.csv',all_rows,cols)
    hold={(r['run'],r['seed'],r['alpha']):r for r in all_rows if r['scope']=='holdout'}
    fixed=list(hold.values())
    print('\n=== Fixed alpha on same holdout (mean±sample SD over seeds) ===')
    by=defaultdict(list)
    for r in fixed: by[(r['run'],r['alpha'])].append(r)
    print('run | alpha | n | '+ ' | '.join(FIELDS))
    for (run,alpha),rr in sorted(by.items()):
        print(f'{run} | {alpha:g} | {len(rr)} | '+ ' | '.join(fmt([r[k] for r in rr]) for k in FIELDS))
    paired=[]
    for r in fixed:
        if r['run']==a.base: continue
        b=hold.get((a.base,r['seed'],r['alpha']))
        if b is None: continue
        dr={'run':r['run'],'baseline':a.base,'seed':r['seed'],'alpha':r['alpha'],'selection':a.selection}
        dr.update({k:r[k]-b[k] for k in FIELDS})
        paired.append(dr)
    print(f'\n=== Paired deltas versus {a.base} (same seed, alpha, holdout) ===')
    bydelta=defaultdict(list)
    for r in paired: bydelta[(r['run'],r['alpha'])].append(r)
    print('run | alpha | n | '+ ' | '.join(FIELDS))
    for (run,alpha),rr in sorted(bydelta.items()):
        print(f'{run} | {alpha:g} | {len(rr)} | '+ ' | '.join(fmt([r[k] for r in rr],paired=True) for k in FIELDS))
    write_csv(out/f'C6_retrieval_paired_{a.selection}.csv',paired,
              ['run','baseline','seed','alpha','selection',*FIELDS])

    # Tune alpha only on calibration samples, then read the selected alpha's holdout metrics.
    calibr=defaultdict(list)
    for r in all_rows:
        if r['scope']=='calibration': calibr[(r['run'],r['seed'])].append(r)
    selected=[]
    for key,rr in sorted(calibr.items()):
        # Tie-break: other train-bank metric, then val 10-way; do not use holdout to select alpha.
        other='train_bank_cat_top5' if a.select_by=='train_bank_cat_top1' else 'train_bank_cat_top1'
        best=max(rr,key=lambda r:(r[a.select_by],r[other],r['val_10way']))
        h=hold.get((key[0],key[1],best['alpha']))
        if h is None: continue
        selected.append({**h,'selected_by':a.select_by,'calibration_score':best[a.select_by]})
    print(f'\n=== Calibration-selected alpha by {a.select_by}; metrics on disjoint holdout ===')
    grouped=defaultdict(list)
    for r in selected: grouped[r['run']].append(r)
    print('run | alpha by seed | n | '+ ' | '.join(FIELDS))
    for run,rr in sorted(grouped.items()):
        print(f"{run} | "+','.join(f"{r['seed']}:{r['alpha']:g}" for r in rr)+f" | {len(rr)} | "+
              ' | '.join(fmt([r[k] for r in rr]) for k in FIELDS))
    write_csv(out/f'C6_retrieval_calibrated_{a.selection}.csv',selected,cols+['selected_by','calibration_score'])
    print('\nCaveat: same original validation was used to select training checkpoints; holdout is')
    print('disjoint only from alpha calibration, not from checkpoint selection. No official test used.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
