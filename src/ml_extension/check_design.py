"""Verify evaluation coverage and training boundaries; save only aggregate fold fingerprints."""
import hashlib
import json
import numpy as np
import pandas as pd
from run_extension import ROOT, OUT, TARGETS, SETS, SEED, attach, within_folds, country_folds


def fingerprint(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def record(train, test, kind, target, outer, inner):
    assert train.row_id.is_unique and test.row_id.is_unique
    assert not set(train.row_id).intersection(test.row_id)
    overlap = set(train.iso3).intersection(test.iso3)
    if kind == 'known_country':
        assert overlap == set(test.iso3) == set(train.iso3)
    elif kind == 'country_holdout' or inner >= 0:
        assert not overlap
    else:
        assert len(overlap) == 30 and len(set(test.iso3)-overlap) == 8
    return dict(task=kind, target=target, outer=outer, inner=inner,
                training_n=len(train), evaluation_n=len(test), shared_countries=len(overlap),
                training_sha256=fingerprint(train.row_id), evaluation_sha256=fingerprint(test.row_id))


def main():
    base = ROOT/'results/geography_thesis/analysis'
    a, b = [pd.read_parquet(base/f'r{w}_prepared.parquet') for w in (8,10)]
    for data in (a,b):
        assert data.row_id.is_unique and data.iso3.notna().all()
        assert np.isfinite(data.weight_raw).all() and data.weight_raw.gt(0).all()
        for cols in SETS.values():
            assert set(cols) <= set(data) and not set(cols).intersection([*TARGETS,'row_id','iso3','country','wave','weight_raw'])
    rows=[]
    for target in TARGETS:
        d=attach(a,target)
        assert set(d.y)=={0,1}
        for task in ('known_country','country_holdout','temporal'):
            if task=='temporal' and target not in ('cultural','political'): continue
            splitter=within_folds if task=='known_country' else country_folds
            if task=='temporal': pairs=[(d,attach(b,target))]
            else:
                outer=splitter(d,3 if task=='known_country' else 5,SEED)
                assert np.array_equal(np.sort(np.concatenate([te for _,te in outer])),np.arange(len(d)))
                pairs=[(d.iloc[tr].reset_index(drop=True),d.iloc[te]) for tr,te in outer]
            for f,(train,test) in enumerate(pairs):
                rows.append(record(train,test,task,target,f,-1))
                splits=splitter(train,3,SEED+f)
                assert np.array_equal(np.sort(np.concatenate([v for _,v in splits])),np.arange(len(train)))
                for j,(u,v) in enumerate(splits):rows.append(record(train.iloc[u],train.iloc[v],task,target,f,j))
    pd.DataFrame(rows).to_csv(OUT/'fold_design_audit.csv',index=False)
    print(f'{len(rows)} outer/inner partition checks passed.')


if __name__=='__main__':main()
