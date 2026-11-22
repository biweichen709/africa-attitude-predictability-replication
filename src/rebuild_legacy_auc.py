"""Refit locked C-logistic specifications; no tuning or shipped models."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/geography_thesis'))
from preprocessing import make_preprocessor
from descriptive_and_fe import weights,auc_pairs
def main():
 a=pd.read_parquet(ROOT/'data/processed/analysis.parquet');b=pd.read_parquet(ROOT/'data/processed/external_r10.parquet');cfg=json.loads((ROOT/'metadata/analysis_config.json').read_text());spec=json.loads((ROOT/'metadata/legacy_logistic_specifications.json').read_text());rows=[]
 for target,s in spec.items():
  tr=a[a[target].notna()].reset_index(drop=True);te=b[b[target].notna()].reset_index(drop=True);cols=s['features'];pre=make_preprocessor(cols,cfg['categorical']);x=np.asarray(pre.fit_transform(tr[cols]),dtype=np.float32);z=np.asarray(pre.transform(te[cols]),dtype=np.float32)
  model=LogisticRegression(C=s['parameters']['C'],max_iter=800,solver='lbfgs',random_state=20261121);model.fit(x,tr[target].to_numpy(int),sample_weight=weights(tr))
  d=te[['iso3','weight_raw']].copy();d['y']=te[target].to_numpy(int);d['raw']=model.predict_proba(z)[:,1];matrix,r=auc_pairs(d,'raw');matrix.to_csv(ROOT/f'results/geography_thesis/analysis/legacy_pair_matrix_{target}.csv',index=False);rows.append(dict(target=target,**r))
 pd.DataFrame(rows).to_csv(ROOT/'results/geography_thesis/analysis/legacy_auc_pair_decomposition.csv',index=False)
if __name__=='__main__':main()
