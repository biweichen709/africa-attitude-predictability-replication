import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import t
from prepare_analysis import CORE
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/geography_thesis/analysis'
def main():
 a=pd.read_parquet(OUT/'r8_prepared.parquet');b=pd.read_parquet(OUT/'r10_prepared.parquet');common=set(a.iso3)&set(b.iso3);assert len(common)==30;all=pd.concat([a,b],ignore_index=True);rows=[]
 for target in ['cultural','political']:
  d=all[all.iso3.isin(common)&all.core_complete&all[target].notna()].reset_index(drop=True);d['country_wave']=d.iso3+'_'+d.wave.astype(str);late=(d.wave==10).astype(float);w=d.weight_raw.to_numpy()/d.groupby('country_wave').weight_raw.transform('sum').to_numpy()*len(d)/d.country_wave.nunique();x=d[CORE].astype(float).copy()
  for col in CORE:x['change_'+col]=x[col]*late
  x=x.join(pd.get_dummies(d.country_wave,prefix='cell',drop_first=True,dtype=float));x=sm.add_constant(x);y=d[target].to_numpy();fit=sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=100);p=fit.predict();xx=x.to_numpy();bread=np.linalg.pinv(xx.T@((w*p*(1-p))[:,None]*xx));scores=pd.DataFrame(xx*(w*(y-p))[:,None]).groupby(d.iso3).sum().to_numpy();J=30;cov=bread@(scores.T@scores)@bread*J/(J-1)*(len(d)-1)/(len(d)-xx.shape[1]);se=np.sqrt(np.maximum(np.diag(cov),0));critical=t.ppf(.975,J-1)
  for col in CORE:
   i=x.columns.get_loc('change_'+col);base=float(fit.params[col]);delta=float(fit.params.iloc[i]);rows.append(dict(target=target,variable=col,n=len(d),countries=J,n_r8=int((d.wave==8).sum()),n_r10=int((d.wave==10).sum()),r8_slope=base,r10_slope=base+delta,change=delta,country_cluster_se=se[i],lower=delta-critical*se[i],upper=delta+critical*se[i],p_value=2*t.sf(abs(delta/se[i]),J-1),converged=fit.converged,status='Post-specification exploratory sensitivity; logit-scale association change, not individual or causal change'))
 r=pd.DataFrame(rows);r.to_csv(OUT/'temporal_slope_changes.csv',index=False);print(r.to_string(index=False))
if __name__=='__main__':main()
