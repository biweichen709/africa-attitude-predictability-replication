import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import roc_auc_score,log_loss
from descriptive_and_fe import weights,auc_pairs
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/geography_thesis/analysis'
MODELS=['pooled','country_only','centred','multilevel','context_only','pooled_context','global','centred_transductive']
def metrics(d,m):
 y=d.y.to_numpy();p=np.clip(d[m].to_numpy(),1e-8,1-1e-8);w=weights(d);ca=[];country=[]
 for c,g in d.groupby('iso3'):
  yy=g.y.to_numpy();pp=np.clip(g[m].to_numpy(),1e-8,1-1e-8);ww=g.weight_raw.to_numpy();auc=roc_auc_score(yy,pp,sample_weight=ww) if len(np.unique(yy))==2 else np.nan;ca.append(auc);country.append(dict(iso3=c,n=len(g),auc=auc,log_loss=log_loss(yy,pp,sample_weight=ww,labels=[0,1]),brier=np.average((yy-pp)**2,weights=ww),prevalence=np.average(yy,weights=ww),mean_prediction=np.average(pp,weights=ww)))
 slope=intercept=np.nan
 if np.std(p)>1e-9:
  cal=sm.GLM(y,sm.add_constant(logit(p)),family=sm.families.Binomial(),freq_weights=w).fit();intercept,slope=cal.params
 bins=np.clip((p*10).astype(int),0,9);ec=0;binrows=[]
 for k in np.unique(bins):
  ix=bins==k;pred=np.average(p[ix],weights=w[ix]);obs=np.average(y[ix],weights=w[ix]);mass=w[ix].sum()/w.sum();ec+=mass*abs(pred-obs);binrows.append(dict(bin=int(k),n=int(ix.sum()),predicted=pred,observed=obs,mass=mass))
 return dict(n=len(d),countries=d.iso3.nunique(),pooled_auc=roc_auc_score(y,p,sample_weight=w),mean_country_auc=np.nanmean(ca),sd_country_auc=np.nanstd(ca),min_country_auc=np.nanmin(ca),max_country_auc=np.nanmax(ca),log_loss=log_loss(y,p,sample_weight=w),brier=np.average((y-p)**2,weights=w),calibration_intercept=intercept,calibration_slope=slope,ece10=ec,mean_prediction=np.average(p,weights=w),prevalence=np.average(y,weights=w)),pd.DataFrame(country),pd.DataFrame(binrows)
def main():
 summaries=[];country=[];bins=[];contrasts=[];pairs=[];rng=np.random.default_rng(20261122)
 for path in sorted(OUT.glob('pred_*.csv')):
  d=pd.read_csv(path);target=d.target.iloc[0];setting=d.setting.iloc[0];assert d.row_id.is_unique
  if setting=='loco':assert d.iso3.nunique()==34 and (d.iso3==d.fold).all()
  subsets={'all':d}
  if setting=='temporal':subsets.update({c:g for c,g in d.groupby('cohort')})
  for cohort,g in subsets.items():
   bymodel={}
   for m in MODELS:
    if m not in g:continue
    assert np.isfinite(g[m]).all() and g[m].between(0,1).all();meta=dict(target=target,setting=setting,cohort=cohort,model=m);r,c,b=metrics(g,m);summaries.append(meta|r);country.append(c.assign(**meta));bins.append(b.assign(**meta));bymodel[m]=c.set_index('iso3')
   for new,old in [('multilevel','country_only'),('multilevel','pooled'),('pooled_context','pooled'),('centred','global'),('context_only','global'),('centred_transductive','centred')]:
    if new not in bymodel:continue
    for metric in ['log_loss','brier','auc']:
     delta=bymodel[new][metric]-bymodel[old][metric];delta=delta.dropna();n=len(delta);idx=rng.integers(0,n,(2000,n));bt=delta.to_numpy()[idx].mean(axis=1);lo,hi=np.quantile(bt,[.025,.975]);contrasts.append(dict(target=target,setting=setting,cohort=cohort,new=new,reference=old,metric=metric,delta=delta.mean(),lower=lo,upper=hi,countries=n,inference='paired country bootstrap conditional on fitted predictions'))
   if cohort=='all' and setting=='known_country':
    for m in ['pooled','country_only','multilevel']:
     matrix,r=auc_pairs(g,m);pairs.append(dict(target=target,model=m,setting=setting,**r));matrix.to_csv(OUT/f'pair_{setting}_{target}_{m}.csv',index=False)
 pd.DataFrame(summaries).to_csv(OUT/'prediction_metrics.csv',index=False);pd.concat(country).to_csv(OUT/'prediction_country_metrics.csv',index=False);pd.concat(bins).to_csv(OUT/'calibration_bins.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'paired_country_bootstrap.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'matched_auc_pair_decomposition.csv',index=False)
 print(pd.DataFrame(summaries)[['target','setting','cohort','model','n','pooled_auc','mean_country_auc','log_loss','brier']].to_string(index=False))
if __name__=='__main__':main()
