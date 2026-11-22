import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import t
from sklearn.metrics import roc_auc_score
from prepare_analysis import make,CORE
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/geography_thesis/analysis';OUT.mkdir(parents=True,exist_ok=True)
TARGETS=['economic','cultural','political','welfare']
def weights(d):
 return d.weight_raw.to_numpy()/d.groupby('iso3').weight_raw.transform('sum').to_numpy()*len(d)/d.iso3.nunique()
def auc_pairs(d,score):
 d=d.copy();d['w']=weights(d);countries=sorted(d.iso3.unique()); pos=d.y==1; wp=d.loc[pos,'w'].sum();wn=d.loc[~pos,'w'].sum();rows=[]
 for c in countries:
  a=d[(d.iso3==c)&pos];ap=a[score].to_numpy();aw=a.w.to_numpy()
  for e in countries:
   b=d[(d.iso3==e)&~pos].sort_values(score);bp=b[score].to_numpy();bw=b.w.to_numpy();cum=np.r_[0,bw.cumsum()];lo=np.searchsorted(bp,ap,'left');hi=np.searchsorted(bp,ap,'right');numer=float(np.sum(aw*(cum[lo]+.5*(cum[hi]-cum[lo]))));mass=aw.sum()*bw.sum()/(wp*wn)
   rows.append(dict(positive_country=c,negative_country=e,pair_mass=mass,pair_auc=numer/(aw.sum()*bw.sum()) if mass>0 else np.nan,contribution=numer/(wp*wn),diagonal=c==e))
 r=pd.DataFrame(rows);diag=r[r.diagonal];cross=r[~r.diagonal];pooled=roc_auc_score(d.y,d[score],sample_weight=d.w);assert abs(r.contribution.sum()-pooled)<1e-10
 return r,dict(pooled_auc=pooled,within_pair_mass=diag.pair_mass.sum(),within_pair_auc=diag.contribution.sum()/diag.pair_mass.sum(),cross_pair_auc=cross.contribution.sum()/cross.pair_mass.sum(),diagonal_contribution=diag.contribution.sum(),cross_contribution=cross.contribution.sum(),mean_country_auc=diag.pair_auc.mean(),reconstruction_error=r.contribution.sum()-pooled)
def variance(d,target):
 d=d.copy();d['w']=weights(d);d['edgroup']=pd.cut(d.education,[-1,2,5,10],labels=['low','middle','high']);d['depgroup']=pd.cut(d.deprivation,[-1,.999999,1.999999,5],labels=['low','middle','high']);d['cell']=d.edgroup.astype(str)+'_'+d.depgroup.astype(str)+'_'+d.employed.astype(str);cells=[];countries=[]
 for c,g in d.groupby('iso3'):
  pj=np.average(g[target],weights=g.w);within=0;group=0
  for k,h in g.groupby('cell'):
   q=np.average(h[target],weights=h.w);mass=h.w.sum()/g.w.sum();within+=mass*q*(1-q);group+=mass*(q-pj)**2;neff=h.w.sum()**2/(h.w**2).sum();cells.append(dict(target=target,iso3=c,cell=k,n=len(h),n_eff=neff,prevalence=q,discordance=2*q*(1-q),country_mass=mass,both_responses=h[target].nunique()==2))
  countries.append(dict(target=target,iso3=c,n=len(g),prevalence=pj,within_cell=within,between_cell=group))
 c=pd.DataFrame(countries);p=c.prevalence.mean();between=float(c.prevalence.var(ddof=0));v=p*(1-p);within=c.within_cell.mean();group=c.between_cell.mean();assert abs(v-between-within-group)<1e-10
 rng=np.random.default_rng(20261122);ix=rng.integers(0,len(c),(2000,len(c)));means=c.prevalence.to_numpy()[ix];vt=means.mean(axis=1)*(1-means.mean(axis=1));bv=means.var(axis=1);bounds=np.quantile(bv/vt,[.025,.975]);r=dict(target=target,n=len(d),countries=len(c),prevalence=p,total_variance=v,between_country=between,between_positions_within_country=group,within_position=within,between_country_share=between/v,between_position_share=group/v,within_position_share=within/v,between_country_share_lo=bounds[0],between_country_share_hi=bounds[1]);return r,c,pd.DataFrame(cells)
def fe_fit(d,target,label,mode='equal',occupation=False,education_cat=False):
 d=d.copy();cols=CORE.copy();w=weights(d) if mode=='equal' else np.ones(len(d)) if mode=='unweighted' else d.weight_raw.to_numpy()/d.weight_raw.mean();x=d[cols].astype(float)
 if education_cat:x=x.drop(columns='education').join(pd.get_dummies(d.education.astype(int).astype(str),prefix='education',drop_first=True,dtype=float))
 if occupation:x=x.join(pd.get_dummies(d.occupation_factor,prefix='occupation',drop_first=True,dtype=float))
 x=x.join(pd.get_dummies(d.iso3,prefix='country',drop_first=True,dtype=float));x=sm.add_constant(x);y=d[target].to_numpy();fit=sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=100);pr=fit.predict();xx=x.to_numpy();bread=np.linalg.pinv(xx.T@((w*pr*(1-pr))[:,None]*xx));score=xx*(w*(y-pr))[:,None];s=pd.DataFrame(score).groupby(d.iso3.to_numpy()).sum().to_numpy();j=d.iso3.nunique();cov=bread@(s.T@s)@bread*j/(j-1)*(len(d)-1)/(len(d)-xx.shape[1]);se=np.sqrt(np.maximum(np.diag(cov),0));q=t.ppf(.975,j-1);rows=[]
 for i,name in enumerate(x.columns):
  if name.startswith('country_') or name=='const':continue
  beta=fit.params.iloc[i];rows.append(dict(target=target,specification=label,term=name,n=len(d),countries=j,estimate=beta,cluster_se=se[i],lower=beta-q*se[i],upper=beta+q*se[i],p_value=2*t.sf(abs(beta/se[i]),j-1),converged=fit.converged))
 return rows
def main():
 a=make(pd.read_parquet(ROOT/'data/processed/analysis.parquet'));b=make(pd.read_parquet(ROOT/'data/processed/external_r10.parquet'));vs=[];cs=[];cells=[];fes=[];prev=[]
 for wave,d,ts in [(8,a,TARGETS),(10,b,['cultural','political'])]:
  for target in ts:
   for c,g in d[d[target].notna()].groupby('iso3'):prev.append(dict(wave=wave,target=target,iso3=c,n=len(g),prevalence=np.average(g[target],weights=g.weight_raw)))
 for target in TARGETS:
  d=a[a.core_complete&a[target].notna()].reset_index(drop=True);r,c,h=variance(d,target);vs.append(r);cs.append(c);cells.append(h)
  for label,mode,occ,cat in [('primary','equal',False,False),('unweighted','unweighted',False,False),('raw_weights','raw',False,False),('occupation','equal',True,False),('categorical_education','equal',False,True)]:
   fes.extend(fe_fit(d,target,label,mode,occ,cat));print('FE',target,label,flush=True)
  common=a[a.core_complete&a[TARGETS].notna().all(axis=1)].reset_index(drop=True);fes.extend(fe_fit(common,target,'common_four_outcomes'));rv,_,_=variance(common,target);rv['sample']='common_four';vs.append(rv)
  if target in ['cultural','economic']:
   ss=d[d[target+'_raw']!=3].copy();ss[target]=(ss[target+'_raw']>=4).astype(float);ss=ss.reset_index(drop=True);fes.extend(fe_fit(ss,target,'exclude_neutral'));rv,_,_=variance(ss,target);rv['sample']='exclude_neutral';vs.append(rv)
  if (d.age>100).any():fes.extend(fe_fit(d[d.age<=100].reset_index(drop=True),target,'exclude_age_above100'))
 pd.DataFrame(vs).fillna({'sample':'primary'}).to_csv(OUT/'observed_variance.csv',index=False);pd.concat(cs).to_csv(OUT/'country_descriptives.csv',index=False);pd.concat(cells).to_csv(OUT/'position_cells.csv',index=False);pd.DataFrame(fes).to_csv(OUT/'fixed_effects_cluster_robust.csv',index=False);pd.DataFrame(prev).to_csv(OUT/'prevalence_by_wave_country.csv',index=False)
 h6=[]
 for target,g in pd.concat(cells).groupby('target'):
  good=g[(g.n>=50)&(g.n_eff>=30)];h6.append(dict(target=target,total_cells=len(g),adequate_cells=len(good),adequate_cells_mixed=int(good.both_responses.sum()),adequate_mass=good.country_mass.sum()/34,mixed_mass_all=g.loc[g.both_responses,'country_mass'].sum()/34,adequate_mixed_fraction=float(good.both_responses.mean()),adequate_weighted_discordance=float(np.average(good.discordance,weights=good.country_mass))))
 pd.DataFrame(h6).to_csv(OUT/'within_position_diagnostics.csv',index=False);print('DESCRIPTIVE_FE_COMPLETE',flush=True)
if __name__=='__main__':main()
