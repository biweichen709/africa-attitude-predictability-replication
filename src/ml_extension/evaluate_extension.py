"""Aggregate strict-validation predictions without releasing respondent-level records."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/ml_extension"; PRED=OUT/"predictions"; SEED=20261124

def weights(d):
    raw=d.weight_raw.to_numpy(float); sums=d.assign(_w=raw).groupby("iso3")._w.transform("sum").to_numpy()
    return raw/sums*len(d)/d.iso3.nunique()

def ll(y,p,w):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7); y=np.asarray(y,int)
    return float(np.average(-(y*np.log(p)+(1-y)*np.log(1-p)),weights=w))

def metric(d,col):
    y=d.y.to_numpy(int); p=np.clip(d[col].to_numpy(float),1e-7,1-1e-7); w=weights(d)
    auc=roc_auc_score(y,p,sample_weight=w) if len(np.unique(y))==2 else np.nan
    country_auc=[]
    for _,g in d.groupby("iso3"):
        if g.y.nunique()==2: country_auc.append(roc_auc_score(g.y,g[col],sample_weight=g.weight_raw))
    intercept=slope=np.nan
    if np.std(p)>1e-10:
        try:
            fit=sm.GLM(y,sm.add_constant(logit(p)),family=sm.families.Binomial(),freq_weights=w).fit()
            intercept,slope=map(float,fit.params)
        except Exception: pass
    bins=np.minimum((p*10).astype(int),9); ece=0.0
    for b in range(10):
        ix=bins==b
        if ix.any(): ece += w[ix].sum()/w.sum()*abs(np.average(y[ix],weights=w[ix])-np.average(p[ix],weights=w[ix]))
    return {"n":len(d),"countries":d.iso3.nunique(),"prevalence":float(np.average(y,weights=w)),"mean_prediction":float(np.average(p,weights=w)),"equal_country_log_loss":ll(y,p,w),"equal_country_brier":float(np.average((y-p)**2,weights=w)),"pooled_auc":float(auc),"mean_country_auc":float(np.nanmean(country_auc)),"calibration_intercept":intercept,"calibration_slope":slope,"ece10":float(ece)}

def load_all():
    groups={}
    for f in sorted(PRED.glob("*.parquet")):
        d=pd.read_parquet(f); key=tuple(d[x].iloc[0] for x in ["task","target","information_set","model"])
        groups.setdefault(key,[]).append(d)
    return {k:pd.concat(v,ignore_index=True).sort_values("row_id").reset_index(drop=True) for k,v in groups.items()}

def cohorts(task,d):
    if task!="temporal": return [("all",d)]
    return [("all",d),("shared",d[d.temporal_cohort=="shared"]),("new",d[d.temporal_cohort=="new"])]

def country_losses(d,col):
    return pd.Series({c:ll(g.y,g[col],g.weight_raw) for c,g in d.groupby("iso3")},name=col)

def paired_boot(a,b,reps=2000):
    z=pd.concat([a.rename("new"),b.rename("reference")],axis=1).dropna(); delta=(z.new-z.reference).to_numpy(); rng=np.random.default_rng(SEED); draw=rng.integers(0,len(delta),(reps,len(delta))); boot=delta[draw].mean(1)
    return {"countries":len(delta),"delta_log_loss":float(delta.mean()),"lower":float(np.quantile(boot,.025)),"upper":float(np.quantile(boot,.975))}

def decision(d,col):
    y=d.y.to_numpy(int);p=d[col].to_numpy(float);w=weights(d);prev=np.average(y,weights=w);rows=[]
    for t in np.arange(.10,.501,.05):
        pos=p>=t; nb=(w[pos]*(y[pos]==1)).sum()/w.sum()-(w[pos]*(y[pos]==0)).sum()/w.sum()*t/(1-t)
        all_nb=prev-(1-prev)*t/(1-t)
        rows.append({"threshold":round(float(t),2),"net_benefit":float(nb),"standardized_net_benefit":float(nb/prev),"treat_all_net_benefit":float(all_nb),"treat_none_net_benefit":0.0})
    return rows

def main():
    groups=load_all(); metrics=[]; country=[]; subgroup=[]; curves=[]
    sample=pd.read_csv(OUT/'sample_audit.csv').set_index(['round','target'])
    assert len(groups)==40
    for (task,target,info,model),g in groups.items():
        assert g.row_id.is_unique and len(g)==sample.loc[(10 if task=='temporal' else 8,target),'outcome_valid']
        assert g.fold.nunique()=={'temporal':1,'known_country':3,'country_holdout':5}[task]
        assert np.isfinite(g[['calibrated','baseline','raw']]).all().all()
    for (task,target,info,model),d in groups.items():
        for cohort,g in cohorts(task,d):
            for prob in ["raw","calibrated","baseline"]:
                metrics.append({"task":task,"target":target,"cohort":cohort,"information_set":info,"model":model,"probability":prob,**metric(g,prob)})
                for iso,h in g.groupby("iso3"):
                    country.append({"task":task,"target":target,"cohort":cohort,"information_set":info,"model":model,"probability":prob,"iso3":iso,"n":len(h),"log_loss":ll(h.y,h[prob],h.weight_raw),"brier":float(np.average((h.y-h[prob])**2,weights=h.weight_raw)),"auc":roc_auc_score(h.y,h[prob],sample_weight=h.weight_raw) if h.y.nunique()==2 else np.nan})
            for variable in ["sex_group","urban_group","education_group","deprivation_group"]:
                for level,h in g.groupby(variable,dropna=False):
                    for prob in ["calibrated","baseline"]:
                        subgroup.append({"task":task,"target":target,"cohort":cohort,"information_set":info,"model":model,"variable":variable,"group":str(level),"probability":prob,**metric(h,prob)})
            for prob in ["calibrated","baseline"]:
                for row in decision(g,prob): curves.append({"task":task,"target":target,"cohort":cohort,"information_set":info,"model":model,"probability":prob,**row})
    pd.DataFrame(metrics).to_csv(OUT/"metrics.csv",index=False);pd.DataFrame(country).to_csv(OUT/"country_metrics.csv",index=False);pd.DataFrame(subgroup).to_csv(OUT/"subgroup_metrics.csv",index=False);pd.DataFrame(curves).to_csv(OUT/"decision_curves.csv",index=False)
    contrasts=[]
    for (task,target,info,model),d in groups.items():
        for cohort,g in cohorts(task,d): contrasts.append({"task":task,"target":target,"cohort":cohort,"contrast_type":"model_vs_prevalence","information_set":info,"new_model":model,"reference_model":"prevalence","new_information":info,"reference_information":info,**paired_boot(country_losses(g,"calibrated"),country_losses(g,"baseline"))})
    for task in sorted({k[0] for k in groups}):
      for target in sorted({k[1] for k in groups if k[0]==task}):
       for info in ["socioeconomic","socioeconomic_context"]:
        if (task,target,info,"elastic_net") not in groups or (task,target,info,"lightgbm") not in groups: continue
        a=groups[(task,target,info,"lightgbm")];b=groups[(task,target,info,"elastic_net")]
        for cohort,ga in cohorts(task,a):
            gb=dict(cohorts(task,b))[cohort]; assert ga.row_id.tolist()==gb.row_id.tolist()
            contrasts.append({"task":task,"target":target,"cohort":cohort,"contrast_type":"algorithm","information_set":info,"new_model":"lightgbm","reference_model":"elastic_net","new_information":info,"reference_information":info,**paired_boot(country_losses(ga,"calibrated"),country_losses(gb,"calibrated"))})
       for model in ["elastic_net","lightgbm"]:
        a=groups[(task,target,"socioeconomic_context",model)];b=groups[(task,target,"socioeconomic",model)]
        for cohort,ga in cohorts(task,a):
            gb=dict(cohorts(task,b))[cohort]; assert ga.row_id.tolist()==gb.row_id.tolist()
            contrasts.append({"task":task,"target":target,"cohort":cohort,"contrast_type":"information","information_set":"context_minus_socioeconomic","new_model":model,"reference_model":model,"new_information":"socioeconomic_context","reference_information":"socioeconomic",**paired_boot(country_losses(ga,"calibrated"),country_losses(gb,"calibrated"))})
    pd.DataFrame(contrasts).to_csv(OUT/"paired_country_contrasts.csv",index=False)
    sg=pd.DataFrame(subgroup); disp=[]
    for keys,g in sg[sg.probability=="calibrated"].groupby(["task","target","cohort","information_set","model","variable"]):
        disp.append(dict(zip(["task","target","cohort","information_set","model","variable"],keys),max_minus_min_log_loss=float(g.equal_country_log_loss.max()-g.equal_country_log_loss.min()),worst_group=g.loc[g.equal_country_log_loss.idxmax(),"group"],best_group=g.loc[g.equal_country_log_loss.idxmin(),"group"]))
    pd.DataFrame(disp).to_csv(OUT/"subgroup_disparities.csv",index=False)
    audit={"prediction_groups":len(groups),"metric_rows":len(metrics),"country_rows":len(country),"subgroup_rows":len(subgroup),"bootstrap_reps":2000,"primary_metric":"equal-country weighted log loss","respondent_outputs_released":False}
    (OUT/"evaluation_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print("EVALUATED",audit,flush=True)

if __name__=="__main__": main()
