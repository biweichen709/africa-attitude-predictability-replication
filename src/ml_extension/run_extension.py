"""Strict ML extension: train-only tuning/calibration across three transport tasks."""
from __future__ import annotations
import json, os, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from lightgbm import LGBMClassifier

os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ml_extension"
PRED = OUT / "predictions"
OUT.mkdir(parents=True, exist_ok=True); PRED.mkdir(exist_ok=True)
SEED = 20261124
TARGETS = ["economic", "cultural", "political", "welfare"]
SOCIO = ["age","sex","urban","education","employment","occupation","food_deprivation","water_deprivation","medical_deprivation","fuel_deprivation","cash_deprivation","asset_radio","asset_tv","asset_vehicle","asset_computer","asset_bank","asset_phone","electricity_home"]
CONTEXT = ["unsafe_neighbourhood","area_electricity","area_water","area_school","area_clinic","log_gdp_pc","unemployment","gini","education_spending","public_health_spending"]
SETS = {"socioeconomic": SOCIO, "socioeconomic_context": SOCIO + CONTEXT}
CATEGORICAL = ["sex","urban","employment","occupation","asset_radio","asset_tv","asset_vehicle","asset_computer","asset_bank","asset_phone","electricity_home","area_electricity","area_water","area_school","area_clinic"]

def weights(d):
    raw=d.weight_raw.to_numpy(float); key="iso3" if "iso3" in d else "country"
    sums=d.assign(_w=raw).groupby(key)._w.transform("sum").to_numpy()
    return raw/sums*len(d)/d[key].nunique()

def loss(y,p,w):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7); y=np.asarray(y,int)
    return float(np.average(-(y*np.log(p)+(1-y)*np.log(1-p)),weights=w))

def make_preprocessor(cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    cat=[x for x in cols if x in CATEGORICAL]; num=[x for x in cols if x not in CATEGORICAL]
    return ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",StandardScaler())]),num),
        ("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=True))]),cat)
    ])

PARAMS={
 "elastic_net":[{"C":c,"l1_ratio":r} for c,r in [(0.03,0),(0.1,.25),(.3,.5),(1,.5),(3,.75),(10,1)]],
 "lightgbm":[
  {"n_estimators":120,"learning_rate":.03,"num_leaves":15,"max_depth":4,"min_child_samples":100,"reg_alpha":.1,"reg_lambda":3},
  {"n_estimators":220,"learning_rate":.025,"num_leaves":15,"max_depth":5,"min_child_samples":80,"reg_alpha":.5,"reg_lambda":5},
  {"n_estimators":160,"learning_rate":.05,"num_leaves":7,"max_depth":3,"min_child_samples":120,"reg_alpha":.1,"reg_lambda":8},
  {"n_estimators":260,"learning_rate":.02,"num_leaves":31,"max_depth":6,"min_child_samples":160,"reg_alpha":1,"reg_lambda":10},
  {"n_estimators":100,"learning_rate":.07,"num_leaves":15,"max_depth":4,"min_child_samples":160,"reg_alpha":2,"reg_lambda":15},
  {"n_estimators":200,"learning_rate":.035,"num_leaves":7,"max_depth":5,"min_child_samples":60,"reg_alpha":.01,"reg_lambda":2},
 ]}

def model(name,p):
    if name=="elastic_net":
        return LogisticRegression(penalty="elasticnet",solver="saga",C=p["C"],l1_ratio=p["l1_ratio"],max_iter=2500,tol=2e-4,random_state=SEED,n_jobs=2)
    return LGBMClassifier(**p,subsample=.85,colsample_bytree=.8,random_state=SEED,n_jobs=2,verbosity=-1)

def within_folds(d,n,seed):
    assignment=np.full(len(d),-1)
    for j,(_,idx) in enumerate(d.groupby("iso3",sort=True).groups.items()):
        idx=np.asarray(list(idx)); y=d.loc[idx,"y"].to_numpy(int)
        sk=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed+j)
        for f,(_,te) in enumerate(sk.split(idx,y)): assignment[idx[te]]=f
    assert (assignment>=0).all()
    return [(np.flatnonzero(assignment!=f),np.flatnonzero(assignment==f)) for f in range(n)]

def country_folds(d,n,seed):
    return list(GroupKFold(n_splits=n,shuffle=True,random_state=seed).split(d,groups=d.iso3))

def transform_fit(train,valid,cols):
    pre=make_preprocessor(cols); a=pre.fit_transform(train[cols]); b=pre.transform(valid[cols]); return pre,a,b

def fit_predict(train,valid,cols,name,p):
    pre,a,b=transform_fit(train,valid,cols); m=model(name,p); m.fit(a,train.y.to_numpy(int),sample_weight=weights(train)); return m.predict_proba(b)[:,1],pre,m

def tune(train,cols,name,fold_kind,seed):
    splits=within_folds(train,3,seed) if fold_kind=="within" else country_folds(train,3,seed)
    scores=[]
    for p in PARAMS[name]:
        vals=[]; masses=[]
        for tr,va in splits:
            pred,_,_=fit_predict(train.iloc[tr],train.iloc[va],cols,name,p)
            vals.append(loss(train.y.iloc[va],pred,weights(train.iloc[va]))); masses.append(train.iso3.iloc[va].nunique())
        scores.append(float(np.average(vals,weights=masses)))
    best=PARAMS[name][int(np.argmin(scores))]
    oof=np.full(len(train),np.nan)
    for tr,va in splits: oof[va]=fit_predict(train.iloc[tr],train.iloc[va],cols,name,best)[0]
    z=logit(np.clip(oof,1e-6,1-1e-6)).reshape(-1,1)
    cal=LogisticRegression(C=1e6,max_iter=1000).fit(z,train.y.to_numpy(int),sample_weight=weights(train))
    audit={"candidate_losses":scores,"selected":best,"inner_folds":3,"calibration_intercept":float(cal.intercept_[0]),"calibration_slope":float(cal.coef_[0,0]),"calibration_source":"training-only inner-fold OOF predictions"}
    return best,cal,audit

def calibrated(raw,cal): return expit(cal.intercept_[0]+cal.coef_[0,0]*logit(np.clip(raw,1e-6,1-1e-6)))

def attach(d,target):
    z=d[d[target].notna()].copy().reset_index(drop=True); z["y"]=z[target].astype(int)
    z["education_group"]=pd.cut(z.education,[-np.inf,2,5,np.inf],labels=["low","middle","high"]).astype("string").fillna("missing")
    dep=z[[x for x in SOCIO if x.endswith("_deprivation")]].mean(axis=1)
    z["deprivation_group"]=pd.cut(dep,[-np.inf,.99,1.99,np.inf],labels=["low","middle","high"]).astype("string").fillna("missing")
    z["sex_group"]=z.sex.map({1:"male",2:"female"}).fillna("missing")
    z["urban_group"]=z.urban.map({1:"urban",2:"rural",3:"peri-urban"}).fillna("missing")
    return z

def prevalence_baseline(train,test,known_country):
    w=weights(train); grand=float(np.average(train.y,weights=w))
    if not known_country: return np.repeat(grand,len(test))
    rates={c:float(np.average(g.y,weights=weights(train)[g.index])) for c,g in train.groupby("iso3")}
    return test.iso3.map(rates).fillna(grand).to_numpy(float)

def save_cell(task,target,info,name,fold,test,raw,cal,base,audit):
    keep=["row_id","iso3","country","weight_raw","y","sex_group","urban_group","education_group","deprivation_group"]
    p=test[keep].copy();p["task"]=task;p["target"]=target;p["information_set"]=info;p["model"]=name;p["fold"]=fold;p["raw"]=raw;p["calibrated"]=calibrated(raw,cal);p["baseline"]=base
    path=PRED/f"{task}_{target}_{info}_{name}_{fold}.parquet";p.to_parquet(path,index=False)
    audit.update(task=task,target=target,information_set=info,model=name,fold=str(fold),train_only_selection=True,prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/f"audit_{task}_{target}_{info}_{name}_{fold}.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")

def run_nested(r8):
    for target in TARGETS:
        d=attach(r8,target)
        for task in ["known_country","country_holdout"]:
            outer=within_folds(d,3,SEED) if task=="known_country" else country_folds(d,5,SEED)
            for f,(tr,te) in enumerate(outer):
                train=d.iloc[tr].reset_index(drop=True); test=d.iloc[te].reset_index(drop=True)
                if task=="country_holdout": assert set(train.iso3).isdisjoint(test.iso3)
                for info,cols in SETS.items():
                    for name in ["elastic_net","lightgbm"]:
                        path=PRED/f"{task}_{target}_{info}_{name}_{f}.parquet"
                        if path.exists(): continue
                        best,cal,audit=tune(train,cols,name,"within" if task=="known_country" else "country",SEED+f)
                        raw,_,_=fit_predict(train,test,cols,name,best)
                        base=prevalence_baseline(train,test,task=="known_country")
                        save_cell(task,target,info,name,f,test,raw,cal,base,audit)
                        print("COMPLETE",task,target,info,name,f,flush=True)

def run_temporal(r8,r10):
    shared=set(r8.iso3)&set(r10.iso3); assert len(shared)==30
    for target in ["cultural","political"]:
        train=attach(r8,target); test=attach(r10,target)
        test["temporal_cohort"]=np.where(test.iso3.isin(shared),"shared","new")
        for info,cols in SETS.items():
            for name in ["elastic_net","lightgbm"]:
                path=PRED/f"temporal_{target}_{info}_{name}_R10.parquet"
                if path.exists(): continue
                best,cal,audit=tune(train,cols,name,"country",SEED)
                raw,_,_=fit_predict(train,test,cols,name,best); base=prevalence_baseline(train,test,True)
                save_cell("temporal",target,info,name,"R10",test,raw,cal,base,audit)
                p=pd.read_parquet(path);p["temporal_cohort"]=test.temporal_cohort.to_numpy();p.to_parquet(path,index=False)
                print("COMPLETE temporal",target,info,name,flush=True)

if __name__=="__main__":
    r8=pd.read_parquet(ROOT/"results/geography_thesis/analysis/r8_prepared.parquet")
    r10=pd.read_parquet(ROOT/"results/geography_thesis/analysis/r10_prepared.parquet")
    run_nested(r8);run_temporal(r8,r10)
    manifest={"seed":SEED,"countries_r8":int(r8.iso3.nunique()),"countries_r10":int(r10.iso3.nunique()),"shared_iso3":sorted(set(r8.iso3)&set(r10.iso3)),"prediction_files":len(list(PRED.glob("*.parquet"))),"specification":"reports/ml_extension/analysis_specification.md"}
    (OUT/"completion_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("ML EXTENSION COMPLETE",manifest["prediction_files"],flush=True)
