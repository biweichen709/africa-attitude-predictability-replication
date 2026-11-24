"""ISO3/sample audit and weighted Round 8 to Round 10 predictor shift."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"results/ml_extension";OUT.mkdir(exist_ok=True)
from run_extension import SOCIO, CONTEXT, CATEGORICAL, weights

def main():
    a=pd.read_parquet(ROOT/"results/geography_thesis/analysis/r8_prepared.parquet");b=pd.read_parquet(ROOT/"results/geography_thesis/analysis/r10_prepared.parquet")
    shared=sorted(set(a.iso3)&set(b.iso3)); assert len(shared)==30
    rows=[]
    for cohort,z in [("shared",b[b.iso3.isin(shared)]),("new",b[~b.iso3.isin(shared)])]:
      for x in SOCIO+CONTEXT:
        aa=a[a[x].notna()];bb=z[z[x].notna()]
        wa=weights(aa);wb=weights(bb);ma=np.average(aa[x],weights=wa);mb=np.average(bb[x],weights=wb);va=np.average((aa[x]-ma)**2,weights=wa);vb=np.average((bb[x]-mb)**2,weights=wb)
        rows.append({"cohort":cohort,"variable":x,"variable_type":"categorical_code" if x in CATEGORICAL else "numeric","r8_mean":ma,"r10_mean":mb,"standardized_mean_difference":float((mb-ma)/np.sqrt((va+vb)/2)) if va+vb>0 else np.nan,"r8_missing":float(a[x].isna().mean()),"r10_missing":float(z[x].isna().mean())})
    pd.DataFrame(rows).to_csv(OUT/"predictor_shift.csv",index=False)
    sample=[]
    for wave,d,targets in [(8,a,["economic","cultural","political","welfare"]),(10,b,["cultural","political"])]:
      for t in targets: sample.append({"round":wave,"target":t,"all_respondents":len(d),"countries":d.iso3.nunique(),"outcome_valid":int(d[t].notna().sum()),"complete_core":int((d[t].notna()&d.core_complete).sum())})
    pd.DataFrame(sample).to_csv(OUT/"sample_audit.csv",index=False)
    identity=pd.concat([a[["country","iso3"]],b[["country","iso3"]]]).drop_duplicates().sort_values(["iso3","country"]);identity.to_csv(OUT/"country_identity_audit.csv",index=False)
    report={"shared_countries":30,"r8_countries":34,"r10_countries":38,"shared_iso3":shared,"label_discrepancy":{"R8":"Gambia","R10":"Gambia, The","iso3":"GMB"},"sample_definitions":{"outcome_valid":"nonmissing binary outcome","complete_core":"outcome valid and all eight parsimonious core predictors observed"}}
    (OUT/"identity_and_sample_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("AUDITED",report,flush=True)
if __name__=="__main__":main()
