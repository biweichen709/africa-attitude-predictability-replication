import sys,json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,pyreadstat
from feature_engineering import FEATURES
from data_loader import macro_join
MAP={'age':'Q1','sex':'Q101','urban':'URBRUR','education':'Q96','employment':'Q94A','occupation':'Q94B','food_deprivation':'Q7A','water_deprivation':'Q7B','medical_deprivation':'Q7C','fuel_deprivation':'Q7D','cash_deprivation':'Q7E','asset_radio':'Q90A','asset_tv':'Q90B','asset_vehicle':'Q90C','asset_computer':'Q90D','asset_bank':'Q90E','asset_phone':'Q90F','electricity_home':'Q93A','unsafe_neighbourhood':'Q9','area_electricity':'EA_SVC_A','area_water':'EA_SVC_B','area_school':'EA_FAC_B','area_clinic':'EA_FAC_D'}
def main():
    p=ROOT/'data/raw/afro_r10.sav'
    if not p.exists():
        with zipfile.ZipFile(ROOT/'data/raw/afro_r10.sav.zip') as z:
            ns=[n for n in z.namelist() if n.lower().endswith('.sav') and not n.startswith('__MACOSX/')]; assert len(ns)==1; p.write_bytes(z.read(ns[0]))
    cols=list(MAP.values())+['COUNTRY','REGION','DATEINTR','withinwt_hh','Q88C','Q22']; raw,m=pyreadstat.read_sav(p,usecols=list(dict.fromkeys(cols)),user_missing=True); out=pd.DataFrame(index=raw.index); mappings=[]
    out['row_id']=['R10_'+str(i) for i in range(len(raw))]; out['country']=raw.COUNTRY.map(m.variable_value_labels['COUNTRY']); out['region']=raw.REGION.astype(str); out['psu']=out.country+'::'+out.region; out['survey_year']=pd.to_datetime(raw.DATEINTR).dt.year; out['weight_raw']=raw.withinwt_hh.astype(float); assert out.weight_raw.gt(0).all()
    for name,var in MAP.items():
        lo,hi=FEATURES[name][2:4]; x=pd.to_numeric(raw[var],errors='coerce')
        if name=='occupation': x=x.replace({9995:95}); x=x.where(x.isin(list(range(13))+[95]))
        else: x=x.where(x.between(lo,hi))
        out[name]=x.astype(float); mappings.append({'analysis':name,'r10':var,'label':m.column_names_to_labels[var],'valid_range':[lo,hi],'missing_fraction':float(x.isna().mean())})
    out['cultural']=raw.Q88C.isin([3,4,5]).astype(float).where(raw.Q88C.isin([1,2,3,4,5])); out['political']=(raw.Q22==3).astype(float).where(raw.Q22.isin([1,2,3])); out,audit=macro_join(out); out.to_parquet(ROOT/'data/processed/external_r10.parquet',index=False); pd.DataFrame(audit).to_csv(ROOT/'results/v2/r10_macro_join_audit.csv',index=False)
    report={'rows':len(out),'countries':sorted(out.country.unique().tolist()),'years':sorted(out.survey_year.unique().tolist()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mapping':mappings,'target_mapping':{'cultural':'R10 Q88C; positive 3/4/5; valid 1–5','political':'R10 Q22; positive 3; valid 1–3'},'codebook_discrepancy':'The downloaded merged codebook shows 45,600 tabulated records, while the actual February 2026 release has 50,961 rows. Use the SAV value labels and explicitly mapped codes; do not copy codebook frequencies as this dataset results.','excluded':'All GPS coordinates, interview comments, free text, personal follow-up information, and political predictors excluded.'}
    (ROOT/'reports/v2/round10_harmonisation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print('HARMONISED',len(out),out.country.nunique(),report['years'],'(no model evaluation performed)',flush=True)
if __name__=='__main__': main()
