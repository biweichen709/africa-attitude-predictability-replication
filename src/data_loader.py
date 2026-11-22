import json,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd,pyreadstat
from feature_engineering import FEATURES,TARGETS,MACROS,A,B,C
ROOT=Path(__file__).resolve().parents[1]; NL=chr(10)
ALIASES={'Cape Verde':'Cabo Verde','Cote d’Ivoire':"Cote d'Ivoire","Côte d'Ivoire":"Cote d'Ivoire",'Ivory Coast':"Cote d'Ivoire",'Gambia':'Gambia, The','The Gambia':'Gambia, The','Swaziland':'Eswatini','São Tomé and Príncipe':'Sao Tome and Principe','São Tomé & Príncipe':'Sao Tome and Principe'}
def macro_join(out):
    ALIASES['Congo-Brazzaville']='Congo, Rep.'
    countries=json.loads((ROOT/'data/raw/worldbank_countries.json').read_text())[1]; names={x['name']:x['id'] for x in countries}; names.update({'Cote d’Ivoire':'CIV',"Cote d'Ivoire":'CIV','São Tomé and Príncipe':'STP'})
    out['iso3']=out.country.map(lambda x:names.get(ALIASES.get(x,x))); assert out.iso3.notna().all(),out.loc[out.iso3.isna(),'country'].unique()
    audit=[]
    for name in MACROS:
        path=ROOT/'data/raw'/f'worldbank_{name}.json'; vals=json.loads(path.read_text(encoding='utf-8'))['data'][1] if path.exists() else []; table={}
        for x in vals:
            if x['value'] is not None: table[(x['countryiso3code'],int(x['date']))]=x['value']
        joins={}
        for iso,year in out[['iso3','survey_year']].drop_duplicates().itertuples(index=False,name=None):
            candidates=[yr for yr in range(int(year)-5,int(year)) if (iso,yr) in table]; yr=max(candidates) if candidates else None; value=table.get((iso,yr),np.nan); joins[(iso,year)]=value; audit.append({'iso3':iso,'survey_year':year,'indicator':name,'source_year':yr,'value':value})
        out[name]=[joins[(i,y)] for i,y in zip(out.iso3,out.survey_year)]
    out['log_gdp_pc']=np.log1p(out.gdp_pc); return out,audit
def load(round):
    file=ROOT/f'data/raw/afro_r{round}.sav'; _,meta=pyreadstat.read_sav(str(file),metadataonly=True)
    selected=[f[round-8] for f in FEATURES.values()]+[t[f'r{round}'] for t in TARGETS.values() if t[f'r{round}']]+['COUNTRY','REGION','RESPNO','DATEINTR','withinwt_hh','withinwt_ea','Combinwt_new_hh','COMBINWT','withinwt','WITHINWT']
    selected=[v for v in dict.fromkeys(selected) if v in meta.column_names]; df,m=pyreadstat.read_sav(str(file),usecols=selected,user_missing=True); out=pd.DataFrame(index=df.index); mapping=[]
    out['row_id']=[f'R{round}_{i}' for i in range(len(df))]; countrylabels=m.variable_value_labels['COUNTRY']; out['country']=df.COUNTRY.map(countrylabels); assert out.country.notna().all()
    out['region']=df.REGION.astype(str); out['psu']=out.country+'::'+out.region; out['stratum']=out.country
    # The public merged files do not release EA/PSU IDs. psu is only a legacy column name for a country-region block.
    out['survey_year']=pd.to_datetime(df.DATEINTR).dt.year
    weight_col=next((c for c in ['withinwt_hh','withinwt_ea','withinwt','WITHINWT','Combinwt_new_hh','COMBINWT'] if c in df),None)
    if weight_col is None: raise ValueError('No documented survey weights')
    w=pd.to_numeric(df[weight_col],errors='coerce'); assert w.notna().all() and (w>0).all(); out['weight_raw']=w
    # Country-equal mixture estimand; do not claim population-weighted Africa.
    out['weight']=w/out.assign(w=w).groupby('country').w.transform('sum')*len(df)/out.country.nunique()
    for name,(r8,r9,lo,hi,kind,why) in FEATURES.items():
        raw=r8 if round==8 else r9
        if raw not in df: raise ValueError(f'Missing mapped variable {raw}')
        x=pd.to_numeric(df[raw],errors='coerce')
        if name=='occupation':
            x=x.replace({220:90,1500:90,1580:90,15800:90,1700:90,700:91}); x=x.where(x.isin(list(range(13))+[90,91,95]))
        else: x=x.where(x.between(lo,hi))
        out[name]=x.astype(float); mapping.append({'variable':raw,'name':name,'label':m.column_names_to_labels[raw],'type':kind,'encoding':f'Valid {lo}..{hi}; raw category preserved; occupation special codes to retired=90/self-employed=91','missing':'Invalid/refused/DK to NA; training-fold median or mode; numeric NA flag','role':'feature','reason':why,'missing_rate':float(x.isna().mean())})
    for name,t in TARGETS.items():
        raw=t[f'r{round}']; out[name]=np.nan; out[name+'_raw']=np.nan
        if raw:
            x=df[raw]; out[name+'_raw']=x.where(x.isin(t['valid'])); out[name]=x.isin(t['positive']).astype(float).where(x.isin(t['valid'])); mapping.append({'variable':raw,'name':name,'label':m.column_names_to_labels[raw],'type':'binary outcome from observed item','encoding':f'Valid {t["valid"]}; positive {t["positive"]}','missing':'No target imputation; exclude invalid per task','role':'target','reason':t['description'],'missing_rate':float(out[name].isna().mean())})
    out,audit=macro_join(out); out.to_parquet(ROOT/('data/processed/analysis.parquet' if round==8 else 'data/processed/external_r9.parquet'),index=False)
    metaout={'round':round,'rows':len(df),'countries':out.country.nunique(),'country_names':sorted(out.country.unique().tolist()),'survey_years':sorted(out.survey_year.unique().tolist()),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'weight_column':weight_col,'geographic_group':'country x REGION, NOT released PSU','raw_columns':m.column_names_to_labels,'value_labels':m.variable_value_labels,'targets':{t:{'valid_n':int(out[t].notna().sum()),'weighted_prevalence':float(np.average(out.loc[out[t].notna(),t],weights=out.loc[out[t].notna(),'weight'])) if out[t].notna().any() else None} for t in TARGETS}}
    (ROOT/f'data/processed/r{round}_audit.json').write_text(json.dumps(metaout,ensure_ascii=False,indent=2),encoding='utf-8'); pd.DataFrame(mapping).to_csv(ROOT/f'data/processed/r{round}_variable_mapping.csv',index=False); pd.DataFrame(audit).to_csv(ROOT/f'data/processed/r{round}_macro_join_audit.csv',index=False)
    out.groupby('country').agg(n=('row_id','size'),blocks=('psu','nunique'),first_year=('survey_year','min'),last_year=('survey_year','max')).to_csv(ROOT/f'data/processed/r{round}_country_samples.csv')
    return out,mapping,metaout
def main():
    out,mapping,audit=load(8); ext,_,_=load(9)
    config={'seed':20261121,'targets':TARGETS,'experiments':{'A':A,'B':B,'C':C},'categorical':[k for k,v in FEATURES.items() if v[4]=='categorical'],'grouping':'country-region proxy, not true PSU','weight_estimand':'equal-country weighted mixture','primary_metric':'weighted log loss'}
    (ROOT/'data/processed/config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    table=['# Variable mapping','','Original label, questionnaire-led valid range, encoding, missingness, role and rationale are recorded below. Raw labels and all SPSS value labels are in r8_audit.json.','','| Raw | Analysis name | Original label | Type | Encoding | Missingness rule | Role | Reason |','|---|---|---|---|---|---|---|---|']
    for x in mapping: table.append('| '+' | '.join(str(x[k]).replace('|','/') for k in ['variable','name','label','type','encoding','missing','role','reason'])+' |')
    table+=['','## Macro indicators','World Bank indicators: '+json.dumps(MACROS)+'. Values come from the latest nonmissing year strictly before the interview year and no more than five years earlier. Every source year is retained in macro_join_audit.csv. GDP is log(1+x); other macro features are numeric and fold-imputed. Public health spending is not total welfare spending.','','## Explicit omissions','No parental education, parental occupation, childhood class, monetary income, household net wealth, education-resource history or social-mobility expectations are inferred. Household assets and experienced deprivation are observed proxies with narrower meanings. Religion, ethnicity, voting, trust, participation, ideological items, IDs, sampling weights and target-derived composites are excluded as predictors. COUNTRY and REGION identify evaluation groups only; macro features can still proxy country.','','## Grouping limitation','The merged public files do not contain true enumeration-area identifiers. The internal column named psu is a country-region block used for geographic validation and uncertainty; it must not be described as a true survey PSU. Country is the bootstrap stratum. Exact design-based variances and full survey-stratum reconstruction are unavailable.']
    (ROOT/'reports/variable_mapping.md').write_text(NL.join(table),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['rows','countries','survey_years','weight_column','targets']},indent=2),flush=True)
if __name__=='__main__': main()
