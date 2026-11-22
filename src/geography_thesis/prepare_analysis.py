from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/geography_thesis/analysis';OUT.mkdir(parents=True,exist_ok=True)
CORE=['age10','female','urban_i','education','deprivation','assets','electricity','employed']
DEPR=[x+'_deprivation' for x in ['food','water','medical','fuel','cash']]
ASSETS=['asset_'+x for x in ['radio','tv','vehicle','computer','bank','phone']]
def make(d):
 d=d.copy();d['age10']=d.age/10;d['female']=(d.sex==2).astype(float);d['urban_i']=(d.urban==1).astype(float)
 d['deprivation']=d[DEPR].mean(axis=1,skipna=False);d['assets']=d[ASSETS].gt(0).sum(axis=1).astype(float);d.loc[d[ASSETS].isna().any(axis=1),'assets']=np.nan
 d['electricity']=d.electricity_home;d['employed']=d.employment.isin([2,3]).astype(float);d.loc[d.employment.isna(),'employed']=np.nan
 d['occupation_factor']=d.occupation.astype('Int64').astype('string').fillna('missing')
 d['core_complete']=d[CORE].notna().all(axis=1);d['context_year']=d.survey_year.astype(int)-1
 return d
def main():
 a=make(pd.read_parquet(ROOT/'data/processed/analysis.parquet'));b=make(pd.read_parquet(ROOT/'data/processed/external_r10.parquet'))
 v=pd.read_csv(ROOT/'data/raw/geography/vdem_context_2018_2025.csv');print(v[['country_name','country_text_id']].drop_duplicates().to_string(index=False))
 # country_text_id uses ISO-like identifiers; explicit exceptions checked against country_name.
 aliases={'GAM':'GMB','SAF':'ZAF','SWA':'SWZ','ZAM':'ZMB','ZIM':'ZWE','CON':'COG','DRC':'COD','MAU':'MRT','GUI':'GIN','BFO':'BFA','CHA':'TCD'}
 v['iso3']=v.country_text_id.replace(aliases); assert not v.duplicated(['iso3','year']).any()
 records=[];coverage=[]
 for wave,d,targets in [(8,a,['economic','cultural','political','welfare']),(10,b,['cultural','political'])]:
  d=d.merge(v[['iso3','year','country_name','v2x_polyarchy','v2x_libdem']].rename(columns={'year':'context_year','country_name':'vdem_country'}),on=['iso3','context_year'],how='left',validate='many_to_one')
  assert d.v2x_polyarchy.notna().all(),d.loc[d.v2x_polyarchy.isna(),['iso3','country']].drop_duplicates().to_dict('records')
  coverage.append(d[['iso3','country','survey_year','context_year','vdem_country','v2x_polyarchy','log_gdp_pc']].drop_duplicates().assign(round=wave))
  d['wave']=wave;d.to_parquet(OUT/f'r{wave}_prepared.parquet',index=False);d.to_csv(OUT/f'r{wave}_prepared.csv',index=False)
  for target in targets:
   for c,g in d.groupby('iso3'):
    records.append(dict(round=wave,target=target,iso3=c,total_n=len(g),outcome_n=int(g[target].notna().sum()),analysis_n=int((g[target].notna()&g.core_complete).sum()),outcome_missing=float(g[target].isna().mean()),core_missing=float((~g.core_complete).mean())))
 pd.concat(coverage).to_csv(OUT/'context_join_audit.csv',index=False);pd.DataFrame(records).to_csv(OUT/'sample_flow.csv',index=False)
 dictionary=[]
 descriptions={'age10':'Age divided by ten; no age imputation','female':'sex=2 versus sex=1; survey binary sex measure','urban_i':'urban=1 versus rural/periurban codes 2/3','education':'Original ordered 0–9 education code; linear approximation','deprivation':'Arithmetic mean of five 0–4 deprivation items, all required','assets':'Count of six assets accessible personally or through household (code>0), all required','electricity':'Household electricity 0/1','employed':'employment codes 2/3 versus 0/1','v2x_polyarchy':'V-Dem v16 electoral democracy index, prior calendar year','log_gdp_pc':'Inherited lagged log GDP per capita; original World Bank provenance retained'}
 for name,definition in descriptions.items(): dictionary.append(dict(variable=name,definition=definition,level='country-year' if name in ['v2x_polyarchy','log_gdp_pc'] else 'individual',r8_missing=float(a[name].isna().mean()) if name in a else 0,r10_missing=float(b[name].isna().mean()) if name in b else 0))
 pd.DataFrame(dictionary).to_csv(OUT/'analysis_dictionary.csv',index=False)
 cfg=json.loads((ROOT/'data/processed/config.json').read_text());full=[]
 for name in a.columns:
  full.append(dict(variable=name,r8_dtype=str(a[name].dtype),r8_missing=float(a[name].isna().mean()),r10_available=name in b,r10_missing=float(b[name].isna().mean()) if name in b else None,role='primary core' if name in CORE else 'outcome' if name in cfg['targets'] else 'identifier/context/legacy diagnostic',definition=descriptions.get(name,cfg['targets'].get(name,{}).get('description','See inherited config and metadata; not newly interpreted'))))
 pd.DataFrame(full).to_csv(OUT/'full_variable_inventory.csv',index=False)
 spec=ROOT/'docs/analysis_specification.md';(OUT/'execution_freeze.json').write_text(json.dumps(dict(date='2026-11-22',specification_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),core=CORE,legacy_guard_passed=None,analysis_status='retrospective exploratory'),indent=2))
 print('PREPARED',len(a),len(b),flush=True)
if __name__=='__main__':main()
