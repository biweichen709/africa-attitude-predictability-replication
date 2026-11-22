from pathlib import Path
import sys,json
import pandas as pd,pyreadstat
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/v2'))
from feature_engineering import FEATURES,TARGETS,MACROS
from prepare_round10 import MAP
out=ROOT/'results/geography_thesis/analysis';records=[]
for wave,path in [(8,ROOT/'data/raw/afro_r8.sav'),(10,ROOT/'data/raw/afro_r10.sav')]:
 _,m=pyreadstat.read_sav(path,metadataonly=True);d=pd.read_parquet(ROOT/('data/processed/analysis.parquet' if wave==8 else 'data/processed/external_r10.parquet'))
 for col in d.columns:
  var=FEATURES[col][0] if col in FEATURES and wave==8 else MAP.get(col) if col in FEATURES else TARGETS[col]['r8'] if col in TARGETS and wave==8 else {'cultural':'Q88C','political':'Q22'}.get(col)
  level='country-year' if col in MACROS or col=='log_gdp_pc' else 'individual/identifier'
  definition=m.column_names_to_labels.get(var,'') if var else ''
  if col in FEATURES:definition=FEATURES[col][5]+'; '+definition
  if col in TARGETS:definition=TARGETS[col]['description']
  if col in MACROS:definition='World Bank '+MACROS[col]+'; lag rule and historical observation in inherited macro join audit'
  if col=='log_gdp_pc':definition='Natural logarithm log(1 + lagged GDP per capita), preserving the inherited WDI series'
  if col.endswith('_raw') and col[:-4] in TARGETS:var=TARGETS[col[:-4]]['r8'];definition='Valid original response codes before binary recoding'
  special={'row_id':'Stable source-wave row identifier','country':'Country label from SAV metadata','region':'Original survey-region code stored as string','psu':'Country::region proxy; NOT verified original sampling PSU','stratum':'Country proxy, not original full survey strata','survey_year':'Calendar year extracted from interview date','weight_raw':'Original withinwt_hh household weight, with inherited R8 positive floor','weight':'Inherited equal-country rescaled household weight; recomputed per eligible sample in new analyses','iso3':'Country ISO3 identifier'}
  definition=special.get(col,definition);records.append(dict(round=wave,variable=col,source_item=var,definition=definition,measurement_level=level,valid_codes=json.dumps(m.variable_value_labels.get(var,{}),ensure_ascii=False),missing_n=int(d[col].isna().sum()),missing_fraction=float(d[col].isna().mean()),distinct_observed=int(d[col].nunique())))
pd.DataFrame(records).to_csv(out/'complete_data_dictionary.csv',index=False)
assert all(r['definition'] for r in records),[r['variable'] for r in records if not r['definition']]
print('DICTIONARY',len(records),'wave-variable records')
