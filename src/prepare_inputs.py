"""Clean official SAV inputs using the archived national macro lookup."""
from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/v2'))
import data_loader,prepare_round10
from feature_engineering import MACROS
def macro_join(out):
 identity=pd.read_csv(ROOT/'metadata/country_identity.csv');names=identity.set_index('country').iso3.to_dict()
 out['iso3']=out.country.map(names)
 if out.iso3.isna().any():raise ValueError('Country label absent from archived identity table; review release differences.')
 lookup=pd.read_csv(ROOT/'metadata/country_year_macro_lookup.csv');wave=10 if str(out.row_id.iloc[0]).startswith('R10_') else 8;lookup=lookup[lookup['round']==wave];audit=[]
 for name in MACROS:
  q=lookup[lookup.indicator==name];assert not q.duplicated(['iso3','survey_year']).any()
  table={(r.iso3,int(r.survey_year)):r.value for r in q.itertuples()}
  keys=list(zip(out.iso3,out.survey_year.astype(int)))
  if not all(k in table for k in keys):raise ValueError('Unarchived country/year; review macro vintage explicitly.')
  out[name]=[table[k] for k in keys];audit.extend(q.drop(columns='round').to_dict('records'))
 out['log_gdp_pc']=np.log1p(out.gdp_pc);return out,audit
def main():
 for p in ['data/processed','results/v2','reports/v2','results/geography_thesis/analysis','logs/geography_thesis','models/geography_thesis']:(ROOT/p).mkdir(parents=True,exist_ok=True)
 shutil.copyfile(ROOT/'metadata/analysis_config.json',ROOT/'data/processed/config.json')
 data_loader.macro_join=macro_join;prepare_round10.macro_join=macro_join
 data_loader.load(8);prepare_round10.main()
if __name__=='__main__':main()
