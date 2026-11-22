from pathlib import Path
import numpy as np,pandas as pd
from prepare_analysis import CORE
from descriptive_and_fe import weights
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/geography_thesis/analysis'
def correlation(x,y,w):
 x=x-np.average(x,weights=w);y=y-np.average(y,weights=w);den=np.sqrt(np.average(x*x,weights=w)*np.average(y*y,weights=w));return np.average(x*y,weights=w)/den if den>0 else np.nan
def main():
 records=[];summaries=[]
 for wave in [8,10]:
  all=pd.read_parquet(OUT/f'r{wave}_prepared.parquet')
  for col in CORE:
   d=all[all[col].notna()].copy();w=weights(d);summaries.append(dict(wave=wave,variable=col,n=len(d),missing_n=int(all[col].isna().sum()),weighted_mean=np.average(d[col],weights=w),weighted_sd=np.sqrt(np.average((d[col]-np.average(d[col],weights=w))**2,weights=w)),minimum=d[col].min(),maximum=d[col].max()))
  targets=['economic','cultural','political','welfare'] if wave==8 else ['cultural','political']
  for target in targets:
   d=all[all.core_complete&all[target].notna()].copy();means=[]
   for c,g in d.groupby('iso3'):
    means.append(dict(iso3=c,**{x:np.average(g[x],weights=g.weight_raw) for x in CORE+[target,'log_gdp_pc','v2x_polyarchy']}))
   means=pd.DataFrame(means).set_index('iso3');w=weights(d);cy=d[target].to_numpy()-d.iso3.map(means[target]).to_numpy()
   for col in CORE+['log_gdp_pc','v2x_polyarchy']:
    records.append(dict(wave=wave,target=target,variable=col,level='between_countries',correlation=correlation(means[col].to_numpy(),means[target].to_numpy(),np.ones(len(means))),n_units=len(means)))
    if col in CORE:
     cx=d[col].to_numpy()-d.iso3.map(means[col]).to_numpy();records.append(dict(wave=wave,target=target,variable=col,level='within_country_centred',correlation=correlation(cx,cy,w),n_units=len(d)))
 pd.DataFrame(records).to_csv(OUT/'within_between_correlations.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'weighted_predictor_distributions.csv',index=False);print('LEVEL_CORRELATIONS',len(records))
if __name__=='__main__':main()
