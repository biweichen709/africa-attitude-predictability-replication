"""Compare frozen procedures on identical complete-core Round 10 respondents."""
from evaluate_extension import OUT,ROOT,load_all,paired_boot,country_losses,metric
import numpy as np
import pandas as pd

def main():
    rows=[]
    groups=load_all()
    for target in ('cultural','political'):
        old=pd.read_csv(ROOT/f'results/geography_thesis/analysis/pred_temporal_{target}.csv')
        for info in ('socioeconomic','socioeconomic_context'):
            for model in ('elastic_net','lightgbm'):
                new=groups['temporal',target,info,model]
                both=old.merge(new,on=['row_id','iso3','y'],validate='one_to_one',suffixes=('','_ml'))
                assert len(both)==len(old) and np.allclose(both.weight_raw,both.weight_raw_ml)
                assert (both.cohort.replace({'common':'shared'})==both.temporal_cohort).all()
                for cohort,g in both.groupby('temporal_cohort'):
                    for ref in ('multilevel','country_only','pooled_context'):
                        est=metric(g,'calibrated');baseline=metric(g,ref)
                        rows.append(dict(target=target,cohort=cohort,information_set=info,model=model,reference=ref,n=len(g),model_loss=est['equal_country_log_loss'],reference_loss=baseline['equal_country_log_loss'],**paired_boot(country_losses(g,'calibrated'),country_losses(g,ref))))
    pd.DataFrame(rows).to_csv(OUT/'matched_temporal_complete_core.csv',index=False)

if __name__=='__main__':main()
