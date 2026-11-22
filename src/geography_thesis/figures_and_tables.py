from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/geography_thesis/analysis';FIG=ROOT/'figures/geography_thesis/final';TAB=ROOT/'reports/geography_thesis/tables';FIG.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
ORDER=['economic','cultural','political','welfare'];LABEL={'economic':'Redistribution fairness','cultural':'Cultural non-rejection','political':'Democracy preference','welfare':'Tax–service preference'}
COLORS=['#235789','#DB7C26','#428C74','#8A659B'];MODELS=['pooled','country_only','centred','multilevel','context_only','pooled_context'];SHORT={'pooled':'Pooled core','country_only':'Country baseline','centred':'Centred core','multilevel':'Multilevel core','context_only':'Context only','pooled_context':'Core + context','global':'Global baseline','centred_transductive':'Centred transductive'}
manifest=[]
def save(fig,n,name,caption):
 if n==1:fig.axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,frameon=False)
 if n==2:fig.axes[0].set_xticks(range(4),[LABEL[t].replace(' ',chr(10),1) for t in ORDER],rotation=0)
 if n==7:
  for ax in fig.axes:ax.set_title(ax.get_title().replace(' ',chr(10),1))
 if n in [8,9]:
  for ax in fig.axes:ax.set_xlabel('Δ log loss vs global baseline')
 if n==14:
  for ax in fig.axes:ax.set_xlabel('Δ log loss')
 scale=min(6.47/fig.get_figwidth(),7.8/fig.get_figheight());minimum=8.5/scale
 for text in fig.findobj(matplotlib.text.Text):text.set_fontsize(max(text.get_fontsize(),minimum))
 fig.tight_layout();file=f'{n:02d}_{name}';fig.savefig(FIG/(file+'.png'),dpi=190,bbox_inches='tight');fig.savefig(FIG/(file+'.svg'),bbox_inches='tight');plt.close(fig);manifest.append(dict(number=n,name=name,path=f'../../figures/geography_thesis/final/{file}.png',caption=caption))
def table(df,n,title,note):
 if n==6:note='All models share held-out respondents and evaluation weights in the 75/25 known-country split. Country means and prevalence are learned from training observations only; no comparison with unmatched legacy samples is implied.'
 df=df.copy()
 if 'Item' in df:df['Item']=df['Item'].map(lambda x:LABEL.get(x,x))
 df.to_csv(TAB/f'table_{n:02d}.csv',index=False);head='| '+' | '.join(df.columns)+' |';sep='| '+' | '.join(['---']*len(df.columns))+' |';rows=['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in df.itertuples(index=False,name=None)];text=f'**Table {n}**\n\n*{title}*\n\n'+ '\n'.join([head,sep]+rows)+f'\n\n*Note.* {note}\n';(TAB/f'table_{n:02d}.md').write_text(text,encoding='utf-8')
def main():
 flow=pd.read_csv(OUT/'sample_flow.csv').groupby(['round','target'])[['total_n','outcome_n','analysis_n']].sum().reset_index();table(flow.rename(columns={'round':'Round','target':'Item','total_n':'All respondents','outcome_n':'Valid outcome','analysis_n':'Complete core'}),1,'Outcome eligibility and primary analytical samples','Each matched model comparison uses the same complete-core cases within its task. Economic and welfare outcomes are not transported to Round 10.')
 short={'economic':'Redistribution','cultural':'Cultural','political':'Democracy','welfare':'Tax–service'}
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(flow));ax.bar(x,flow.outcome_n,color='#C8D1D8',label='Valid outcome');ax.bar(x,flow.analysis_n,color=COLORS[0],label='Complete core');ax.set_xticks(x,[f'R{r.round}'+chr(10)+short[r.target] for r in flow.itertuples()],fontsize=8);ax.set_ylabel('Respondents');ax.legend();save(fig,1,'sample_flow','Outcome-valid and complete-core sample sizes by round; counts are unweighted.')
 v=pd.read_csv(OUT/'observed_variance.csv');primary=v[v['sample']=='primary'].set_index('target').loc[ORDER];country=pd.read_csv(OUT/'country_descriptives.csv');mat=country.pivot(index='iso3',columns='target',values='prevalence')[ORDER].sort_values('cultural');fig,ax=plt.subplots(figsize=(7,9));im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu');ax.set_xticks(range(4),[LABEL[x] for x in ORDER],rotation=20,ha='right',fontsize=9);ax.set_yticks(range(len(mat)),mat.index,fontsize=8);fig.colorbar(im,ax=ax,label='Household-weighted prevalence',shrink=.7);save(fig,2,'country_prevalence','Round 8 weighted prevalences in complete-core samples. Countries are ordered by cultural-item prevalence; item meanings are distinct.')
 fig,ax=plt.subplots(figsize=(8,4));left=np.zeros(4)
 for key,label,color in [('between_country_share','Between countries',COLORS[0]),('between_position_share','Between positions within countries',COLORS[1]),('within_position_share','Within measured positions','#BBC6CE')]:
  values=primary[key].to_numpy()*100;ax.barh(range(4),values,left=left,label=label,color=color);left+=values
 ax.set_yticks(range(4),[LABEL[t] for t in ORDER]);ax.set_xlim(0,100);ax.set_xlabel('Share of observed binary-response variance (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=1);save(fig,3,'observed_variance','Exact descriptive variance decomposition under equal-country weights. Within-position variation is not a measure of free will or pure response error.')
 t=primary[['n','prevalence','between_country_share','between_position_share','within_position_share']].reset_index();t.columns=['Item','N','Prevalence','Between country','Between position','Within position'];
 for c in t.columns[2:]:t[c]=t[c].map(lambda x:f'{x:.4f}')
 table(t,2,'Observed variance decomposition','Shares sum to one; positions combine education, deprivation and employment within countries. Between-cell shares include sampling noise.')
 pair=pd.read_csv(OUT/'legacy_auc_pair_decomposition.csv');fig,ax=plt.subplots(figsize=(8,4));x=np.arange(2);
 for j,(key,label,col) in enumerate([('pooled_auc','Pooled',COLORS[0]),('within_pair_auc','Within-country pair-weighted',COLORS[1]),('cross_pair_auc','Cross-country pair-weighted',COLORS[2])]):ax.bar(x+(j-1)*.24,pair[key],width=.24,label=label,color=col)
 ax.set_xticks(x,[LABEL[t] for t in pair.target]);ax.set_ylim(0,1);ax.axhline(.5,c='gray',ls='--',lw=.8);ax.set_ylabel('AUC');ax.legend(fontsize=8);save(fig,4,'legacy_auc_pairs','Exact AUC pair decomposition of frozen Round 10 C-logistic scores. Within-country pair mass is reported separately, not interpreted as explanation.')
 t=pair[['target','pooled_auc','within_pair_mass','within_pair_auc','cross_pair_auc','mean_country_auc']].copy();t.columns=['Item','Pooled AUC','Within pair mass','Within pair AUC','Cross pair AUC','Mean country AUC'];
 for c in t.columns[1:]:t[c]=t[c].map(lambda x:f'{x:.4f}')
 table(t,3,'Exact decomposition of inherited Round 10 discrimination','Cultural N=50,055; democracy N=49,399. These legacy samples differ from the new complete-core samples. Pair mass is a probability of selecting that type of positive–negative pair, not a variance share.')
 fe=pd.read_csv(OUT/'fixed_effects_cluster_robust.csv');fp=fe[(fe.specification=='primary') & fe.term.isin(['education','deprivation','assets','employed'])];fig,axes=plt.subplots(2,2,figsize=(9,6))
 for ax,target in zip(axes.flat,ORDER):
  g=fp[fp.target==target];ax.errorbar(g.estimate,np.arange(len(g)),xerr=np.vstack([g.estimate-g.lower,g.upper-g.estimate]),fmt='o',color=COLORS[0],capsize=3);ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(len(g)),g.term);ax.set_title(LABEL[target]);ax.set_xlabel('Log-odds coefficient (original units)')
 save(fig,5,'within_coefficients','Country-fixed-effect logistic associations, with country-cluster-robust 95% t intervals. Coefficients are not causal effects or directly comparable standardized magnitudes.')
 t=fp[['target','term','estimate','lower','upper']].copy();t['95% interval']=[f'[{a:.3f}, {b:.3f}]' for a,b in zip(t.lower,t.upper)];t=t.drop(columns=['lower','upper']);t.estimate=t.estimate.map(lambda x:f'{x:.3f}');t.columns=['Item','Predictor','Coefficient','95% interval'];table(t,4,'Within-country associations with selected socioeconomic characteristics','Country-fixed-effect logistic models adjust for all eight core characteristics. Intervals use a country-cluster sandwich and t(33); associations are exploratory.')
 vc=pd.concat([pd.read_csv(p,keep_default_na=False) for p in OUT.glob('variance_components_*.csv')]);co=pd.concat([pd.read_csv(p,keep_default_na=False) for p in OUT.glob('multilevel_coefficients_*.csv')]);vc.to_csv(OUT/'all_variance_components.csv',index=False);co.to_csv(OUT/'all_multilevel_coefficients.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
 for j,model in enumerate(['null','within_between']):
  values=[float(vc[(vc.target==t)&(vc.model==model)&(vc.var1=='(Intercept)')].vcov.iloc[0]) for t in ORDER];ax.bar(x+(j-.5)*.3,values,width=.3,label='Null' if j==0 else 'Within–between + context',color=COLORS[j])
 ax.set_xticks(x,[LABEL[t] for t in ORDER],rotation=15,ha='right');ax.set_ylabel('Country-intercept variance (latent log-odds)');ax.legend();save(fig,6,'latent_variance','Estimated random-intercept variance before and after the within–between/context specification. Changes are descriptive, not causal percentages explained.')
 rows=[]
 for t in ORDER:
  nul=float(vc[(vc.target==t)&(vc.model=='null')].vcov.iloc[0]);adj=float(vc[(vc.target==t)&(vc.model=='within_between')].vcov.iloc[0]);slope=float(vc[(vc.target==t)&(vc.model=='random_slope')&(vc.var1=='w_education')].sdcor.iloc[0]);r=co[(co.target==t)&(co.model=='moderation')&(co.term=='w_education:institution_z')].iloc[0];rows.append([LABEL[t],f'{nul:.3f}',f'{nul/(nul+np.pi**2/3):.3f}',f'{adj:.3f}',f'{slope:.3f}',f'{float(r.estimate):.3f} [{float(r.lower):.3f}, {float(r.upper):.3f}]'])
 table(pd.DataFrame(rows,columns=['Item','Null variance','Latent ICC','Adjusted variance','Education slope SD','Institution interaction [95%]']),5,'Multilevel heterogeneity and institutional moderation','Null fits use nine-point adaptive quadrature. Adjusted and random-slope models use glmmTMB Laplace, with independent random intercept and slope. Interaction is per one education-code step and one training SD of electoral democracy; its interval is model-based.')
 fig,axes=plt.subplots(1,4,figsize=(11,9),sharey=True)
 for ax,t in zip(axes,ORDER):
  r=pd.read_csv(OUT/f'random_effects_{t}.csv').sort_values('iso3');fixed=float(co[(co.target==t)&(co.model=='random_slope')&(co.term=='w_education')].estimate.iloc[0]);ax.scatter(r.w_education+fixed,np.arange(len(r)),s=18,color=COLORS[0]);ax.axvline(0,c='gray',lw=.8);ax.set_yticks(range(len(r)),r.iso3,fontsize=8);ax.set_title(LABEL[t],fontsize=9);ax.set_xlabel('Education slope',fontsize=9)
 save(fig,7,'random_education_slopes','Conditional country education-slope estimates after partial pooling. Points lack uncertainty intervals and must not be used to rank countries conclusively.')
 if '--descriptive-only' in sys.argv:
  (OUT/'descriptive_figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('DESCRIPTIVE_FIGURES_READY',len(manifest));return
 metrics=pd.read_csv(OUT/'prediction_metrics.csv');boot=pd.read_csv(OUT/'paired_country_bootstrap.csv')
 for n,setting,name in [(8,'known_country','known_country_scores'),(9,'loco','loco_scores')]:
  fig,axes=plt.subplots(2,2,figsize=(10,7));rows=[]
  for ax,t in zip(axes.flat,ORDER):
   g=metrics[(metrics.target==t)&(metrics.setting==setting)&(metrics.cohort=='all')].set_index('model');base=g.loc['global','log_loss'];vals=[g.loc[m,'log_loss']-base for m in MODELS];ax.barh(range(len(MODELS)),vals,color=[COLORS[0],COLORS[1],COLORS[2],COLORS[3],'#708090','#526677']);ax.axvline(0,c='black',lw=.7);ax.set_yticks(range(len(MODELS)),[SHORT[m] for m in MODELS],fontsize=8);ax.set_title(LABEL[t]);ax.set_xlabel('Log loss minus global baseline (lower better)')
   for m in ['pooled','country_only','centred','multilevel','context_only','pooled_context']:r=g.loc[m];rows.append([LABEL[t],SHORT[m],int(r.n),f'{r.pooled_auc:.3f}',f'{r.mean_country_auc:.3f}',f'{r.log_loss:.4f}',f'{r.brier:.4f}'])
  save(fig,n,name,f'{setting} matched-model proper scores. Negative differences improve on the global training-prevalence baseline; all comparisons are within the same outcome/task sample.')
  table(pd.DataFrame(rows,columns=['Item','Model','N','Pooled AUC','Mean AUC','Log loss','Brier']),6 if setting=='known_country' else 7,'Known-country validation' if setting=='known_country' else 'Strict leave-one-country-out validation','All models share test respondents and evaluation weights. Strict LOCO cannot estimate target-country prevalence; centred core uses an inductive fallback. Pooled LOCO AUC compares scores from different fits.')
 fig,axes=plt.subplots(2,2,figsize=(10,7));rows=[]
 for ax,(t,cohort) in zip(axes.flat,[('cultural','common'),('cultural','new'),('political','common'),('political','new')]):
  g=metrics[(metrics.target==t)&(metrics.setting=='temporal')&(metrics.cohort==cohort)].set_index('model');vals=[g.loc[m,'log_loss'] for m in MODELS];ax.barh(range(6),vals,color=[COLORS[0],COLORS[1],COLORS[2],COLORS[3],'#708090','#526677']);ax.set_yticks(range(6),[SHORT[m] for m in MODELS],fontsize=8);ax.set_title(LABEL[t]+' — '+cohort);ax.set_xlabel('Log loss (lower better)')
  for m in MODELS:r=g.loc[m];rows.append([LABEL[t],cohort,SHORT[m],int(r.n),f'{r.pooled_auc:.3f}',f'{r.mean_country_auc:.3f}',f'{r.log_loss:.4f}'])
 save(fig,10,'temporal_transport','Round 8 to Round 10 transport, separating shared and newly represented countries. No Round 10 outcome-based recalibration is performed.')
 table(pd.DataFrame(rows,columns=['Item','Cohort','Model','N','Pooled AUC','Mean AUC','Log loss']),8,'Temporal transport under declared national-information rules','Common/new country comparisons are different populations. Shared-country random effects and means are retained from Round 8; new effects are marginalized. All-country metrics and calibration are in the supplementary CSV.')
 cb=pd.read_csv(OUT/'calibration_bins.csv');fig,axes=plt.subplots(2,3,figsize=(11,7))
 for i,t in enumerate(['cultural','political']):
  for j,setting in enumerate(['known_country','loco','temporal']):
   ax=axes[i,j];ax.plot([0,1],[0,1],ls='--',color='gray',lw=.8)
   for m,col in [('pooled',COLORS[0]),('country_only',COLORS[1]),('multilevel',COLORS[2])]:
    g=cb[(cb.target==t)&(cb.setting==setting)&(cb.cohort=='all')&(cb.model==m)];ax.plot(g.predicted,g.observed,'o-',ms=3,label=SHORT[m],color=col)
   ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted probability',ylabel='Observed prevalence',title=LABEL[t]+'\n'+setting);ax.legend(fontsize=7)
 save(fig,11,'calibration','Ten equal-width probability bins. Points are descriptive and may contain unequal numbers of respondents; curves are not fitted recalibration adjustments.')
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
 for j,sample in enumerate(['primary','common_four']):
  g=v[v['sample']==sample].set_index('target').loc[ORDER];ax.bar(x+(j-.5)*.25,g.between_country_share*100,width=.25,label=sample.replace('_',' '),color=COLORS[j])
 for t in ['economic','cultural']:
  q=v[(v.target==t)&(v['sample']=='exclude_neutral')].iloc[0];ax.scatter(ORDER.index(t),q.between_country_share*100,marker='D',s=45,color=COLORS[3],label='Exclude neutral' if t=='economic' else None)
 ax.set_xticks(x,[LABEL[t] for t in ORDER],rotation=15,ha='right');ax.set_ylabel('Between-country share of observed variance (%)');ax.legend();save(fig,12,'coding_sensitivity','Common-four-outcome and neutral-response sensitivity. Excluding neutral responses changes both coding and the eligible population; it is not a uniquely correct alternative.')
 p=pd.read_csv(OUT/'prevalence_by_wave_country.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.5))
 for ax,t in zip(axes,['cultural','political']):
  g=p[p.target==t].pivot(index='iso3',columns='wave',values='prevalence').dropna();ax.plot([0,1],[0,1],c='gray',ls='--',lw=.8);ax.scatter(g[8],g[10],s=24,color=COLORS[0]);
  for c,r in g.loc[(g[10]-g[8]).abs().nlargest(8).index].iterrows():ax.annotate(c,(r[8],r[10]),fontsize=6,xytext=(3,3),textcoords='offset points')
  ax.set(xlim=(0,1),ylim=(0,1),xlabel='Round 8 prevalence',ylabel='Round 10 prevalence',title=LABEL[t])
 save(fig,13,'prevalence_transport','Weighted outcome-valid country prevalences across repeated cross-sections. Labels identify the eight largest absolute changes per item; all common countries are plotted. Differences are not individual changes and use outcome-valid denominators.')
 fig,axes=plt.subplots(1,2,figsize=(10,5))
 for ax,t in zip(axes,['cultural','political']):
  g=boot[(boot.target==t)&(boot.metric=='log_loss')&(boot.new=='multilevel')&(boot.reference=='country_only')&((boot.cohort=='all')|(boot.setting=='temporal'))].copy();labs=g.setting+' / '+g.cohort;ax.errorbar(g.delta,range(len(g)),xerr=np.vstack([g.delta-g.lower,g.upper-g.delta]),fmt='o',capsize=3,color=COLORS[0]);ax.axvline(0,c='gray',lw=.8);ax.set_yticks(range(len(g)),labs,fontsize=8);ax.set_title(LABEL[t]);ax.set_xlabel('Multilevel minus country-baseline log loss')
 save(fig,14,'paired_uncertainty','Paired 95% country-bootstrap intervals conditional on fitted predictions. Negative differences favour multilevel core; intervals do not include refitting uncertainty.')
 fig,axes=plt.subplots(1,2,figsize=(9,5))
 for ax,t in zip(axes,['cultural','political']):
  g=metrics[(metrics.target==t)&(metrics.cohort=='all')&metrics.model.isin(['pooled','multilevel'])].copy();labs=g.setting.replace({'known_country':'Known','loco':'LOCO','temporal':'R10'})+' / '+g.model;ax.scatter(g.calibration_slope,range(len(g)),c=[COLORS[0] if m=='pooled' else COLORS[1] for m in g.model],s=35);ax.axvline(1,c='gray',ls='--');ax.set_yticks(range(len(g)),labs,fontsize=9);ax.set_title(LABEL[t]);ax.set_xlabel('Calibration slope')
 save(fig,15,'calibration_slopes','Descriptive calibration slopes for pooled and multilevel core models. The reference value is one; intercepts and binned curves must also be considered. No test outcomes are used to repair predictions.')
 change=pd.read_csv(OUT/'temporal_slope_changes.csv');fig,axes=plt.subplots(1,2,figsize=(9,6))
 for ax,t in zip(axes,['cultural','political']):
  g=change[change.target==t];ax.errorbar(g.change,range(len(g)),xerr=np.vstack([g.change-g.lower,g.upper-g.change]),fmt='o',capsize=3,color=COLORS[0]);ax.axvline(0,c='gray');ax.set_yticks(range(len(g)),g.variable,fontsize=9);ax.set_title(LABEL[t]);ax.set_xlabel('R10 minus R8 log-odds slope')
 save(fig,16,'temporal_slope_changes','Post-specification exploratory country-by-wave fixed-effect sensitivity in 30 shared countries. Country-cluster 95% t intervals are unadjusted for multiplicity; every interval includes zero. Units differ by predictor.')
 corr=pd.read_csv(OUT/'within_between_correlations.csv');fig,axes=plt.subplots(2,2,figsize=(9,8))
 for ax,t in zip(axes.flat,ORDER):
  g=corr[(corr.wave==8)&(corr.target==t)&corr.variable.isin(['age10','female','urban_i','education','deprivation','assets','electricity','employed'])].pivot(index='variable',columns='level',values='correlation');y=np.arange(len(g));ax.hlines(y,g.within_country_centred,g.between_countries,color='#B8C4CE');ax.scatter(g.within_country_centred,y,color=COLORS[0],label='Within',s=25);ax.scatter(g.between_countries,y,color=COLORS[1],marker='s',label='Between',s=25);ax.axvline(0,color='gray',lw=.8);ax.set_yticks(y,g.index,fontsize=8);ax.set_xlim(-1,1);ax.set_title(LABEL[t]);ax.set_xlabel('Unadjusted correlation');ax.legend(fontsize=8,loc='lower right')
 save(fig,17,'level_specific_correlations','Weighted within-country-centred correlations and equal-country correlations of country means. These are unadjusted descriptive associations, not mixed-model coefficients or causal effects.')
 g=boot[(boot.setting=='known_country')&(boot.cohort=='all')&(boot.new=='multilevel')&(boot.reference=='country_only')&(boot.metric=='log_loss')].set_index('target').loc[ORDER];fig,ax=plt.subplots(figsize=(8,4));ax.errorbar(g.delta,range(4),xerr=np.vstack([g.delta-g.lower,g.upper-g.delta]),fmt='o',capsize=4,color=COLORS[0]);ax.axvline(0,color='gray');ax.set_yticks(range(4),[LABEL[t] for t in ORDER]);ax.set_xlabel('Log-loss change: multilevel minus country baseline')
 save(fig,18,'individual_information_gain','Incremental known-country predictive value of core individual information across all four items. Negative values favour multilevel core; paired 95% country-bootstrap intervals condition on fitted models.')
 (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('FIGURES_TABLES_COMPLETE',len(manifest),flush=True)
if __name__=='__main__':main()
