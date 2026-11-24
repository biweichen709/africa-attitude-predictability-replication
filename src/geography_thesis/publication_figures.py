"""Publication-quality redraw of Figures 1-18 from saved aggregate outputs."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/geography_thesis/analysis"; FIG=ROOT/"figures/geography_thesis/final"; BACK=ROOT/"figures/geography_thesis/backup_2026-11-24"
ORDER=["economic","cultural","political","welfare"]
LAB={"economic":"Redistribution fairness","cultural":"Cultural non-rejection","political":"Democracy preference","welfare":"Tax-service preference"}
SHORT={"economic":"Redistribution","cultural":"Non-rejection","political":"Democracy","welfare":"Tax-service"}
BLUE="#0072B2"; ORANGE="#D55E00"; GREEN="#009E73"; PURPLE="#CC79A7"; GREY="#6B6B6B"; LIGHT="#D9D9D9"; BLACK="#202020"
COL=[BLUE,ORANGE,GREEN,PURPLE]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"figure.dpi":150,"savefig.dpi":450,"svg.fonttype":"none"})

def archive():
    BACK.mkdir(parents=True,exist_ok=True)
    for p in FIG.glob("*.*"):
        q=BACK/p.name
        if not q.exists(): shutil.copy2(p,q)

def panel(ax,label): ax.text(-.10,1.04,label,transform=ax.transAxes,fontweight="bold",fontsize=10,va="bottom")
def finish(fig,n,name):
    fig.align_labels();fig.tight_layout(pad=.8);fig.savefig(FIG/f"{n:02d}_{name}.png",bbox_inches="tight",facecolor="white");fig.savefig(FIG/f"{n:02d}_{name}.svg",bbox_inches="tight",facecolor="white");plt.close(fig)
def hzero(ax): ax.axvline(0,color=BLACK,lw=.7,ls=(0,(3,2)),zorder=0)
def load(x): return pd.read_csv(OUT/x,keep_default_na=False,na_values=['','NA','NaN'])

def figure1():
 d=load("sample_flow.csv").groupby(["round","target"],as_index=False)[["outcome_n","analysis_n"]].sum();d=d[d.target.isin(ORDER)];d["ord"]=d.target.map({x:i for i,x in enumerate(ORDER)});d=d.sort_values(["round","ord"]);x=np.arange(len(d));fig,ax=plt.subplots(figsize=(7.2,3.4));ax.bar(x-.18,d.outcome_n,.36,color="white",edgecolor=BLACK,hatch="///",label="Outcome valid");ax.bar(x+.18,d.analysis_n,.36,color=BLUE,label="Complete core");ax.set_xticks(x,[f"R{r['round']}\n{SHORT[r.target]}" for _,r in d.iterrows()]);ax.set_ylabel("Respondents");ax.yaxis.set_major_locator(MaxNLocator(5,integer=True));ax.legend(frameon=False,ncol=2,loc="upper right");finish(fig,1,"sample_flow")
def figure2():
 d=load("country_descriptives.csv");mat=d.pivot(index="iso3",columns="target",values="prevalence")[ORDER].sort_values("cultural");fig,ax=plt.subplots(figsize=(6.4,7.5));im=ax.imshow(mat,aspect="auto",vmin=0,vmax=1,cmap="cividis");ax.set_xticks(range(4),[SHORT[x] for x in ORDER]);ax.set_yticks(range(len(mat)),mat.index);cb=fig.colorbar(im,ax=ax,pad=.02,shrink=.8);cb.set_label("Weighted prevalence");finish(fig,2,"country_prevalence")
def figure3():
 d=load("observed_variance.csv");d=d[d["sample"]=="primary"].set_index("target").loc[ORDER];fig,ax=plt.subplots(figsize=(7.2,3.5));left=np.zeros(4);parts=[("between_country_share","Between countries",BLUE,""),("between_position_share","Between positions within country",ORANGE,"///"),("within_position_share","Within position",LIGHT,"...")]
 for c,l,color,h in parts: ax.barh(range(4),100*d[c],left=left,color=color,edgecolor=BLACK,lw=.4,hatch=h,label=l);left+=100*d[c].to_numpy()
 ax.set_yticks(range(4),[LAB[x] for x in ORDER]);ax.set_xlim(0,100);ax.set_xlabel("Share of observed binary-response variance (%)");ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.01));finish(fig,3,"observed_variance")
def figure4():
 d=load("legacy_auc_pair_decomposition.csv");fig,ax=plt.subplots(figsize=(6.5,3.6));x=np.arange(len(d));cols=[("within_pair_auc","Within-country pairs",BLUE,"o"),("cross_pair_auc","Between-country pairs",ORANGE,"s"),("pooled_auc","Pooled AUC",BLACK,"D")]
 for c,l,color,m in cols: ax.plot(x,d[c],marker=m,color=color,lw=1.4,label=l)
 ax.axhline(.5,color=GREY,lw=.7,ls="--");ax.set_xticks(x,[LAB[t] for t in d.target],rotation=15,ha="right");ax.set_ylabel("AUC");ax.set_ylim(.45,.85);ax.legend(frameon=False,ncol=3);finish(fig,4,"legacy_auc_pairs")
def figure5():
 d=load("fixed_effects_cluster_robust.csv");d=d[(d.specification=="primary")&d.term.isin(["education","deprivation","assets","employed"])];terms=["education","deprivation","assets","employed"];fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),sharex=False)
 for k,(ax,t) in enumerate(zip(axs.flat,terms)):
  z=d[d.term==t].set_index("target").loc[ORDER];ax.errorbar(z.estimate,range(4),xerr=[z.estimate-z.lower,z.upper-z.estimate],fmt="o",color=BLUE,ecolor=BLACK,capsize=2);hzero(ax);ax.set_yticks(range(4),[SHORT[x] for x in ORDER]);ax.set_title(t.capitalize());ax.set_xlabel("Log-odds coefficient");panel(ax,chr(65+k))
 finish(fig,5,"within_coefficients")
def figure6():
 rows=[]
 for t in ORDER:
  d=load(f"variance_components_{t}.csv");
  for m,l in [("null","Null"),("within_between","Adjusted")]: rows.append((t,l,d[(d.model==m)&(d.grp=="iso3")].vcov.iloc[0]))
 d=pd.DataFrame(rows,columns=["target","model","value"]);fig,ax=plt.subplots(figsize=(7,3.6));x=np.arange(4);w=.34
 for i,(m,c,h) in enumerate([("Null",LIGHT,"///"),("Adjusted",BLUE,"")]):z=d[d.model==m].set_index("target").loc[ORDER];ax.bar(x+(i-.5)*w,z.value,w,label=m,color=c,edgecolor=BLACK,hatch=h)
 ax.set_xticks(x,[SHORT[t] for t in ORDER]);ax.set_ylabel("Country-intercept variance");ax.legend(frameon=False);finish(fig,6,"latent_variance")
def figure7():
 fig,axs=plt.subplots(2,2,figsize=(7.2,7),sharex=False)
 for k,(ax,t) in enumerate(zip(axs.flat,ORDER)):
  z=load(f"random_effects_{t}.csv");z["slope"]=z.w_education+load(f"multilevel_coefficients_{t}.csv").query("model=='random_slope' and term=='w_education'").estimate.iloc[0];z=z.sort_values("slope");ax.scatter(z.slope,range(len(z)),s=13,color=BLUE);ax.axvline(z.slope.mean(),color=ORANGE,lw=1);ax.axvline(0,color=BLACK,lw=.6,ls="--");ax.set_yticks(range(len(z)),z.iso3,fontsize=6);ax.set_title(LAB[t]);ax.set_xlabel("Conditional education slope");panel(ax,chr(65+k))
 finish(fig,7,"random_education_slopes")
def scorefig(n,setting,name):
 d=load("prediction_metrics.csv");d=d[(d.setting==setting)&(d.cohort=="all")];mods=[x for x in ["country_only","pooled","centred","multilevel","context_only","pooled_context"] if x in set(d.model)];labels={"country_only":"Country prevalence","pooled":"Pooled core","centred":"Centred core","multilevel":"Multilevel core","context_only":"Context only","pooled_context":"Core + context"};fig,axs=plt.subplots(2,2,figsize=(7.2,5.6))
 for k,(ax,t) in enumerate(zip(axs.flat,ORDER)):
  z=d[d.target==t].set_index("model");base=z.loc["global","log_loss"] if "global" in z.index else z.log_loss.max();vals=[z.loc[m,"log_loss"]-base for m in mods];ax.barh(range(len(mods)),vals,color=[COL[i%4] for i in range(len(mods))],edgecolor=BLACK,lw=.3);hzero(ax);ax.set_yticks(range(len(mods)),[labels[m] for m in mods]);ax.invert_yaxis();ax.set_title(LAB[t]);ax.set_xlabel("Log-loss difference from global prevalence");panel(ax,chr(65+k))
 finish(fig,n,name)
def figure10():
 d=load("prediction_metrics.csv");d=d[d.setting=="temporal"];mods=["global","country_only","multilevel","pooled_context"];labels={"global":"Global","country_only":"Country prevalence","multilevel":"Multilevel core","pooled_context":"Core + context"};fig,axs=plt.subplots(2,2,figsize=(7.2,5.2))
 for k,(ax,(t,c)) in enumerate(zip(axs.flat,[("cultural","common"),("cultural","new"),("political","common"),("political","new")])):
  z=d[(d.target==t)&(d.cohort==c)].set_index("model");use=[m for m in mods if m in z.index];ax.barh(range(len(use)),[z.loc[m,"log_loss"] for m in use],color=[LIGHT,ORANGE,BLUE,GREEN][:len(use)],edgecolor=BLACK,lw=.4);ax.set_yticks(range(len(use)),[labels[m] for m in use]);ax.invert_yaxis();ax.set_title(f"{LAB[t]}: {'shared' if c=='common' else 'new'} countries");ax.set_xlabel("Log loss");panel(ax,chr(65+k))
 finish(fig,10,"temporal_transport")
def figure11():
 d=load("calibration_bins.csv");fig,axs=plt.subplots(2,2,figsize=(7,6),sharex=True,sharey=True)
 cells=[("cultural","known_country"),("political","known_country"),("cultural","temporal"),("political","temporal")]
 for k,(ax,(t,s)) in enumerate(zip(axs.flat,cells)):
  z=d[(d.target==t)&(d.setting==s)&(d.cohort=="all")]
  for m,c,mark in [("country_only",ORANGE,"s"),("multilevel",BLUE,"o")]:
   q=z[z.model==m];ax.plot(q.predicted,q.observed,marker=mark,color=c,lw=1,label="Country prevalence" if m=="country_only" else "Multilevel core")
  ax.plot([0,1],[0,1],color=BLACK,lw=.7,ls="--");ax.set_title(f"{LAB[t]}\n{'Known country' if s=='known_country' else 'R8 to R10'}");ax.set_xlabel("Predicted probability");ax.set_ylabel("Observed proportion");panel(ax,chr(65+k))
 axs.flat[0].legend(frameon=False);finish(fig,11,"calibration")
def figure12():
 d=load("observed_variance.csv");samples=["primary","common_four","exclude_neutral"];fig,ax=plt.subplots(figsize=(7,3.7));x=np.arange(4);w=.24
 for i,s in enumerate(samples):
  z=d[d["sample"]==s].set_index("target").reindex(ORDER);ax.bar(x+(i-(len(samples)-1)/2)*w,100*z.between_country_share,w,label=s.replace("_"," "),color=[BLUE,LIGHT,ORANGE][i],edgecolor=BLACK,hatch="///" if i==1 else "")
 ax.set_xticks(x,[SHORT[t] for t in ORDER]);ax.set_ylabel("Between-country share (%)");ax.legend(frameon=False);finish(fig,12,"coding_sensitivity")
def figure13():
 d=load("prevalence_by_wave_country.csv");fig,axs=plt.subplots(1,2,figsize=(7.2,3.6))
 for k,(ax,t) in enumerate(zip(axs,["cultural","political"])):
  z=d[d.target==t].pivot(index="iso3",columns="wave",values="prevalence").dropna();ax.scatter(z[8],z[10],s=20,color=BLUE,alpha=.8);ax.plot([0,1],[0,1],color=BLACK,lw=.7,ls="--");chg=(z[10]-z[8]).abs().nlargest(6)
  for iso in chg.index:ax.annotate(iso,(z.loc[iso,8],z.loc[iso,10]),xytext=(3,3),textcoords="offset points",fontsize=7)
  ax.set_title(LAB[t]);ax.set_xlabel("Round 8 prevalence");ax.set_ylabel("Round 10 prevalence");ax.set_xlim(0,1);ax.set_ylim(0,1);panel(ax,chr(65+k))
 finish(fig,13,"prevalence_transport")
def figure14():
 d=load("paired_country_bootstrap.csv");d=d[(d.metric=="log_loss")&(d["new"]=="multilevel")&(d.reference=="country_only")&d.target.isin(["cultural","political"])];d=d[((d.setting=="known_country")&(d.cohort=="all"))|((d.setting=="temporal")&d.cohort.isin(["common","new"]))];d["label"]=d.target.map(SHORT)+" | "+d.setting.str.replace("known_country","Known country").str.replace("temporal","R8 to R10")+" | "+d.cohort.str.replace("common","shared");d=d.sort_values(["target","setting","cohort"]);fig,ax=plt.subplots(figsize=(7,4));y=np.arange(len(d));ax.errorbar(d.delta,y,xerr=[d.delta-d.lower,d.upper-d.delta],fmt="o",color=BLUE,ecolor=BLACK,capsize=2);hzero(ax);ax.set_yticks(y,d.label);ax.invert_yaxis();ax.set_xlabel("Multilevel minus country-prevalence log loss");finish(fig,14,"paired_uncertainty")
def figure15():
 d=load("prediction_metrics.csv");d=d[(d.model.isin(["pooled","multilevel"]))&(d.setting.isin(["known_country","loco","temporal"]))&(d.cohort=="all")];fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),sharey=True)
 for k,(ax,t) in enumerate(zip(axs.flat,ORDER)):
  z=d[d.target==t];labels=[]
  for i,(s,m) in enumerate([(s,m) for s in ["known_country","loco","temporal"] for m in ["pooled","multilevel"]]):
   q=z[(z.setting==s)&(z.model==m)];
   if len(q):ax.scatter(i,q.calibration_slope.iloc[0],marker="o" if m=="multilevel" else "s",color=BLUE if m=="multilevel" else ORANGE);labels.append((s[:1].upper(),m[:1].upper()))
  ax.axhline(1,color=BLACK,lw=.7,ls="--");ax.set_xticks(range(len(labels)),[a+"/"+b for a,b in labels]);ax.set_title(LAB[t]);ax.set_ylabel("Calibration slope");panel(ax,chr(65+k))
 finish(fig,15,"calibration_slopes")
def figure16():
 d=load("temporal_slope_changes.csv");predictors=list(d.variable.drop_duplicates());fig,axs=plt.subplots(1,2,figsize=(7.2,4.5),sharey=True)
 for k,(ax,t) in enumerate(zip(axs,["cultural","political"])):
  z=d[d.target==t].set_index("variable").loc[predictors];y=np.arange(len(z));ax.errorbar(z.change,y,xerr=[z.change-z.lower,z.upper-z.change],fmt="o",color=BLUE,ecolor=BLACK,capsize=2);hzero(ax);ax.set_yticks(y,[x.replace("_"," ") for x in predictors]);ax.invert_yaxis();ax.set_title(LAB[t]);ax.set_xlabel("Round 10 minus Round 8 slope");panel(ax,chr(65+k))
 finish(fig,16,"temporal_slope_changes")
def figure17():
 d=load("within_between_correlations.csv");d=d[d.wave==8];w=d[d.level=="within_country_centred"].rename(columns={"correlation":"within_correlation"});b=d[d.level=="between_countries"].rename(columns={"correlation":"between_correlation"});d=w.merge(b,on=["wave","target","variable"],suffixes=("_w","_b"));preds=list(d.variable.drop_duplicates());fig,axs=plt.subplots(2,2,figsize=(7.2,7),sharey=True)
 for k,(ax,t) in enumerate(zip(axs.flat,ORDER)):
  z=d[d.target==t].set_index("variable").reindex(preds);y=np.arange(len(z));ax.scatter(z.within_correlation,y,color=BLUE,label="Within country",marker="o");ax.scatter(z.between_correlation,y,color=ORANGE,label="Country means",marker="s");hzero(ax);ax.set_yticks(y,[x.replace("_"," ") for x in preds]);ax.invert_yaxis();ax.set_title(LAB[t]);ax.set_xlabel("Weighted correlation");panel(ax,chr(65+k))
 axs.flat[0].legend(frameon=False);finish(fig,17,"level_specific_correlations")
def figure18():
 d=load("paired_country_bootstrap.csv");d=d[(d.setting=="known_country")&(d.cohort=="all")&(d["new"]=="multilevel")&(d.reference=="country_only")&(d.metric=="log_loss")].set_index("target").loc[ORDER];fig,ax=plt.subplots(figsize=(6.8,3.5));y=np.arange(4);ax.errorbar(d.delta,y,xerr=[d.delta-d.lower,d.upper-d.delta],fmt="o",color=BLUE,ecolor=BLACK,capsize=2);hzero(ax);ax.set_yticks(y,[LAB[t] for t in ORDER]);ax.invert_yaxis();ax.set_xlabel("Multilevel minus country-prevalence log loss");finish(fig,18,"individual_information_gain")
def main():
    FIG.mkdir(parents=True,exist_ok=True);archive();figure1();figure2();figure3();figure4();figure5();figure6();figure7();scorefig(8,"known_country","known_country_scores");scorefig(9,"loco","loco_scores");figure10();figure11();figure12();figure13();figure14();figure15();figure16();figure17();figure18();print("REDREW 18 FIGURES",flush=True)
if __name__=="__main__":main()
