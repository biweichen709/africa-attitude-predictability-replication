"""Portable runner. --figures-only reads included aggregates; --full refits."""
from pathlib import Path
import argparse,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--figures-only',action='store_true');g.add_argument('--full',action='store_true');p.add_argument('--rscript',default='Rscript');args=p.parse_args()
 def py(name):subprocess.run([sys.executable,str(ROOT/'src'/name)],cwd=ROOT,check=True)
 def r(name,*extra):subprocess.run([args.rscript,str(ROOT/'src/geography_thesis'/name),*extra],cwd=ROOT,check=True)
 if args.full:
  py('prepare_inputs.py');py('geography_thesis/acquire_context.py');r('extract_context.R');py('geography_thesis/prepare_analysis.py');py('geography_thesis/complete_dictionary.py');py('geography_thesis/descriptive_and_fe.py');py('rebuild_legacy_auc.py');py('geography_thesis/level_correlations.py')
  for target in ['economic','cultural','political','welfare']:r('fit_multilevel.R',target)
  py('geography_thesis/temporal_slope_sensitivity.py');r('audit_convergence.R');r('within_between_contrasts.R');r('audit_fitted_models.R');py('geography_thesis/evaluate_models.py')
 py('geography_thesis/figures_and_tables.py')
if __name__=='__main__':main()
