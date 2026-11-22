# Reproduction guide

Run commands from the repository root. Publication assembly did not execute numerical reproduction, model fitting or figure rendering, at the author's request. Entry points are provided, not certified as freshly rerun.

## Environment

Recorded Python version: 3.12.13. Create an isolated environment; activate it using the command for your operating system:

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
Rscript setup_R.R
```

Recorded R environment: R 4.6.2, lme4 2.1-0, glmmTMB 1.1.15.2, sandwich 3.1-3, jsonlite 2.0.0. clubSandwich is included as requested but was not used in the existing estimators and no historical version was recorded. R installation above uses available CRAN versions, not a locked R environment. Python requirements record installed analysis versions; numerical differences across platforms/versions remain possible.

## A. Render from included aggregates (no microdata or R needed)

```sh
python src/run_replication.py --figures-only
```

This writes Figures 1–18 to `figures/geography_thesis/final/` and Tables 1–8 to `reports/geography_thesis/tables/`. It also recombines coefficient summaries in the aggregate directory. Included table CSVs are the archived publication outputs. Copy the repository first if you wish to retain an untouched archive.

## B. Rebuild from provider inputs

Download Afrobarometer inputs using the official links in README and place the extracted SAVs at `data/raw/afro_r8.sav` and `data/raw/afro_r10.sav`. Use the English merged releases. Compare file SHA-256 values with `metadata/input_checksums.json` and investigate differences before analysis. Do not silently substitute a different release. The R10 codebook lists 45,600 observations whereas the workspace SAV has 50,961; the archived R8 SAV has 48,084. These are source discrepancies, not edited sample counts.

```sh
python src/run_replication.py --full
# If Rscript is not on PATH:
# python src/run_replication.py --full --rscript /path/to/Rscript
```

The runner performs: SAV cleaning; V-Dem v16 acquisition and context extraction; analytical preparation and dictionaries; observed variance and fixed effects; locked legacy logistic refits and exact AUC decomposition; level-specific correlations; four outcome-specific mixed-model suites including strict LOCO and temporal transport; temporal slope sensitivity; convergence/contrast/model audits; aggregate prediction evaluation; figure/table rendering. R fitting can be expensive. Intermediate respondent-level files and fitted models remain local and ignored by Git.

The raw cleaning deliberately uses the archived country/year WDI lookup in metadata, including source years and missing values. It preserves the latest nonmissing observation strictly before interview year within five preceding years and `log_gdp_pc = log1p(gdp_pc)`. The original WDI acquisition range ended in 2023; new API retrieval or later revisions may change R10 contextual values. No claim is made that today's API reconstructs the historical vintage. `src/data_loader.py` retains the original API-JSON join implementation for inspection; use `src/prepare_inputs.py`, not its legacy main routine (which expects Round 9).

Legacy Figure 4/Table 3 use all outcome-valid cases rather than the new complete-core samples. `src/rebuild_legacy_auc.py` refits the archived C parameters on R8 with training-only preprocessing and equal-country weights, then evaluates raw probabilities on R10. Calibration parameters are archived but not used for this raw-score decomposition. No binary models or respondent predictions are distributed. Hyperparameter search is not repeated: this is a locked-specification rebuild. Exact equivalence to the previously frozen binaries has not been tested in this release.

The removed private historical checksum guard cannot validate externally obtained files. `metadata/input_checksums.json` records workspace source hashes instead; newly generated execution metadata does not claim that the historical external guard passed. See `docs/analysis_specification.md` and `docs/execution_amendments.md` for the original exploratory specification and estimator changes. Country-region blocks are not true released PSUs. Inference is not causal.
