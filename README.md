# The Mirage of Predictability: National Contexts, Within-Country Stratification, and Political Attitudes in Africa

Replication code, analytical dictionaries and aggregate publication outputs for an Afrobarometer Round 8 / Round 10 study. Code is MIT-licensed; eligible original documentation and outputs are CC BY 4.0, without overriding provider rights. See LICENSE and LICENSE-DATA.md.

**Release status:** prepared from completed analyses. Numerical reproduction, refitting and rendering were not rerun during publication assembly at the author's request. This is not a claim of journal approval or independently validated reproducibility. Public GitHub ownership remains visible despite a neutral commit identity.

## Layout

- `src/prepare_inputs.py`: portable SAV cleaning using the archived national macro lookup.
- `src/data_loader.py`, `src/v2/prepare_round10.py`, `src/feature_engineering.py`: source item mappings and recodes.
- `src/geography_thesis/`: analytical preparation, country-fixed-effect regressions, within–between lme4/glmmTMB models, exact AUC-pair decomposition, known-country/LOCO/temporal evaluation, sensitivities and figures.
- `src/rebuild_legacy_auc.py`: locked legacy C-logistic refits without shipped binaries.
- `src/run_replication.py`: portable aggregate-only/full entry point.
- `metadata/`: analytical config, country identities, historical national WDI lookup, input hashes, legacy model specifications and aggregate manifest.
- `results/geography_thesis/analysis/`: aggregate results and analytical dictionaries only.
- `reports/geography_thesis/tables/`: archived Tables 1–8 as CSV.
- `docs/`: specification and execution amendments.

## Official inputs and exact placement

Source data are not redistributed. Download from the providers and review their conditions.

1. **Afrobarometer merged R8:** official resource page https://www.afrobarometer.org/survey-resource/merged-round-8-data-34-countries-2022/ ; the page labels the round 2022, while the selected English release filename is dated 1 March 2023: https://www.afrobarometer.org/wp-content/uploads/2023/03/afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav . Save as `data/raw/afro_r8.sav`.
2. **Afrobarometer merged R10:** English 38-country release identified by the filename date 25 February 2026: https://www.afrobarometer.org/wp-content/uploads/2026/10/AB_R10.Merge-Dataset_25Feb26.38ctry.release.sav.zip . Extract the SAV and save as `data/raw/afro_r10.sav`. The official resource listing is https://www.afrobarometer.org/data/merged-data/ . Filename release date and website upload-directory date differ. Compare the independently obtained files with `metadata/input_checksums.json`; do not assume matching release names guarantee matching bytes.
3. **V-Dem v16:** https://github.com/vdeminstitute/vdemdata/releases/tag/V16 . `python src/geography_thesis/acquire_context.py` obtains `vdem.RData` and `codebook.RData` from the official V16 tag into `data/raw/geography/`, retaining checksums/provenance locally. `Rscript src/geography_thesis/extract_context.R` extracts 2018–2025 national context. The full runner performs both steps.
4. **World Bank WDI:** https://api.worldbank.org/v2/ ; indicator codes are in `src/feature_engineering.py` (`MACROS`). The exact archived country/year/value/source-year lookup is `metadata/country_year_macro_lookup.csv`; no new WDI download is necessary for the historical-vintage workflow. To inspect current source values, use `https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.KD?date=2013:2023&format=json&per_page=20000` and substitute each documented indicator code. Revisions can change values; current downloads are not claimed identical to the archive.

Afrobarometer access/copyright conditions: https://www.afrobarometer.org/data/data-usage-and-access-policy/ . Provider licences apply independently of this repository's software/documentation licences.

## Setup and execution

See [REPRODUCE_GEOGRAPHY.md](REPRODUCE_GEOGRAPHY.md) for complete environment, ordered commands, local output exclusions and limitations.

```sh
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
Rscript setup_R.R
python src/run_replication.py --figures-only
# After downloading the two SAV files, to refit instead:
python src/run_replication.py --full
```

R setup installs lme4, glmmTMB, sandwich, clubSandwich and jsonlite. Existing fixed-effect cluster-robust covariance is implemented explicitly; installing clubSandwich does not imply that its estimators were used. Full execution may take substantial time and produces private intermediates; do not upload them.

## Script-to-table/figure mapping

All final rendering is in `src/geography_thesis/figures_and_tables.py`. Analytical script paths below are relative to `src/geography_thesis/` unless prefixed otherwise.

| Outputs | Analytical producer | Included aggregate inputs |
|---|---|---|
| Table 1; Figure 1 | `prepare_analysis.py` | `sample_flow.csv` |
| Figure 2 | `descriptive_and_fe.py` | `country_descriptives.csv` |
| Table 2; Figure 3 | `descriptive_and_fe.py` | `observed_variance.csv` |
| Table 3; Figure 4 | `src/rebuild_legacy_auc.py`; `auc_pairs` in `descriptive_and_fe.py` | `legacy_auc_pair_decomposition.csv`, `legacy_pair_matrix_*.csv` |
| Table 4; Figure 5 | `descriptive_and_fe.py` | `fixed_effects_cluster_robust.csv` |
| Table 5; Figures 6–7 | `fit_multilevel.R` | `variance_components_*.csv`, `multilevel_coefficients_*.csv`, `random_effects_*.csv` |
| Table 6; Figure 8 | `fit_multilevel.R`; `evaluate_models.py` | `prediction_metrics.csv`, `paired_country_bootstrap.csv` (known country) |
| Table 7; Figure 9 | same scripts, strict LOCO | same CSVs (LOCO) |
| Table 8; Figure 10 | same scripts, R8-to-R10 transport | same CSVs (temporal, shared/new cohorts) |
| Figure 11 | `evaluate_models.py` | `calibration_bins.csv` |
| Figure 12 | `descriptive_and_fe.py` | `observed_variance.csv` sensitivity samples |
| Figure 13 | `descriptive_and_fe.py` | `prevalence_by_wave_country.csv` |
| Figure 14 | `evaluate_models.py` | `paired_country_bootstrap.csv` |
| Figure 15 | `evaluate_models.py` | `prediction_metrics.csv` calibration slopes |
| Figure 16 | `temporal_slope_sensitivity.py` | `temporal_slope_changes.csv` |
| Figure 17 | `level_correlations.py` | `within_between_correlations.csv` |
| Figure 18 | `evaluate_models.py` | `paired_country_bootstrap.csv` (individual information gain) |

`within_between_contrasts.R`, `audit_convergence.R` and `audit_fitted_models.R` provide supplementary contrasts and diagnostics. `complete_dictionary.py` rebuilds source mappings when licensed SAV inputs are present. Included `analysis_dictionary.csv`, `complete_data_dictionary.csv` and `full_variable_inventory.csv` document analytical variables.

## Scope and disclosure

No respondent records, source SAV/CSV files, individual predictions, model checkpoints, credentials or manuscript drafts are included. Aggregates describe country samples, performance, model coefficients and pairwise national AUC components. They are not population-weighted estimates for Africa. Source data release discrepancies, historical WDI-vintage dependence and untested locked legacy refits are documented rather than silently corrected. A new software release is not a permanent archival DOI deposit.
# Machine-learning extension and revised figures

Run `python src/geography_thesis/publication_figures.py` from the repository root to render Figures 1 to 18 from the released aggregates. Output files are editable SVGs and 450 dpi PNGs in `figures/geography_thesis/final/`.

For the validation extension, first build the respondent files following `REPRODUCE_GEOGRAPHY.md`. Then run the following commands from the repository root.

```sh
python src/ml_extension/check_design.py
python src/ml_extension/audit_shift.py
python src/ml_extension/run_extension.py
python src/ml_extension/evaluate_extension.py
python src/ml_extension/matched_transport.py
```

`reports/ml_extension/analysis_specification.md` describes the tasks and information sets. `results/ml_extension/fold_design_audit.csv` records the sample counts, overlap and membership fingerprints of the outer and inner folds. `legacy_l2_algorithm_contrasts.csv` contains the four previously fitted five-fold LightGBM versus L2 logistic comparisons reported in the manuscript. The elastic-net search is a separate analysis.
