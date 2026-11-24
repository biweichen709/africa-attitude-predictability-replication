# Machine-learning extension specification

Date frozen: 24 November 2026. Status: retrospective extension to previously inspected data.

## Political-science question

The extension tests whether individual socioeconomic information transports across respondents, countries, and survey rounds. Algorithm choice is secondary to the information available at prediction time.

## Samples and information sets

All algorithm comparisons use the same outcome-valid respondents. Missing predictors are imputed from each training fold. The socioeconomic information set contains age, sex, urban residence, education, employment, occupation, five deprivation items, six assets, and household electricity. The context information set adds area services and lagged country-level economic indicators. `fear_crime_home` is excluded because Round 10 does not contain a harmonized measure.

Countries are identified by ISO3 codes. Round 8 and Round 10 share 30 countries; `Gambia` and `Gambia, The` are the same country (`GMB`). Round 10 has 50,055 valid cultural outcomes and 49,399 valid political outcomes. The corresponding complete eight-variable core samples are 48,026 and 47,383.

## Validation tasks

Known-country interpolation uses three outer respondent folds formed separately within each country and outcome. Every country appears in training and testing. Inner folds repeat this construction using training respondents only.

Country holdout uses five outer country folds and three inner country folds. No held-out country contributes outcomes, predictor transformations, hyperparameter selection, or calibration.

Temporal transport fits on Round 8 and evaluates Round 10 once. Three inner country folds select hyperparameters and estimate Platt calibration from Round 8 out-of-fold predictions. Round 10 outcomes do not alter the model. Results are separated into the 30 shared and eight new countries.

## Models and selection

The algorithmic models are elastic-net logistic regression and LightGBM. Candidate hyperparameters are fixed in code. The selected candidate minimizes equal-country weighted log loss across inner folds. Preprocessing, imputation, scaling, category encoding, model fitting, and calibration are fitted within training data. Existing pooled logistic, country-prevalence, centred logistic, and multilevel logistic results remain the interpretable reference models.

## Evaluation

The primary metric is equal-country weighted log loss. Secondary metrics are Brier score, pooled AUC, arithmetic mean-country AUC, calibration intercept and slope, and expected calibration error. Country bootstrap intervals use 2,000 paired resamples and condition on the fitted predictions. Social-group diagnostics report group-specific loss and calibration error for sex, urban residence, education, and deprivation. Decision curves report standardized net benefit over probability thresholds from 0.10 to 0.50. They describe hypothetical classification tradeoffs and do not represent a specified policy deployment.

All respondent-level predictions and fitted models remain local. The public release contains code, specifications, audit files, and aggregate outputs only.
