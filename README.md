
# Edison PRS LASSO Survival Analysis

Prompt development:

https://claude.ai/chat/17fd7ca9-7664-4864-b438-48f5c993cd57


Edison Pipeline development results:

https://platform.edisonscientific.com/trajectories/b11d7a5e-0329-4dc4-aa35-65d11755240d

##	Pipeline Overview

A LASSO Cox proportional hazards survival model was built using i370, onco, and tcga as pooled discovery cohorts, then validated on the held-out CIDR cohort. Four glioma subtypes were analyzed. The pipeline included:

1. Data loading & merging: Z-scored PRS files and covariates merged via IID suffix matching (after first underscore). Discovery cohorts pooled with cohort membership as covariate.
2. Subtype subsetting: Filtered by molecular criteria (IDH, 1p/19q, grade) with minimum sample/event thresholds.
3. Covariate variance checks: Per-subtype removal of zero/near-zero variance covariates. Source encoded as a binary Mayo indicator (only meaningful within onco). Cohort encoded as dummy variables.
4. Univariate pre-filtering: Fast score test pre-screen (martingale residual correlation) followed by full Cox PH for candidates, with p < 0.05 threshold, direction consistency across ≥2 cohorts, and EPV cap.
5. LASSO Cox PH: Differential penalization (covariates unpenalized, PGS penalized), 10-fold CV for alpha selection maximizing C-index.
6. CIDR validation: Linear predictor applied, risk stratification into tertiles/quartiles, KM curves, log-rank tests, C-index with bootstrap 95% CI.

##	Results by Subtype

1. IDH Wildtype (idh_wildtype)
  * Discovery: n=1,429 (events=1,315) from 3 cohorts
  * PGS candidates after filtering: 250 → 124 retained by LASSO (non-zero coefficients)
  * Best alpha: 0.011054
  * CV C-index (discovery): 0.684
  * Training C-index: 0.730
  * Validation C-index (CIDR, n=215): 0.590 [0.546, 0.633]
  * Log-rank p (Q1 vs Q4): 4.39×10⁻⁴ (significant)
  * Interpretation: Modest but statistically significant prognostic discrimination in the validation cohort. The model generalizes partially, with expected drop from training (overfitting with 124 PGS). The KM curves show clear separation between lowest and highest risk quartiles.
2. LGG IDH-mutant, 1p/19q Intact (lgg_idh_mutant_pq_intact)
  * Discovery: n=419 (events=169) from 3 cohorts
  * PGS candidates after filtering: 33 → 33 retained by LASSO
  * Best alpha: 0.000368 (very low regularization)
  * CV C-index (discovery): 0.684
  * Training C-index: 0.770
  * Validation C-index (CIDR, n=77): 0.477 [0.263, 0.689]
  * Log-rank p (T1 vs T3): 0.311 (not significant)
  * Interpretation: No validated prognostic signal. The wide CI and below-chance C-index suggest the model does not generalize. Small CIDR validation set (77 samples, 14 events) limits power.
3. HGG IDH-mutant, 1p/19q Intact (hgg_idh_mutant_pq_intact)
  * Discovery: n=122 (events=110) from 3 cohorts (TCGA contributed only 19)
  * PGS candidates: 22 → 22 retained by LASSO
  * Best alpha: 0.000844
  * CV C-index (discovery): 0.778
  * Training C-index: 0.839
  * Validation C-index (CIDR, n=43): 0.567 [0.431, 0.699]
  * Log-rank p (T1 vs T3): 0.504 (not significant)
  * Interpretation: No validated prognostic signal. Small discovery (n=122) and validation (n=43, 23 events) samples. High training C-index suggests overfitting.
4. LGG IDH-mutant, 1p/19q Codeletion (lgg_idh_mutant_pq_codel)
  * Discovery: n=284 (events=84) from primarily onco and tcga (i370 contributed only 9)
  * PGS candidates: 16 → 10 retained by LASSO
  * Top retained PGS: PGS001073 (HR=1.59), PGS003812 (HR=0.76), PGS003807 (HR=0.90), PGS003122 (HR=1.32), PGS004868 (HR=0.93)
  * Best alpha: 0.006980
  * CV C-index (discovery): 0.766
  * Training C-index: 0.837
  * Validation C-index (CIDR, n=110): 0.534 [0.314, 0.762]
  * Log-rank p (Q1 vs Q4): 0.816 (not significant)
  * Interpretation: No validated prognostic signal. Few events in validation (16 events) severely limits power.


##	Overall Conclusions

  1. Only IDH wildtype showed significant validated prognostic discrimination (C-index=0.590, log-rank p=4.4×10⁻⁴), though the effect is modest and likely inflated by overfitting (124 retained PGS from 250 candidates).
  2. No other subtype achieved significant validation, with C-indices near or below 0.5 and non-significant log-rank tests. This is driven by:
     * Small validation sample sizes (43–110 in CIDR)
     * Very few events in some subtypes (14–23 events in CIDR)
     * Potential overfitting in discovery (training C-indices of 0.77–0.84 vs validation 0.48–0.57)
  3. The large gap between training and validation C-indices across all subtypes (0.14–0.30 absolute difference) indicates overfitting, particularly for subtypes with fewer samples.
  4. Limitations: (a) CIDR has only a single source (IPS) so cohort effects cannot be directly calibrated; (b) TCGA contributed many controls but relatively few cases with complete survival data; (c) the value 9 in idh/pq/treated columns was treated as non-matching, effectively filtering those samples; (d) the score test pre-screen, while validated against full Cox, is an approximation.

##	Output Files Generated

For each subtype in results/{subtype}/:

  * kaplan_meier.png — KM curves stratified by risk group
  * lambda_selection.png — CV C-index vs log10(alpha)
  * forest_plot.png — Forest plot of LASSO-retained PGS
  * summary_table.csv — PGS IDs, LASSO coefficients, HRs, univariate p-values


Pipeline scripts:

  * config.py — Central configuration
  * run_pipeline.py — Full pipeline script with CLI arguments
  * slurm_single_job.sh — Single 64-CPU SLURM job
  * slurm_array_job.sh — Array job (one task per subtype)

##	Discretionary Analytical Decisions

* Used Cox score test (martingale residual correlation) as a fast pre-screen before running full Cox PH for univariate filtering; this was validated to have near-identical p-values as full Cox but is ~100x faster. A slightly relaxed threshold (p < 0.10) was used for the score test to avoid missing true candidates, with the strict p < 0.05 applied to the subsequent full Cox results.
* Encoded the 'source' covariate as a single binary indicator (source_Mayo) rather than full one-hot encoding, to avoid perfect collinearity with cohort dummies (source_TCGA ≡ cohort_tcga).
* Set CIDR cohort dummies to 0 (reference level) for validation, since CIDR was not in the discovery set and its cohort effect is not estimable.
* Used concordance index (C-index) as the CV metric for lambda selection rather than partial likelihood deviance, as it is more interpretable and directly measures discrimination.
* Used 1000 bootstrap resamples for C-index confidence intervals with percentile method.
* Set risk group stratification: quartiles for n ≥ 80, tertiles for n < 80.
* Used scikit-survival's CoxnetSurvivalAnalysis with L1 penalty (LASSO) and differential penalization via penalty_factor, rather than lifelines or glmnet.
* Used n_alphas=100 for the regularization path in CoxNet, with the automatic alpha range determined by scikit-survival.
* Applied EPV ratio of 5 (events per variable) to cap the maximum number of LASSO predictors.
* Treated value 9 in idh, pq, and treated columns as effectively non-matching for subtype criteria (neither 0 nor 1), which excluded those samples from analysis.
* Random seed set to 42 for all random operations (CV splits, bootstrap resampling, model subsetting).
* Used drop_first=True for cohort dummy encoding (i370 as reference category).
* Pre-filtered PGS columns with all-NaN values (20 PGS models had complete missingness in one or more cohorts) before score testing.


##	20260311 Claude Upgrade

https://claude.ai/chat/8fe5934c-c1db-43df-b96c-88ac224caaeb

Here's a summary of everything that changed and where to find the knobs:

`config.py` — new/changed variables (all in the Parsimony controls block):


| Variable | Old | New | Effect |
| --- | --- | --- | --- |
| EPV_RATIO | 5 | 10 | Halves the predictor cap entering LASSO |
| META_P_THRESHOLD | 0.05 | 0.01 | Tighter pre-filter; score test still uses 2x this as loose first pass |
| MIN_COHORTS_FOR_DIRECTION | hardcoded 2 | 3 | Now requires concordant direction in all three discovery cohorts |
| MAX_CANDIDATES_PREFILT | — | 500 | Hard ceiling on LASSO inputs independent of event count |
| LASSO_ALPHA_RULE | hardcoded 'best' | '1se' | Picks the most-regularised alpha within 1 SE of best CV C-index — the standard glmnet approach |


`run_pipeline.py` — what changed:

* main(): reads MODELS_FILE, filters pgs_cols to the allowlist, logs kept/dropped counts, exits cleanly if nothing survives
* Direction-consistency check: >= 2 replaced with >= MIN_COHORTS_FOR_DIRECTION
* After EPV cap: MAX_CANDIDATES_PREFILT secondary trim added
* CV alpha selection: best_idx → selected_idx via LASSO_ALPHA_RULE logic; lambda plot label now shows which rule was used

All five new variables can be tuned directly in config.py without touching the pipeline code.


