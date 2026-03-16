#!/usr/bin/env python3
"""
PRS LASSO Cox Survival Analysis Pipeline

Builds a LASSO Cox PH survival model using pooled discovery cohorts (i370, onco, tcga),
then validates on the held-out CIDR cohort. Analyzes 4 glioma subtypes.

Usage:
    python3 run_pipeline.py --data-dir /path/to/data --output-dir results
    python3 run_pipeline.py --data-dir /path/to/data --n-models 100 --debug  # dev test
    python3 run_pipeline.py --test  # smoke test
"""

import os
import sys
import time
import json
import gzip
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import config
import config
from config import *

def dprint(*args, **kwargs):
    """Debug print - only prints if DEBUG is True."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def load_cohort_data(cohort, data_dir):
    """Load z-scored PRS and covariates for a cohort, merge on IID part."""
    print(f"  Loading {cohort}...")

    cov = pd.read_csv(os.path.join(data_dir, f'{cohort}-covariates.csv'))

    zsc = pd.read_csv(os.path.join(data_dir, f'{cohort}.scores.z-scores.txt.gz'), compression='gzip')

    merged = cov.merge(zsc, left_on='IID', right_on='sample', how='inner')
    merged['cohort'] = cohort

    # Encode sex
    merged['sex'] = merged['sex'].map({'M': 0, 'F': 1})
    # Encode grade
    merged['grade_numeric'] = merged['grade'].map({'LGG': 0, 'HGG': 1})

    print(f"    Loaded: {len(merged)} samples (cov={len(cov)}, zsc={len(zsc)})")
    return merged


def subset_by_subtype(df, subtype_name, criteria):
    """Filter to matching subtype with valid survival data."""
    mask = pd.Series(True, index=df.index)
    for col, val in criteria.items():
        mask &= (df[col] == val)

    subset = df[mask].copy()
    valid = subset.dropna(subset=['survdays', 'vstatus'])
    valid = valid[valid['survdays'] > 0]
    valid = valid[valid['vstatus'].isin([0, 1])]

    n_events = int((valid['vstatus'] == 1).sum())
    print(f"  {subtype_name}: n={len(valid)}, events={n_events}")

    if len(valid) < MIN_SAMPLES_PER_SUBTYPE:
        print(f"  WARNING: < {MIN_SAMPLES_PER_SUBTYPE} samples. Skipping.")
        return None
    if n_events < MIN_EVENTS_PER_SUBTYPE:
        print(f"  WARNING: < {MIN_EVENTS_PER_SUBTYPE} events. Skipping.")
        return None

    return valid


def check_covariate_variance(df, covariates, label=""):
    """Check and filter covariates with zero/near-zero variance.

    Uses pd.api.types.is_numeric_dtype() rather than dtype == 'object' so that
    Arrow-backed string columns (pandas >= 2.0 with pyarrow backend) are handled
    correctly alongside legacy object-dtype strings.
    """
    surviving = []
    for cov in covariates:
        col = 'grade_numeric' if cov == 'grade' else cov
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if pd.api.types.is_numeric_dtype(vals):
            if vals.var() < 1e-10:
                dprint(f"  Dropping {cov} ({label}): near-zero variance")
                continue
        else:
            # Categorical / string column — check cardinality only
            if vals.nunique() <= 1:
                dprint(f"  Dropping {cov} ({label}): single value")
                continue
        if df[col].isna().sum() / len(df) > 0.5:
            dprint(f"  Dropping {cov} ({label}): >50% missing")
            continue
        surviving.append(cov)
    return surviving


def compute_score_tests(df, pgs_cols, cov_cols):
    """Fast univariate pre-screening using Cox score test via martingale residuals."""
    base_data = df[cov_cols + ['survdays', 'vstatus']].dropna()
    base_idx = base_data.index

    if len(base_data) < 20 or base_data['vstatus'].sum() < 10:
        return pd.DataFrame()

    cph_null = CoxPHFitter()
    cph_null.fit(base_data, duration_col='survdays', event_col='vstatus', show_progress=False)

    lp = cph_null.predict_partial_hazard(base_data).values.flatten()
    baseline_cum_haz = cph_null.baseline_cumulative_hazard_
    times = base_data['survdays'].values
    events = base_data['vstatus'].values
    H0_at_t = np.interp(times, baseline_cum_haz.index.values, baseline_cum_haz.values.flatten())
    martingale_resid = events - H0_at_t * lp

    available_pgs = []
    pgs_data_list = []
    for pgs in pgs_cols:
        vals = df.loc[base_idx, pgs].values
        if np.isnan(vals).sum() == 0 and np.var(vals) > 1e-10:
            available_pgs.append(pgs)
            pgs_data_list.append(vals)

    if not available_pgs:
        return pd.DataFrame()

    pgs_matrix = np.column_stack(pgs_data_list)
    X_cov = base_data[cov_cols].values.astype(np.float64)
    XtX_inv_Xt = np.linalg.pinv(X_cov)
    pgs_projected = X_cov @ (XtX_inv_Xt @ pgs_matrix)
    pgs_residualized = pgs_matrix - pgs_projected

    U_adj = pgs_residualized.T @ martingale_resid
    var_M = np.var(martingale_resid)
    var_U = np.sum(pgs_residualized**2, axis=0) * var_M
    T_score = U_adj**2 / (var_U + 1e-10)
    p_score = 1 - stats.chi2.cdf(T_score, df=1)
    direction = np.sign(U_adj)

    return pd.DataFrame({'pgs': available_pgs, 'score_p': p_score, 'direction': direction})


def _fit_one_cox(df, pgs, cov_cols):
    """Fit a single univariate Cox PH model. Used by fit_full_cox_batch."""
    try:
        cols = [pgs] + cov_cols + ['survdays', 'vstatus']
        sub = df[cols].dropna()
        if len(sub) < 10 or sub['vstatus'].sum() < 5 or sub[pgs].var() < 1e-10:
            return None
        cph = CoxPHFitter()
        cph.fit(sub, duration_col='survdays', event_col='vstatus', show_progress=False)
        s = cph.summary.loc[pgs]
        return {
            'pgs': pgs, 'coef': s['coef'], 'hr': s['exp(coef)'],
            'se': s['se(coef)'], 'z': s['z'], 'p': s['p'],
            'ci_lower': s['exp(coef) lower 95%'], 'ci_upper': s['exp(coef) upper 95%'],
            'n': len(sub), 'events': int(sub['vstatus'].sum()),
        }
    except Exception:
        return None


def fit_full_cox_batch(df, pgs_list, cov_cols):
    """Fit full univariate Cox PH for a list of PGS models (parallel)."""
    results = Parallel(n_jobs=N_JOBS, prefer='threads')(
        delayed(_fit_one_cox)(df, pgs, cov_cols) for pgs in pgs_list
    )
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


def fit_baseline_cox(train_df, val_df, cov_cols, label):
    """Fit a covariate-only Cox PH model and return train/val C-indices.

    Parameters
    ----------
    train_df : DataFrame  discovery subset (must contain cov_cols + survdays + vstatus)
    val_df   : DataFrame  validation subset (same columns required)
    cov_cols : list       covariate column names to use
    label    : str        human-readable label for log output

    Returns
    -------
    dict with keys: train_cindex, val_cindex (None if val fitting fails)
    """
    result = {'label': label, 'train_cindex': None, 'val_cindex': None}
    try:
        cols = cov_cols + ['survdays', 'vstatus']
        tr = train_df[cols].dropna()
        if len(tr) < 10 or tr['vstatus'].sum() < 5:
            return result
        cph = CoxPHFitter()
        cph.fit(tr, duration_col='survdays', event_col='vstatus', show_progress=False)
        risk_tr = cph.predict_partial_hazard(tr).values.flatten()
        result['train_cindex'] = concordance_index_censored(
            tr['vstatus'].astype(bool).values, tr['survdays'].values, risk_tr)[0]

        va = val_df[cols].dropna()
        if len(va) >= 10 and va['vstatus'].sum() >= 5:
            risk_va = cph.predict_partial_hazard(va).values.flatten()
            result['val_cindex'] = concordance_index_censored(
                va['vstatus'].astype(bool).values, va['survdays'].values, risk_va)[0]
    except Exception as e:
        dprint(f"  Baseline Cox failed for '{label}': {e}")
    return result


def _cv_fold_one(fi, tri, vai, X, y, penalty_factor, alpha_path):
    """Fit CoxNet on one CV fold and return (fold_idx, scores array).
    Used by process_subtype to parallelize the CV loop."""
    scores = np.full(len(alpha_path), np.nan)
    try:
        cv_m = CoxnetSurvivalAnalysis(l1_ratio=1.0, penalty_factor=penalty_factor,
                                       alphas=alpha_path, fit_baseline_model=True)
        cv_m.fit(X[tri], y[tri])
        for ai in range(len(alpha_path)):
            risk = X[vai] @ cv_m.coef_[:, ai]
            try:
                scores[ai] = concordance_index_censored(
                    y[vai]['event'], y[vai]['time'], risk)[0]
            except Exception:
                pass
    except Exception:
        pass
    return fi, scores


def _check_direction_one(pgs, df_sub, active_cohorts, per_cohort_cov):
    """Check direction consistency for a single PGS across active cohorts.
    Returns pgs name if consistent, None otherwise. Used by process_subtype."""
    signs = {}
    for cohort in active_cohorts:
        try:
            cdf = df_sub[df_sub['cohort'] == cohort]
            cv  = [c for c in per_cohort_cov[cohort] if cdf[c].var() > 1e-10]
            cols = [pgs] + cv + ['survdays', 'vstatus']
            sub = cdf[cols].dropna()
            if len(sub) < 10 or sub['vstatus'].sum() < 5 or sub[pgs].var() < 1e-10:
                continue
            cph = CoxPHFitter()
            cph.fit(sub, duration_col='survdays', event_col='vstatus', show_progress=False)
            signs[cohort] = np.sign(cph.summary.loc[pgs, 'coef'])
        except Exception:
            continue
    if len(signs) >= MIN_COHORTS_FOR_DIRECTION and len(set(signs.values())) == 1:
        return pgs
    return None


def process_subtype(subtype_name, criteria, discovery_pooled, val_df, val_name,
                    pgs_cols, output_dir, train_cohorts=None):
    """Process a single glioma subtype through the full pipeline.

    Parameters
    ----------
    subtype_name     : str        e.g. 'idh_wildtype'
    criteria         : dict       column filters defining the subtype
    discovery_pooled : DataFrame  pooled training cohorts
    val_df           : DataFrame  held-out validation cohort (any single cohort)
    val_name         : str        cohort name for logging/output (e.g. 'cidr')
    pgs_cols         : list       PGS column names to consider
    output_dir       : str        root results directory
    train_cohorts    : list|None  cohort names present in discovery_pooled; used
                                  for direction-consistency check.  Defaults to
                                  DISCOVERY_COHORTS if None (fixed strategy).
    """
    if train_cohorts is None:
        train_cohorts = DISCOVERY_COHORTS
    print(f"\n{'='*60}")
    print(f"SUBTYPE: {subtype_name}")
    print(f"{'='*60}")

    subtype_dir = os.path.join(output_dir, subtype_name)
    os.makedirs(subtype_dir, exist_ok=True)

    # Step 2: Subset
    df_sub = subset_by_subtype(discovery_pooled, subtype_name, criteria)
    if df_sub is None:
        return {'status': 'insufficient_samples'}

    df_sub = df_sub.copy()
    n_events = int((df_sub['vstatus'] == 1).sum())

    # Step 3: Covariate variance check
    surviving_covs = check_covariate_variance(df_sub, COVARIATES, subtype_name)
    surviving_covs_no_source = [c for c in surviving_covs if c != 'source']

    if 'source' in surviving_covs:
        df_sub['source_Mayo'] = (df_sub['source'] == 'Mayo').astype(float)
        source_dummy_cols = ['source_Mayo'] if df_sub['source_Mayo'].var() > 1e-10 else []
    else:
        source_dummy_cols = []

    cohort_dummies = pd.get_dummies(df_sub['cohort'], prefix='cohort', drop_first=True)
    cohort_dummy_cols = list(cohort_dummies.columns)
    for col in cohort_dummy_cols:
        df_sub[col] = cohort_dummies[col].values

    cov_col_list = []
    for cov in surviving_covs_no_source:
        cov_col_list.append('grade_numeric' if cov == 'grade' else cov)
    cov_col_list += source_dummy_cols + cohort_dummy_cols

    # Final variance check
    base_check = df_sub[cov_col_list + ['survdays', 'vstatus']].dropna()
    cov_col_list = [c for c in cov_col_list if base_check[c].var() > 1e-10 and base_check[c].nunique() > 1]
    print(f"  Covariates: {cov_col_list}")

    # Step 4: Score test pre-screening
    print(f"  Score test for {len(pgs_cols)} PGS...")
    score_results = compute_score_tests(df_sub, pgs_cols, cov_col_list)

    if len(score_results) == 0:
        return {'status': 'no_models_tested'}

    score_sig = score_results[score_results['score_p'] < META_P_THRESHOLD * 2]

    if len(score_sig) == 0:
        return {'status': 'no_significant_pgs'}

    # Full Cox for pre-screened candidates
    print(f"  Full Cox for {len(score_sig)} candidates...")
    pooled_df = fit_full_cox_batch(df_sub, score_sig['pgs'].tolist(), cov_col_list)

    if len(pooled_df) == 0:
        return {'status': 'no_fitted_models'}

    sig_pooled = pooled_df[pooled_df['p'] < META_P_THRESHOLD]
    print(f"  Significant (p<{META_P_THRESHOLD}): {len(sig_pooled)}")

    if len(sig_pooled) == 0:
        return {'status': 'no_significant_pgs_fullcox'}

    # Direction consistency
    if REQUIRE_CONSISTENT_DIR:
        per_cohort_cov = {}
        for cohort in train_cohorts:
            csub = df_sub[df_sub['cohort'] == cohort]
            if len(csub) < 10 or (csub['vstatus'] == 1).sum() < 5:
                per_cohort_cov[cohort] = None
                continue
            valid = []
            for col in cov_col_list:
                if col.startswith('cohort_'):
                    continue
                if col in csub.columns and csub[col].dropna().var() > 1e-10:
                    valid.append(col)
            per_cohort_cov[cohort] = valid

        active_cohorts = [c for c, v in per_cohort_cov.items() if v is not None]

        dir_results = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_check_direction_one)(
                row['pgs'], df_sub, active_cohorts, per_cohort_cov)
            for _, row in sig_pooled.iterrows()
        )
        consistent = [pgs for pgs in dir_results if pgs is not None]

        print(f"  Consistent direction: {len(consistent)} / {len(sig_pooled)}")
        filtered_pgs = consistent
    else:
        filtered_pgs = sig_pooled['pgs'].tolist()

    if not filtered_pgs:
        return {'status': 'no_filtered_pgs'}

    filtered_df = pooled_df[pooled_df['pgs'].isin(filtered_pgs)].sort_values('p')

    # Step 5: EPV cap
    max_pred = n_events // EPV_RATIO
    print(f"  EPV cap: {max_pred} (events={n_events})")
    if len(filtered_df) > max_pred:
        filtered_df = filtered_df.head(max_pred)

    # Secondary hard cap (MAX_CANDIDATES_PREFILT)
    if MAX_CANDIDATES_PREFILT is not None and len(filtered_df) > MAX_CANDIDATES_PREFILT:
        print(f"  Hard cap: trimming {len(filtered_df)} -> {MAX_CANDIDATES_PREFILT} candidates")
        filtered_df = filtered_df.head(MAX_CANDIDATES_PREFILT)

    final_pgs = filtered_df['pgs'].tolist()

    # Step 6: LASSO Cox
    print(f"  LASSO Cox: {len(final_pgs)} PGS + {len(cov_col_list)} covariates")
    feature_cols = final_pgs + cov_col_list
    model_df = df_sub[feature_cols + ['survdays', 'vstatus']].dropna()

    X = model_df[feature_cols].values.astype(np.float64)
    y = np.array([(bool(e), t) for e, t in zip(model_df['vstatus'], model_df['survdays'])],
                  dtype=[('event', bool), ('time', float)])

    n_pgs = len(final_pgs)
    n_cov = len(cov_col_list)
    penalty_factor = np.array([1.0]*n_pgs + [0.0]*n_cov)

    coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0, penalty_factor=penalty_factor,
                                     n_alphas=100, fit_baseline_model=True)
    coxnet.fit(X, y)
    alpha_path = coxnet.alphas_

    # CV — parallel over folds
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = np.full((CV_FOLDS, len(alpha_path)), np.nan)

    fold_results_cv = Parallel(n_jobs=min(N_JOBS, CV_FOLDS), prefer='threads')(
        delayed(_cv_fold_one)(fi, tri, vai, X, y, penalty_factor, alpha_path)
        for fi, (tri, vai) in enumerate(kf.split(X))
    )
    for fi, scores in fold_results_cv:
        cv_scores[fi] = scores

    mean_cv = np.nanmean(cv_scores, axis=0)
    std_cv = np.nanstd(cv_scores, axis=0)

    # Alpha selection: 'best' maximises CV C-index; '1se' picks the most
    # regularised alpha whose mean is within 1 SE of the best (glmnet convention).
    best_idx = np.nanargmax(mean_cv)
    if LASSO_ALPHA_RULE == '1se':
        threshold = mean_cv[best_idx] - std_cv[best_idx]
        # alpha_path is descending (more regularisation = larger alpha = earlier index)
        # so we want the smallest index (largest alpha) that still clears the threshold
        eligible = np.where(mean_cv >= threshold)[0]
        selected_idx = int(eligible[0])  # highest regularisation among eligible
        print(f"  Alpha rule: 1se  (best_idx={best_idx}, selected_idx={selected_idx})")
    else:
        selected_idx = best_idx
        print(f"  Alpha rule: best (selected_idx={selected_idx})")
    best_alpha = alpha_path[selected_idx]

    final_m = CoxnetSurvivalAnalysis(l1_ratio=1.0, penalty_factor=penalty_factor,
                                      alphas=[best_alpha], fit_baseline_model=True)
    final_m.fit(X, y)
    final_coef = final_m.coef_[:, 0]

    pgs_coef = final_coef[:n_pgs]
    nonzero = np.abs(pgs_coef) > 1e-10

    train_risk = X @ final_coef
    train_ci = concordance_index_censored(y['event'], y['time'], train_risk)[0]

    print(f"  Best alpha: {best_alpha:.6f}, CV C-index: {mean_cv[selected_idx]:.4f}")
    print(f"  Non-zero PGS: {nonzero.sum()}, Train C-index: {train_ci:.4f}")

    # Save lambda plot
    fig, ax = plt.subplots(figsize=(10, 6))
    vm = ~np.isnan(mean_cv)
    ax.errorbar(np.log10(alpha_path[vm]), mean_cv[vm], yerr=std_cv[vm],
                fmt='o-', markersize=2, linewidth=0.5, capsize=1)
    ax.axvline(np.log10(best_alpha), color='red', linestyle='--',
               label=f'selected ({LASSO_ALPHA_RULE})')
    ax.set_xlabel('log10(alpha)'); ax.set_ylabel('CV C-index')
    ax.set_title(f'{subtype_name}: Lambda Selection')
    plt.tight_layout()
    fig.savefig(os.path.join(subtype_dir, 'lambda_selection.png'), dpi=150)
    plt.close(fig)

    # Step 7: Held-out cohort validation
    val_sub = subset_by_subtype(val_df, subtype_name, criteria)
    if val_sub is None:
        return {'status': 'no_val_validation', 'train_cindex': train_ci,
                'n_nonzero_pgs': int(nonzero.sum()), 'best_cv_cindex': mean_cv[selected_idx],
                'baselines': {}}

    val_sub = val_sub.copy()

    # Prepare validation covariates:
    # source_Mayo: set from actual source column if present, else 0
    if 'source_Mayo' in cov_col_list:
        if 'source' in val_sub.columns:
            val_sub['source_Mayo'] = (val_sub['source'] == 'Mayo').astype(float)
        else:
            val_sub['source_Mayo'] = 0.0

    # Cohort dummies: val cohort is not one of the training cohorts, so all are 0
    for cd in cohort_dummy_cols:
        val_sub[cd] = 0.0

    val_model = val_sub[feature_cols + ['survdays', 'vstatus']].dropna()
    X_v = val_model[feature_cols].values.astype(np.float64)
    y_v = np.array([(bool(e), t) for e, t in zip(val_model['vstatus'], val_model['survdays'])],
                    dtype=[('event', bool), ('time', float)])

    val_risk = X_v @ final_coef
    val_ci = concordance_index_censored(y_v['event'], y_v['time'], val_risk)[0]

    # Bootstrap CI
    rng = np.random.RandomState(RANDOM_SEED)
    boot = [concordance_index_censored(y_v['event'][idx], y_v['time'][idx], val_risk[idx])[0]
            for idx in (rng.choice(len(val_risk), len(val_risk), True) for _ in range(1000))
            if True]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    print(f"  Validation C-index ({val_name}): {val_ci:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ------------------------------------------------------------------
    # Baseline models: covariate-only C-indices for comparison
    #
    # Three baselines:
    #   1. clinical_only   : age, sex, grade_numeric, treated
    #   2. clinical_pcs    : clinical + PC1-PC8
    #   3. full_covariates : all covariates (cov_col_list, same as LASSO)
    #
    # For each we fit on the discovery subset and evaluate on CIDR,
    # mirroring exactly what the PGS model does.  cohort dummies are
    # included in full_covariates but zeroed-out in CIDR (same as above).
    # ------------------------------------------------------------------
    CLINICAL_COLS  = ['age', 'sex', 'grade_numeric', 'treated']
    PC_COLS        = [f'PC{i}' for i in range(1, 9)]

    clinical_cols  = [c for c in CLINICAL_COLS if c in cov_col_list]
    clinical_pc_cols = [c for c in CLINICAL_COLS + PC_COLS if c in cov_col_list]
    full_cov_cols  = cov_col_list   # already variance-filtered, includes dummies

    # CIDR baseline df needs the same dummy-zeroing as the PGS validation
    val_base = val_sub.copy()   # val_sub already has source_Mayo and zeroed cohort dummies

    baselines = {}
    for bl_label, bl_cols in [
        ('clinical_only',   clinical_cols),
        ('clinical_pcs',    clinical_pc_cols),
        ('full_covariates', full_cov_cols),
    ]:
        bl = fit_baseline_cox(df_sub, val_base, bl_cols, bl_label)
        baselines[bl_label] = bl
        tr_str = f"{bl['train_cindex']:.4f}" if bl['train_cindex'] is not None else "n/a"
        va_str = f"{bl['val_cindex']:.4f}"   if bl['val_cindex']  is not None else "n/a"
        print(f"  Baseline [{bl_label}]: train={tr_str}, val={va_str}")

    print(f"  PGS model:           train={train_ci:.4f}, val={val_ci:.4f}")

    # KM plot
    n_groups = 4 if len(val_model) >= 80 else 3
    gl = 'Quartile' if n_groups == 4 else 'Tertile'
    val_model = val_model.copy()
    val_model['risk'] = val_risk
    val_model['rg'] = pd.qcut(val_risk, n_groups, labels=False, duplicates='drop') + 1

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    for g in sorted(val_model['rg'].unique()):
        gd = val_model[val_model['rg'] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(gd['survdays'], gd['vstatus'], label=f'{gl} {g} (n={len(gd)})')
        kmf.plot_survival_function(ax=ax, color=colors[g-1], ci_show=True)

    lr = logrank_test(val_model[val_model['rg']==val_model['rg'].min()]['survdays'],
                      val_model[val_model['rg']==val_model['rg'].max()]['survdays'],
                      val_model[val_model['rg']==val_model['rg'].min()]['vstatus'],
                      val_model[val_model['rg']==val_model['rg'].max()]['vstatus'])

    ax.set_title(f'{subtype_name} [{val_name}]: KM ({gl}s)\n'
                 f'C-index={val_ci:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], LR p={lr.p_value:.2e}')
    ax.set_xlabel('Time (days)'); ax.set_ylabel('Survival Probability')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(subtype_dir, f'kaplan_meier_{val_name}.png'), dpi=150)
    plt.close(fig)

    # Summary table
    nz_names = [final_pgs[i] for i in range(n_pgs) if nonzero[i]]
    nz_coefs = pgs_coef[nonzero]

    if nz_names:
        st = pd.DataFrame({'PGS_ID': nz_names, 'LASSO_coef': nz_coefs, 'LASSO_HR': np.exp(nz_coefs)})
        for i, pn in enumerate(nz_names):
            r = pooled_df[pooled_df['pgs'] == pn]
            if len(r):
                st.loc[i, 'univar_p'] = r.iloc[0]['p']
                st.loc[i, 'univar_HR'] = r.iloc[0]['hr']
                st.loc[i, 'univar_CI_lo'] = r.iloc[0]['ci_lower']
                st.loc[i, 'univar_CI_hi'] = r.iloc[0]['ci_upper']
        st = st.sort_values('univar_p')
        st.to_csv(os.path.join(subtype_dir, 'summary_table.csv'), index=False)

        # Forest plot
        fig, ax = plt.subplots(figsize=(10, max(4, len(nz_names)*0.5+2)))
        yp = list(range(len(st)))
        ax.errorbar(st['univar_HR'].values, yp,
                    xerr=[st['univar_HR'].values-st['univar_CI_lo'].values,
                          st['univar_CI_hi'].values-st['univar_HR'].values],
                    fmt='s', color='steelblue', markersize=6, capsize=3, label='Univar HR (95% CI)')
        ax.scatter(st['LASSO_HR'].values, yp, color='red', marker='D', s=50, zorder=5, label='LASSO HR')
        ax.axvline(1, color='grey', linestyle='--')
        ax.set_yticks(yp); ax.set_yticklabels(st['PGS_ID'])
        ax.set_xlabel('HR'); ax.set_title(f'{subtype_name}: Forest Plot')
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(subtype_dir, 'forest_plot.png'), dpi=150)
        plt.close(fig)

    return {
        'status': 'complete', 'val_cohort': val_name,
        'n_discovery': len(model_df), 'n_events': n_events,
        'n_pgs_candidates': len(final_pgs), 'n_nonzero_pgs': int(nonzero.sum()),
        'best_alpha': best_alpha, 'best_cv_cindex': float(mean_cv[selected_idx]),
        'train_cindex': float(train_ci), 'val_cindex': float(val_ci),
        'val_ci': (float(ci_lo), float(ci_hi)), 'logrank_p': float(lr.p_value),
        'baselines': {
            k: {
                'train_cindex': float(v['train_cindex']) if v['train_cindex'] is not None else None,
                'val_cindex':   float(v['val_cindex'])   if v['val_cindex']   is not None else None,
            }
            for k, v in baselines.items()
        },
    }


def _build_pgs_table(all_rows, run_labels):
    """
    Shared helper: given a list of dicts (one per selected PGS per run),
    build a consistency DataFrame with per-run LASSO coef columns,
    fold/split counts, mean univar stats, and sign consistency flag.

    Parameters
    ----------
    all_rows   : list of dict, each with keys:
                   run_label, subtype, PGS_ID, LASSO_coef, LASSO_HR,
                   and optionally univar_p, univar_HR, univar_CI_lo, univar_CI_hi
    run_labels : ordered list of all run labels (folds or split IDs)

    Returns
    -------
    consistency : DataFrame, one row per (subtype, PGS_ID), sorted by
                  subtype then descending run_count then ascending mean_univar_p
    presence    : DataFrame, binary matrix (index=PGS_ID, cols=run_labels + ['run_count','subtype'])
    """
    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_rows)
    subtypes = sorted(df['subtype'].unique())
    cons_parts = []
    pres_parts = []

    for subtype in subtypes:
        sub = df[df['subtype'] == subtype]
        pgs_ids = sorted(sub['PGS_ID'].unique())

        presence = pd.DataFrame(0, index=pgs_ids, columns=run_labels)
        for _, row in sub.iterrows():
            presence.loc[row['PGS_ID'], row['run_label']] = 1
        presence['run_count'] = presence[run_labels].sum(axis=1)
        presence['subtype']   = subtype
        pres_parts.append(presence.reset_index().rename(columns={'index': 'PGS_ID'}))

        rows = []
        for pgs in pgs_ids:
            pgs_rows = sub[sub['PGS_ID'] == pgs]
            run_count = len(pgs_rows)
            row = {
                'subtype':          subtype,
                'PGS_ID':           pgs,
                'run_count':        run_count,
                'n_runs_total':     len(run_labels),
                'select_fraction':  run_count / len(run_labels),
                'runs':             ','.join(sorted(pgs_rows['run_label'].tolist())),
                'mean_LASSO_coef':  pgs_rows['LASSO_coef'].mean(),
                'mean_LASSO_HR':    pgs_rows['LASSO_HR'].mean(),
                'sign_consistent':  int(pgs_rows['LASSO_coef'].apply(np.sign).nunique() == 1),
            }
            for rl in run_labels:
                rr = pgs_rows[pgs_rows['run_label'] == rl]
                row[f'coef_{rl}'] = rr['LASSO_coef'].iloc[0] if len(rr) else np.nan

            for col, out in [('univar_p',     'mean_univar_p'),
                              ('univar_HR',    'mean_univar_HR'),
                              ('univar_CI_lo', 'mean_univar_CI_lo'),
                              ('univar_CI_hi', 'mean_univar_CI_hi')]:
                if col in pgs_rows.columns:
                    row[out] = pgs_rows[col].mean()

            rows.append(row)

        cons = pd.DataFrame(rows).sort_values(
            ['run_count', 'mean_univar_p'] if 'mean_univar_p' in pd.DataFrame(rows).columns
            else ['run_count'],
            ascending=[False, True] if 'mean_univar_p' in pd.DataFrame(rows).columns
            else [False])
        cons_parts.append(cons)

    consistency = pd.concat(cons_parts, ignore_index=True) if cons_parts else pd.DataFrame()
    presence    = pd.concat(pres_parts, ignore_index=True) if pres_parts else pd.DataFrame()
    return consistency, presence


def _print_pgs_summary(consistency, min_fraction, n_runs_total, strategy_label):
    """Print a human-readable PGS consistency summary to stdout."""
    min_count = max(1, int(np.ceil(min_fraction * n_runs_total)))
    subtypes  = consistency['subtype'].unique() if not consistency.empty else []

    print(f"\n{'='*60}")
    print(f"PGS CONSISTENCY SUMMARY  [{strategy_label}]")
    print(f"  Reporting threshold : >= {min_fraction:.0%} of runs "
          f"(>= {min_count}/{n_runs_total})")
    print(f"{'='*60}")

    coef_cols = [c for c in consistency.columns if c.startswith('coef_')]

    for subtype in subtypes:
        sub = consistency[
            (consistency['subtype'] == subtype) &
            (consistency['run_count'] >= min_count)
        ]
        print(f"\n  Subtype: {subtype}")
        print(f"  PGS selected in >= {min_count}/{n_runs_total} runs: {len(sub)}")

        if sub.empty:
            print("    (none)")
            continue

        for _, row in sub.iterrows():
            frac_str = f"{int(row['run_count'])}/{n_runs_total} " \
                       f"({row['select_fraction']:.0%})"
            sign_str = "yes" if row['sign_consistent'] else "NO — mixed signs"
            print(f"\n    {row['PGS_ID']}")
            print(f"      Selected in    : {frac_str}  runs={row['runs']}")
            print(f"      Sign consistent: {sign_str}")
            print(f"      Mean LASSO HR  : {row['mean_LASSO_HR']:.4f}  "
                  f"(coef={row['mean_LASSO_coef']:.4f})")
            for cc in coef_cols:
                run_name = cc[len('coef_'):]
                val = row[cc]
                print(f"        {run_name:12s}: "
                      f"{'—' if np.isnan(val) else f'{val:.4f}'}")
            if 'mean_univar_p' in row and pd.notna(row.get('mean_univar_p')):
                print(f"      Mean univar p  : {row['mean_univar_p']:.4e}")
                print(f"      Mean univar HR : {row['mean_univar_HR']:.4f} "
                      f"[{row['mean_univar_CI_lo']:.4f}, "
                      f"{row['mean_univar_CI_hi']:.4f}]")


def summarize_loco_results(results, output_dir, min_fraction=0.25):
    """
    Build and save PGS consistency tables for a completed LOCO run.
    Reads per-fold summary_table.csv files from output_dir/loco_folds/.
    Called automatically at the end of a loco pipeline run.

    Parameters
    ----------
    results      : pipeline results dict (from main loop)
    output_dir   : pipeline OUTPUT_DIR
    min_fraction : fraction threshold for console reporting
    """
    print(f"\n{'='*60}")
    print("POST-RUN: LOCO PGS CONSISTENCY ANALYSIS")
    print(f"{'='*60}")

    loco_dir  = os.path.join(output_dir, 'loco_folds')
    all_rows  = []
    run_labels = []

    for cohort in sorted(os.listdir(loco_dir)):
        cohort_dir = os.path.join(loco_dir, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        run_labels.append(cohort)
        for subtype in sorted(os.listdir(cohort_dir)):
            csv_path = os.path.join(cohort_dir, subtype, 'summary_table.csv')
            if not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    entry = {
                        'run_label': cohort,
                        'subtype':   subtype,
                        'PGS_ID':    row['PGS_ID'],
                        'LASSO_coef': row['LASSO_coef'],
                        'LASSO_HR':   row['LASSO_HR'],
                    }
                    for col in ('univar_p', 'univar_HR', 'univar_CI_lo', 'univar_CI_hi'):
                        if col in df.columns:
                            entry[col] = row[col]
                    all_rows.append(entry)
            except Exception as e:
                print(f"  WARNING: could not read {csv_path}: {e}")

    if not all_rows:
        print("  No summary tables found — skipping consistency analysis.")
        return

    n_runs = len(run_labels)
    print(f"  Folds found: {run_labels}  (n={n_runs})")

    consistency, presence = _build_pgs_table(all_rows, run_labels)

    out_dir   = loco_dir
    cons_path = os.path.join(out_dir, 'pgs_consistency.csv')
    pres_path = os.path.join(out_dir, 'pgs_presence_matrix.csv')
    consistency.to_csv(cons_path, index=False)
    presence.to_csv(pres_path,    index=False)
    print(f"  Saved: {cons_path}")
    print(f"  Saved: {pres_path}")

    _print_pgs_summary(consistency, min_fraction, n_runs, 'LOCO')


def summarize_random_split_results(results, output_dir, n_splits, min_fraction=0.25):
    """
    Build and save PGS consistency tables for a completed random_split run.
    Reads per-split summary_table.csv files from output_dir/random_splits/.
    Called automatically at the end of a random_split pipeline run.

    Parameters
    ----------
    results      : pipeline results dict (from main loop)
    output_dir   : pipeline OUTPUT_DIR
    n_splits     : total number of splits attempted (for denominator)
    min_fraction : fraction threshold for console reporting
    """
    print(f"\n{'='*60}")
    print("POST-RUN: RANDOM SPLIT PGS CONSISTENCY ANALYSIS")
    print(f"{'='*60}")

    splits_dir = os.path.join(output_dir, 'random_splits')
    all_rows   = []
    run_labels = []

    for split in sorted(os.listdir(splits_dir)):
        split_dir = os.path.join(splits_dir, split)
        if not os.path.isdir(split_dir):
            continue
        # Collect run labels from subtype subdirs
        has_tables = False
        for subtype in sorted(os.listdir(split_dir)):
            csv_path = os.path.join(split_dir, subtype, 'summary_table.csv')
            if not os.path.isfile(csv_path):
                continue
            has_tables = True
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    entry = {
                        'run_label':  split,
                        'subtype':    subtype,
                        'PGS_ID':     row['PGS_ID'],
                        'LASSO_coef': row['LASSO_coef'],
                        'LASSO_HR':   row['LASSO_HR'],
                    }
                    for col in ('univar_p', 'univar_HR', 'univar_CI_lo', 'univar_CI_hi'):
                        if col in df.columns:
                            entry[col] = row[col]
                    all_rows.append(entry)
            except Exception as e:
                print(f"  WARNING: could not read {csv_path}: {e}")
        if has_tables:
            run_labels.append(split)

    # Use n_splits as the denominator even if some splits produced no output
    # (failed splits should still count against the selection fraction)
    all_split_labels = [f'split_{i+1:02d}' for i in range(n_splits)]

    if not all_rows:
        print("  No summary tables found — skipping consistency analysis.")
        return

    n_runs = n_splits
    print(f"  Splits with output : {len(run_labels)}/{n_runs}")

    consistency, presence = _build_pgs_table(all_rows, all_split_labels)

    out_dir   = splits_dir
    cons_path = os.path.join(out_dir, 'pgs_consistency.csv')
    pres_path = os.path.join(out_dir, 'pgs_presence_matrix.csv')
    consistency.to_csv(cons_path, index=False)
    presence.to_csv(pres_path,    index=False)
    print(f"  Saved: {cons_path}")
    print(f"  Saved: {pres_path}")

    _print_pgs_summary(consistency, min_fraction, n_runs,
                       f'random_split  N={n_runs}, train={TRAIN_FRACTION:.0%}')


def run_smoke_test():
    """Run a quick smoke test with subsampled data."""
    print("=== SMOKE TEST ===")
    # Minimal check that pipeline functions work
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'survdays': np.random.exponential(1000, n),
        'vstatus': np.random.binomial(1, 0.7, n).astype(float),
        'age': np.random.normal(55, 10, n),
        'sex': np.random.binomial(1, 0.5, n).astype(float),
        'PGS_test1': np.random.normal(0, 1, n),
        'PGS_test2': np.random.normal(0, 1, n),
    })

    # Test score test
    st = compute_score_tests(df, ['PGS_test1', 'PGS_test2'], ['age', 'sex'])
    assert len(st) == 2, "Score test failed"

    # Test full Cox
    fc = fit_full_cox_batch(df, ['PGS_test1', 'PGS_test2'], ['age', 'sex'])
    assert len(fc) == 2, "Full Cox failed"

    # Test CoxNet
    X = df[['PGS_test1', 'PGS_test2', 'age', 'sex']].values
    y = np.array([(bool(e), t) for e, t in zip(df['vstatus'], df['survdays'])],
                  dtype=[('event', bool), ('time', float)])
    cn = CoxnetSurvivalAnalysis(l1_ratio=1.0, n_alphas=10)
    cn.fit(X, y)
    assert cn.coef_.shape[0] == 4, "CoxNet failed"

    print("=== SMOKE TEST PASSED ===")
    return True


def main():
    args = parse_args()

    # Re-import all config globals that parse_args() may have mutated.
    from config import (DATA_DIR, OUTPUT_DIR, N_MODELS, N_JOBS,
                        DEBUG, MODELS_FILE, CV_STRATEGY, N_SPLITS, TRAIN_FRACTION,
                        MIN_REPORT_FRACTION)

    if args.test:
        success = run_smoke_test()
        sys.exit(0 if success else 1)

    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load cohort data
    # ------------------------------------------------------------------
    if CV_STRATEGY in ('loco', 'random_split'):
        label = 'LOCO' if CV_STRATEGY == 'loco' else \
                f'random_split (N={N_SPLITS}, train={TRAIN_FRACTION:.0%})'
        print(f"CV strategy: {label} over {ALL_COHORTS}")
        cohort_dfs = {c: load_cohort_data(c, DATA_DIR) for c in ALL_COHORTS}
    else:
        print(f"CV strategy: fixed  (train={DISCOVERY_COHORTS}, val={VALIDATION_COHORT})")
        print("Loading discovery cohorts...")
        dfs = [load_cohort_data(c, DATA_DIR) for c in DISCOVERY_COHORTS]
        pooled = pd.concat(dfs, ignore_index=True)
        print("Loading validation cohort...")
        cidr = load_cohort_data(VALIDATION_COHORT, DATA_DIR)

    # ------------------------------------------------------------------
    # Identify PGS columns
    # ------------------------------------------------------------------
    meta = ['IID', 'dataset', 'sample', 'cohort', 'source', 'age', 'sex',
            'case', 'grade', 'idh', 'pq', 'tert', 'rad', 'chemo', 'treated',
            'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
            'survdays', 'vstatus', 'grade_numeric']

    if CV_STRATEGY in ('loco', 'random_split'):
        ref_df = pd.concat(list(cohort_dfs.values()), ignore_index=True)
    else:
        ref_df = pooled

    pgs_cols = [c for c in ref_df.columns if c not in meta]

    if N_MODELS is not None:
        pgs_cols = list(np.random.choice(pgs_cols, min(N_MODELS, len(pgs_cols)), replace=False))
        print(f"Subsampled to {len(pgs_cols)} PGS models")

    if MODELS_FILE is not None:
        with open(MODELS_FILE) as fh:
            allowed = {line.strip() for line in fh if line.strip() and not line.startswith('#')}
        before = len(pgs_cols)
        pgs_cols = [c for c in pgs_cols if c in allowed]
        print(f"Model allowlist '{MODELS_FILE}': kept {len(pgs_cols)} / {before} models "
              f"({before - len(pgs_cols)} dropped)")
        if len(pgs_cols) == 0:
            print("ERROR: No PGS models remain after applying allowlist.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Determine which subtypes to run
    # ------------------------------------------------------------------
    if args.subtype:
        subtypes_to_run = {args.subtype: SUBTYPES[args.subtype]}
    else:
        subtypes_to_run = SUBTYPES

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    results = {}

    if CV_STRATEGY == 'loco':
        # LOCO: for each held-out cohort, train on the rest, validate on it.
        for name, criteria in subtypes_to_run.items():
            fold_results = {}
            for val_cohort in ALL_COHORTS:
                train_cohorts = [c for c in ALL_COHORTS if c != val_cohort]
                train_df = pd.concat([cohort_dfs[c] for c in train_cohorts],
                                     ignore_index=True)
                val_df   = cohort_dfs[val_cohort]
                fold_out = os.path.join(OUTPUT_DIR, 'loco_folds', val_cohort)
                os.makedirs(fold_out, exist_ok=True)
                try:
                    fold_results[val_cohort] = process_subtype(
                        name, criteria, train_df, val_df, val_cohort,
                        pgs_cols, fold_out, train_cohorts=train_cohorts)
                except Exception as e:
                    print(f"ERROR in {name} / val={val_cohort}: {e}")
                    traceback.print_exc()
                    fold_results[val_cohort] = {'status': 'error', 'error': str(e)}

            val_cis = [v['val_cindex'] for v in fold_results.values()
                       if v.get('status') == 'complete' and v.get('val_cindex') is not None]
            results[name] = {
                'status': 'complete' if val_cis else 'all_folds_failed',
                'cv_strategy': 'loco',
                'folds': fold_results,
                'mean_val_cindex': float(np.mean(val_cis)) if val_cis else None,
                'std_val_cindex':  float(np.std(val_cis))  if val_cis else None,
                'n_folds_complete': len(val_cis),
            }
            if val_cis:
                print(f"\n  {name} LOCO summary: mean val C-index = "
                      f"{results[name]['mean_val_cindex']:.4f} "
                      f"± {results[name]['std_val_cindex']:.4f} "
                      f"({len(val_cis)}/{len(ALL_COHORTS)} folds complete)")

    elif CV_STRATEGY == 'random_split':
        # Random split: pool all cohorts, perform N_SPLITS independent
        # stratified random splits, average results.
        all_data = pd.concat(list(cohort_dfs.values()), ignore_index=True)
        split_out = os.path.join(OUTPUT_DIR, 'random_splits')
        os.makedirs(split_out, exist_ok=True)

        # Use a seeded RNG so splits are reproducible but distinct
        rng = np.random.RandomState(RANDOM_SEED)
        split_seeds = rng.randint(0, 2**31, size=N_SPLITS).tolist()

        for name, criteria in subtypes_to_run.items():
            print(f"\n{'='*60}")
            print(f"SUBTYPE: {name}  [random_split: {N_SPLITS} splits, "
                  f"train={TRAIN_FRACTION:.0%}]")
            print(f"{'='*60}")

            # Subset to this subtype first so split proportions apply to
            # the relevant population, not the full pooled dataset
            subtype_data = subset_by_subtype(all_data, name, criteria)
            if subtype_data is None:
                results[name] = {'status': 'insufficient_samples',
                                 'cv_strategy': 'random_split'}
                continue

            n_total = len(subtype_data)
            split_results = {}

            for i, seed in enumerate(split_seeds):
                split_rng = np.random.RandomState(seed)
                idx = subtype_data.index.to_numpy().copy()
                split_rng.shuffle(idx)
                n_train = int(np.floor(n_total * TRAIN_FRACTION))
                train_idx = idx[:n_train]
                val_idx   = idx[n_train:]

                train_df = all_data.loc[train_idx]
                val_df   = all_data.loc[val_idx]

                # Use all cohorts present in the training split for direction check
                train_cohorts_present = list(train_df['cohort'].unique())

                split_label = f'split_{i+1:02d}'
                fold_out    = os.path.join(split_out, split_label)
                os.makedirs(fold_out, exist_ok=True)

                print(f"\n--- Split {i+1}/{N_SPLITS} "
                      f"(train n={len(train_df)}, val n={len(val_df)}) ---")
                try:
                    split_results[split_label] = process_subtype(
                        name, criteria, train_df, val_df, split_label,
                        pgs_cols, fold_out,
                        train_cohorts=train_cohorts_present)
                except Exception as e:
                    print(f"ERROR in {name} / {split_label}: {e}")
                    traceback.print_exc()
                    split_results[split_label] = {'status': 'error', 'error': str(e)}

            val_cis = [v['val_cindex'] for v in split_results.values()
                       if v.get('status') == 'complete' and v.get('val_cindex') is not None]
            results[name] = {
                'status': 'complete' if val_cis else 'all_splits_failed',
                'cv_strategy': 'random_split',
                'n_splits': N_SPLITS,
                'train_fraction': TRAIN_FRACTION,
                'splits': split_results,
                'mean_val_cindex': float(np.mean(val_cis)) if val_cis else None,
                'std_val_cindex':  float(np.std(val_cis))  if val_cis else None,
                'n_splits_complete': len(val_cis),
            }
            if val_cis:
                print(f"\n  {name} random_split summary: "
                      f"mean val C-index = {results[name]['mean_val_cindex']:.4f} "
                      f"± {results[name]['std_val_cindex']:.4f} "
                      f"({len(val_cis)}/{N_SPLITS} splits complete)")

    else:
        # Fixed: original single train/val split
        for name, criteria in subtypes_to_run.items():
            try:
                results[name] = process_subtype(
                    name, criteria, pooled, cidr, VALIDATION_COHORT,
                    pgs_cols, OUTPUT_DIR)
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                traceback.print_exc()
                results[name] = {'status': 'error', 'error': str(e)}

    # ------------------------------------------------------------------
    # Post-run PGS consistency summary
    # ------------------------------------------------------------------
    if CV_STRATEGY == 'loco':
        summarize_loco_results(results, OUTPUT_DIR,
                               min_fraction=MIN_REPORT_FRACTION)
    elif CV_STRATEGY == 'random_split':
        summarize_random_split_results(results, OUTPUT_DIR, N_SPLITS,
                                       min_fraction=MIN_REPORT_FRACTION)

    # ------------------------------------------------------------------
    # Save results summary
    # ------------------------------------------------------------------
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== PIPELINE COMPLETE ===")
    for name, res in results.items():
        if CV_STRATEGY in ('loco', 'random_split'):
            mean = res.get('mean_val_cindex')
            std  = res.get('std_val_cindex')
            summary = (f"mean val C-index={mean:.4f} ± {std:.4f}" if mean is not None
                       else res.get('status', 'unknown'))
        else:
            summary = res.get('status', 'unknown')
        print(f"  {name}: {summary}")


if __name__ == '__main__':
    main()
