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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Import config
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
    """Check and filter covariates with zero/near-zero variance."""
    surviving = []
    for cov in covariates:
        col = 'grade_numeric' if cov == 'grade' else cov
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if vals.dtype == 'object':
            if vals.nunique() <= 1:
                dprint(f"  Dropping {cov} ({label}): single value")
                continue
        else:
            if vals.var() < 1e-10:
                dprint(f"  Dropping {cov} ({label}): near-zero variance")
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


def fit_full_cox_batch(df, pgs_list, cov_cols):
    """Fit full univariate Cox PH for a list of PGS models."""
    results = []
    for pgs in pgs_list:
        try:
            cols = [pgs] + cov_cols + ['survdays', 'vstatus']
            sub = df[cols].dropna()
            if len(sub) < 10 or sub['vstatus'].sum() < 5 or sub[pgs].var() < 1e-10:
                continue
            cph = CoxPHFitter()
            cph.fit(sub, duration_col='survdays', event_col='vstatus', show_progress=False)
            s = cph.summary.loc[pgs]
            results.append({
                'pgs': pgs, 'coef': s['coef'], 'hr': s['exp(coef)'],
                'se': s['se(coef)'], 'z': s['z'], 'p': s['p'],
                'ci_lower': s['exp(coef) lower 95%'], 'ci_upper': s['exp(coef) upper 95%'],
                'n': len(sub), 'events': int(sub['vstatus'].sum()),
            })
        except Exception:
            pass
    return pd.DataFrame(results)


def process_subtype(subtype_name, criteria, discovery_pooled, cidr_df, pgs_cols, output_dir):
    """Process a single glioma subtype through the full pipeline."""
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
        for cohort in DISCOVERY_COHORTS:
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
        consistent = []

        for _, row in sig_pooled.iterrows():
            pgs = row['pgs']
            signs = {}
            for cohort in active_cohorts:
                try:
                    cdf = df_sub[df_sub['cohort'] == cohort]
                    cv = [c for c in per_cohort_cov[cohort] if cdf[c].var() > 1e-10]
                    cols = [pgs] + cv + ['survdays', 'vstatus']
                    sub = cdf[cols].dropna()
                    if len(sub) < 10 or sub['vstatus'].sum() < 5 or sub[pgs].var() < 1e-10:
                        continue
                    cph = CoxPHFitter()
                    cph.fit(sub, duration_col='survdays', event_col='vstatus', show_progress=False)
                    signs[cohort] = np.sign(cph.summary.loc[pgs, 'coef'])
                except:
                    continue
            if len(signs) >= MIN_COHORTS_FOR_DIRECTION and len(set(signs.values())) == 1:
                consistent.append(pgs)

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

    # CV
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = np.full((CV_FOLDS, len(alpha_path)), np.nan)

    for fi, (tri, vai) in enumerate(kf.split(X)):
        try:
            cv_m = CoxnetSurvivalAnalysis(l1_ratio=1.0, penalty_factor=penalty_factor,
                                           alphas=alpha_path, fit_baseline_model=True)
            cv_m.fit(X[tri], y[tri])
            for ai in range(len(alpha_path)):
                risk = X[vai] @ cv_m.coef_[:, ai]
                try:
                    cv_scores[fi, ai] = concordance_index_censored(y[vai]['event'], y[vai]['time'], risk)[0]
                except:
                    pass
        except:
            pass

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

    # Step 7: CIDR validation
    cidr_sub = subset_by_subtype(cidr_df, subtype_name, criteria)
    if cidr_sub is None:
        return {'status': 'no_cidr_validation', 'train_cindex': train_ci,
                'n_nonzero_pgs': int(nonzero.sum()), 'best_cv_cindex': mean_cv[selected_idx]}

    cidr_sub = cidr_sub.copy()
    if 'source_Mayo' in cov_col_list:
        cidr_sub['source_Mayo'] = (cidr_sub['source'] == 'Mayo').astype(float)
    for cd in cohort_dummy_cols:
        cidr_sub[cd] = 0.0

    cidr_model = cidr_sub[feature_cols + ['survdays', 'vstatus']].dropna()
    X_c = cidr_model[feature_cols].values.astype(np.float64)
    y_c = np.array([(bool(e), t) for e, t in zip(cidr_model['vstatus'], cidr_model['survdays'])],
                    dtype=[('event', bool), ('time', float)])

    cidr_risk = X_c @ final_coef
    val_ci = concordance_index_censored(y_c['event'], y_c['time'], cidr_risk)[0]

    # Bootstrap CI
    rng = np.random.RandomState(RANDOM_SEED)
    boot = [concordance_index_censored(y_c['event'][idx], y_c['time'][idx], cidr_risk[idx])[0]
            for idx in (rng.choice(len(cidr_risk), len(cidr_risk), True) for _ in range(1000))
            if True]  # simplified
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    print(f"  Validation C-index: {val_ci:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    # KM plot
    n_groups = 4 if len(cidr_model) >= 80 else 3
    gl = 'Quartile' if n_groups == 4 else 'Tertile'
    cidr_model = cidr_model.copy()
    cidr_model['risk'] = cidr_risk
    cidr_model['rg'] = pd.qcut(cidr_risk, n_groups, labels=False, duplicates='drop') + 1

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    for g in sorted(cidr_model['rg'].unique()):
        gd = cidr_model[cidr_model['rg'] == g]
        kmf = KaplanMeierFitter()
        kmf.fit(gd['survdays'], gd['vstatus'], label=f'{gl} {g} (n={len(gd)})')
        kmf.plot_survival_function(ax=ax, color=colors[g-1], ci_show=True)

    lr = logrank_test(cidr_model[cidr_model['rg']==cidr_model['rg'].min()]['survdays'],
                      cidr_model[cidr_model['rg']==cidr_model['rg'].max()]['survdays'],
                      cidr_model[cidr_model['rg']==cidr_model['rg'].min()]['vstatus'],
                      cidr_model[cidr_model['rg']==cidr_model['rg'].max()]['vstatus'])

    ax.set_title(f'{subtype_name}: KM ({gl}s)\nC-index={val_ci:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], LR p={lr.p_value:.2e}')
    ax.set_xlabel('Time (days)'); ax.set_ylabel('Survival Probability')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(subtype_dir, 'kaplan_meier.png'), dpi=150)
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
        'status': 'complete', 'n_discovery': len(model_df), 'n_events': n_events,
        'n_pgs_candidates': len(final_pgs), 'n_nonzero_pgs': int(nonzero.sum()),
        'best_alpha': best_alpha, 'best_cv_cindex': float(mean_cv[selected_idx]),
        'train_cindex': float(train_ci), 'val_cindex': float(val_ci),
        'val_ci': (float(ci_lo), float(ci_hi)), 'logrank_p': float(lr.p_value),
    }


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

    if args.test:
        success = run_smoke_test()
        sys.exit(0 if success else 1)

    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load discovery data
    print("Loading discovery cohorts...")
    dfs = [load_cohort_data(c, DATA_DIR) for c in DISCOVERY_COHORTS]
    pooled = pd.concat(dfs, ignore_index=True)

    # Load validation data
    print("Loading validation cohort...")
    cidr = load_cohort_data(VALIDATION_COHORT, DATA_DIR)

    # Identify PGS columns
    meta = ['IID', 'dataset', 'sample', 'cohort', 'source', 'age', 'sex',
            'case', 'grade', 'idh', 'pq', 'tert', 'rad', 'chemo', 'treated',
            'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
            'survdays', 'vstatus', 'grade_numeric']
    pgs_cols = [c for c in pooled.columns if c not in meta]

    if N_MODELS is not None:
        pgs_cols = list(np.random.choice(pgs_cols, min(N_MODELS, len(pgs_cols)), replace=False))
        print(f"Subsampled to {len(pgs_cols)} PGS models")

    # Apply model allowlist if provided (--models flag / MODELS_FILE config)
    if MODELS_FILE is not None:
        with open(MODELS_FILE) as fh:
            allowed = {line.strip() for line in fh if line.strip() and not line.startswith('#')}
        before = len(pgs_cols)
        pgs_cols = [c for c in pgs_cols if c in allowed]
        print(f"Model allowlist '{MODELS_FILE}': kept {len(pgs_cols)} / {before} models "
              f"({before - len(pgs_cols)} dropped)")
        if len(pgs_cols) == 0:
            print("ERROR: No PGS models remain after applying allowlist. "
                  "Check that model IDs in the file match column names in the score files.")
            sys.exit(1)

    # Determine which subtypes to run
    if args.subtype:
        subtypes_to_run = {args.subtype: SUBTYPES[args.subtype]}
    else:
        subtypes_to_run = SUBTYPES

    # Process each subtype
    results = {}
    for name, criteria in subtypes_to_run.items():
        try:
            results[name] = process_subtype(name, criteria, pooled, cidr, pgs_cols, OUTPUT_DIR)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            traceback.print_exc()
            results[name] = {'status': 'error', 'error': str(e)}

    # Save results summary
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== PIPELINE COMPLETE ===")
    for name, res in results.items():
        print(f"  {name}: {res.get('status', 'unknown')}")


if __name__ == '__main__':
    main()
