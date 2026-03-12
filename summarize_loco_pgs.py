#!/usr/bin/env python3
"""
summarize_loco_pgs.py

Post-processing script for LOCO LASSO Cox pipeline results.
Reads per-fold summary_table.csv files, identifies PGS models selected
across multiple folds, and reports:

  1. pgs_consistency.csv  — one row per (subtype, PGS_ID) with per-fold
                            LASSO coefficients, fold count, and mean
                            univariate HR/p across folds where selected.
  2. pgs_presence_matrix.csv — binary presence/absence matrix
                               (rows=PGS_ID, cols=folds).
  3. Console summary of models selected in >= MIN_FOLDS folds.

Usage:
    python3 summarize_loco_pgs.py --results-dir results --min-folds 2
    python3 summarize_loco_pgs.py --results-dir results --min-folds 1 --subtype idh_wildtype
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config defaults (can be overridden via CLI)
# ---------------------------------------------------------------------------
RESULTS_DIR = 'results'
MIN_FOLDS   = 2          # minimum fold count to include in console summary
LOCO_SUBDIR = 'loco_folds'


def find_summary_tables(results_dir, subtype_filter=None):
    """
    Walk results/loco_folds/{cohort}/{subtype}/summary_table.csv
    and return a list of (cohort, subtype, path) tuples.
    """
    loco_dir = os.path.join(results_dir, LOCO_SUBDIR)
    if not os.path.isdir(loco_dir):
        print(f"ERROR: LOCO folds directory not found: {loco_dir}")
        sys.exit(1)

    found = []
    for cohort in sorted(os.listdir(loco_dir)):
        cohort_dir = os.path.join(loco_dir, cohort)
        if not os.path.isdir(cohort_dir):
            continue
        for subtype in sorted(os.listdir(cohort_dir)):
            if subtype_filter and subtype != subtype_filter:
                continue
            csv_path = os.path.join(cohort_dir, subtype, 'summary_table.csv')
            if os.path.isfile(csv_path):
                found.append((cohort, subtype, csv_path))

    return found


def load_fold_tables(entries):
    """
    Load all summary_table.csv files and return a combined DataFrame
    with added 'fold' (held-out cohort) and 'subtype' columns.
    """
    dfs = []
    for cohort, subtype, path in entries:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            df['fold']    = cohort
            df['subtype'] = subtype
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")

    if not dfs:
        print("ERROR: No summary tables could be loaded.")
        sys.exit(1)

    return pd.concat(dfs, ignore_index=True)


def build_consistency_table(combined, subtypes):
    """
    For each subtype, build:
      - consistency_df : one row per (subtype, PGS_ID) with per-fold coefs,
                         fold_count, mean univar stats
      - presence_df    : binary matrix (index=PGS_ID, cols=fold names)
    """
    all_consistency = []
    all_presence    = []

    for subtype in subtypes:
        sub = combined[combined['subtype'] == subtype]
        folds = sorted(sub['fold'].unique())
        pgs_ids = sorted(sub['PGS_ID'].unique())

        if not pgs_ids:
            continue

        # Presence/absence matrix
        presence = pd.DataFrame(0, index=pgs_ids, columns=folds)
        coef_cols = {}
        for fold in folds:
            fold_df = sub[sub['fold'] == fold].set_index('PGS_ID')
            for pgs in pgs_ids:
                if pgs in fold_df.index:
                    presence.loc[pgs, fold] = 1
            coef_cols[fold] = fold_df['LASSO_coef'].reindex(pgs_ids)

        presence['fold_count'] = presence[folds].sum(axis=1)
        presence['subtype']    = subtype
        presence = presence.reset_index().rename(columns={'index': 'PGS_ID'})
        all_presence.append(presence)

        # Consistency table
        rows = []
        for pgs in pgs_ids:
            pgs_rows = sub[sub['PGS_ID'] == pgs]
            fold_count = len(pgs_rows)

            row = {
                'subtype':    subtype,
                'PGS_ID':     pgs,
                'fold_count': fold_count,
                'folds':      ','.join(sorted(pgs_rows['fold'].tolist())),
                'mean_LASSO_coef': pgs_rows['LASSO_coef'].mean(),
                'mean_LASSO_HR':   pgs_rows['LASSO_HR'].mean(),
                'sign_consistent': int(pgs_rows['LASSO_coef'].apply(np.sign).nunique() == 1),
            }

            # Per-fold LASSO coef columns
            for fold in folds:
                fold_row = pgs_rows[pgs_rows['fold'] == fold]
                row[f'LASSO_coef_{fold}'] = (fold_row['LASSO_coef'].iloc[0]
                                              if len(fold_row) else np.nan)

            # Mean univar stats across folds where available
            for col, out_col in [('univar_p',      'mean_univar_p'),
                                  ('univar_HR',     'mean_univar_HR'),
                                  ('univar_CI_lo',  'mean_univar_CI_lo'),
                                  ('univar_CI_hi',  'mean_univar_CI_hi')]:
                if col in pgs_rows.columns:
                    row[out_col] = pgs_rows[col].mean()

            rows.append(row)

        consistency_df = pd.DataFrame(rows)
        consistency_df = consistency_df.sort_values(
            ['fold_count', 'mean_univar_p'], ascending=[False, True])
        all_consistency.append(consistency_df)

    consistency = pd.concat(all_consistency, ignore_index=True) if all_consistency else pd.DataFrame()
    presence    = pd.concat(all_presence,    ignore_index=True) if all_presence    else pd.DataFrame()
    return consistency, presence


def print_summary(consistency, min_folds, folds_available):
    """Print a readable summary to stdout."""
    n_total_folds = len(folds_available)
    subtypes = consistency['subtype'].unique()

    for subtype in subtypes:
        sub = consistency[
            (consistency['subtype'] == subtype) &
            (consistency['fold_count'] >= min_folds)
        ]
        print(f"\n{'='*60}")
        print(f"SUBTYPE: {subtype}")
        print(f"{'='*60}")
        print(f"  Folds available: {folds_available}")
        print(f"  PGS selected in >= {min_folds}/{n_total_folds} folds: {len(sub)}")

        if sub.empty:
            print("  (none)")
            continue

        # Determine coef columns present
        coef_cols = [c for c in sub.columns if c.startswith('LASSO_coef_')]

        for _, row in sub.iterrows():
            print(f"\n  {row['PGS_ID']}")
            print(f"    Folds selected : {row['folds']}  ({int(row['fold_count'])}/{n_total_folds})")
            print(f"    Sign consistent: {'yes' if row['sign_consistent'] else 'NO'}")
            print(f"    Mean LASSO HR  : {row['mean_LASSO_HR']:.4f}  "
                  f"(coef={row['mean_LASSO_coef']:.4f})")
            for cc in coef_cols:
                fold_name = cc.replace('LASSO_coef_', '')
                val = row[cc]
                val_str = f"{val:.4f}" if not np.isnan(val) else "—"
                print(f"      {fold_name:10s}: {val_str}")
            if 'mean_univar_p' in row and not np.isnan(row['mean_univar_p']):
                print(f"    Mean univar p  : {row['mean_univar_p']:.4e}")
                print(f"    Mean univar HR : {row['mean_univar_HR']:.4f} "
                      f"[{row['mean_univar_CI_lo']:.4f}, {row['mean_univar_CI_hi']:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize PGS consistency across LOCO folds')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Pipeline results directory (default: results)')
    parser.add_argument('--min-folds', type=int, default=MIN_FOLDS,
                        help='Minimum fold count for console summary (default: 2)')
    parser.add_argument('--subtype', type=str, default=None,
                        help='Restrict to a single subtype (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for CSV files '
                             '(default: <results-dir>/loco_folds)')
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, LOCO_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    # Discover and load tables
    entries = find_summary_tables(args.results_dir, subtype_filter=args.subtype)
    if not entries:
        print("No summary_table.csv files found. "
              "Run the pipeline with CV_STRATEGY='loco' first.")
        sys.exit(1)

    print(f"Found {len(entries)} summary table(s) across "
          f"{len(set(e[0] for e in entries))} fold(s) and "
          f"{len(set(e[1] for e in entries))} subtype(s).")

    combined = load_fold_tables(entries)
    subtypes = sorted(combined['subtype'].unique())
    folds    = sorted(combined['fold'].unique())

    # Build tables
    consistency, presence = build_consistency_table(combined, subtypes)

    # Save CSVs
    cons_path = os.path.join(out_dir, 'pgs_consistency.csv')
    pres_path = os.path.join(out_dir, 'pgs_presence_matrix.csv')
    consistency.to_csv(cons_path, index=False)
    presence.to_csv(pres_path,    index=False)
    print(f"\nSaved: {cons_path}")
    print(f"Saved: {pres_path}")

    # Console summary
    print_summary(consistency, args.min_folds, folds)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == '__main__':
    main()
