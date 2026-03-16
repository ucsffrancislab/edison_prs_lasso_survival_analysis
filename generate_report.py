#!/usr/bin/env python3
"""
generate_report.py

Generates a self-contained HTML report from PRS LASSO Cox pipeline results.
Reads results_summary.json and per-split/fold PNG files, embeds everything
as base64, and writes a single portable HTML file.

Supports both 'random_split' and 'loco' CV strategies.

Usage:
    python3 generate_report.py --results-dir results
    python3 generate_report.py --results-dir results --output report.html
    python3 generate_report.py --results-dir run8-random_test
"""

import os
import sys
import json
import base64
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — summary figures will be skipped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def b64_png(path):
    """Return a base64-encoded data URI for a PNG file, or None if missing."""
    if path and os.path.isfile(path):
        with open(path, 'rb') as f:
            return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
    return None


def fmt(v, decimals=4):
    """Format a float or return '—' for None/nan."""
    if v is None:
        return '—'
    try:
        if np.isnan(float(v)):
            return '—'
        return f'{float(v):.{decimals}f}'
    except (TypeError, ValueError):
        return str(v)


def delta_str(pgs_val, base_val):
    """Return a coloured delta string (PGS − baseline)."""
    if pgs_val is None or base_val is None:
        return ''
    try:
        d = float(pgs_val) - float(base_val)
        colour = '#2a9d5c' if d >= 0 else '#e05252'
        sign = '+' if d >= 0 else ''
        return f'<span style="color:{colour};font-size:0.85em">{sign}{d:.4f}</span>'
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Summary figures (generated fresh, not loaded from disk)
# ---------------------------------------------------------------------------

def make_cindex_distribution_fig(subtype, runs_data, strategy):
    """
    Violin + strip plot of val C-index per model (PGS, baselines) across splits.
    Returns base64 PNG string.
    """
    if not HAS_MPL:
        return None

    model_keys = ['pgs_model', 'clinical_only', 'clinical_pcs', 'full_covariates']
    model_labels = {
        'pgs_model':       'PGS Model',
        'clinical_only':   'Clinical Only',
        'clinical_pcs':    'Clinical + PCs',
        'full_covariates': 'Full Covariates',
    }
    colors = {
        'pgs_model':       '#1a5276',
        'clinical_only':   '#a9cce3',
        'clinical_pcs':    '#7fb3d3',
        'full_covariates': '#2e86c1',
    }

    data = {k: [] for k in model_keys}
    for run_res in runs_data:
        if run_res.get('status') != 'complete':
            continue
        vc = run_res.get('val_cindex')
        if vc is not None:
            data['pgs_model'].append(float(vc))
        bl = run_res.get('baselines', {})
        for bk in ('clinical_only', 'clinical_pcs', 'full_covariates'):
            bv = bl.get(bk, {}).get('val_cindex')
            if bv is not None:
                data[bk].append(float(bv))

    present = [k for k in model_keys if len(data[k]) >= 2]
    if not present:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.6), 5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    for i, k in enumerate(present):
        vals = data[k]
        vp = ax.violinplot(vals, positions=[i], widths=0.6,
                           showmeans=False, showmedians=True, showextrema=True)
        for pc in vp['bodies']:
            pc.set_facecolor(colors[k])
            pc.set_alpha(0.55)
        for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
            if part in vp:
                vp[part].set_color(colors[k])
                vp[part].set_linewidth(1.5)
        # strip
        jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=colors[k], s=18, alpha=0.7, zorder=3, edgecolors='white', linewidths=0.4)
        mean_v = np.mean(vals)
        ax.plot([i - 0.25, i + 0.25], [mean_v, mean_v],
                color='#222', linewidth=2, zorder=4)

    ax.axhline(0.5, color='#999', linestyle=':', linewidth=1, label='Chance (0.5)')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([model_labels[k] for k in present], fontsize=10)
    ax.set_ylabel('Validation C-index', fontsize=11)
    ax.set_title(f'{subtype.replace("_", " ").title()}: C-index Distribution Across {strategy.upper()} Runs',
                 fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    plt.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def make_mean_bar_fig(subtype, runs_data, strategy):
    """
    Grouped bar chart: mean ± SD val C-index for each model type.
    Returns base64 PNG string.
    """
    if not HAS_MPL:
        return None

    model_keys   = ['pgs_model', 'clinical_only', 'clinical_pcs', 'full_covariates']
    model_labels = ['PGS Model', 'Clinical Only', 'Clinical + PCs', 'Full Covariates']
    colors       = ['#1a5276', '#a9cce3', '#7fb3d3', '#2e86c1']

    means, sds = [], []
    for k in model_keys:
        if k == 'pgs_model':
            vals = [float(r['val_cindex']) for r in runs_data
                    if r.get('status') == 'complete' and r.get('val_cindex') is not None]
        else:
            vals = [float(r['baselines'][k]['val_cindex'])
                    for r in runs_data
                    if r.get('status') == 'complete'
                    and r.get('baselines', {}).get(k, {}).get('val_cindex') is not None]
        means.append(np.mean(vals) if vals else np.nan)
        sds.append(np.std(vals)   if vals else np.nan)

    present_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
    if not present_idx:
        return None

    fig, ax = plt.subplots(figsize=(max(5, len(present_idx) * 1.4), 4.5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    xs = range(len(present_idx))
    bars = ax.bar(xs,
                  [means[i] for i in present_idx],
                  yerr=[sds[i] for i in present_idx],
                  color=[colors[i] for i in present_idx],
                  width=0.55, capsize=5, error_kw={'linewidth': 1.5, 'ecolor': '#555'},
                  zorder=3)

    for bar, i in zip(bars, present_idx):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (sds[i] or 0) + 0.004,
                f'{means[i]:.3f}', ha='center', va='bottom', fontsize=9, color='#333')

    ax.axhline(0.5, color='#999', linestyle=':', linewidth=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([model_labels[i] for i in present_idx], fontsize=10)
    ax.set_ylabel('Mean Validation C-index ± SD', fontsize=11)
    ax.set_ylim(bottom=max(0, min(m for m in means if not np.isnan(m)) - 0.08))
    ax.set_title(f'{subtype.replace("_", " ").title()}: Mean Validation C-index',
                 fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    ax.yaxis.grid(True, color='#e0e0e0', zorder=0)
    plt.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


def make_delta_fig(subtype, runs_data, strategy):
    """
    Strip + box plot of (PGS − full_covariates) delta across splits.
    Highlights whether PGS adds value over the best covariate baseline.
    Returns base64 PNG string.
    """
    if not HAS_MPL:
        return None

    deltas = []
    for r in runs_data:
        if r.get('status') != 'complete':
            continue
        pgs_v = r.get('val_cindex')
        cov_v = r.get('baselines', {}).get('full_covariates', {}).get('val_cindex')
        if pgs_v is not None and cov_v is not None:
            deltas.append(float(pgs_v) - float(cov_v))

    if len(deltas) < 2:
        return None

    fig, ax = plt.subplots(figsize=(4, 4.5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    color = '#1a5276'
    bp = ax.boxplot(deltas, positions=[0], widths=0.4, patch_artist=True,
                    medianprops={'color': '#e05252', 'linewidth': 2},
                    boxprops={'facecolor': color, 'alpha': 0.3},
                    whiskerprops={'color': color},
                    capprops={'color': color},
                    flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5})

    jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(deltas))
    for d, j in zip(deltas, jitter):
        ax.scatter(j, d, color=color, s=22, alpha=0.7,
                   zorder=3, edgecolors='white', linewidths=0.4)

    ax.axhline(0, color='#e05252', linestyle='--', linewidth=1.2, zorder=2,
               label='No difference')
    mean_d = np.mean(deltas)
    ax.axhline(mean_d, color='#1a5276', linestyle='-', linewidth=1.5, zorder=2,
               label=f'Mean Δ={mean_d:+.4f}')

    ax.set_xticks([0])
    ax.set_xticklabels(['PGS − Full Covariates'], fontsize=10)
    ax.set_ylabel('ΔC-index', fontsize=11)
    ax.set_title(f'{subtype.replace("_", " ").title()}\nPGS Marginal Contribution',
                 fontsize=11, fontweight='bold', pad=8)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    ax.yaxis.grid(True, color='#e0e0e0', zorder=0)
    plt.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Per-run row builder
# ---------------------------------------------------------------------------

def run_row_html(run_label, run_res, run_dir, subtype):
    """Build HTML for a single split/fold row (collapsible)."""
    if run_res.get('status') != 'complete':
        status_badge = f'<span class="badge badge-fail">{run_res.get("status","error")}</span>'
        return f'''
        <div class="run-row failed">
          <div class="run-header">
            <span class="run-label">{run_label}</span>
            {status_badge}
          </div>
        </div>'''

    pgs_val = run_res.get('val_cindex')
    pgs_tr  = run_res.get('train_cindex')
    bl      = run_res.get('baselines', {})
    ci_lo, ci_hi = (run_res.get('val_ci') or [None, None])

    def bl_cell(key):
        v = bl.get(key, {}).get('val_cindex')
        return f'{fmt(v)} {delta_str(pgs_val, v)}'

    # Per-run figures
    km_path     = os.path.join(run_dir, subtype, f'kaplan_meier_{run_label}.png')
    forest_path = os.path.join(run_dir, subtype, 'forest_plot.png')
    lambda_path = os.path.join(run_dir, subtype, 'lambda_selection.png')

    km_img     = b64_png(km_path)
    forest_img = b64_png(forest_path)
    lambda_img = b64_png(lambda_path)

    imgs_html = ''
    if km_img or forest_img or lambda_img:
        parts = []
        for img, cap in [(km_img, 'Kaplan–Meier'),
                         (forest_img, 'Forest Plot'),
                         (lambda_img, 'Lambda Selection')]:
            if img:
                parts.append(f'''
                  <figure>
                    <img src="{img}" alt="{cap}">
                    <figcaption>{cap}</figcaption>
                  </figure>''')
        imgs_html = f'<div class="figure-row">{"".join(parts)}</div>'

    nonzero = run_res.get('n_nonzero_pgs', '—')
    candidates = run_res.get('n_pgs_candidates', '—')
    alpha = run_res.get('best_alpha')
    cv_ci = run_res.get('best_cv_cindex')
    lr_p  = run_res.get('logrank_p')
    n_disc = run_res.get('n_discovery', '—')
    n_ev   = run_res.get('n_events', '—')

    ci_str = f'[{fmt(ci_lo)}, {fmt(ci_hi)}]' if ci_lo is not None else ''

    return f'''
    <div class="run-row">
      <details>
        <summary class="run-header">
          <span class="run-label">{run_label}</span>
          <span class="run-stats">
            <span class="stat-chip pgs-chip">PGS val: {fmt(pgs_val)}</span>
            <span class="stat-chip cov-chip">Cov val: {fmt(bl.get("full_covariates",{}).get("val_cindex"))}</span>
            <span class="stat-chip delta-chip">Δ: {delta_str(pgs_val, bl.get("full_covariates",{}).get("val_cindex"))}</span>
            <span class="stat-chip">Non-zero PGS: {nonzero}</span>
          </span>
        </summary>
        <div class="run-body">
          <table class="stats-table">
            <thead><tr>
              <th>Metric</th><th>Train</th><th>Val</th>
            </tr></thead>
            <tbody>
              <tr><td>PGS Model</td><td>{fmt(pgs_tr)}</td>
                  <td><strong>{fmt(pgs_val)}</strong> <span class="ci-str">{ci_str}</span></td></tr>
              <tr><td>Clinical Only</td>
                  <td>{fmt(bl.get("clinical_only",{}).get("train_cindex"))}</td>
                  <td>{bl_cell("clinical_only")}</td></tr>
              <tr><td>Clinical + PCs</td>
                  <td>{fmt(bl.get("clinical_pcs",{}).get("train_cindex"))}</td>
                  <td>{bl_cell("clinical_pcs")}</td></tr>
              <tr><td>Full Covariates</td>
                  <td>{fmt(bl.get("full_covariates",{}).get("train_cindex"))}</td>
                  <td>{bl_cell("full_covariates")}</td></tr>
            </tbody>
          </table>
          <table class="stats-table meta-table">
            <tbody>
              <tr><td>Training n</td><td>{n_disc}</td>
                  <td>Events</td><td>{n_ev}</td></tr>
              <tr><td>PGS candidates</td><td>{candidates}</td>
                  <td>Non-zero PGS</td><td>{nonzero}</td></tr>
              <tr><td>Best α</td><td>{fmt(alpha, 6)}</td>
                  <td>CV C-index</td><td>{fmt(cv_ci)}</td></tr>
              <tr><td>Log-rank p</td><td colspan="3">{fmt(lr_p, 4) if lr_p else "—"}</td></tr>
            </tbody>
          </table>
          {imgs_html}
        </div>
      </details>
    </div>'''


# ---------------------------------------------------------------------------
# Subtype section builder
# ---------------------------------------------------------------------------

def subtype_section_html(subtype, subtype_res, results_dir, strategy):
    """Build the full HTML section for one subtype."""

    cv_strat = subtype_res.get('cv_strategy', strategy)

    # Collect per-run results and their directories
    if cv_strat == 'random_split':
        runs = subtype_res.get('splits', {})
        runs_dir_root = os.path.join(results_dir, 'random_splits')
        run_dir_fn = lambda lbl: os.path.join(runs_dir_root, lbl)
    else:
        runs = subtype_res.get('folds', {})
        runs_dir_root = os.path.join(results_dir, 'loco_folds')
        run_dir_fn = lambda lbl: os.path.join(runs_dir_root, lbl)

    runs_data = list(runs.values())
    n_complete = sum(1 for r in runs_data if r.get('status') == 'complete')
    n_total    = len(runs_data)
    mean_ci    = subtype_res.get('mean_val_cindex')
    std_ci     = subtype_res.get('std_val_cindex')

    status = subtype_res.get('status', 'unknown')
    status_class = 'complete' if status == 'complete' else 'failed'

    # Summary figures
    dist_img  = make_cindex_distribution_fig(subtype, runs_data, cv_strat)
    bar_img   = make_mean_bar_fig(subtype, runs_data, cv_strat)
    delta_img = make_delta_fig(subtype, runs_data, cv_strat)

    summary_figs = ''
    if dist_img or bar_img or delta_img:
        parts = []
        for img, cap in [(dist_img,  'C-index Distribution'),
                         (bar_img,   'Mean ± SD C-index'),
                         (delta_img, 'PGS Marginal Contribution (Δ vs Full Covariates)')]:
            if img:
                parts.append(f'''
                <figure class="summary-fig">
                  <img src="{img}" alt="{cap}">
                  <figcaption>{cap}</figcaption>
                </figure>''')
        summary_figs = f'<div class="summary-figs-row">{"".join(parts)}</div>'

    mean_str = f'{fmt(mean_ci)} ± {fmt(std_ci)}' if mean_ci is not None else '—'

    # Compute mean delta (PGS - full_covariates)
    deltas = []
    for r in runs_data:
        if r.get('status') != 'complete':
            continue
        pv = r.get('val_cindex')
        cv = r.get('baselines', {}).get('full_covariates', {}).get('val_cindex')
        if pv is not None and cv is not None:
            deltas.append(float(pv) - float(cv))
    delta_mean = np.mean(deltas) if deltas else None
    delta_str_val = (f'<span style="color:{"#2a9d5c" if delta_mean >= 0 else "#e05252"}">'
                     f'{delta_mean:+.4f}</span>') if delta_mean is not None else '—'

    # Per-run rows
    run_rows = '\n'.join(
        run_row_html(lbl, res, run_dir_fn(lbl), subtype)
        for lbl, res in runs.items()
    )

    subtype_display = subtype.replace('_', ' ').title()

    return f'''
    <section class="subtype-section {status_class}" id="{subtype}">
      <div class="subtype-header">
        <h2>{subtype_display}</h2>
        <div class="subtype-meta">
          <span class="badge badge-{"ok" if status == "complete" else "fail"}">{status}</span>
          <span class="meta-chip">Runs: {n_complete}/{n_total} complete</span>
          <span class="meta-chip">Mean val C-index: <strong>{mean_str}</strong></span>
          <span class="meta-chip">Mean ΔPGS vs Covariates: <strong>{delta_str_val}</strong></span>
        </div>
      </div>

      <div class="summary-block">
        <h3>Summary Across All Runs</h3>
        {summary_figs if summary_figs else '<p class="muted">Insufficient data for summary figures.</p>'}
      </div>

      <div class="runs-block">
        <h3>Per-Run Detail</h3>
        {run_rows if run_rows else '<p class="muted">No run data available.</p>'}
      </div>
    </section>'''


# ---------------------------------------------------------------------------
# Full HTML assembly
# ---------------------------------------------------------------------------

CSS = '''
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #f4f2ee;
  --surface:   #ffffff;
  --border:    #ddd8cf;
  --text:      #1c1c1c;
  --muted:     #7a7268;
  --accent:    #1a3a5c;
  --accent2:   #c9622f;
  --good:      #2a9d5c;
  --bad:       #e05252;
  --radius:    6px;
  --shadow:    0 2px 8px rgba(0,0,0,0.07);
  --mono:      'DM Mono', monospace;
  --serif:     'DM Serif Display', serif;
  --sans:      'Source Sans 3', sans-serif;
}

body {
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.65;
  color: var(--text);
  background: var(--bg);
}

/* ---- Header ---- */
.report-header {
  background: var(--accent);
  color: #fff;
  padding: 2.5rem 3rem 2rem;
  border-bottom: 4px solid var(--accent2);
}
.report-header h1 {
  font-family: var(--serif);
  font-size: 2.1rem;
  font-weight: 400;
  letter-spacing: 0.01em;
  margin-bottom: 0.4rem;
}
.report-header .subtitle {
  font-size: 0.95rem;
  opacity: 0.75;
  font-weight: 300;
  font-style: italic;
}
.report-header .meta-row {
  margin-top: 1rem;
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  font-size: 0.85rem;
  opacity: 0.85;
}
.report-header .meta-row span::before { content: ''; }

/* ---- Nav ---- */
nav.toc {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0.75rem 3rem;
  display: flex;
  gap: 0.5rem 1.5rem;
  flex-wrap: wrap;
  font-size: 0.88rem;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: var(--shadow);
}
nav.toc a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
  padding: 0.15rem 0;
  border-bottom: 2px solid transparent;
  transition: border-color 0.15s;
}
nav.toc a:hover { border-bottom-color: var(--accent2); }

/* ---- Main ---- */
main { max-width: 1200px; margin: 0 auto; padding: 2rem 2rem 4rem; }

/* ---- Subtype sections ---- */
.subtype-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 2.5rem;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.subtype-section.failed { opacity: 0.65; }

.subtype-header {
  padding: 1.2rem 1.5rem;
  border-bottom: 1px solid var(--border);
  background: #faf9f7;
}
.subtype-header h2 {
  font-family: var(--serif);
  font-size: 1.4rem;
  font-weight: 400;
  margin-bottom: 0.4rem;
  color: var(--accent);
}
.subtype-meta {
  display: flex;
  gap: 0.5rem 1rem;
  flex-wrap: wrap;
  align-items: center;
  font-size: 0.88rem;
}
.meta-chip {
  background: #eef0f2;
  padding: 0.15rem 0.6rem;
  border-radius: 3px;
  color: var(--muted);
}
.meta-chip strong { color: var(--text); }

/* ---- Summary block ---- */
.summary-block, .runs-block {
  padding: 1.2rem 1.5rem;
}
.summary-block { border-bottom: 1px solid var(--border); }
.summary-block h3, .runs-block h3 {
  font-family: var(--sans);
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 1rem;
}

.summary-figs-row {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}
.summary-fig {
  flex: 1 1 300px;
  text-align: center;
}
.summary-fig img {
  max-width: 100%;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.summary-fig figcaption {
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 0.35rem;
  font-style: italic;
}

/* ---- Run rows ---- */
.run-row {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 0.6rem;
  overflow: hidden;
}
.run-row.failed { background: #fdf6f6; }

details > summary { cursor: pointer; list-style: none; }
details > summary::-webkit-details-marker { display: none; }

.run-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.65rem 1rem;
  background: #faf9f7;
  flex-wrap: wrap;
}
details[open] .run-header { background: #eef3f8; }

.run-label {
  font-family: var(--mono);
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--accent);
  min-width: 90px;
}

.run-stats {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  align-items: center;
  font-size: 0.82rem;
}

.stat-chip {
  padding: 0.12rem 0.5rem;
  border-radius: 3px;
  font-size: 0.8rem;
}
.pgs-chip   { background: #dce8f5; color: var(--accent); }
.cov-chip   { background: #e8f0e8; color: #2a5c3a; }
.delta-chip { background: #f5f0e0; color: #6b5a2a; }

.run-body {
  padding: 1rem 1rem 0.75rem;
  border-top: 1px solid var(--border);
}

/* ---- Tables ---- */
.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  margin-bottom: 0.75rem;
}
.stats-table th {
  text-align: left;
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  padding: 0.3rem 0.6rem;
  border-bottom: 2px solid var(--border);
}
.stats-table td {
  padding: 0.3rem 0.6rem;
  border-bottom: 1px solid #f0ece6;
  font-variant-numeric: tabular-nums;
}
.stats-table tr:last-child td { border-bottom: none; }
.stats-table td:first-child { color: var(--muted); font-size: 0.85rem; }
.meta-table td { font-family: var(--mono); font-size: 0.82rem; }
.meta-table td:first-child, .meta-table td:nth-child(3) {
  color: var(--muted);
  font-family: var(--sans);
  font-size: 0.82rem;
}

.ci-str { font-size: 0.8rem; color: var(--muted); font-family: var(--mono); }

/* ---- Figures row ---- */
.figure-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 0.75rem;
}
.figure-row figure {
  flex: 1 1 260px;
  text-align: center;
}
.figure-row img {
  max-width: 100%;
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.figure-row figcaption {
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.3rem;
  font-style: italic;
}

/* ---- Badges ---- */
.badge {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 3px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.badge-ok   { background: #d4edda; color: #1a5c2e; }
.badge-fail { background: #fde8e8; color: #8b1a1a; }

.muted { color: var(--muted); font-style: italic; font-size: 0.9rem; }

/* ---- Footer ---- */
footer {
  text-align: center;
  padding: 2rem;
  color: var(--muted);
  font-size: 0.82rem;
  border-top: 1px solid var(--border);
  margin-top: 2rem;
}
'''


def build_html(results_dir, results, strategy, args_meta):
    """Assemble the full HTML document."""
    subtypes = list(results.keys())

    # Table of contents
    toc_links = ' '.join(
        f'<a href="#{s}">{s.replace("_"," ").title()}</a>'
        for s in subtypes
    )

    # Subtype sections
    sections = '\n'.join(
        subtype_section_html(s, results[s], results_dir, strategy)
        for s in subtypes
    )

    # Strategy metadata line
    strat_info = args_meta.get('strategy_info', strategy)
    timestamp  = datetime.now().strftime('%Y-%m-%d %H:%M')
    run_dir    = os.path.abspath(results_dir)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PRS LASSO Cox — Results Report</title>
  <style>{CSS}</style>
</head>
<body>

<header class="report-header">
  <h1>PRS LASSO Cox Survival Analysis</h1>
  <div class="subtitle">Glioma Polygenic Risk Score Pipeline — Results Report</div>
  <div class="meta-row">
    <span>Strategy: <strong>{strat_info}</strong></span>
    <span>Subtypes: <strong>{len(subtypes)}</strong></span>
    <span>Results dir: <code>{run_dir}</code></span>
    <span>Generated: <strong>{timestamp}</strong></span>
  </div>
</header>

<nav class="toc">
  <strong>Jump to:</strong>
  {toc_links}
</nav>

<main>
{sections}
</main>

<footer>
  Generated by <code>generate_report.py</code> &mdash; PRS LASSO Cox Pipeline
</footer>

</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML report from PRS LASSO Cox pipeline results')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Pipeline results directory (default: results)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path '
                             '(default: <results-dir>/report.html)')
    args = parser.parse_args()

    results_dir = args.results_dir
    json_path   = os.path.join(results_dir, 'results_summary.json')

    if not os.path.isfile(json_path):
        print(f"ERROR: results_summary.json not found in {results_dir}")
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)

    if not results:
        print("ERROR: results_summary.json is empty.")
        sys.exit(1)

    # Infer strategy from first result that has one
    strategy = 'unknown'
    n_splits = None
    train_frac = None
    for v in results.values():
        if 'cv_strategy' in v:
            strategy = v['cv_strategy']
            n_splits   = v.get('n_splits')
            train_frac = v.get('train_fraction')
            break

    if strategy == 'random_split' and n_splits and train_frac:
        strat_info = f'Random Split  N={n_splits}, train={train_frac:.0%}'
    elif strategy == 'loco':
        strat_info = 'Leave-One-Cohort-Out (LOCO)'
    else:
        strat_info = strategy

    out_path = args.output or os.path.join(results_dir, 'report.html')

    print(f"Building report for strategy: {strat_info}")
    print(f"Subtypes: {list(results.keys())}")

    html = build_html(results_dir, results,
                      strategy, {'strategy_info': strat_info})

    with open(out_path, 'w') as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Report written: {out_path}  ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
