"""
Configuration for PRS LASSO Cox Survival Analysis Pipeline
All pipeline variables defined in one place.
"""
import os
import argparse

# Random seed for reproducibility
RANDOM_SEED = 42

# Cohorts
DISCOVERY_COHORTS = ['i370', 'onco', 'tcga']
VALIDATION_COHORT = 'cidr'

# ---------------------------------------------------------------------------
# Validation strategy
# ---------------------------------------------------------------------------
# 'fixed'        : original behaviour — train on DISCOVERY_COHORTS, validate
#                  on VALIDATION_COHORT.
# 'loco'         : Leave-One-Cohort-Out — all cohorts are pooled, then each
#                  is held out in turn as the validation set while the
#                  remaining cohorts train.  Per-fold results are averaged
#                  to give a final mean ± SD C-index.  VALIDATION_COHORT and
#                  DISCOVERY_COHORTS are ignored; ALL_COHORTS is used instead.
# 'random_split' : all cohorts are pooled, then N_SPLITS independent random
#                  train/test splits (TRAIN_FRACTION / 1-TRAIN_FRACTION) are
#                  performed.  Results are averaged over splits to give a
#                  mean ± SD C-index.
CV_STRATEGY     = 'random_split'
ALL_COHORTS     = ['i370', 'onco', 'tcga', 'cidr']  # used by loco and random_split

# Random-split parameters (used only when CV_STRATEGY = 'random_split')
N_SPLITS        = 20     # number of independent random splits
TRAIN_FRACTION  = 0.70   # fraction of samples used for training

# Covariates used in all Cox PH models
# 'source' distinguishes two recruitment sources within the onco cohort only;
# it should be included as a covariate when fitting models within or across cohorts
# that include onco samples, but checked for variance before use.
COVARIATES = ['source', 'age', 'sex', 'grade', 'treated', 'PC1', 'PC2', 'PC3',
              'PC4', 'PC5', 'PC6', 'PC7', 'PC8']

# Glioma subtypes to analyze
# grade column contains string values: 'LGG' or 'HGG'
SUBTYPES = {
    'idh_wildtype':              {'case': 1, 'idh': 0},
    'lgg_idh_mutant_pq_intact':  {'case': 1, 'grade': 'LGG', 'idh': 1, 'pq': 0},
    'hgg_idh_mutant_pq_intact':  {'case': 1, 'grade': 'HGG', 'idh': 1, 'pq': 0},
    'lgg_idh_mutant_pq_codel':   {'case': 1, 'grade': 'LGG', 'idh': 1, 'pq': 1},
}

# Minimum thresholds
MIN_SAMPLES_PER_SUBTYPE = 20      # Skip subtype if fewer samples than this
MIN_EVENTS_PER_SUBTYPE  = 10      # Skip subtype if fewer events than this

# ---------------------------------------------------------------------------
# Parsimony controls
# ---------------------------------------------------------------------------

# Events-per-variable ratio for EPV cap: max predictors entering LASSO = floor(n_events / EPV_RATIO)
# Raised from 5 -> 10 to halve the predictor cap and reduce overfitting risk.
EPV_RATIO               = 10

# Nominal p-value threshold for pre-filtering PGS candidates via full univariate Cox.
# The score-test screen uses META_P_THRESHOLD * 2 as a looser first pass.
# Lowered from 0.05 -> 0.01 to admit fewer candidates into LASSO.
META_P_THRESHOLD        = 0.01

# Require the same direction of effect in at least this many discovery cohorts
# before a PGS is admitted to LASSO.  Set to len(DISCOVERY_COHORTS) (i.e. 3)
# to require unanimity across all cohorts; set to 2 for the original behaviour.
MIN_COHORTS_FOR_DIRECTION = 3

# Hard cap on the number of PGS candidates entering LASSO, applied after the
# EPV cap.  Acts as a secondary safety net independent of event count.
# Set to None to disable.
MAX_CANDIDATES_PREFILT  = 500

# Alpha-selection rule for the LASSO CV path.
#   'best' : alpha that maximises mean CV C-index  (original behaviour)
#   '1se'  : largest alpha (most regularisation) whose mean CV C-index is
#            within 1 SE of the best -- standard glmnet heuristic, yields
#            sparser models and is generally preferred to reduce overfitting.
LASSO_ALPHA_RULE        = '1se'

# Require consistent direction of effect across cohorts (uses MIN_COHORTS_FOR_DIRECTION above)
REQUIRE_CONSISTENT_DIR  = True

# ---------------------------------------------------------------------------

# Cross-validation
CV_FOLDS = 10

# Output
OUTPUT_DIR = 'results'

# Parallelization
N_JOBS = 64   # Match available CPUs on the HPC node

# Debug mode (set via --debug flag; triggers verbose printing)
DEBUG = False

# Data directory
DATA_DIR = '.'

# Model subsetting for development/testing
N_MODELS = None  # Set to integer for testing (e.g., 100); None for all models

# Model allowlist file (set via --models flag)
# If set, only PGS IDs listed in this file (one per line) will be used.
MODELS_FILE = None


def parse_args():
    """Parse command-line arguments and update config accordingly."""
    global DATA_DIR, OUTPUT_DIR, N_MODELS, N_JOBS, DEBUG, MODELS_FILE, \
           CV_STRATEGY, N_SPLITS, TRAIN_FRACTION
    parser = argparse.ArgumentParser(description='PRS LASSO Cox Survival Analysis Pipeline')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory containing input data files')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory for output results')
    parser.add_argument('--n-models', type=int, default=None,
                        help='Number of PGS models to sample for testing (default: all)')
    parser.add_argument('--models', type=str, default=None,
                        help='Path to file with allowlisted PGS model IDs, one per line. '
                             'All other models are dropped before the pipeline runs.')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS,
                        help='Number of parallel jobs')
    parser.add_argument('--cv-strategy', type=str, default=CV_STRATEGY,
                        choices=['fixed', 'loco', 'random_split'],
                        help="Validation strategy: 'fixed' (original), 'loco' "
                             "(Leave-One-Cohort-Out), or 'random_split' "
                             "(N random train/test splits)")
    parser.add_argument('--n-splits', type=int, default=N_SPLITS,
                        help='Number of random splits (random_split strategy only, '
                             'default: %(default)s)')
    parser.add_argument('--train-fraction', type=float, default=TRAIN_FRACTION,
                        help='Fraction of samples used for training in each random split '
                             '(default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable verbose debug output')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run smoke test with synthetic data')
    parser.add_argument('--subtype', type=str, default=None,
                        help='Run only this subtype (for array jobs)')
    args = parser.parse_args()

    DATA_DIR       = args.data_dir
    OUTPUT_DIR     = args.output_dir
    N_MODELS       = args.n_models
    N_JOBS         = args.n_jobs
    DEBUG          = args.debug
    MODELS_FILE    = args.models
    CV_STRATEGY    = args.cv_strategy
    N_SPLITS       = args.n_splits
    TRAIN_FRACTION = args.train_fraction

    return args
