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
EPV_RATIO               = 5       # Events-per-variable: max predictors = n_events / EPV_RATIO

# Pre-filtering thresholds (applied to pooled univariate Cox PH results)
META_P_THRESHOLD        = 0.05    # Nominal p-value threshold for pre-filtering PGS candidates
REQUIRE_CONSISTENT_DIR  = True    # Require consistent direction of effect across cohorts

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


def parse_args():
    """Parse command-line arguments and update config accordingly."""
    global DATA_DIR, OUTPUT_DIR, N_MODELS, N_JOBS, DEBUG
    parser = argparse.ArgumentParser(description='PRS LASSO Cox Survival Analysis Pipeline')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help='Directory containing input data files')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory for output results')
    parser.add_argument('--n-models', type=int, default=None,
                        help='Number of PGS models to sample for testing (default: all)')
    parser.add_argument('--n-jobs', type=int, default=N_JOBS,
                        help='Number of parallel jobs')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable verbose debug output')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run smoke test with synthetic data')
    parser.add_argument('--subtype', type=str, default=None,
                        help='Run only this subtype (for array jobs)')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    N_MODELS = args.n_models
    N_JOBS = args.n_jobs
    DEBUG = args.debug

    return args
