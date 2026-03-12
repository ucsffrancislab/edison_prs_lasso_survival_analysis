#!/bin/bash
#SBATCH --job-name=prs_lasso_array
#SBATCH --output=prs_lasso_%A_%a.out
#SBATCH --error=prs_lasso_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=490G
#SBATCH --time=12:00:00

# PRS LASSO Cox Survival Analysis - Array Job
# One array task per glioma subtype

set -euo pipefail

#mkdir -p logs

# Load modules
#module load python/3.12 2>/dev/null || true
#source venv/bin/activate 2>/dev/null || true

# Map array index to subtype
SUBTYPES=("idh_wildtype" "lgg_idh_mutant_pq_intact" "hgg_idh_mutant_pq_intact" "lgg_idh_mutant_pq_codel")
SUBTYPE=${SUBTYPES[$SLURM_ARRAY_TASK_ID]}

DATA_DIR="${1:-.}"
OUTPUT_DIR="${2:-results}"
EXTRA_ARGS="${@:3}"

echo "=== PRS LASSO Cox Pipeline - Array Task ==="
echo "Start time: $(date)"
echo "Subtype: ${SUBTYPE}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

python3 run_pipeline.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-jobs ${SLURM_CPUS_PER_TASK:-64} \
    --subtype "${SUBTYPE}" \
    ${EXTRA_ARGS}

echo "End time: $(date)"
echo "=== Task Complete ==="
