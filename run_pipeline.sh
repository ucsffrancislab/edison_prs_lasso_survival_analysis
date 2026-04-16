#!/bin/bash
#SBATCH --job-name=prs_lasso_cox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=14-0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# PRS LASSO Cox Survival Analysis - Single Large Job
# Runs all subtypes using Python multiprocessing on a 64-CPU node

set -euo pipefail

# ── Locate the directory this script lives in ────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    PIPELINE_DIR=$(dirname "$(scontrol show job "$SLURM_JOB_ID" \
        | awk '/Command=/{sub(/.*Command=/, ""); print $1}')")
else
    PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# Create log directory
#mkdir -p logs

# Load modules (adjust for your HPC)
#module load python/3.12 2>/dev/null || true

# Activate virtual environment
#source venv/bin/activate 2>/dev/null || true

# Set data and output directories (override via command line)
#DATA_DIR="${1:-input}"
#OUTPUT_DIR="${2:-results}"
#EXTRA_ARGS="${@:3}"

# ── Parse --outdir from args, collect remainder for Python ───────────────────
DATA_DIR="input"
OUTPUT_DIR="results"
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datadir)
            DATA_DIR="$2"
            shift 2
            ;;
        --outdir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --outdir is required" >&2
    echo "Usage: sbatch run_pipeline.sh --outdir <directory> [pipeline options]" >&2
    exit 1
fi

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Redirect all subsequent stdout/stderr to a log file in outdir ────────────
exec > "${OUTPUT_DIR}/pipeline.out.txt" 2>&1


echo "=== PRS LASSO Cox Pipeline ==="
echo "Start time: $(date)"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Pipeline dir: ${PIPELINE_DIR}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-64}"
#echo "Extra args: ${EXTRA_ARGS}"
echo "Extra args: ${PYTHON_ARGS}"

# Run the main pipeline
python3 "$PIPELINE_DIR/run_pipeline.py" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-jobs ${SLURM_CPUS_PER_TASK:-64} \
    "${PYTHON_ARGS[@]}"

#    ${EXTRA_ARGS}

echo "End time: $(date)"
echo "=== Pipeline Complete ==="
