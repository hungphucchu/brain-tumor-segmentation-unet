#!/bin/bash
#SBATCH --job-name=brain_unet
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Train U-Net until epochs finish, early stopping, or time limit.
# Submit from repo root:  sbatch slurm/train_unet.slurm
# Or:  sbatch --chdir=/path/to/brain-tumor-segmentation-unet slurm/train_unet.slurm
#
# Resume after a kill (once checkpoints/last.pt exists):
#   SBATCH_EXTRA_ARGS='--resume checkpoints/last.pt' sbatch slurm/train_unet.slurm
# Or edit the python line below.

set -eo pipefail
set +u
source ~/.bashrc 2>/dev/null || true
set -u

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

mkdir -p logs checkpoints

echo "[brain_unet] started at $(date) on $(hostname)"
echo "[brain_unet] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "[brain_unet] PROJECT_ROOT=$PROJECT_ROOT"
pwd

# Python: conda env path (set before submit or edit here)
# Example ARC: module load anaconda3 && conda activate /path/to/env
if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi
if [ -n "${CONDA_ENV:-}" ]; then
  conda activate "$CONDA_ENV"
elif [ -n "${CONDA_PREFIX:-}" ]; then
  conda activate "$CONDA_PREFIX"
fi

# Optional: cluster module (uncomment if needed)
# module load anaconda3

export PYTHONUNBUFFERED=1

# Optional .env (Kaggle, etc.) — not required for training if data is already on disk
if [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

python -m src.train --config configs/default.yaml ${SBATCH_EXTRA_ARGS:-}

echo "[brain_unet] finished at $(date)"
