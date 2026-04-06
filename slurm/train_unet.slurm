#!/bin/bash
#SBATCH --job-name=brain_unet
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# If you see "Memory specification can not be satisfied" or "node configuration is not available":
#   - Do NOT set --mem here unless ARC docs require it (many GPU nodes only allow the partition default).
#   - Override at submit time:  sbatch --mem=16G --partition=YOUR_GPU_PART slurm/train_unet.slurm
#   - List partitions:  sinfo -s   or   sinfo -o "%P %G %m %c"
#   - Match configs/default.yaml num_workers to cpus-per-task (e.g. num_workers: 2–4).

# Train U-Net until epochs finish, early stopping, or time limit.
# Submit from repo root:  sbatch slurm/train_unet.slurm
# Or:  sbatch --chdir=/path/to/brain-tumor-segmentation-unet slurm/train_unet.slurm
#
# Resume after a kill (once checkpoints/last.pt exists):
#   SBATCH_EXTRA_ARGS='--resume checkpoints/last.pt' sbatch slurm/train_unet.slurm
#
# Conda in batch jobs: either set CONDA_ENV to the env *directory* (recommended), e.g.
#   export CONDA_ENV=$HOME/.conda/envs/brain-tumor-unet
#   sbatch --chdir="$PWD" slurm/train_unet.slurm
# or set CONDA_SH to conda.sh then CONDA_ENV to env name, e.g.
#   export CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
#   export CONDA_ENV=brain-tumor-unet

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

# Python: SLURM is non-interactive — `conda activate` needs conda.sh first, or use env's python binary.
export PYTHONUNBUFFERED=1

PYTHON_BIN=""

if [ -n "${CONDA_ENV:-}" ] && [ -x "${CONDA_ENV}/bin/python" ]; then
  PYTHON_BIN="${CONDA_ENV}/bin/python"
  echo "[brain_unet] using CONDA_ENV python: $PYTHON_BIN"
else
  module load anaconda3 2>/dev/null || true

  _conda_sh="${CONDA_SH:-}"
  if [ -z "$_conda_sh" ] && [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    _conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
  if [ -z "$_conda_sh" ] && command -v conda >/dev/null 2>&1; then
    _root="$(conda info --base 2>/dev/null || true)"
    if [ -n "$_root" ] && [ -f "$_root/etc/profile.d/conda.sh" ]; then
      _conda_sh="$_root/etc/profile.d/conda.sh"
    fi
  fi
  if [ -z "$_conda_sh" ]; then
    for _root in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
      if [ -f "$_root/etc/profile.d/conda.sh" ]; then
        _conda_sh="$_root/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if [ -n "$_conda_sh" ] && [ -f "$_conda_sh" ]; then
    # shellcheck source=/dev/null
    source "$_conda_sh"
    hash -r 2>/dev/null || true
  fi

  if [ -n "${CONDA_ENV:-}" ]; then
    conda activate "$CONDA_ENV"
  elif [ -n "${CONDA_PREFIX:-}" ]; then
    conda activate "$CONDA_PREFIX"
  fi

  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

echo "[brain_unet] python: $(command -v "$PYTHON_BIN" 2>/dev/null || echo "$PYTHON_BIN")"
"$PYTHON_BIN" -V

# Optional .env (Kaggle, etc.) — not required for training if data is already on disk
if [ -f ".env" ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

"$PYTHON_BIN" -m src.train --config configs/default.yaml ${SBATCH_EXTRA_ARGS:-}

echo "[brain_unet] finished at $(date)"
