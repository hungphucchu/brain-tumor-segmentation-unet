# Brain Tumor Segmentation (U-Net)

This project trains and evaluates a U-Net model to segment brain tumor regions from MRI slices.
It uses the LGG MRI Segmentation dataset (`kaggle_3m`) and predicts a binary tumor mask for each image.

Main outputs:
- trained checkpoint (`checkpoints/best.pt`)
- quantitative metrics (Dice, IoU, Precision, Recall)
- qualitative overlays (`outputs/eval_overlays/`)
- optional Flask demo for local inference (`python -m src.webapp`)

Fast local setup for grading and demo.

## 1) Create environment

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2) Download dataset

Dataset: [LGG MRI Segmentation (kaggle_3m)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

Set Kaggle credentials (`~/.kaggle/kaggle.json` or env vars), then:

```bash
python -m src.download_dataset
```

## 3) Train model

```bash
python -m src.train --config configs/default.yaml
```

Output checkpoint:
- `checkpoints/best.pt`

## 4) Evaluate on test set

```bash
python -m src.eval --config configs/default.yaml --checkpoint checkpoints/best.pt --split test --max-overlays 20
```

Output overlays:
- `outputs/eval_overlays/`

## 5) Run web demo

```bash
python -m src.webapp
```

Open:
- [http://localhost:5002](http://localhost:5002)

Demo supports:
- file upload
- image URL
- local file path
- local demo-file dropdown

---

## HPC training (SLURM)

Use the provided batch script:
- `slurm/train_unet.slurm`

### 1) Prepare environment on HPC

Create/install your environment on the cluster first (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2) Submit training job

From repo root:

```bash
mkdir -p logs
sbatch --chdir="$PWD" slurm/train_unet.slurm
```

If your site uses a conda env folder, you can set:

```bash
export CONDA_ENV="$HOME/.conda/envs/brain-tumor-unet"
sbatch --chdir="$PWD" slurm/train_unet.slurm
```

### 3) Check job status

```bash
squeue -u "$USER"
```

### 4) Resume from last checkpoint (if job stopped)

```bash
python -m src.train --config configs/default.yaml --resume checkpoints/last.pt
```

Or with SLURM script override:

```bash
SBATCH_EXTRA_ARGS='--resume checkpoints/last.pt' sbatch --chdir="$PWD" slurm/train_unet.slurm
```

Notes:
- Best model is saved as `checkpoints/best.pt`.
- If job is slow or OOM, reduce `batch_size` / `num_workers` in `configs/default.yaml`.

---

## If you want a super quick check (no retraining)

If `checkpoints/best.pt` already exists:

```bash
python -m src.eval --config configs/default.yaml --checkpoint checkpoints/best.pt --split test --max-overlays 20
python -m src.webapp
```

---

## Common issues

- **`No module named ...`**  
  Activate venv and reinstall:
  ```bash
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  ```

- **Dataset not found**  
  Re-run:
  ```bash
  python -m src.download_dataset
  ```

- **CPU fallback / no CUDA**  
  Training still works, just slower.
