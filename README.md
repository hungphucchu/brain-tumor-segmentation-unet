# Brain tumor segmentation (U-Net)

PyTorch implementation for binary tumor masks on 2D brain MRI slices, following [implementation.md](implementation.md) and the brief in [idea.txt](idea.txt).

## Dataset

Use the Kaggle **Brain MRI segmentation** dataset (LGG / TCGA-style TIFF slices):

- [https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### Option A — Download with kagglehub (full dataset, no zip)

Configure Kaggle API access (same as the Kaggle CLI): put `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY`.

From the repo root:

```bash
pip install -r requirements.txt
python -m src.download_dataset
```

This downloads via `kagglehub`, then adds `data/raw/lgg_mri_kagglehub` as a **symlink** to the cached dataset so `data_root: data/raw` in the config still works.

- `--no-link` — only download and print the path (use with `python -m src.eda --data-root ...`).
- If symlink fails on Windows, use the printed path with `--data-root`.

### Option B — Manual zip

Unzip so that slice folders live under `data/raw/` (recursive scan is supported). Expected pairing:

- Image: `<name>.tif`
- Mask: `<name>_mask.tif`  
  in the same directory (case folders such as `TCGA_*`).

**Split policy:** `data/splits/splits.json` is built with a **case-level** 80% / 10% / 10% split (parent folder name = case ID) and a fixed seed from `configs/default.yaml` so runs are reproducible.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NumPy is pinned to **1.x** (`numpy<2`) so it matches common PyTorch wheels and avoids “compiled using NumPy 1.x” / `_ARRAY_API` errors. If you already installed NumPy 2, run `pip install -r requirements.txt` again to downgrade.

## Commands

Run from the repository root.

1. **EDA** — verify shapes, dtypes, and a few overlays:

   ```bash
   python -m src.eda --config configs/default.yaml
   ```

   If the dataset lives outside `data/raw/`, point to the folder that contains the case subdirectories (often `kaggle_3m`):

   ```bash
   python -m src.eda --config configs/default.yaml --data-root /path/to/kaggle_3m
   ```

2. **Prepare splits only** (optional; training also creates splits if missing):

   ```bash
   python -m src.train --config configs/default.yaml --prepare-splits
   ```

3. **Train**

   ```bash
   python -m src.train --config configs/default.yaml
   ```

   Checkpoints: `checkpoints/best.pt` (best validation Dice).

   **HPC / second machine:** `data/splits/splits.json` stores paths **relative** to `data/raw` so clones stay portable. If you still see `FileNotFoundError` pointing at another computer’s path, regenerate splits on this host: `python -m src.train --config configs/default.yaml --force-splits`.

   **CUDA “driver too old”:** Install a PyTorch build that matches the cluster’s CUDA/driver (see [pytorch.org](https://pytorch.org)), or training falls back to CPU if `torch.cuda.is_available()` is false. You can set `amp: false` in `configs/default.yaml` when on CPU.

4. **Evaluate** on test set and save overlay PNGs:

   ```bash
   python -m src.eval --config configs/default.yaml --checkpoint checkpoints/best.pt --split test --max-overlays 20
   ```

   Overlays: `outputs/eval_overlays/`.

## Configuration

Edit [configs/default.yaml](configs/default.yaml) for paths, batch size, learning rate, loss (`bce_dice` or `dice`), mixed precision (`amp`), and early stopping.

## References

- Ronneberger et al., U-Net (2015): [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Buda et al. (2019), LGG shape features / dataset context: *Computers in Biology and Medicine*

## Note on metrics

Pixel accuracy is **not** emphasized: most voxels are background. Training monitors **validation Dice**; evaluation reports **Dice**, **IoU**, and foreground **precision/recall**.
