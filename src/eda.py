from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml

from .data_files import (
    _normalize_image,
    _normalize_mask,
    data_root_hint,
    discover_pairs,
    resolve_effective_data_root,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick EDA: shapes, dtypes, sample overlay.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder with TIFF pairs (overrides data_root in config). Relative paths use current working directory.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = load_config(args.config)
    data_root, picked = resolve_effective_data_root(
        repo_root, str(cfg["data_root"]), args.data_root
    )
    pairs = discover_pairs(data_root)
    print(f"Data root: {data_root} ({picked})")
    if not pairs:
        raise SystemExit(
            f"No image/mask pairs under {data_root}.\n"
            f"{data_root_hint(data_root)}\n\n"
            "This usually means the Kaggle archive is not unpacked here yet. "
            "Either unzip the LGG dataset into data/raw/ (see README), or pass your folder, e.g.:\n"
            "  python -m src.eda --config configs/default.yaml --data-root /path/to/kaggle_3m"
        )

    rng = random.Random(int(cfg["seed"]))
    pick = rng.sample(range(len(pairs)), min(args.samples, len(pairs)))

    out_dir = repo_root / cfg.get("outputs_dir", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(pick):
        img_path, mask_path, case_id = pairs[idx]
        image = tifffile.imread(str(img_path))
        mask = tifffile.imread(str(mask_path))
        print(f"[{i}] case={case_id}")
        print(f"    image: {img_path} shape={image.shape} dtype={image.dtype}")
        print(f"    mask:  {mask_path} shape={mask.shape} dtype={mask.dtype}")

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        vis = _normalize_image(image)
        m = _normalize_mask(mask)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(np.clip(vis, 0, 1))
        ax[0].set_title("Image (normalized preview)")
        ax[0].axis("off")
        ax[1].imshow(np.clip(vis, 0, 1))
        ax[1].imshow(m, alpha=0.45, cmap="Reds")
        ax[1].set_title("Overlay (mask)")
        ax[1].axis("off")
        fig.suptitle(f"{case_id}")
        fig.tight_layout()
        fig_path = out_dir / f"eda_sample_{i}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"    saved {fig_path}")

    print("EDA done.")


if __name__ == "__main__":
    main()
