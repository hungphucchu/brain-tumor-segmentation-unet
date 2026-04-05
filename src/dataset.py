from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .data_files import _normalize_image, _normalize_mask, discover_pairs

__all__ = [
    "BrainTumorDataset",
    "discover_pairs",
    "load_split_entries",
    "write_splits",
]


def write_splits(
    pairs: list[tuple[Path, Path, str]],
    split_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_case: dict[str, list[tuple[Path, Path]]] = {}
    for img, msk, case in pairs:
        by_case.setdefault(case, []).append((img, msk))

    cases = list(by_case.keys())
    rng.shuffle(cases)
    n = len(cases)
    if n < 3:
        raise ValueError(
            f"Need at least 3 case folders for train/val/test; found {n}. "
            "Check data_root and *_mask.tif / *.tif pairing."
        )

    i_train = int(round(train_ratio * n))
    i_val_end = int(round((train_ratio + val_ratio) * n))
    i_train = max(1, min(i_train, n - 2))
    i_val_end = max(i_train + 1, min(i_val_end, n - 1))

    train_cases = set(cases[:i_train])
    val_cases = set(cases[i_train:i_val_end])
    test_cases = set(cases[i_val_end:])

    def collect(case_set: set[str]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for c in sorted(case_set):
            for img, msk in by_case[c]:
                out.append({"image": str(img), "mask": str(msk)})
        return sorted(out, key=lambda x: x["image"])

    payload = {
        "train": collect(train_cases),
        "val": collect(val_cases),
        "test": collect(test_cases),
        "meta": {
            "seed": seed,
            "n_cases": n,
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "test_cases": len(test_cases),
            "split_indices": {"train_end": i_train, "val_end": i_val_end},
        },
    }
    path = split_dir / "splits.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_entries(split_dir: Path, split: str) -> list[dict[str, str]]:
    path = split_dir / "splits.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run training with --prepare-splits first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if split not in data:
        raise KeyError(f"Split '{split}' not in {path}")
    return data[split]


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        entries: list[dict[str, str]],
        height: int,
        width: int,
        augment: bool = False,
    ) -> None:
        self.entries = entries
        self.height = height
        self.width = width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.entries[idx]
        img_path = Path(e["image"])
        mask_path = Path(e["mask"])
        image = tifffile.imread(str(img_path))
        mask = tifffile.imread(str(mask_path))

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]

        image = _normalize_image(image)
        mask = _normalize_mask(mask)

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        image_chw = np.transpose(image, (2, 0, 1))
        mask_hw = mask[np.newaxis, ...]

        return {
            "image": torch.from_numpy(image_chw),
            "mask": torch.from_numpy(mask_hw),
        }
