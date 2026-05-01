from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from .data_files import _iter_files, _normalize_image, _normalize_mask, discover_pairs

__all__ = [
    "BrainTumorDataset",
    "discover_pairs",
    "load_split_entries",
    "resolve_pair_paths",
    "write_splits",
]


def resolve_pair_paths(image_s: str, mask_s: str, data_root: Path) -> tuple[Path, Path]:
    data_root = data_root.resolve()

    def one(p: Path) -> Path:
        if not p.is_absolute():
            out = (data_root / p).resolve()
            if out.is_file():
                return out
            return out
        p = p.resolve()
        if p.is_file():
            return p
        if len(p.parts) < 2:
            return p
        case_dir, fname = p.parts[-2], p.parts[-1]
        for base in (data_root, data_root / "lgg_mri_kagglehub"):
            cand = (base / case_dir / fname).resolve()
            if cand.is_file():
                return cand
        for f in _iter_files(data_root):
            if f.name == fname and f.parent.name == case_dir:
                return f.resolve()
        return p

    return one(Path(image_s)), one(Path(mask_s))


def write_splits(
    pairs: list[tuple[Path, Path, str]],
    split_dir: Path,
    data_root: Path,
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

    anchor = data_root.resolve()

    def to_rel(p: Path) -> str:
        p = p.resolve()
        try:
            return str(p.relative_to(anchor))
        except ValueError:
            return str(p)

    def collect(case_set: set[str]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for c in sorted(case_set):
            for img, msk in by_case[c]:
                out.append({"image": to_rel(img), "mask": to_rel(msk)})
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
            "path_format": "relative_to_data_root",
            "data_root_used_for_split": str(anchor),
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
        path_anchor: Path | None = None,
    ) -> None:
        self.entries = entries
        self.height = height
        self.width = width
        self.augment = augment
        self.path_anchor = path_anchor.resolve() if path_anchor is not None else None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        e = self.entries[idx]
        if self.path_anchor is not None:
            img_path, mask_path = resolve_pair_paths(
                e["image"], e["mask"], self.path_anchor
            )
        else:
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
