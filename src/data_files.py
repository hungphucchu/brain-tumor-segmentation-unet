from __future__ import annotations
import os
from pathlib import Path

import numpy as np


def resolve_data_root(repo_root: Path, config_data_root: str, override: Path | None) -> Path:
    if override is not None:
        p = Path(override)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()
    return (repo_root / config_data_root).resolve()


def _iter_files(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            out.append(Path(dirpath) / name)
    return out


def _mask_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_mask", "_MASK", "-mask"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return ""


def discover_pairs(data_root: Path) -> list[tuple[Path, Path, str]]:
    data_root = data_root.resolve()
    pairs: list[tuple[Path, Path, str]] = []
    for mask_path in sorted(_iter_files(data_root)):
        if not mask_path.is_file():
            continue
        if mask_path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        stem = _mask_stem(mask_path)
        if not stem:
            continue
        image_path = mask_path.with_name(f"{stem}{mask_path.suffix}")
        if not image_path.is_file():
            alt = mask_path.with_name(f"{stem}.tif")
            if alt.is_file():
                image_path = alt
            else:
                continue
        case_id = mask_path.parent.name
        pairs.append((image_path, mask_path, case_id))
    return pairs


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0 + 1e-6 and arr.min() >= -1e-6:
        return np.clip(arr, 0.0, 1.0)
    if arr.max() <= 255.0 + 1e-6:
        return np.clip(arr / 255.0, 0.0, 1.0)
    cmin = float(arr.min())
    cmax = float(arr.max())
    if cmax - cmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - cmin) / (cmax - cmin), 0.0, 1.0)


def _normalize_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[..., 0]
    m = arr.astype(np.float32)
    if m.max() > 1.5:
        m = (m > 127).astype(np.float32)
    else:
        m = (m > 0.5).astype(np.float32)
    return m


def data_root_hint(data_root: Path) -> str:
    data_root = data_root.resolve()
    if not data_root.is_dir():
        return f"Directory does not exist: {data_root}"
    tifs = [p for p in _iter_files(data_root) if p.suffix.lower() in {".tif", ".tiff"}]
    masks = [p for p in tifs if _mask_stem(p)]
    return (
        f"Under {data_root}: {len(tifs)} TIFF files, {len(masks)} look like masks (*_mask.tif). "
        "Unzip the Kaggle LGG dataset so case folders (e.g. kaggle_3m/TCGA_*) live under data/raw, "
        "or run: python -m src.download_dataset"
    )


def resolve_effective_data_root(
    repo_root: Path,
    config_data_root: str,
    override: Path | None,
) -> tuple[Path, str]:
    primary = resolve_data_root(repo_root, config_data_root, override)
    if override is not None:
        return primary, "from --data-root"

    raw = repo_root / "data" / "raw"
    candidates: list[tuple[Path, str]] = [
        (primary, "from config data_root"),
        (raw / "lgg_mri_kagglehub", "data/raw/lgg_mri_kagglehub"),
        (raw / "lgg_mri_kagglehub" / "kaggle_3m", "data/raw/lgg_mri_kagglehub/kaggle_3m"),
    ]

    for path, label in candidates:
        try:
            r = path.resolve()
        except OSError:
            continue
        if r.is_dir() and discover_pairs(r):
            return r, label

    try:
        import kagglehub

        hub = Path(kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")).resolve()
        for sub, label in (
            (hub / "kaggle_3m", "kagglehub cache …/kaggle_3m"),
            (hub, "kagglehub cache (dataset root)"),
        ):
            if sub.is_dir() and discover_pairs(sub):
                return sub, label
    except Exception:
        pass

    return primary, "from config data_root (no pairs found here)"
