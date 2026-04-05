from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .data_files import discover_pairs, resolve_effective_data_root
from .dataset import BrainTumorDataset, load_split_entries, write_splits
from .losses import bce_dice_loss, dice_loss
from .metrics import dice_iou_from_logits
from .model import UNet


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_splits(cfg: dict, repo_root: Path, data_root: Path, force: bool) -> None:
    split_dir = repo_root / cfg["split_dir"]
    split_file = split_dir / "splits.json"
    if split_file.is_file() and not force:
        return
    pairs = discover_pairs(data_root)
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs under {data_root}. "
            "Expected structure like LGG Kaggle: <case>/*.tif and <case>/*_mask.tif"
        )
    write_splits(
        pairs,
        split_dir,
        data_root,
        float(cfg["train_ratio"]),
        float(cfg["val_ratio"]),
        float(cfg["test_ratio"]),
        int(cfg["seed"]),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with autocast(enabled=True):
                logits = model(images)
        else:
            logits = model(images)
        d, j = dice_iou_from_logits(logits, masks)
        dice_sum += float(d.cpu())
        iou_sum += float(j.cpu())
        n += 1
    return dice_sum / max(n, 1), iou_sum / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net on LGG-style brain MRI pairs.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Only scan data_root, write data/splits/splits.json, then exit.",
    )
    parser.add_argument(
        "--force-splits",
        action="store_true",
        help="Regenerate splits even if splits.json exists.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder with TIFF pairs (overrides data_root in config). Relative paths use cwd.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    data_root, picked = resolve_effective_data_root(
        repo_root, str(cfg["data_root"]), args.data_root
    )
    ensure_splits(cfg, repo_root, data_root, force=args.force_splits)
    if args.prepare_splits:
        print(f"Using data root: {data_root} ({picked})")
        print(f"Wrote splits under {repo_root / cfg['split_dir']}")
        return

    print(f"Training data root: {data_root} ({picked})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = int(cfg["img_height"]), int(cfg["img_width"])

    train_entries = load_split_entries(repo_root / cfg["split_dir"], "train")
    val_entries = load_split_entries(repo_root / cfg["split_dir"], "val")

    train_ds = BrainTumorDataset(
        train_entries,
        height=h,
        width=w,
        augment=bool(cfg.get("augment_train", True)),
        path_anchor=data_root,
    )
    val_ds = BrainTumorDataset(
        val_entries, height=h, width=w, augment=False, path_anchor=data_root
    )

    loader_kw = dict(
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = UNet(in_channels=3, base=64).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    loss_name = str(cfg.get("loss", "bce_dice"))
    bce_w = float(cfg.get("bce_weight", 0.5))

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if loss_name == "dice":
            return dice_loss(logits, targets)
        return bce_dice_loss(logits, targets, bce_weight=bce_w)

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler: GradScaler | None = GradScaler(enabled=True) if use_amp else None

    ckpt_dir = repo_root / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    best_dice = -1.0
    patience = int(cfg.get("early_stopping_patience", 15))
    stale = 0
    epochs = int(cfg["epochs"])

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        val_dice, val_iou = evaluate_epoch(model, val_loader, device, use_amp)
        scheduler.step(val_dice)
        print(
            f"Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  "
            f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "config": cfg,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping (no val Dice improvement for {patience} epochs).")
                break

    print(f"Best checkpoint: {best_path}  val_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
