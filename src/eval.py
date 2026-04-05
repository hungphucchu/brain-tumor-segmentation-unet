from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .dataset import BrainTumorDataset, load_split_entries
from .metrics import dice_iou_from_logits
from .model import UNet


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path | None,
    max_save: int,
) -> dict[str, float]:
    model.eval()
    w_dice = 0.0
    w_iou = 0.0
    n_samples = 0
    tp = fp = fn = 0.0
    saved = 0
    eps = 1e-6

    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bs = images.size(0)
        logits = model(images)
        d, j = dice_iou_from_logits(logits, masks)
        w_dice += float(d.cpu()) * bs
        w_iou += float(j.cpu()) * bs
        n_samples += bs

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        tp += float((pred * masks).sum().cpu())
        fp += float((pred * (1.0 - masks)).sum().cpu())
        fn += float(((1.0 - pred) * masks).sum().cpu())

        if save_dir is not None and saved < max_save:
            probs_viz = torch.sigmoid(logits)
            b = images.size(0)
            for i in range(b):
                if saved >= max_save:
                    break
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                pred_np = (probs_viz[i, 0].cpu().numpy() >= 0.5).astype(np.float32)
                gt = masks[i, 0].cpu().numpy()

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(np.clip(img, 0, 1))
                ax[0].set_title("Input")
                ax[0].axis("off")
                ax[1].imshow(img)
                ax[1].imshow(pred_np, alpha=0.45, cmap="Reds")
                ax[1].set_title("Prediction overlay")
                ax[1].axis("off")
                ax[2].imshow(img)
                ax[2].imshow(gt, alpha=0.45, cmap="Greens")
                ax[2].set_title("Ground truth overlay")
                ax[2].axis("off")
                fig.tight_layout()
                fig.savefig(save_dir / f"overlay_{saved:04d}.png", dpi=120)
                plt.close(fig)
                saved += 1

    if n_samples == 0:
        return {
            "dice_mean": 0.0,
            "iou_mean": 0.0,
            "precision_mean": 0.0,
            "recall_mean": 0.0,
            "n_samples": 0.0,
        }
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "dice_mean": w_dice / n_samples,
        "iou_mean": w_iou / n_samples,
        "precision_mean": float(precision),
        "recall_mean": float(recall),
        "n_samples": float(n_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net; save overlay PNGs.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-overlays", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path.cwd()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = load_split_entries(repo_root / cfg["split_dir"], args.split)
    ds = BrainTumorDataset(
        entries,
        height=int(cfg["img_height"]),
        width=int(cfg["img_width"]),
        augment=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else repo_root / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(in_channels=3, base=64).to(device)
    model.load_state_dict(ckpt["model"])

    out_dir = repo_root / cfg.get("outputs_dir", "outputs") / "eval_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_eval(model, loader, device, out_dir, max_save=int(args.max_overlays))
    print(
        f"Split={args.split}  "
        f"Dice={metrics['dice_mean']:.4f}  IoU={metrics['iou_mean']:.4f}  "
        f"precision={metrics['precision_mean']:.4f}  recall={metrics['recall_mean']:.4f}"
    )
    print(
        "Pixel accuracy is not reported: with ~98% background it is misleading; "
        "use Dice / IoU as in implementation.md."
    )
    print(f"Saved up to {args.max_overlays} overlays under {out_dir}")


if __name__ == "__main__":
    main()
