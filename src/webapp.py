from __future__ import annotations

import base64
import io
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import cv2
import numpy as np
import requests
import tifffile
import torch
import yaml
from flask import Flask, render_template, request
from PIL import Image

from .data_files import _normalize_image
from .model import UNet, infer_unet_params_from_state_dict

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
DEFAULT_CKPT = ROOT / "checkpoints" / "best.pt"
DEMO_DIRS = [
    ROOT / "presentation" / "imgs",
    ROOT / "test_img",
]
DEMO_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class Predictor:
    def __init__(self, config_path: Path, checkpoint_path: Path, min_tumor_pixels: int = 25) -> None:
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.height = int(cfg["img_height"])
        self.width = int(cfg["img_width"])
        self.min_tumor_pixels = int(min_tumor_pixels)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        in_channels, base, depth = infer_unet_params_from_state_dict(ckpt["model"])
        self.model = UNet(in_channels=in_channels, base=base, depth=depth).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _read_bytes_as_image(self, data: bytes, hint: str = "") -> np.ndarray:
        lower = hint.lower()
        if lower.endswith((".tif", ".tiff")):
            arr = tifffile.imread(io.BytesIO(data))
        else:
            try:
                arr = tifffile.imread(io.BytesIO(data))
            except Exception:
                arr = np.array(Image.open(io.BytesIO(data)))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    def _png_b64(self, img: np.ndarray) -> str:
        img = np.clip(img, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _apply_clahe(self, image01: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
        gray_u8 = (np.clip(image01.mean(axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
        enhanced = clahe.apply(gray_u8).astype(np.float32) / 255.0
        return np.repeat(enhanced[..., None], 3, axis=-1)

    def _remove_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        if min_area <= 1:
            return mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= min_area:
                keep[labels == lab] = 1
        return keep

    def predict(
        self,
        image: np.ndarray,
        *,
        threshold: float = 0.5,
        use_clahe: bool = False,
        min_area: int = 1,
    ) -> dict[str, Any]:
        vis = _normalize_image(image)
        vis = cv2.resize(vis, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if use_clahe:
            vis = self._apply_clahe(vis)

        x = np.transpose(vis, (2, 0, 1))[None, ...]
        x_t = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        mask = (probs >= threshold).astype(np.uint8)
        mask = self._remove_small_components(mask, min_area=min_area)
        tumor_pixels = int(mask.sum())
        tumor_ratio = float(tumor_pixels / mask.size)
        has_tumor = tumor_pixels >= self.min_tumor_pixels

        input_img = (vis * 255.0).astype(np.uint8)
        mask_img = (mask * 255).astype(np.uint8)
        prob_img = np.clip(probs * 255.0, 0, 255).astype(np.uint8)
        prob_color = cv2.applyColorMap(prob_img, cv2.COLORMAP_INFERNO)
        prob_color = cv2.cvtColor(prob_color, cv2.COLOR_BGR2RGB)
        mask_rgb = np.stack([mask_img, np.zeros_like(mask_img), np.zeros_like(mask_img)], axis=-1)
        overlay = (0.75 * input_img + 0.25 * mask_rgb).astype(np.uint8)

        return {
            "has_tumor": has_tumor,
            "tumor_pixels": tumor_pixels,
            "tumor_ratio": tumor_ratio,
            "threshold": threshold,
            "use_clahe": use_clahe,
            "min_area": min_area,
            "input_png": self._png_b64(input_img),
            "mask_png": self._png_b64(mask_img),
            "prob_png": self._png_b64(prob_color),
            "overlay_png": self._png_b64(overlay),
        }


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATES))

    ckpt_path = Path(app.config.get("CHECKPOINT_PATH", DEFAULT_CKPT))
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    predictor = Predictor(CONFIG_PATH, ckpt_path)

    def infer_demo_label(path: Path) -> str:
        name = path.name.lower()
        if "no_tumor" in name or "notumor" in name or "normal" in name:
            return "No tumor"
        if name.startswith("demo_"):
            return "Tumor"
        return "Unlabeled"

    def list_demo_options() -> list[dict[str, str]]:
        opts: list[dict[str, str]] = []
        seen: set[Path] = set()
        for base in DEMO_DIRS:
            if not base.is_dir():
                continue
            for p in sorted(base.iterdir()):
                if not p.is_file() or p.suffix.lower() not in DEMO_SUFFIXES:
                    continue
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                rel = rp.relative_to(ROOT) if str(rp).startswith(str(ROOT)) else rp
                tag = infer_demo_label(rp)
                opts.append(
                    {
                        "value": str(rp),
                        "label": f"{rel} [{tag}]",
                    }
                )
        return opts

    def parse_float(name: str, default: float, low: float, high: float) -> float:
        raw = request.form.get(name, str(default)).strip()
        try:
            val = float(raw)
        except ValueError:
            return default
        return float(min(max(val, low), high))

    def parse_int(name: str, default: int, low: int, high: int) -> int:
        raw = request.form.get(name, str(default)).strip()
        try:
            val = int(raw)
        except ValueError:
            return default
        return int(min(max(val, low), high))

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        error = None
        result = None
        demo_options = list_demo_options()
        controls = {
            "threshold": 0.5,
            "use_clahe": False,
            "min_area": 1,
            "image_path": "",
            "demo_path": "",
        }

        if request.method == "POST":
            file = request.files.get("image_file")
            url = (request.form.get("image_url") or "").strip()
            image_path_s = (request.form.get("image_path") or "").strip()
            demo_path_s = (request.form.get("demo_path") or "").strip()
            controls["threshold"] = parse_float("threshold", 0.5, 0.05, 0.95)
            controls["use_clahe"] = request.form.get("use_clahe") == "on"
            controls["min_area"] = parse_int("min_area", 1, 1, 5000)
            controls["image_path"] = image_path_s
            controls["demo_path"] = demo_path_s

            def load_local_path(path_s: str) -> tuple[bytes, str]:
                p = Path(path_s).expanduser()
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if not p.is_file():
                    raise FileNotFoundError(f"Local file not found: {p}")
                return p.read_bytes(), str(p)

            if file and file.filename:
                data = file.read()
                hint = file.filename
            elif url:
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"}:
                    resp = requests.get(url, timeout=20)
                    resp.raise_for_status()
                    data = resp.content
                    hint = url
                else:
                    data, hint = load_local_path(url)
            elif image_path_s:
                data, hint = load_local_path(image_path_s)
            elif demo_path_s:
                data, hint = load_local_path(demo_path_s)
            else:
                error = (
                    "Please upload an image file, provide an image URL, choose a local demo file, "
                    "or enter a local file path."
                )
                return render_template(
                    "index.html",
                    error=error,
                    result=None,
                    min_tumor_pixels=predictor.min_tumor_pixels,
                    controls=controls,
                    demo_options=demo_options,
                )

            try:
                image = predictor._read_bytes_as_image(data, hint=hint)
                result = predictor.predict(
                    image,
                    threshold=controls["threshold"],
                    use_clahe=controls["use_clahe"],
                    min_area=controls["min_area"],
                )
            except (FileNotFoundError, requests.RequestException, ValueError) as exc:
                error = f"Prediction failed: {exc}"
            except Exception as exc:
                error = f"Prediction failed: {exc}"

        return render_template(
            "index.html",
            error=error,
            result=result,
            min_tumor_pixels=predictor.min_tumor_pixels,
            controls=controls,
            demo_options=demo_options,
        )

    return app


def main() -> None:
    app = create_app()
    host = "0.0.0.0"
    port = 5002
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
