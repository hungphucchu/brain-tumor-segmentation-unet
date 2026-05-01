from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net for binary segmentation. Returns logits (apply sigmoid for probabilities).
    """

    def __init__(self, in_channels: int = 3, base: int = 64, depth: int = 3) -> None:
        super().__init__()
        if depth not in (3, 4):
            raise ValueError(f"Unsupported U-Net depth: {depth}. Expected 3 or 4.")
        self.depth = depth

        c1, c2, c3 = base, base * 2, base * 4

        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        if self.depth == 4:
            c4 = base * 8
            self.enc4 = DoubleConv(c3, c4)
            self.pool4 = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(c4, c4 * 2)
            self.up4 = nn.ConvTranspose2d(c4 * 2, c4, 2, stride=2)
            self.dec4 = DoubleConv(c4 * 2, c4)
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        else:
            self.bottleneck = DoubleConv(c3, c3 * 2)
            self.up3 = nn.ConvTranspose2d(c3 * 2, c3, 2, stride=2)

        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        if self.depth == 4:
            e4 = self.enc4(self.pool3(e3))
            b = self.bottleneck(self.pool4(e4))
            d4 = self.up4(b)
            d4 = self.dec4(torch.cat([d4, e4], dim=1))
            d3 = self.up3(d4)
        else:
            b = self.bottleneck(self.pool3(e3))
            d3 = self.up3(b)

        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


def infer_unet_params_from_state_dict(
    state_dict: dict[str, torch.Tensor], default_in_channels: int = 3
) -> tuple[int, int, int]:
    in_channels = default_in_channels
    base = 64
    depth = 4 if "enc4.block.0.weight" in state_dict else 3

    enc1_w = state_dict.get("enc1.block.0.weight")
    if enc1_w is not None and enc1_w.ndim == 4:
        in_channels = int(enc1_w.shape[1])
        base = int(enc1_w.shape[0])

    return in_channels, base, depth
