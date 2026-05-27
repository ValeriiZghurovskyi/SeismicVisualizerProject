"""Seed-conditioned U-Net for seismic feature segmentation.

Input : (B, 2, H, W) — channel 0 = seismic amplitude, channel 1 = seed hint.
Output: (B, 1, H, W) — raw logits; apply torch.sigmoid for probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """2-channel U-Net with skip connections.

    Args:
        in_channels: Input channels (default 2: seismic + seed hint).
        base_filters: Filters in first encoder block; doubles at each depth level.
        depth: Number of encoder/decoder levels.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_filters: int = 16,
        depth: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        # Dropout only at deeper levels (bottleneck + last encoder block) — keeps
        # early features sharp while regularising the high-capacity layers.
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            drop = dropout if i == depth - 1 else 0.0
            self.encoders.append(_DoubleConv(ch, out_ch, dropout=drop))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck = _DoubleConv(ch, base_filters * (2 ** depth), dropout=dropout)
        ch = base_filters * (2 ** depth)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = base_filters * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, skip_ch, kernel_size=2, stride=2))
            drop = dropout if i == depth - 1 else 0.0
            self.decoders.append(_DoubleConv(skip_ch * 2, skip_ch, dropout=drop))
            ch = skip_ch

        self.out_conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)
