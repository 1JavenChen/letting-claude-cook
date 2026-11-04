"""The trainable part: audio features -> mouth blendshape weights per frame."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, ch, kernel=5, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, C]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(torch.nn.functional.gelu(self.norm(y)))


class MouthHead(nn.Module):
    """Project features down, mix over time with a few 1D convolutions, then
    predict each blendshape in [0, 1].

    Each conv sees +/-2 frames, so 3 blocks see roughly +/-200 ms of context:
    enough for coarticulation, and only ~200 ms of look-ahead for real time.
    """

    def __init__(self, in_dim, n_out, hidden=256, blocks=3, dropout=0.1):
        super().__init__()
        self.config = dict(in_dim=in_dim, n_out=n_out, hidden=hidden, blocks=blocks, dropout=dropout)
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ConvBlock(hidden, dropout=dropout) for _ in range(blocks)])
        self.out = nn.Linear(hidden, n_out)

    def forward(self, feats):  # [B, T, in_dim] -> [B, T, n_out]
        return torch.sigmoid(self.out(self.blocks(self.inp(feats))))
