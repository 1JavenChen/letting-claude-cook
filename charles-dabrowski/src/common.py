"""Shared pieces: feature extraction, dataset, model."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

SR = 16_000
N_MELS = 64
N_CLASSES = 5  # 0..4 speakers (also used for the male / female count heads)

LABEL_KEYS = ["n_speakers", "n_male", "n_female"]

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=512, win_length=400, hop_length=160, n_mels=N_MELS, f_min=50, f_max=8000
)


def log_mel(audio: np.ndarray) -> torch.Tensor:
    """waveform (T,) -> log-mel spectrogram (N_MELS, frames)."""
    x = torch.from_numpy(audio.astype(np.float32))
    return torch.log(_mel(x) + 1e-6)


# Plain (linear-frequency) power spectrogram with the same STFT settings as the mel front end.
# 257 bins at 31.25 Hz; adjacent bins are averaged -> N_STFT_BINS = 128 bins at 62.5 Hz, 0-8 kHz.
N_STFT_BINS = 128
_spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, power=2.0)


def log_stft(audio: np.ndarray) -> torch.Tensor:
    """waveform (T,) -> log linear-frequency spectrogram (N_STFT_BINS, frames)."""
    x = torch.from_numpy(audio.astype(np.float32))
    S = _spec(x)[:-1]                                   # drop the Nyquist bin -> (256, frames)
    S = S.view(N_STFT_BINS, 2, -1).mean(1)              # average adjacent bins -> (128, frames)
    return torch.log(S + 1e-6)


FEATURES = {"mel": log_mel, "stft": log_stft}


def read_labels(root: Path, split: str) -> list[dict]:
    rows = list(csv.DictReader(open(root / "labels.csv")))
    return [r for r in rows if r["split"] == split]


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- dataset
class MixtureDataset(torch.utils.data.Dataset):
    """Loads a split, caches log-mel features on disk so later runs are fast."""

    def __init__(self, root: str | Path, split: str, augment: bool = False, feature: str = "mel"):
        self.root = Path(root)
        self.rows = read_labels(self.root, split)
        self.augment = augment
        self.feature = feature
        extract = FEATURES[feature]

        # "mel" keeps the original cache name so existing caches stay valid
        cache = self.root / (f"features_{split}.pt" if feature == "mel" else f"features_{feature}_{split}.pt")
        if cache.exists():
            self.feats = torch.load(cache)
        else:
            print(f"computing {feature} features for {split} ({len(self.rows)} clips)...", flush=True)
            feats = []
            for r in self.rows:
                audio, sr = sf.read(self.root / r["file"], dtype="float32")
                assert sr == SR
                feats.append(extract(audio).half())
            self.feats = torch.stack(feats)
            torch.save(self.feats, cache)

        self.labels = torch.tensor([[int(r[k]) for k in LABEL_KEYS] for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.feats[i].float()
        if self.augment:
            x = self._spec_augment(x)
        return x.unsqueeze(0), self.labels[i]  # (1, n_bins, frames), (3,)

    @staticmethod
    def _spec_augment(x: torch.Tensor) -> torch.Tensor:
        """Mask a random frequency band and time span (SpecAugment-lite)."""
        x = x.clone()
        n_mels, n_frames = x.shape
        f = np.random.randint(0, max(2, n_mels // 8))  # mask up to 1/8 of the frequency axis
        f0 = np.random.randint(0, n_mels - f)
        x[f0 : f0 + f, :] = x.mean()
        t = np.random.randint(0, 40)
        t0 = np.random.randint(0, n_frames - t)
        x[:, t0 : t0 + t] = x.mean()
        return x


# ----------------------------------------------------------------------------- model
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SpeakerCountCNN(nn.Module):
    """Small CNN on log-mel spectrograms with three classification heads:
    total speaker count, number of male voices, number of female voices."""

    def __init__(self, n_classes: int = N_CLASSES, width: int = 32):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)  # per-input normalisation of the log-mel
        self.features = nn.Sequential(
            conv_block(1, width),
            conv_block(width, width * 2),
            conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({k: nn.Linear(width * 4, n_classes) for k in LABEL_KEYS})

    def forward(self, x):
        h = self.pool(self.features(self.norm(x))).flatten(1)
        h = self.dropout(h)
        return {k: head(h) for k, head in self.heads.items()}
