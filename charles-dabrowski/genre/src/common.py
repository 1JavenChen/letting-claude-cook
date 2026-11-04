"""Shared pieces for the genre classifier: features, dataset, model."""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchaudio, soundfile as sf

GENRES = ["classical", "pop", "jazz", "hiphop", "rock", "electronic", "country", "score"]
G2I = {g: i for i, g in enumerate(GENRES)}
SR, N_MELS, N_FFT, HOP, CLIP_S = 22_050, 128, 2048, 1024, 30
N_FRAMES = CLIP_S * SR // HOP + 1          # 646 frames per 30 s clip
ROOT = Path(__file__).resolve().parents[1]

_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=20, f_max=SR // 2)

def load_audio(path, where="center") -> np.ndarray:
    """mp3/wav -> mono float32 at SR, cropped / zero-padded to CLIP_S seconds.
    where="center" (dataset previews) or "start" (only the first CLIP_S seconds are even decoded)."""
    if where == "start":
        with sf.SoundFile(str(path)) as f:
            x = f.read(frames=CLIP_S * f.samplerate, dtype="float32", always_2d=True); sr = f.samplerate
    else:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    n = CLIP_S * SR
    if len(x) >= n:
        s = (len(x) - n) // 2; x = x[s:s + n]
    else:
        x = torch.nn.functional.pad(x, (0, n - len(x)))
    return x.numpy()

def log_mel(audio: np.ndarray) -> torch.Tensor:
    """waveform (T,) -> log-mel (N_MELS, N_FRAMES)."""
    return torch.log(_mel(torch.from_numpy(audio.astype(np.float32))) + 1e-6)

def load_model(name, dev):
    """Build a GenreCNN whose head size matches the checkpoint (7-class cnn_v1/v2 or 8-class)."""
    sd = torch.load(ROOT / "models" / name / "best.pt", map_location=dev)
    n = sd["head.4.weight"].shape[0]; width = sd["features.0.0.weight"].shape[0]
    m = GenreCNN(n_classes=n, width=width).to(dev); m.load_state_dict(sd); return m.eval()

def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------- dataset
class GenreDataset(torch.utils.data.Dataset):
    """Cached log-mel clips. Training: random crop of `crop` frames + SpecAugment. Eval: full clip."""
    def __init__(self, split, crop=None, augment=False, root=ROOT / "data"):
        d = torch.load(root / f"features_{split}.pt")
        self.x, self.y, self.ids = d["x"], d["y"], d["ids"]
        self.crop, self.augment = crop, augment
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = self.x[i].float()
        if self.crop:
            t0 = np.random.randint(0, x.shape[1] - self.crop + 1)
            x = x[:, t0:t0 + self.crop]
        if self.augment:
            x = x.clone()
            for _ in range(2):
                f = np.random.randint(0, N_MELS // 8); f0 = np.random.randint(0, N_MELS - f); x[f0:f0 + f] = x.mean()
                t = np.random.randint(0, x.shape[1] // 10); t0 = np.random.randint(0, x.shape[1] - t); x[:, t0:t0 + t] = x.mean()
            x = x + 0.1 * np.random.randn()            # random gain (log domain)
        return x.unsqueeze(0), self.y[i]

# ----------------------------------------------------------------------------- model
def conv_block(cin, cout, pool=(2, 2)):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(pool))

class GenreCNN(nn.Module):
    """VGG-style CNN on log-mel spectrograms; global avg+max pooling so any clip length works."""
    def __init__(self, n_classes=len(GENRES), width=32, p=0.3):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.features = nn.Sequential(
            conv_block(1, width), conv_block(width, width * 2), conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 8), conv_block(width * 8, width * 8))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(width * 16, 256), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(256, n_classes))
    def forward(self, x):
        h = self.features(self.norm(x))
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.head(h)
