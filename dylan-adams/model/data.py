"""Training windows: cached audio features + MediaPipe mouth labels."""

import os

import numpy as np
import torch

from common.blendshapes import MOUTH_SHAPES


class Clip:
    """One processed video. `shapes` picks the label columns (default: the 20
    mouth shapes; "all" = every tracker blendshape). If the .npz has a
    low-resolution tracking pass (tools/degrade_track.py) it is loaded too.

    target="labels" uses MediaPipe's scores; target="labels_hyb" the hybrid geometric
    labels (tools/make_hybrid.py; 51 shapes, no "_neutral"), with labels_hyb_lo as the
    low-res input and frames kept only where both trackers saw the face."""

    def __init__(self, processed_path, feature_path, shapes=None, target="labels"):
        data = np.load(processed_path)
        names = list(data["names"])
        if target != "labels":
            names = [n for n in names if n != "_neutral"]
        shapes = names if shapes == "all" else (shapes or MOUTH_SHAPES)
        cols = [names.index(s) for s in shapes]
        self.names = [str(s) for s in shapes]
        self.name = os.path.basename(processed_path)[:-4]
        self.path = processed_path
        ok_key = "face_ok" if target == "labels" else target.replace("labels", "face_ok")
        self.labels = data[target][:, cols].astype(np.float32)
        self.face_ok = data[ok_key]
        self.keep = data["keep"] & self.face_ok
        if target + "_lo" in data:
            self.labels_lo = data[target + "_lo"][:, cols].astype(np.float32)
            self.face_ok_lo = data[ok_key + "_lo"]
        self.feats = np.load(feature_path, mmap_mode="r")
        assert len(self.feats) == len(self.labels), f"{self.name}: features/labels length mismatch"

    @property
    def audio(self):
        """Raw audio, loaded on demand (fusion training only needs the cached features)."""
        return np.load(self.path)["audio"]

    def __len__(self):
        return len(self.labels)


def load_clips(root, encoder, pattern=None, shapes=None, target="labels"):
    """Clips with cached features; `pattern` = substring the name must contain."""
    processed = os.path.join(root, "data", "processed")
    feat_dir = os.path.join(root, "data", "features", encoder)
    clips = []
    for f in sorted(os.listdir(processed)):
        if pattern and not any(p in f for p in pattern.split(",")):
            continue
        if f.endswith(".npz") and os.path.exists(os.path.join(feat_dir, f[:-4] + ".npy")):
            path = os.path.join(processed, f)
            if target != "labels":
                with np.load(path) as d:
                    if target not in d.files:
                        continue  # not relabelled (yet)
            clips.append(Clip(path, os.path.join(feat_dir, f[:-4] + ".npy"), shapes, target))
    return clips


def split(clip, val_frac):
    """(train_range, val_range): the last val_frac of each clip's usable
    (kept) frames is held out, so b-roll at the end can't empty the val set."""
    kept = np.cumsum(clip.keep)
    cut = int(np.searchsorted(kept, kept[-1] * (1 - val_frac)))
    return (0, cut), (cut, len(clip))


class WindowDataset(torch.utils.data.Dataset):
    """Random fixed-length windows from the given (clip, start, end) ranges,
    only starting where at least `min_keep` of the window has usable labels."""

    def __init__(self, ranges, window, samples, min_keep=0.6, seed=0, with_clip=False):
        self.with_clip = with_clip  # return (clip, start) and let the caller build tensors
        self.window = window
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.starts = []
        for clip, lo, hi in ranges:
            if hi - lo < window:
                continue
            k = np.convolve(clip.keep[lo:hi].astype(float), np.ones(window), "valid") / window
            for s in np.nonzero(k >= min_keep)[0]:
                self.starts.append((clip, lo + s))
        if not self.starts:
            raise ValueError("No usable training windows - is there enough kept data?")

    def __len__(self):
        return self.samples

    def __getitem__(self, _):
        clip, s = self.starts[self.rng.integers(len(self.starts))]
        e = s + self.window
        if self.with_clip:
            return clip, s
        return (torch.from_numpy(np.asarray(clip.feats[s:e], np.float32)),
                torch.from_numpy(clip.labels[s:e]),
                torch.from_numpy(clip.keep[s:e].astype(np.float32)))
