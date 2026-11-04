"""Jitter removal for blendshape curves.

OneEuro: the standard adaptive low-pass for live tracking (Casiez et al. 2012).
It smooths hard while a value is nearly still (where jitter is most visible)
and opens up when it moves fast (lip closures, blinks), so it removes shake
without making the mouth lag or mush.

zero_phase(): the same filter run forwards and backwards over a recording,
for cleaning training labels offline (no lag at all, since it can see ahead).
"""

import numpy as np

# (min_cutoff Hz, beta) per face region: lower min_cutoff = smoother when still;
# higher beta = opens up faster on quick moves. Blinks must stay snappy.
REGION_PARAMS = {
    "eyeBlink": (3.0, 0.5),
    "eye": (1.5, 0.3),
    "brow": (1.0, 0.3),
    "jaw": (2.5, 0.8),
    "mouth": (2.5, 0.8),
    "": (1.5, 0.3),  # everything else
}


def params_for(names):
    out = []
    for n in names:
        for prefix, prm in REGION_PARAMS.items():
            if n.startswith(prefix):
                out.append(prm)
                break
    return np.array(out, np.float32).T  # (min_cutoff[S], beta[S])


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, names, fps=30.0, d_cutoff=1.0, scale=1.0):
        """scale < 1 smooths more, > 1 less (multiplies every min_cutoff)."""
        self.min_cutoff, self.beta = params_for(names)
        self.min_cutoff = self.min_cutoff * scale
        self.dt = 1.0 / fps
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = None

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / self.dt
        self.dx = self.dx + _alpha(self.d_cutoff, self.dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + _alpha(cutoff, self.dt) * (x - self.x)
        return self.x.copy()


def filter_sequence(X, names, fps=30.0, scale=1.0):
    f = OneEuro(names, fps, scale=scale)
    return np.stack([f(x) for x in X])


def zero_phase(X, names, fps=30.0, scale=1.0):
    """Forward + backward pass, averaged: smooth with no time shift (offline only)."""
    fwd = filter_sequence(X, names, fps, scale)
    bwd = filter_sequence(X[::-1], names, fps, scale)[::-1]
    return (fwd + bwd) / 2
