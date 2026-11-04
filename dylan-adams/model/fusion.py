"""Audio + webcam fusion: (speech features, webcam blendshapes) -> blendshapes.

The webcam tracker sees the whole face but is noisy, loses the face, and is
weak on fast lip motion; the audio model nails timing and lip closures but
can't see brows, blinks or silent expressions. This network takes both and
learns, per blendshape, how much to trust each.

Video input per frame: 52 tracker blendshapes plus two flags, `face_seen`
(the tracker found a face) and `mouth_seen` (the mouth isn't hidden). Missing
inputs are zeroed and flagged, so at run time it degrades gracefully to
audio-only when the camera loses you.

The output is a correction on top of the camera: the network starts from the
tracker's value (or your average face where the camera can't see) and learns
how far to move it, in logit space. With little training data this keeps the
camera's detail instead of re-learning it from scratch.

Unlike MouthHead, the convolutions are lopsided: each looks `lookahead`
frames into the future and the rest into the past, so the whole network
needs only blocks * lookahead frames of future (e.g. 4 * 1 = 133 ms) - that is
the delay it adds when running live.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LookaheadConv(nn.Module):
    def __init__(self, ch, kernel=5, lookahead=1, dropout=0.1):
        super().__init__()
        assert 0 <= lookahead < kernel
        self.pad = (kernel - 1 - lookahead, lookahead)
        self.conv = nn.Conv1d(ch, ch, kernel)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, T, C]
        y = self.conv(F.pad(x.transpose(1, 2), self.pad)).transpose(1, 2)
        return x + self.drop(F.gelu(self.norm(y)))


class FusionHead(nn.Module):
    def __init__(self, audio_dim, n_shapes, hidden=256, blocks=4, kernel=5, lookahead=1,
                 dropout=0.1, prior=None, mouth_cols=None, lookahead_total=None):
        """lookahead: future frames each block sees. lookahead_total: instead, spread this many
        future frames over the first blocks (1 each) - e.g. 2 = 67 ms of delay with 4 blocks."""
        super().__init__()
        # prior: average pose (used where unseen); mouth_cols: which shapes `mouth_seen` covers
        self.register_buffer("prior", torch.as_tensor(prior if prior is not None else
                                                      [0.1] * n_shapes, dtype=torch.float32))
        self.register_buffer("mouth_cols", torch.as_tensor(mouth_cols if mouth_cols is not None else
                                                           [False] * n_shapes, dtype=torch.bool))
        self.config = dict(audio_dim=audio_dim, n_shapes=n_shapes, hidden=hidden, blocks=blocks,
                           kernel=kernel, lookahead=lookahead, dropout=dropout,
                           lookahead_total=lookahead_total)
        per_block = ([1 if i < lookahead_total else 0 for i in range(blocks)] if lookahead_total is not None
                     else [lookahead] * blocks)
        self._lookahead_frames = sum(per_block)
        self.audio = nn.Sequential(nn.Dropout(dropout), nn.Linear(audio_dim, hidden))
        self.video = nn.Linear(n_shapes + 2, hidden)
        self.blocks = nn.Sequential(*[LookaheadConv(hidden, kernel, la, dropout) for la in per_block])
        self.out = nn.Linear(hidden, n_shapes)
        nn.init.zeros_(self.out.weight)  # start as "trust the camera"
        nn.init.zeros_(self.out.bias)

    @property
    def lookahead_frames(self):
        return self._lookahead_frames

    def forward(self, audio, video, face_seen, mouth_seen):
        """audio [B,T,A], video [B,T,S] (zeros where unseen), flags [B,T] -> [B,T,S]."""
        v = torch.cat([video, face_seen[..., None], mouth_seen[..., None]], -1)
        x = F.gelu(self.audio(audio) + self.video(v))
        seen = face_seen[..., None].bool() & ~(self.mouth_cols & ~mouth_seen[..., None].bool())
        base = torch.where(seen, video, self.prior).clamp(0.01, 0.99)
        return torch.sigmoid(torch.logit(base) + self.out(self.blocks(x)))
