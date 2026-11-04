"""Audio encoders: raw 16 kHz audio -> a sequence of feature vectors.

Every encoder exposes `name`, `dim`, and `encode(audio_np, n_frames)`, which
returns float16 features [n_frames, dim] aligned to the 30 fps label grid.

Option A: Wav2Vec2Encoder, a frozen pretrained speech model. Its features are
          cached to disk once (tools/extract_features.py), then only the small
          mouth head is trained.
Option B: a from-scratch encoder (e.g. log-mel spectrogram + conv layers)
          would go here too, but trained end-to-end with the head.
"""

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000


class Wav2Vec2Encoder:
    name = "wav2vec2"
    dim = 768
    hop = 320  # wav2vec2 outputs one vector per 20 ms of audio

    def __init__(self, model_id="facebook/wav2vec2-base-960h", device="cuda", half=False):
        """half=True stores the weights in fp16: ~2x faster per call, for live use."""
        from transformers import Wav2Vec2Model
        from transformers.utils import logging
        logging.set_verbosity_error()  # silence the harmless "unused lm_head" notice
        self.device = device
        self.model = Wav2Vec2Model.from_pretrained(model_id).to(device).eval()
        self.half = half and device == "cuda"
        if self.half:
            self.model.half()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _run(self, seg):
        x = torch.from_numpy(seg).to(self.device)[None]
        x = (x - x.mean()) / (x.std() + 1e-7)  # the model expects normalised input
        if self.half:
            return self.model(x.half()).last_hidden_state[0].float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            return self.model(x).last_hidden_state[0].float()

    def encode(self, audio, n_frames, chunk_s=20.0, context_s=2.0):
        """Encode arbitrarily long audio in chunks, with context on each side so
        chunk boundaries don't affect the features, then resample to n_frames."""
        chunk, ctx = int(chunk_s * SR), int(context_s * SR)
        pieces = []
        for start in range(0, len(audio), chunk):
            lo, hi = max(0, start - ctx), min(len(audio), start + chunk + ctx)
            seg = audio[lo:hi]
            if len(seg) < 800:  # too short for the model's conv front end
                seg = np.pad(seg, (0, 800 - len(seg)))
            feats = self._run(seg)
            skip = (start - lo) // self.hop
            take = int(np.ceil(min(chunk, len(audio) - start) / self.hop))
            pieces.append(feats[skip: skip + take])
        feats = torch.cat(pieces)  # ~50 per second
        feats = F.interpolate(feats.T[None], size=n_frames, mode="linear", align_corners=False)[0].T
        return feats.cpu().numpy().astype(np.float16)


ENCODERS = {"wav2vec2": Wav2Vec2Encoder}
