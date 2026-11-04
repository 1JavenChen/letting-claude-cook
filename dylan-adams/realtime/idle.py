"""Natural eyes when the camera can't see them.

Without a camera, the model can only predict your *average* eyelid position
(blinks are unpredictable from sound), which renders as permanently
half-closed, sleepy eyes. Real eyes are open and blink every few seconds, so
when the face isn't seen we replace the eyelid/eye shapes with that: open
eyes plus randomly timed blinks with a realistic close-fast/open-slower shape.
"""

import numpy as np

EYE_PREFIXES = ("eyeBlink", "eyeSquint", "eyeWide", "eyeLook")


class Blinker:
    def __init__(self, names, fps=30, mean_gap_s=3.5, seed=None):
        self.blink = [i for i, n in enumerate(names) if n.startswith("eyeBlink")]
        self.other = [i for i, n in enumerate(names) if n.startswith(EYE_PREFIXES[1:])]
        self.fps = fps
        self.mean_gap = mean_gap_s
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.next_blink = self._gap()
        self.unseen_for = 0.0

    def _gap(self):
        return self.t + 1.5 + self.rng.exponential(self.mean_gap - 1.5)

    def _lid(self):
        """Eyelid closure 0..1 for the current time: ~70 ms to close, ~130 ms to reopen."""
        dt = self.t - self.next_blink
        if dt < 0:
            return 0.0
        if dt < 0.07:
            return dt / 0.07
        if dt < 0.20:
            return 1 - (dt - 0.07) / 0.13
        self.next_blink = self._gap()
        return 0.0

    def __call__(self, w, face_seen):
        """w: blendshape vector for this frame; face_seen: did the camera see the face."""
        self.t += 1 / self.fps
        lid = self._lid()
        if face_seen:
            self.unseen_for = 0.0
            return w
        self.unseen_for += 1 / self.fps
        mix = min(1.0, self.unseen_for / 0.3)  # fade over 0.3 s so losing the face isn't a jump
        w = np.array(w, np.float32)
        w[self.blink] = (1 - mix) * w[self.blink] + mix * lid
        w[self.other] = (1 - mix) * w[self.other]  # relaxed, open, looking ahead
        return w


class Hold:
    """When the camera drops out, keep the last seen expression (brows, cheeks, smile...) for a
    moment, then fade to what the model gives without a camera (your average face) over `fade_s`.
    The mouth's speech shapes stay with the model, which follows the audio; the eyes are
    left to Blinker."""

    def __init__(self, names, mouth_cols, fps=30, hold_s=0.3, fade_s=1.0):
        eyes = np.array([n.startswith(EYE_PREFIXES) for n in names])
        self.cols = ~(np.asarray(mouth_cols, bool) | eyes)
        self.fps, self.hold_s, self.fade_s = fps, hold_s, fade_s
        self.last = None
        self.unseen_for = 0.0

    def __call__(self, w, face_seen):
        if face_seen or self.last is None:
            self.unseen_for = 0.0
            if face_seen:
                self.last = np.array(w, np.float32)
            return w
        self.unseen_for += 1 / self.fps
        keep = float(np.clip(1 - (self.unseen_for - self.hold_s) / self.fade_s, 0, 1))
        w = np.array(w, np.float32)
        w[self.cols] = keep * self.last[self.cols] + (1 - keep) * w[self.cols]
        return w
