# SMIRK FLAME labels: setup notes and workarounds

SMIRK (CVPR 2024, https://github.com/georgeretsi/smirk, commit c7de404, MIT) predicts FLAME
parameters (50 expression, jaw pose, 2 eyelids, head pose, camera, 300 shape) from a 224x224 crop.

## Setup
- Code: `third_party/smirk/` (cloned with `core.autocrlf=false`; with autocrlf the protocol-0
  pickle `assets/FLAME_masks/FLAME_masks.pkl` and `head_template.obj` get CRLF-corrupted).
- Weights: `third_party/smirk/pretrained_models/SMIRK_em1.pt` (gdown 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE).
  `quick_install.sh` was NOT run (asks for FLAME password). Training assets skipped.
- Env: `.venv-smirk` from `requirements-smirk.txt` (Python 3.12, torch 2.14.1+cu130, works on sm_120).
- FLAME: put `FLAME2020.zip` at `models/flame/FLAME2020.zip`. On first use `tools/smirk_flame.py`
  unzips it to `third_party/smirk/assets/FLAME2020/` and converts `generic_model.pkl` to
  `generic_model.npz` (or run `python tools/smirk_flame.py convert`). Only needed for meshes /
  rendering; `smirk_track.py` labels without it.

## Workarounds
1. **pytorch3d**: only `src/renderer/renderer.py` imports it (Meshes, load_obj, rasterize_meshes).
   Not used. Encoder (`src/smirk_encoder.py`, timm only) and `src/FLAME/lbs.py` (pure torch) are
   imported directly; rendering is a small OpenCV painter's-algorithm renderer in `smirk_flame.py`.
2. **SMIRK's FLAME.py on numpy 2**: it does `np.float = np.float_` etc. at import, which raises on
   numpy 2. Replaced by `smirk_flame.Flame` (same math: lbs + eyelid offsets + MediaPipe-embedded
   landmarks; FAN/dynamic contour landmarks dropped, so `landmark_embedding.npy`, a pickle with torch
   tensors, is never loaded).
3. **chumpy**: not installed. `generic_model.pkl` is read with a restricted unpickler that maps
   `chumpy.*` and `scipy.sparse.*` classes to inert stubs (taking `.x` / rebuilding csc from
   data/indices/indptr) and refuses any other global except numpy array reconstruction.
   The real FLAME2020 pkl only references chumpy.ch.Ch, scipy.sparse.csc.csc_matrix, builtins.set and
   numpy; it converts cleanly (v_template 5023x3, shapedirs 5023x3x400, posedirs 5023x3x36, f 9976x3).
   Fit check on ai_is_dumb (out/smirk_check.jpg): FLAME's 105 MediaPipe-embedded points land 2.4 px
   (of 224) from MediaPipe's own landmarks on average.
4. **timm 1.0 vs 0.9.16**: state dict loads with all keys matching. `create_backbone` is patched to
   `pretrained=False` so timm doesn't download ImageNet weights the checkpoint overwrites anyway.
5. `torch.load(..., weights_only=True)` for the checkpoint (no pickle code execution).
6. MediaPipe: our `models/face_landmarker.task` + mediapipe 1.1.0 (SMIRK pins 0.10.10); VIDEO mode,
   largest face; detection on a <=640 px copy, crop taken from the full-res frame. Crop matches
   SMIRK's demo (`crop_face`, scale 1.4) via `cv2.warpAffine` instead of skimage.

## Licenses
- SMIRK code: MIT. `src/FLAME/lbs.py` inside it is MPG code (header: use requires a license from MPG).
- FLAME 2020: academic license, non-commercial research/education/art only; no commercial products or
  "production of other artifacts for commercial purposes", no training networks for commercial use,
  no redistribution of the model files. Commercial: ps-license@tue.mpg.de.
  (FLAME 2023 Open is CC BY 4.0 but is a different model; SMIRK was trained with FLAME 2020, so its
  parameters are not meant for the 2023 Open basis.)
- SMIRK_em1.pt was trained with FLAME 2020 (and EMOCA/MICA losses), so treat the weights and the
  labels they produce as non-commercial too unless cleared with MPG.

## FLAME -> our 51 ARKit weights (geometric retargeting)

```powershell
.\.venv-smirk\Scripts\python tools/flame_ict_corr.py        # once: models/flame_ict_corr.npz (+ out/corr_*.jpg)
.\.venv-smirk\Scripts\python tools/smirk_track.py <video> [--crop ...] --match data/processed/<name>.npz --out data/processed/<name>_flame.npz
.\.venv-smirk\Scripts\python tools/flame_to_arkit.py data/processed/<name>_flame.npz --match data/processed/<name>.npz
.\.venv-smirk\Scripts\python tools/flame_labels_check.py data/processed/<name>.npz <video> --flame data/processed/<name>_flame.npz [--crop ...]
.\.venv-smirk\Scripts\python tools/flame_public.py --jobs 3   # all public clips with kept frames, log out/logs/flame_public.log
```

- **Correspondences** (`flame_ict_corr.py`): both neutral heads are rendered (ICT with Head3D, FLAME with
  a small OpenCV renderer) at yaw 0 / +-0.15 with the mouth slightly open (so upper and lower inner
  lips separate), MediaPipe finds its 468 landmarks, each is mapped to the front-facing triangle
  under it (gap points snap to the nearest front vertex) and stored as triangle + barycentrics.
  437 of 468 points are valid on both. On FLAME they agree with SMIRK's own 105-point embedding to
  2.1 mm median; SMIRK's points are only a fallback (its inner-lip points sit on the red of the lips).
- **Solve** (`flame_to_arkit.py`): expression-only FLAME displacement of the points (head pose
  removed), similarity-aligned to ICT, then box-constrained least squares for all frames at once
  (FISTA on GPU, ~1 s per 10k frames) with small L2 (`--lam 0.02`) and temporal smoothing
  (`--mu 0.05`). Lip points x3, eye points x2. Two extra kinds of rows:
  - *visible lip gap*: SMIRK closes mouths with thick rolled lips while FLAME's inner rim stays
    ~2 cm open, so inner-lip material points separate even when the lips touch. Instead the
    uncovered length of 3 vertical image lines through the mouth (camera view, between outer lip
    points 0 and 17) must equal ICT's inner-lip separation there; inner-lip points get weight 0.3.
  - *eye opening fraction*: FLAME's eye opening is ~6 mm vs ICT's ~10 mm; lid pairs must open/close
    by the same fraction of the neutral opening (without this eyeBlink peaked at ~0.3).
  - eyeLook* (gaze) is copied from the clip's MediaPipe labels.
- Output in the clip npz: `labels_flame` [n,51] (= `names` without `_neutral`), `face_ok_flame`,
  `flame_fit_err` (rms point residual, cm); re-timed with `drift_src` like degrade_track.py.

Known limits: SMIRK's resting jaw is ~5 deg open, so jawOpen sits at ~0.45 at rest (with
mouthClose ~0.15 keeping the lips together) vs MediaPipe's ~0.1; pure FLAME jaw rotation leaks
into mouthFrown/jawForward/browInnerUp (0.25 rad -> jawOpen 0.88, frown 0.36, jawForward 0.29,
browInnerUp 0.18) because the two jaw rigs move the face differently; browDown, eyeSquint,
cheekSquint hardly move (FLAME/SMIRK barely model them); smiles come out ~60% of MediaPipe's size.
