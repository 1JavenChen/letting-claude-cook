# Speech → Face (VR avatar side project)

Map speech audio to facial animation in real time, starting with a 48-vertex mouth.

## Layout

| Path | What it is |
|---|---|
| `common/blendshapes.py` | Shared vocabulary: 20 ARKit/MediaPipe mouth blendshape names + the animation JSON format |
| `blender/build_mouth.py` | Builds the 48-vertex mouth mesh and its shape keys (runs inside Blender) |
| `blender/animate.py` | Builds the scene, applies an animation JSON, adds audio, saves `.blend` and/or renders `.mp4` |
| `lipsync/rule_based.py` | Phase 1 "brain": hand-written audio → blendshape rules (the baseline to beat) |
| `tools/make_test_audio.ps1` | Makes a test clip with Windows text-to-speech |
| `tools/prepare_data.py` | Phase 2: videos in `data/raw/` → audio + MediaPipe blendshape labels in `data/processed/` |

Any "brain" (rule-based now, neural network in Phase 2) just has to write the same
animation JSON: `{"fps": 30, "names": [...], "frames": [[...], ...]}`.

## Phase 1 quickstart

```powershell
# one-time setup
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt   # includes CUDA 13.0 PyTorch

# audio -> blendshape weights
powershell -ExecutionPolicy Bypass -File tools/make_test_audio.ps1
.\.venv\Scripts\python lipsync/rule_based.py data/test.wav out/test.json

# weights -> animated mouth (.blend to open in Blender, .mp4 to watch)
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender/animate.py -- `
    --anim out/test.json --audio data/test.wav --blend out/test.blend --video out/test.mp4
```

Open `out/test.blend` and press Space to play with sound. Every shape key is in
Properties → Object Data → Shape Keys if you want to sculpt them better by hand.

## Phase 2: training data

Put talking-head videos of yourself in `data/raw/`, then:

```powershell
.\.venv\Scripts\python tools/prepare_data.py
```

For each video this writes `data/processed/<name>.npz` (16 kHz audio, all 52 MediaPipe
blendshapes at 30 fps, and a `keep` mask) plus `<name>.png`, a plot of audio level and
jawOpen with the dropped regions shaded. Frames are dropped when no face is found, or
when there's speech but your mouth isn't moving (probably someone else talking).
If several faces are visible, the largest one is tracked.

Stream layouts: use `--crop x0,y0,x1,y1` (fractions of the frame) to look only at a
webcam box, and `--preview` first to check the crop. `--start`/`--duration` (seconds)
limit processing to part of a video.

**Two webcams, one audio track** (e.g. a stream with a friend on Discord): process the
video once per crop with different tags, then let voiceprints decide who's talking:

```powershell
.\.venv\Scripts\python tools/prepare_data.py --only video.mp4 --crop 0.785,0.035,0.980,0.230 --tag top
.\.venv\Scripts\python tools/prepare_data.py --only video.mp4 --crop 0.785,0.240,0.980,0.435 --tag bottom
.\.venv\Scripts\python tools/pair_speakers.py data/processed/video_top.npz data/processed/video_bottom.npz --listen out/voice_check
```

`pair_speakers.py` groups the audio by voice (SpeechBrain ECAPA voiceprints, no account
needed), assigns each group to whoever's mouth moves during it, and keeps each person's
frames only for their own voice or background audio. Listen to `out/voice_check` to confirm.

**Fetching your own videos:** `python tools/fetch_youtube.py URL --last 15` downloads only
the last 15 minutes (e.g. a fullscreen-webcam ending) into `data/raw/`.

**Check label quality before training:** `python tools/check_labels.py data/processed/<name>.npz`
tests whether fast jaw movement lines up with the speech frame by frame. Small or heavily
compressed facecams (e.g. ~60 px corner boxes in a 720p stream) fail this check: their
labels only say "mouth is active", which isn't enough to learn lip sync from.

## Phase 2: training (Option A, frozen wav2vec2 + small trained head)

```powershell
.\.venv\Scripts\python tools/extract_features.py   # cache wav2vec2 features (once per clip)
.\.venv\Scripts\python train.py                    # -> models/mouth_wav2vec2.pt, out/train_wav2vec2.png
.\.venv\Scripts\python lipsync/neural.py data/test.wav out/test_neural.json
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b -P blender/animate.py -- `
    --anim out/test_neural.json --audio data/test.wav --video out/test_neural.mp4
```

The last 10% of every clip's usable frames is held out. Training prints validation loss next to an
"always-average guess" baseline, and at the end shows per-shape correlation with the
real (MediaPipe) mouth, next to the Phase 1 rule-based jawOpen correlation.

| Path | What it is |
|---|---|
| `model/encoders.py` | Audio encoders (wav2vec2 now; Option B's from-scratch encoder goes here) |
| `model/head.py` | The trained mouth head (~1.2M parameters, 1D convolutions, ±200 ms context) |
| `model/data.py` | Loads clips and samples 4 s training windows |
| `train.py` / `lipsync/neural.py` | Training / inference |

## Phase 3: webcam + voice fusion, live (the MVP)

The webcam tracker (MediaPipe) drives the whole face; the voice sharpens the
mouth and takes over when the camera loses it. `model/fusion.py` starts from
the camera's values and learns corrections, so it never needs to beat the
camera at what the camera already does well.

**Data from edited videos** (talking head mixed with b-roll):

```powershell
$v = "data/raw/<video>.mp4"; $n = "data/processed/<video>.npz"
.\.venv\Scripts\python tools/prepare_data.py --only <video>.mp4 --max-width 1920   # full-res tracking
.\.venv\Scripts\python tools/fix_drift.py $n          # undo slow audio/video drift from editing
.\.venv\Scripts\python tools/sync_segments.py $n --exclude 0-17.5   # keep only shots where the face is talking
.\.venv\Scripts\python tools/kept_sheet.py $n $v out/kept.jpg       # eyeball: is every kept shot you?
.\.venv\Scripts\python tools/degrade_track.py $n $v --width 320     # "weak webcam" input for training
.\.venv\Scripts\python tools/check_labels.py $n       # should now say GOOD
.\.venv\Scripts\python tools/extract_features.py
.\.venv\Scripts\python train_fusion.py --clips <name substrings>   # -> models/fusion.pt, out/fusion_eval.csv
.\.venv\Scripts\python tools/demo_video.py $n $v --start 90 --duration 30   # real | fused | voice-only
```

`train_fusion.py` reports, on held-out footage, how well each part of the face is
recovered by the weak camera alone, camera + voice, voice alone, and camera with
the mouth hidden - per blendshape in `out/fusion_eval.csv`.

**Live:**

```powershell
.\.venv\Scripts\python live.py --list                   # cameras and microphones
.\.venv\Scripts\python live.py --camera 0 --mic Focusrite
.\.venv\Scripts\python live.py --unreal                 # also stream to Unreal Live Link (UDP 11111)
.\.venv\Scripts\python live.py --file <video> --start 82 --no-window --seconds 20   # test on a recording
```

Keys in the preview: `1` fused, `2` camera only, `3` voice only, `m` pretend the mouth is
hidden, `q` quit. Runs at 30 fps with ~17 ms of GPU work per frame and ~150 ms total
latency (133 ms of it is the model's look-ahead; lower `--lookahead` in training to trade
accuracy for latency). Unreal setup: `docs/unreal_livelink.md`.

**3D head preview.** The preview window and `tools/demo_video.py` show a real 3D head
(`--view 3d`, the default; `--view 2d` for the old cartoon). It is the
[ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) Light model (MIT, cloned into
`third_party/ICT-FaceKit/`): 26,719 vertices with 53 ARKit-style shapes. `realtime/head3d.py`
reads the OBJs once into `models/ict_head.npz` (face, head/neck, mouth interior, teeth,
eyeballs; eyelashes and the transparent eye overlays are dropped), maps our 51 names onto
ICT's (`eyeBlinkLeft` -> `eyeBlink_L`, both are the person's left; `browInnerUp` and
`cheekPuff` drive both of ICT's `_L`/`_R` halves), and renders offscreen with moderngl
(smooth normals each frame, wrap-lit skin, key/fill/rim lights). The head follows the
tracked yaw/pitch/roll in the live window (`--view-head` scales it; the demo video has no
pose, so it faces forward). Update + render is ~7 ms (~150 fps).

```powershell
.\.venv\Scripts\python realtime/head3d.py                # cycle through every shape in a window
.\.venv\Scripts\python realtime/head3d.py --bench        # fps
.\.venv\Scripts\python realtime/head3d.py --sheet out/head3d_check.jpg   # single-shape contact sheet
```

| Path | What it is |
|---|---|
| `realtime/head3d.py` | 3D head viewer (ICT-FaceKit Light) driven by the blendshapes |
| `model/fusion.py` | Fusion network (camera + voice → 51 blendshapes, 133 ms look-ahead) |
| `train_fusion.py` | Training with simulated face loss / hidden mouth / weak camera |
| `realtime/` | Live sources (mic, webcam, file playback), Live Link sender, 2D preview face |
| `live.py` | The real-time app |
| `tools/fix_drift.py`, `sync_segments.py`, `kept_sheet.py`, `degrade_track.py` | Data cleaning for edited videos |
| `tools/fetch_datasets.py`, `tools/process_public.py` | Public research datasets (HDTF, RAVDESS, CREMA-D) for pretraining: download, then run the pipeline above on them (`pub_*` clips). Licenses and totals: `docs/datasets.md` (research / non-commercial only) |
| `docs/patent_landscape.md`, `docs/unreal_livelink.md` | Research notes |

## Phase 4: labels that mean the right thing on a 3D face

MediaPipe's blendshape *scores* don't match ARKit semantics: on a relaxed face they read
eyeSquint ~0.3 / mouthPucker ~0.2 (lazy eyes, pre-puckered lips on a MetaHuman), smiles come
without the cheek raise, and frowns/stretches never fire, so a grimace (or disgust) reads as a
smile. Its 478 3D *landmarks* are much more faithful, so labels are now built from geometry:

- `realtime/retarget.py` - removes head pose, measures each landmark's displacement from the
  person's resting face, and solves for the ICT head's ARKit shape weights that reproduce it.
  Then `hybrid()` takes each shape from the source that measures it best: geometry for eyes,
  smile vs grimace, brow lowering; MediaPipe scores for lip protrusion (pucker/funnel - mostly
  depth, which one camera measures poorly), jaw and the rest; and subtracts the resting face.
  On RAVDESS, MediaPipe scores "disgust" as smile 0.36; the hybrid as frown 0.35 + upper-lip
  raise 0.28.
- `tools/retarget_labels.py` (one clip) / `tools/retarget_public.py` (public sets) re-track with
  landmarks, cached as `<clip>_lmk*.npz`; `tools/make_hybrid.py` adds `labels_hyb(_lo)`.
- `tools/label_sheet.py` renders video | head with labels A | head with labels B, to judge label
  sets by eye.
- Train on them: `train_fusion.py --target labels_hyb`; the report now includes the 3D error on
  the head in mm (lips / eyelids). `--lookahead-total 2` trains a 67 ms (vs 133 ms) look-ahead.
- Live: `live.py` computes the same hybrid labels from the webcam (matching the model's target
  automatically). The resting face is estimated from your most relaxed recent moments; press
  **N** while relaxed and looking at the camera to calibrate (saved to `models/calib_live.npz`).

| Path | What it is |
|---|---|
| `realtime/retarget.py` | Landmark retargeting, hybrid labels, resting-face offsets |
| `tools/retarget_labels.py`, `retarget_public.py`, `make_hybrid.py`, `label_sheet.py` | Relabelling + checking |
| `tools/identity_filter.py` | Drop frames whose SMIRK face shape isn't the clip's person |
| `tools/unreal/` | Unreal project/stage/MetaHuman/Live Link scripts (see docstrings) |
