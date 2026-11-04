# Public datasets for pretraining

Compiled 2026-10-16 for pretraining the speech (+ webcam) -> 51-blendshape model on
many speakers before fine-tuning on our own recordings.

> **Research / class use only.** Everything below was fetched for a non-commercial
> class project. Several licenses forbid commercial use (RAVDESS: NC; MEAD and
> HDTF's YouTube sources: no commercial grant at all). **A product version of this
> model must be retrained from data we have a license or recorded consent for**
> (our own recordings, paid/licensed corpora, or consented participants), and the
> weights pretrained on these sets must not ship in it. Keep the data local; don't
> redistribute it or the derived `.npz` files.

## Summary

| Dataset | Status | Speakers | Clips (`data/processed/pub_<ds>_*.npz`) | Tracked | **Kept** (`keep`) | Kept with sound | License |
|---|---|---|---|---|---|---|---|
| HDTF | used | 208 named people (WDA/WRA) + 56 "Radio" videos | 412 (from 347 of 369 YouTube videos) | 936 min | **482 min** | 352 min | CC BY 4.0 annotations; YouTube videos owned by uploaders (research only) |
| RAVDESS | used | 24 actors (12 F / 12 M) | 24 (one per actor, 60 clips each) | 88 min | **60 min** | 24 min | CC BY-NC-SA 4.0 |
| CREMA-D | used | 91 actors | 91 (one per actor, ~82 clips each) | 317 min | **209 min** | 142 min | ODbL 1.0 / DbCL 1.0 |
| MEAD | blocked (Google Drive quota) | 0 | 0 | - | - | - | none stated (research only) |
| **Total** | | **~380** | **527** | **22.4 h** | **12.5 h** | **8.6 h** | |

"Kept" is after `sync_segments.py` (only stretches where the mouth follows the
speech); RAVDESS/CREMA-D "kept" includes the short pauses between joined sentences.
Every clip also has `labels_lo` (320 px re-tracking for fusion training) and cached
wav2vec2 features. Disk: raw downloads + joined videos ~24 GB in `data/public/`,
`.npz` 6.0 GB, features 3.5 GB.

Use them like the other clips, e.g. `python train_fusion.py --clips pub_` (all public)
or `--clips pub_hdtf,pub_cremad`. Hold-out is per clip (last 10% of each), so
speakers are not disjoint between train and validation.

## Datasets

### HDTF (High-Definition Talking Face) — used

- Source: https://github.com/MRzzm/HDTF (annotation files in `HDTF_dataset/`: YouTube
  URL, clip time ranges, crop boxes, per source video; subsets RD / WDA / WRA).
- Access: no form. Videos are fetched from YouTube with yt-dlp (720p-1080p, only the
  annotated time ranges). Dead or unavailable links are skipped and listed in
  `data/public/hdtf/raw/failed.txt`.
- License: the repository states the **HDTF dataset is released under CC BY 4.0**
  (the annotations; the code is GPL-3.0). The **videos themselves belong to their
  YouTube uploaders** (mostly US politicians' public addresses) and are not covered
  by that license: research use only, subject to YouTube's terms.
- Citation:
  ```bibtex
  @inproceedings{zhang2021flow,
    title     = {Flow-Guided One-Shot Talking Face Generation With a High-Resolution Audio-Visual Dataset},
    author    = {Zhang, Zhimeng and Li, Lincheng and Ding, Yu and Fan, Changjie},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages     = {3661--3670},
    year      = {2021}
  }
  ```

### MEAD (Multi-view Emotional Audio-visual Dataset) — not downloaded (Drive quota)

- Source: https://wywu.github.io/projects/MEAD/MEAD.html (the page now 404s; archived
  copy at web.archive.org). Part0 = 48 actors on Google Drive (folder
  `1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j`) and Baidu; one `video.tar` per actor with all 7
  views (6-19 GB each; front view ~1/7 of it, 1920x1080, audio embedded).
- Access: no form, but on 2026-10-16 **every actor's `video.tar` returned Google
  Drive's "Quota exceeded - too many users have downloaded this file"** page, both
  at 09:30 and at 14:20. We respected the quota and did not work around it.
  `python tools/fetch_datasets.py mead --actors 24` retries; it streams each tar and
  keeps only `video/front/<emotion>/level_<n>/*.mp4` (neutral + intensity 3 by default),
  stopping once the front view has passed, then `concat mead` joins them per actor x
  emotion (centre 1080x1080 crop, scaled to 720).
- License: **no license is stated** on the project page or the code repo
  (https://github.com/uniBruce/Mead). Treat as research-only, no redistribution.
- Citation:
  ```bibtex
  @inproceedings{kaisiyuan2020mead,
    author    = {Wang, Kaisiyuan and Wu, Qianyi and Song, Linsen and Yang, Zhuoqian and Wu, Wayne and Qian, Chen and He, Ran and Qiao, Yu and Loy, Chen Change},
    title     = {MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}
  }
  ```

### RAVDESS — used

- Source: https://zenodo.org/records/1188976 (`Video_Speech_Actor_XX.zip`, ~0.5 GB each).
- Access: direct download, no form. We keep only modality `01` (audio+video) speech
  clips: 24 actors x 60 clips (2 sentences, 8 emotions incl. neutral/calm, 2
  intensities), 1280x720, ~3.5 s each.
- License: **CC BY-NC-SA 4.0** (non-commercial, share-alike, attribution).
- Citation:
  ```bibtex
  @article{livingstone2018ravdess,
    title   = {The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
    author  = {Livingstone, Steven R. and Russo, Frank A.},
    journal = {PLoS ONE},
    volume  = {13},
    number  = {5},
    pages   = {e0196391},
    year    = {2018},
    doi     = {10.1371/journal.pone.0196391}
  }
  ```

### CREMA-D — used

- Source: https://github.com/CheyneyComputerScience/CREMA-D (Git LFS; per-file media
  URLs). Mirror: https://gitlab.com/cs-cooper-lab/crema-d-mirror
- Access: direct download, no form. `VideoFlash/*.flv` (480x360, 29.97 fps, audio
  embedded): 91 actors, 12 sentences, 6 emotions, ~2.5 s per clip, 7442 clips.
- License: **Open Database License (ODbL) 1.0**; contents under the Database Contents
  License (DbCL) 1.0. More permissive than the others (attribution + share-alike for
  the database), but these are recordings of identifiable people collected for
  research; get advice before using them in a product.
- Citation:
  ```bibtex
  @article{cao2014cremad,
    title   = {CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
    author  = {Cao, Houwei and Cooper, David G. and Keutmann, Michael K. and Gur, Ruben C. and Nenkova, Ani and Verma, Ragini},
    journal = {IEEE Transactions on Affective Computing},
    volume  = {5},
    number  = {4},
    pages   = {377--390},
    year    = {2014},
    doi     = {10.1109/TAFFC.2014.2336244}
  }
  ```

## How the data was made

```powershell
.\.venv\Scripts\python tools/fetch_datasets.py hdtf --jobs 5          # -> data/public/hdtf/raw/pub_hdtf_<set>_<name>_s<start>.mp4
.\.venv\Scripts\python tools/fetch_datasets.py ravdess                # -> data/public/ravdess/raw/A01..A24/
.\.venv\Scripts\python tools/fetch_datasets.py cremad --actors 91     # -> data/public/cremad/raw/1001..1091/
.\.venv\Scripts\python tools/fetch_datasets.py mead --actors 24       # (blocked by Drive quota on 2026-10-16)
.\.venv\Scripts\python tools/fetch_datasets.py concat ravdess         # -> data/public/ravdess/videos/pub_ravdess_A01.mp4 + .json
.\.venv\Scripts\python tools/fetch_datasets.py concat cremad
.\.venv\Scripts\python tools/process_public.py hdtf --jobs 6 --max-width 960 --fix-drift
.\.venv\Scripts\python tools/process_public.py ravdess --jobs 4 --fix-drift
.\.venv\Scripts\python tools/process_public.py cremad --jobs 4 --fix-drift --min-mouth 30
.\.venv\Scripts\python tools/extract_features.py
```

All steps are resumable. Log: `out/logs/datasets.log`.

- **Short clips are joined** (`concat`) per speaker, because training samples 4 s
  windows: RAVDESS/CREMA-D per actor (~3.5 min each), MEAD per actor x emotion. The
  `.json` next to each joined video lists where every original clip starts. Each clip
  is first re-encoded to exactly N frames at 30 fps with exactly N/30 s of PCM audio:
  a stream-copy join of AAC clips drifted by ~21 ms per clip (encoder priming), i.e.
  >1 s after 60 clips, which made the first CREMA-D test look like bad labels.
- **`process_public.py`** runs, per video: `prepare_data.py` -> `fix_drift.py --mode
  constant` -> `sync_segments.py` -> `degrade_track.py --width 320`, several videos at once.
- **A/V offset:** the clip-level lip-to-sound peak was consistently around -0.13 to
  -0.23 s (mouth leads sound) on HDTF and RAVDESS, vs. the -0.05 s our own clips are
  aligned to by `fix_drift.py`. A forced-keyframe re-download of one HDTF clip had the
  same offset, so it isn't a download artefact; it may simply be the natural lead for
  this measure. To match our own data's convention, each public clip got one constant
  shift to -0.05 s (`fix_drift.py --mode constant`, new; clips without a confident peak
  are left unchanged). The original labels are kept in each `.npz` as
  `labels_unshifted` etc., so this can be undone or changed.
- **CREMA-D faces are small** (480x360 video, median mouth width ~35 px), so
  `sync_segments.py` runs with `--min-mouth 30` there (default 60).

## Label checks

`tools/check_labels.py` on a sample per dataset, after all processing. GOOD needs
peak corr >= 0.15, |lag| <= 0.35 s **and** a peak that stands out >= 3 sd.

| Dataset | Sample | No kept speech | GOOD (strict) | Peak >= 0.15 within 0.35 s | Median peak / lag | Pooled curve over all clips |
|---|---|---|---|---|---|---|
| HDTF | 83 clips (every 5th) | 30 | 27 / 53 (51%) | 53 / 53 (100%) | +0.28 at -0.03 s | +0.26 at -0.03 s (3.7 sd), 314 min |
| RAVDESS | all 24 | 0 | 17 / 24 (71%) | 24 / 24 (100%) | +0.40 at -0.03 s | +0.35 at -0.07 s (3.4 sd), 20 min |
| CREMA-D | 31 clips (every 3rd) | 0 | 10 / 31 (32%) | 31 / 31 (100%) | +0.29 at -0.03 s | +0.29 at -0.03 s (3.1 sd), 99 min |

For comparison, our own `ai_is_dumb` clip pools to +0.27 at -0.07 s. Every
sampled clip with kept speech peaks in the right place; the strict failures are all
on the "stands out" criterion, which short clips (0.5-2 min of speech) and the
3-4 s sentence rhythm of RAVDESS/CREMA-D push down. The **pooled** curves (all clips
of a dataset) pass clearly.

Notes:
- 36% of HDTF clips keep nothing: `sync_segments.py --min-corr 0.25` (per 20 s block)
  is strict for speakers whose jaw barely moves (often the "Radio" subset). Rerunning
  `sync_segments.py` on those `.npz` with a lower `--min-corr` (e.g. 0.15) would keep
  more; `keep_raw` is preserved so this is cheap.
- 3 HDTF downloads came back audio-only and were deleted; 19 HDTF source videos are
  private/removed and 1 failed repeatedly in ffmpeg (`data/public/hdtf/raw/failed.txt`).
