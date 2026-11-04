# Genre from spectrograms

7 classes: classical, pop, jazz, hiphop, rock, electronic, country. Plus an optional 8th class, **score** (film / game scores), see bottom.

## Data (Deezer public API, 30 s previews)
```
seed artists  = chart/{g}/tracks ∪ editorial/{g}/charts ∪ genre/{g}/radios
expand        = BFS over artist/{id}/related
keep artist   if  #albums with genre_id == g  /  #albums  ≥ 0.5
label(track)  = album.genre_id          (must equal g)
caps          : ≤ 12 tracks / artist, ≤ 4 / album, artist in exactly one genre
split         : by artist, stratified per genre, 80 / 10 / 10
```
```
python src/scrape_deezer.py        # tracks.csv + audio/<genre>/<id>.mp3
python src/redownload.py           # preview URLs expire -> refetch
python src/preprocess.py           # log-mel cache + labels.csv
python src/train.py --name cnn_v1 --epochs 30
python src/evaluate.py --name cnn_v1 --split test
python src/predict.py --name cnn_v1 "miles davis so what"
```

## Features
```
mono, 22 050 Hz, 30 s              x ∈ R^{661 500}
STFT n_fft 2048, hop 1024          S ∈ R^{1025 × 646}
mel 128 bins, 20 Hz – 11 025 Hz    M = log(mel(S) + 1e-6) ∈ R^{128 × 646}
train: random 10 s crop (216 frames) + SpecAugment (2 freq, 2 time masks) + gain
eval : full 30 s clip
```

## Model (2.49 M params)
```
BN(1) → 5 × [conv3×3-BN-ReLU, conv3×3-BN-ReLU, maxpool 2]   widths 32-64-128-256-256
      → concat(global avg, global max) ∈ R^512 → dropout → FC 256 → ReLU → dropout → FC 7
loss  : CE, label smoothing 0.1, class weights ∝ (N / n_c)^0.5
optim : AdamW lr 1e-3 wd 1e-2, OneCycle, batch 64
```

## Results (test = unseen artists)
| model  | clips train/val/test | epochs | best val acc | test acc | macro F1 |
|--------|----------------------|--------|--------------|----------|----------|
| cnn_v1 | 8208 / 1019 / 981  (10 208 clips, classical short) | 30 | 0.809 | 0.783 | 0.787 |
| cnn_v2 | 9038 / 1132 / 1068 (11 238 clips, balanced)        | 40 | 0.829 | 0.792 | 0.793 |

Per-class F1, cnn_v2: classical 0.89, pop 0.61, jazz 0.84, hiphop 0.81, rock 0.76, electronic 0.79, country 0.85.
Main confusion: pop → rock 18 %, pop → country 10 %; hiphop / rock / electronic / country → pop 8–13 %.
(cnn_v1 scored 0.796 on the cnn_v2 test split, but that split was redrawn after cnn_v1 trained, so some of its training artists leak in; 0.783 is its clean number.)

Accuracy vs test clip length, cnn_v2 (centre crop, trained on 10 s crops):
| 2 s | 3 s | 5 s | 10 s | 15 s | 30 s |
|-----|-----|-----|------|------|------|
| 0.609 | 0.688 | 0.732 | 0.760 | 0.778 | 0.793 |

Demo (`predict.py`, cnn_v2): Beethoven → classical 94 %, Miles Davis → jazz 91 %, Johnny Cash → country 96 %, Nirvana → rock 94 %, Kendrick Lamar → hiphop 97 %, Taylor Swift → pop 96 %, Billie Eilish → electronic 74 %, Daft Punk → pop 71 % / electronic 22 %, Dolly Parton → pop 60 % / country 34 %.

Figures: `results/dataset/*.png`, `results/curves_cnn_v1_cnn_v2.png`, `results/<model>/confusion_test.png`, `results/cnn_v2/acc_vs_cliplen.png`; metrics in `results/<model>/metrics_test.json`.

## 8th class: film scores (cnn_v3_8cls)
```
source  : Deezer genre 173 "Films/Games", seeded from 65 film/game composers + score playlists + Film Scores radio
          (genre charts skipped: they are pop songs that appear in films); same purity filter and caps
clips   : 1 461 from 137 artists  (train 1176 / val 149 / test 136); the 7 other genres keep identical splits
model   : cnn_v2 weights, new 8-way output layer, fine-tuned 10 epochs, AdamW lr 3e-4 (run stopped in epoch 10, best = epoch 9)
python src/crawl_score.py; python src/preprocess.py
python src/train.py --name cnn_v3_8cls --init cnn_v2 --epochs 10 --lr 3e-4
```
| model       | classes | test n | test acc | macro F1 |
|-------------|---------|--------|----------|----------|
| cnn_v2      | 7       | 1068   | 0.792    | 0.793    |
| cnn_v3_8cls | 8       | 1204   | 0.754    | 0.754    |

Per-class F1, cnn_v3_8cls: classical 0.76, pop 0.61, jazz 0.81, hiphop 0.84, rock 0.76, electronic 0.77, country 0.85, score 0.64.
Cost of the new class: classical recall 0.90 → 0.67 (classical → score confusion); score recall 0.68, mostly confused with classical and electronic.
Demo: Zimmer "Time" → score 90 %, Howard Shore "Concerning Hobbits" → score 73 %, Imperial March (Wiener Philharmoniker) → score 59 % / classical 22 %,
Beethoven → classical 71 %, but a "Baroque Music for Relaxation" Vivaldi → score 61 %: orchestral recordings with modern production look like scores to the model.

## Demo UI
```
python src/app.py          # opens http://127.0.0.1:8765
```
Drop an mp3 / wav / flac / ogg of any length (only the first 30 s are decoded and used) or type a song name (Deezer preview). Shows P(genre), the spectrogram,
and the per-5 s window guess. Model switch: 8 genres (cnn_v3_8cls) or 7 (cnn_v2). m4a / aac are not decodable without ffmpeg.

## Baseline: CLAP zero-shot (no training)
`python src/clap_zero.py` (checkpoint via `CLAP_CKPT`; text prompts in the file). Test split, 8 classes, n = 1 204:
| model | acc | macro F1 |
|---|---|---|
| cnn_v3_8cls (ours) | 0.754 | 0.754 |
| laion/larger_clap_general, zero-shot | 0.694 | 0.696 |
| laion/clap-htsat-unfused, zero-shot | 0.559 | 0.526 |

`laion/larger_clap_music` loads but returns near-constant similarities under transformers 5 (logit scale 1.0): not usable here.

## Fresh test: 504 new clips, 258 artists not in any split (`python src/fresh_test.py 63`)
| model | acc | macro F1 | electronic F1 | country F1 | classical F1 | score F1 |
|---|---|---|---|---|---|---|
| cnn_v3_8cls | 0.778 | 0.778 | 0.73 | 0.80 | 0.80 | 0.71 |
| CLAP general zero-shot | 0.728 | 0.728 | 0.46 | 0.69 | 0.85 | 0.75 |

Agreement 77 %; both right 65 %, only CNN right 12.5 %, only CLAP right 7.5 %, both wrong 15 %. Data in `data/fresh/`, per-clip predictions in `results/fresh_test/predictions.csv`.

## 2026-10-28 session: more data, CLAP head
Data doubled with new artists only (`src/extend_data.py 1650`): 25 943 clips, train 23 458 / val 1 281 / test 1 204 (val and test clips unchanged; fresh-set artists excluded).
CLAP embeddings for every clip (`src/clap_embed.py`) + small classifier (`src/clap_head.py`: 512 → 256 → 8, dropout 0.5, wd 0.1; sweep picked on val).
| model | test acc | test macro F1 | fresh acc | fresh macro F1 |
|---|---|---|---|---|
| cnn_v3_8cls (2.5M, 11k clips) | 0.754 | 0.754 | 0.778 | 0.778 |
| CLAP zero-shot | 0.694 | 0.696 | 0.728 | 0.728 |
| CLAP embeddings + classifier | 0.794 | 0.791 | 0.813 | 0.814 |
| cnn_v4_w48 (5.5M, 23k train clips, 11 epochs) | 0.774 | 0.773 | 0.794 | 0.794 |
| ensemble cnn_v4_w48 + CLAP head (mean of probs) | 0.797 | 0.795 | 0.815 | 0.814 |
`python src/eval_all.py <models…>` scores any model (or `ensemble:a+b`) on both sets -> `results/comparison.json`.

Learning-curve point: same recipe family, 2x data + 2.2x params + 11 epochs (val still rising at the end) = +2.0 pts test, +1.6 pts fresh over cnn_v3.

## Public website (in-browser CNN)

`web/` is a static version of the demo: `src/export_onnx.py` exports cnn_v4_w48 to ONNX (`web/cnn_v4_w48.onnx`, 22 MB),
and `web/index.html` recomputes the log-mel spectrogram in JavaScript (same STFT/mel settings as `common.py`, verified to
3e-4) and runs the network with ONNX Runtime Web. Song search uses Deezer's JSONP API and preview CDN from the browser.
Live: https://huggingface.co/spaces/guczyslaw/which-genre (site: https://guczyslaw-which-genre.static.hf.space) . The CLAP models are too large to run in a browser, so the
site is CNN-only; `space/` holds a Docker version of the full app (needs a paid Hugging Face PRO plan to host).
