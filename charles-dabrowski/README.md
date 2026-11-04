# Charles Dabrowski — teaching neural networks to listen

Live demo: https://guczyslaw-which-genre.static.hf.space · Full repo with history: https://github.com/Guczyslaw/fys-audio-ml

Two audio projects from a Harvard first-year seminar (fall 2026).

| Folder | Project |
|---|---|
| `src/`, `models/`, `results/`, `slides/` | **Speaker counting**: a CNN that counts how many people are talking in an audio clip (synthetic LibriSpeech mixtures, then AMI meetings). |
| `genre/` | **Music genre classification**: CNN on log-mel spectrograms and CLAP embeddings, 8 genres, 26k Deezer previews. See `genre/README.md`. |

Final report: `genre/report/final_report.pdf`. Slides: `slides/` and `genre/slides/`.

Audio data and precomputed feature tensors are not in the repo (tens of GB); `data/demo` and the track CSVs are. Model checkpoints are included.
