---
title: Which genre is this?
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Music genre from a spectrogram, CNN vs CLAP (Harvard FYS project)
---

# Which genre is this?

> Note (2026-11-04): Hugging Face now requires a PRO subscription for Docker Spaces, so this package is not deployed. The free in-browser version lives in `../web/`.

Drop any audio file (first 30 s are used) or type a song name and the app predicts one of eight genres:
classical, pop, jazz, hip-hop, rock, electronic, country, film score.

Two models trained on 26k Deezer previews for a Harvard first-year seminar:

- **Our CNN v4**: VGG-style CNN on log-mel spectrograms (5.5M params), test 77 %, fresh-artist test 79 %.
- **CLAP embeddings + our classifier**: frozen LAION CLAP audio embeddings + a small MLP, test 79 %, fresh 81 %.

Code and report: https://github.com/Guczyslaw/fys-audio-ml. Runs on a free CPU, so each prediction takes a few seconds.
