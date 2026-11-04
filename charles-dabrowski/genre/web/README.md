---
title: Which genre is this?
emoji: 🎵
colorFrom: purple
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Music genre from a spectrogram, CNN running in your browser
---

# Which genre is this?

Drop any audio file (first 30 s are used) or type a song name. A 5.5M-parameter CNN on log-mel
spectrograms predicts one of eight genres: classical, pop, jazz, hip-hop, rock, electronic, country, film score.
The spectrogram and the network run entirely in the browser (ONNX Runtime Web); nothing is uploaded.

Trained on 26k Deezer previews for a Harvard first-year seminar. Test accuracy 77 %, 79 % on 500 clips from unseen artists.
Code and report: https://github.com/Guczyslaw/fys-audio-ml
