"""Entry point for the Hugging Face Space: same app as src/app.py, served on 0.0.0.0:7860 on CPU."""
import os, sys, threading
from http.server import ThreadingHTTPServer
import numpy as np
sys.path.insert(0, "src")
import app, clap_zero
from common import SR

for n in ("cnn_v4_w48", "cnn_v3_8cls", "cnn_v2"):
    app.model(n)

def warm():
    with app.GPU_LOCK:
        clap_zero._load(); app.clap_head_probs(np.zeros(10 * SR, np.float32))
    print("CLAP ready", flush=True)
threading.Thread(target=warm, daemon=True).start()

port = int(os.environ.get("PORT", 7860)); print("serving on", port, flush=True)
ThreadingHTTPServer(("0.0.0.0", port), app.H).serve_forever()
