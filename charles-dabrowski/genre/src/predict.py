"""Demo: predict the genre of a song.  python src/predict.py --name cnn_v1 "daft punk one more time"  (or a local mp3 path)"""
import argparse, sys, tempfile
from pathlib import Path
import requests, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, GenreCNN, load_audio, log_mel, device, load_model

def fetch_preview(query):
    d = requests.get("https://api.deezer.com/search", params={"q": query, "limit": 1}, timeout=20).json()["data"]
    if not d:
        sys.exit(f"no Deezer result for {query!r}")
    t = d[0]; f = Path(tempfile.gettempdir()) / f"deezer_{t['id']}.mp3"
    f.write_bytes(requests.get(t["preview"], timeout=30).content)
    return f, f"{t['artist']['name']} - {t['title']}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("queries", nargs="+"); ap.add_argument("--name", default="cnn_v1")
    ap.add_argument("--width", type=int, default=32); args = ap.parse_args()
    dev = device(); model = load_model(args.name, dev)
    for q in args.queries:
        f, label = (Path(q), q) if Path(q).exists() else fetch_preview(q)
        x = log_mel(load_audio(f))[None, None].to(dev)
        with torch.no_grad():
            p = model(x).softmax(1)[0].cpu()
        top = p.argsort(descending=True)[:3]
        print(f"{label}\n   " + "   ".join(f"{GENRES[i]} {p[i]:.0%}" for i in top))

if __name__ == "__main__":
    main()
