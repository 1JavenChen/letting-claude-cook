"""Tiny local web UI: drop an audio file (mp3 / wav / flac / ogg) or type a song name, get the predicted genre.
Run:  python src/app.py   then open http://127.0.0.1:8765"""
import base64, io, json, sys, tempfile, webbrowser, threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np, requests, torch, soundfile as sf, re
from mutagen import File as MutagenFile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, GENRES, SR, HOP, load_audio, log_mel, load_model, device
import clap_zero

PORT = 8765
GPU_LOCK = threading.Lock()          # MPS crashes if two threads submit work at once
LABELS = {g: g.replace("hiphop", "hip-hop").replace("score", "film score") for g in GENRES}
dev = device(); MODELS = {}
def model(name):
    if name not in MODELS: MODELS[name] = load_model(name, dev)
    return MODELS[name]

NAMES = {"cnn_v3_8cls": "Our CNN (8 genres)", "cnn_v4_w48": "Our CNN v4 (2x data, 5.5M)", "cnn_v2": "Our CNN (7 genres)",
         "clap": "CLAP zero-shot (LAION)", "clap_head": "CLAP embeddings + our classifier", "ensemble": "Ensemble: CNN + CLAP head"}
_head = {}
def clap_head_probs(a):
    """our small classifier trained on frozen CLAP embeddings (models/clap_head/best.pt)"""
    import torch.nn as nn, torchaudio
    if not _head:
        ck = torch.load(ROOT / "models" / "clap_head" / "best.pt"); h = ck["hidden"]; K = len(GENRES)
        net = nn.Sequential(nn.Dropout(0), nn.Linear(512, h), nn.GELU(), nn.Dropout(0), nn.Linear(h, K)) if h else nn.Sequential(nn.Dropout(0), nn.Linear(512, K))
        net.load_state_dict(ck["state"]); net.eval(); _head.update(net=net, mu=ck["mu"], sd=ck["sd"])
    st = clap_zero._load(); a48 = torchaudio.functional.resample(torch.from_numpy(a), SR, 48_000).numpy(); n = 480_000
    chunks = [a48[j:j + n] for j in range(0, max(1, len(a48) - n + 1), n)] or [a48]
    with torch.no_grad():
        e = clap_zero._emb(st["model"].get_audio_features(**st["proc"](audio=chunks, sampling_rate=48_000, return_tensors="pt").to(st["dev"]))).float().cpu().mean(0)
        return _head["net"](((e - _head["mu"]) / _head["sd"])[None]).softmax(1)[0].numpy()

def predict_one(a, m, name):
    """audio (22 kHz) + its log-mel -> probs, per-window labels, spectrogram png (base64)."""
    if name in ("clap", "clap_head"):                    # CLAP-based: 10 s windows (CLAP's native input length)
        fn = clap_zero.probs if name == "clap" else clap_head_probs
        K = len(GENRES); p = fn(a); step = 10
        pw = np.stack([fn(a[i * SR:(i + step) * SR]) for i in range(0, 30, step)])
    elif name == "ensemble":                             # mean of CNN (best available) and CLAP-head probabilities
        cnn = "cnn_v4_w48" if (ROOT / "models" / "cnn_v4_w48" / "best.pt").exists() else "cnn_v3_8cls"
        K = len(GENRES); net = model(cnn); step = 10
        with torch.no_grad():
            pc = net(m[None, None].to(dev)).softmax(1)[0].cpu().numpy()
            win = step * SR // HOP; wins = [m[:, i:i + win] for i in range(0, m.shape[1] - win + 1, win)]
            pcw = net(torch.stack(wins)[:, None].to(dev)).softmax(1).cpu().numpy()
        p = (pc + clap_head_probs(a)) / 2
        pw = (pcw + np.stack([clap_head_probs(a[i * SR:(i + step) * SR]) for i in range(0, 30, step)])) / 2
    else:
        net = model(name); K = net.head[-1].out_features; step = 5
        with torch.no_grad():
            p = net(m[None, None].to(dev)).softmax(1)[0].cpu().numpy()
            win = step * SR // HOP; wins = [m[:, i:i + win] for i in range(0, m.shape[1] - win + 1, win)]
            pw = net(torch.stack(wins)[:, None].to(dev)).softmax(1).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 2.4)); ax.imshow(m, origin="lower", aspect="auto", cmap="magma", extent=[0, 30, 0, 128])
    for i, row in enumerate(pw):
        ax.text(i * step + step / 2, 118, LABELS[GENRES[row.argmax()]], ha="center", color="white", fontsize=9, fontweight="bold")
        ax.axvline(i * step, color="white", lw=.4, alpha=.5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mel band"); ax.set_title(f"{NAMES[name]}: guess per {step} s window", fontsize=10); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    probs = sorted([(LABELS[GENRES[i]], float(p[i])) for i in range(K)], key=lambda t: -t[1])
    return {"name": NAMES[name], "probs": probs, "spectrogram": base64.b64encode(buf.getvalue()).decode()}

def analyse(path, name="cnn_v3_8cls", label="", cover=None):
    a = load_audio(path, where="start"); m = log_mel(a)
    best_cnn = "cnn_v4_w48" if (ROOT / "models" / "cnn_v4_w48" / "best.pt").exists() else "cnn_v3_8cls"
    names = [best_cnn, "clap_head"] if name == "compare" else [name]
    results = [predict_one(a, m, n) for n in names]
    wav = io.BytesIO(); sf.write(wav, a, SR, format="WAV", subtype="PCM_16")   # the full 30 s that was analysed
    return {"label": label or Path(path).name, "probs": results[0]["probs"], "results": results,
            "audio": base64.b64encode(wav.getvalue()).decode(), "cover": cover}

def deezer_preview(query):
    d = requests.get("https://api.deezer.com/search", params={"q": query, "limit": 1}, timeout=20).json().get("data", [])
    if not d: raise ValueError(f"no Deezer result for {query!r}")
    t = d[0]; f = Path(tempfile.gettempdir()) / f"deezer_{t['id']}.mp3"
    f.write_bytes(requests.get(t["preview"], timeout=30).content)
    return f, f"{t['artist']['name']} – {t['title']}", t["album"].get("cover_big") or t["album"].get("cover_medium")

def cover_for_upload(path, filename):
    """Embedded ID3 / FLAC picture if present, else the cover of the best Deezer match for the file name."""
    try:
        tags = MutagenFile(str(path))
        pics = getattr(tags, "pictures", None) or [v for k, v in (tags.tags or {}).items() if k.startswith("APIC")]
        if pics:
            pic = pics[0]; return f"data:{pic.mime};base64," + base64.b64encode(pic.data).decode()
        meta = " ".join(str(tags.tags.get(k, [""])[0]) for k in ("TPE1", "TIT2", "artist", "title") if tags.tags and k in tags.tags).strip()
    except Exception:
        meta = ""
    q = meta or re.sub(r"[\-_]+", " ", Path(filename).stem)
    q = re.sub(r"\b(official|video|audio|lyrics|hd|4k|\d{3,4}p)\b", " ", q, flags=re.I).strip()
    try:
        d = requests.get("https://api.deezer.com/search", params={"q": q, "limit": 1}, timeout=10).json().get("data", [])
        if d: return d[0]["album"].get("cover_big") or d[0]["album"].get("cover_medium")
    except Exception:
        pass
    return None

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Genre from spectrogram</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#1f1d2b;background:#fff}
h1{font-size:30px;margin:0 0 4px}.sub{color:#6b687a;margin-bottom:24px}
.drop{border:2px dashed #b9b2d0;border-radius:14px;padding:34px;text-align:center;color:#4b2c7f;background:#f8f6fc;cursor:pointer}
.drop.over{background:#ece6f7}.row{display:flex;gap:10px;margin:14px 0}
input[type=text]{flex:1;padding:10px 12px;border:1px solid #cfc9df;border-radius:8px;font-size:15px}
button{padding:10px 16px;border:0;border-radius:8px;background:#4b2c7f;color:#fff;font-size:15px;cursor:pointer}
select{padding:8px;border-radius:8px;border:1px solid #cfc9df}
.bar{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:15px}.bar .n{width:90px;text-align:right;color:#6b687a}
.bar .b{height:22px;background:#d8d1ea;border-radius:5px}.bar.top .b{background:#e8702a}.bar .v{width:50px}
#res{margin-top:22px}#res img.spec{width:100%;border-radius:10px;margin-top:16px}
.np{display:flex;gap:20px;align-items:center;background:#f8f6fc;border-radius:16px;padding:18px;margin:4px 0 18px}
.np .cv{width:150px;height:150px;border-radius:12px;object-fit:cover;flex:none;background:linear-gradient(135deg,#4b2c7f,#e8702a);box-shadow:0 6px 18px rgba(75,44,127,.25)}
.np .info{flex:1;min-width:0}.np .t{font-size:20px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.np .g{font-size:15px;color:#6b687a;margin:2px 0 14px}.np .g b{color:#e8702a;font-size:17px}
.pl{display:flex;align-items:center;gap:12px}.pl button{width:46px;height:46px;border-radius:50%;padding:0;font-size:18px;display:flex;align-items:center;justify-content:center}
.pl .tr{flex:1;height:8px;background:#ddd6ec;border-radius:4px;cursor:pointer;position:relative}.pl .tr i{display:block;height:100%;width:0;background:#4b2c7f;border-radius:4px}
.pl .tm{font-variant-numeric:tabular-nums;color:#6b687a;font-size:13px;width:70px;text-align:right}.big{font-size:26px;margin:10px 0 2px}.big b{color:#e8702a}
.err{color:#b3261e;margin-top:14px}
.cols{display:flex;gap:22px}.col{flex:1;min-width:0}.col .bar{font-size:13px}.col .bar .n{width:70px}.col .bar .v{width:40px}.col h3{margin:6px 0 8px;font-size:16px}.col .bar .b{max-width:100%}.verdict{font-size:15px;color:#6b687a;margin:0 0 10px}.verdict b{color:#1f1d2b}
</style></head><body>
<h1>Which genre is this?</h1><div class="sub">Drop any mp3 / wav / flac / ogg, any length: only the first 30 seconds are used. Or type a song name.</div>
<div class="drop" id="drop">Drop an audio file here or click to choose (first 30 s are analysed)<input type="file" id="file" accept="audio/*" hidden></div>
<div class="row"><input type="text" id="q" placeholder="…or a song name, e.g. miles davis so what"><button id="go">Predict</button>
<select id="model"><option value="compare">Compare: our CNN vs CLAP classifier</option><option value="ensemble">Ensemble: CNN + CLAP head</option><option value="clap_head">CLAP embeddings + our classifier</option><option value="cnn_v4_w48">Our CNN v4 (2x data)</option><option value="cnn_v3_8cls">Our CNN, 8 genres</option><option value="cnn_v2">Our CNN, 7 genres</option></select></div>
<div id="res"></div>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),res=document.getElementById('res');
drop.onclick=()=>file.click();file.onchange=()=>file.files[0]&&send(file.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');e.dataTransfer.files[0]&&send(e.dataTransfer.files[0])};
document.getElementById('go').onclick=()=>{const q=document.getElementById('q').value.trim();if(q)send(null,q)};
document.getElementById('q').onkeydown=e=>{if(e.key==='Enter')document.getElementById('go').click()};
async function send(f,q){res.innerHTML='<div class="sub">Listening…</div>';const m=document.getElementById('model').value;let r;
 try{if(f){const fd=new FormData();fd.append('file',f);fd.append('model',m);r=await fetch('/predict',{method:'POST',body:fd});}
 else r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q,model:m})});
 const d=await r.json();if(d.error)throw d.error;show(d);}catch(e){res.innerHTML='<div class="err">'+e+'</div>'}}
function show(d){const top=d.probs[0];const W=d.results.length>2?170:d.results.length>1?260:560;let h='';
 if(d.results.length>1){const a=d.results[0].probs[0][0],b=d.results[1].probs[0][0];
  const tops=d.results.map(r=>r.probs[0][0]);
  h+='<div class="verdict">'+(tops.every(t=>t===tops[0])?'All models say <b>'+tops[0]+'</b>.':d.results.map(r=>'<b>'+r.name+'</b>: <b>'+r.probs[0][0]+'</b>').join(' · '))+'</div>';}
 h+='<div class="cols">';
 d.results.forEach(r=>{h+='<div class="col">'+(d.results.length>1?'<h3>'+r.name+'</h3>':'');
  r.probs.forEach(([g,p],i)=>{h+='<div class="bar'+(i==0?' top':'')+'"><span class="n">'+g+'</span><div class="b" style="width:'+(p*W)+'px"></div><span class="v">'+Math.round(p*100)+' %</span></div>'});
  h+='</div>'});
 h+='</div>';
 d.results.forEach(r=>{h+='<img class="spec" src="data:image/png;base64,'+r.spectrogram+'">'});
 const cv=d.cover?'<img class="cv" src="'+d.cover+'">':'<div class="cv"></div>';
 res.innerHTML='<div class="np">'+cv+'<div class="info"><div class="t">'+d.label+'</div><div class="g">sounds like <b>'+top[0]+'</b> · '+Math.round(top[1]*100)+' % · 30 s clip</div>'+
  '<div class="pl"><button id="pp">▶</button><div class="tr" id="tr"><i id="pr"></i></div><span class="tm" id="tm">0:00 / 0:30</span></div></div></div>'+h;
 const au=new Audio('data:audio/wav;base64,'+d.audio),pp=document.getElementById('pp'),pr=document.getElementById('pr'),tm=document.getElementById('tm'),tr=document.getElementById('tr');
 const fmt=t=>'0:'+String(Math.floor(t)).padStart(2,'0');
 pp.onclick=()=>{au.paused?au.play():au.pause()};au.onplay=()=>pp.textContent='❚❚';au.onpause=()=>pp.textContent='▶';
 au.ontimeupdate=()=>{pr.style.width=(au.currentTime/(au.duration||30)*100)+'%';tm.textContent=fmt(au.currentTime)+' / '+fmt(au.duration||30)};
 au.onended=()=>{pp.textContent='▶';pr.style.width='0%'};tr.onclick=e=>{au.currentTime=(e.offsetX/tr.clientWidth)*(au.duration||30)};
 if(window._au){window._au.pause()}window._au=au;}
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        b = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/search":
                d = json.loads(body); f, label, cover = deezer_preview(d["q"])
                with GPU_LOCK: res = analyse(f, d.get("model", "cnn_v3_8cls"), label, cover)
                self._json(res)
            else:
                msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
                up = parts["file"]; name = up.get_filename() or "clip"; mdl = parts["model"].get_content().strip() if "model" in parts else "cnn_v3_8cls"
                f = Path(tempfile.gettempdir()) / ("upload_" + Path(name).name); f.write_bytes(up.get_payload(decode=True))
                cover = cover_for_upload(f, name)
                with GPU_LOCK: res = analyse(f, mdl, Path(name).stem, cover)
                self._json(res)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)
    def log_message(self, *a): pass

if __name__ == "__main__":
    model("cnn_v3_8cls")
    url = f"http://127.0.0.1:{PORT}"; print("serving", url, flush=True)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
