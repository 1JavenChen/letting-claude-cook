"""Export a trained GenreCNN to ONNX (dynamic batch + time axis) for the in-browser demo in web/."""
import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load_model, log_mel, load_audio, GENRES
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="cnn_v4_w48"); ap.add_argument("--out", default=None); a = ap.parse_args()
dev = torch.device("cpu"); net = load_model(a.name, dev).eval()
out = Path(a.out or ROOT / "web" / f"{a.name}.onnx"); out.parent.mkdir(exist_ok=True)
x = torch.randn(1, 1, 128, 646)
try:
    torch.onnx.export(net, (x,), str(out), input_names=["mel"], output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"mel": {0: "batch", 3: "time"}, "logits": {0: "batch"}})
    print("exported with the TorchScript exporter")
except Exception as e:
    print("legacy exporter failed:", type(e).__name__, str(e)[:200]); B, T = torch.export.Dim("batch"), torch.export.Dim("time")
    torch.onnx.export(net, (x,), str(out), input_names=["mel"], output_names=["logits"], dynamo=True, dynamic_shapes={"x": {0: B, 3: T}})
    print("exported with the dynamo exporter")
print("size MB", round(out.stat().st_size / 1e6, 1))
import onnxruntime as ort, glob
s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
for f in sorted(glob.glob(str(ROOT / "data/audio/*/*.mp3")))[::3000][:4]:
    m = log_mel(load_audio(f, where="start"))
    for inp in (m[None, None], torch.stack([m[:, i:i + 107] for i in range(0, 646 - 107 + 1, 107)])[:, None]):
        with torch.no_grad(): pt = net(inp).softmax(1).numpy()
        ox = s.run(None, {"mel": inp.numpy()})[0]; ox = np.exp(ox - ox.max(1, keepdims=True)); ox /= ox.sum(1, keepdims=True)
        print(Path(f).parent.name, tuple(inp.shape), "top", GENRES[pt[0].argmax()], f"{pt[0].max():.3f}", "max |torch-onnx|", f"{np.abs(pt - ox).max():.2e}")
