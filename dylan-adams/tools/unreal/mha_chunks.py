"""Run MetaHuman Animator (depth) over a whole take in 1-minute chunks, one editor launch each.

    .venv\\Scripts\\python tools/unreal/mha_chunks.py --start 5400 --end 75664

A 3-minute depth chunk ran the machine out of memory; 1 minute peaks at ~10.5 GB.
Chunks already exported (Saved/vrface_mha_depth_<start>.json) are skipped, so this resumes.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVED = os.path.join(ROOT, "unreal", "VRFace", "Saved")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=5400)
    p.add_argument("--end", type=int, default=75664)
    p.add_argument("--chunk", type=int, default=3600)
    p.add_argument("--capture", default="/Game/CaptureManager/Imports/DefaultSlate_11/CD_DefaultSlate_11")
    p.add_argument("--identity", default="MHI_DefaultSlate_11")
    args = p.parse_args()
    for a in range(args.start, args.end, args.chunk):
        tag = f"depth_{a}"
        if os.path.exists(os.path.join(SAVED, f"vrface_mha_{tag}.json")):
            continue
        req = {"capture": args.capture, "identity": args.identity, "neutral_frame": 0,
               "start": a, "end": min(a + args.chunk, args.end), "tag": tag, "mode": "depth"}
        with open(os.path.join(SAVED, "vrface_mha_request.json"), "w") as f:
            json.dump(req, f)
        t = time.time()
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "unreal", "run_editor.py"),
                        os.path.join(ROOT, "tools", "unreal", "mha_take_gui.py")], capture_output=True)
        ok = os.path.exists(os.path.join(SAVED, f"vrface_mha_{tag}.json"))
        print(f"{tag}: {'ok' if ok else 'FAILED'} in {(time.time() - t) / 60:.1f} min", flush=True)
        if not ok:
            return 1


if __name__ == "__main__":
    sys.exit(main())
