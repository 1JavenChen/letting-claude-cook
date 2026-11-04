"""Runs INSIDE Unreal. Dumps the Python API for ingesting Live Link Face takes
(capture sources, ingest, depth generation) to Saved/vrface_capture_api.txt."""

import os

import unreal

out = []
names = sorted(n for n in dir(unreal) if any(k in n.lower() for k in
               ("capturesource", "ingest", "capturedata", "footage", "depth", "takemetadata", "capturemanager",
                "livelinkfacearchive", "mhaingest")))
out.append("types: " + ", ".join(names))
for n in names:
    cls = getattr(unreal, n)
    if not isinstance(cls, type):
        continue
    out.append(f"\n===== {n} =====\n" + (cls.__doc__ or "").strip()[:2500])
    for attr in sorted(dir(cls)):
        if attr.startswith("_") or attr in ("cast", "get_class", "static_class", "get_default_object"):
            continue
        doc = (getattr(cls, attr).__doc__ or "").strip().replace("\n", " ")[:300]
        if "(" in doc:
            out.append(f"  {attr}: {doc}")
path = os.path.join(unreal.Paths.project_saved_dir(), "vrface_capture_api.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
unreal.log(f"[vrface] capture api -> {path}")
