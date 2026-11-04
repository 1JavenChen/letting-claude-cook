"""Runs INSIDE Unreal. Dumps the Python API of the MetaHuman / Live Link classes
we may script, to Saved/vrface_api.txt (method names + docstrings)."""

import os

import unreal

CLASSES = ["MetaHumanCharacterEditorSubsystem", "MetaHumanCharacter",
           "MetaHumanCharacterExportBlueprintLibrary", "MetaHumanCollectionBlueprintLibrary",
           "MetaHumanCharacterInstanceBlueprintLibrary", "LiveLinkFaceSourceBlueprint",
           "MetaHumanLiveLinkSourceBlueprint", "LiveLinkBlueprintLibrary", "LiveLinkComponentController",
           "LiveLinkFaceSubjectSettings"]
out = []
for name in CLASSES:
    cls = getattr(unreal, name, None)
    out.append(f"\n===== {name} =====")
    if cls is None:
        out.append("  (missing)")
        continue
    out.append((cls.__doc__ or "").strip()[:3000])
    for attr in sorted(dir(cls)):
        if attr.startswith("_"):
            continue
        try:
            doc = (getattr(cls, attr).__doc__ or "").strip().replace("\n", " ")[:400]
        except Exception:  # noqa: BLE001
            doc = ""
        out.append(f"  {attr}: {doc}")
path = os.path.join(unreal.Paths.project_saved_dir(), "vrface_api.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
unreal.log(f"[vrface] api dump -> {path}")
