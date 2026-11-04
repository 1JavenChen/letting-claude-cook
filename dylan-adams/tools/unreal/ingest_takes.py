"""Runs INSIDE Unreal. Ingests Live Link Face take folders: unpacks the
Oodle-compressed depth (and the video frames) into ordinary image files.

    python tools/unreal/make_project.py --run tools/unreal/ingest_takes.py
    (takes listed in Saved/vrface_ingest_request.txt, one take folder path per line;
     tools/iphone_take.py writes that file and calls this)

Logs where the output went to Saved/vrface_ingest_result.txt.
"""

import os

import unreal


def log(msg):
    unreal.log(f"[vrface] {msg}")


saved = unreal.Paths.project_saved_dir()
with open(os.path.join(saved, "vrface_ingest_request.txt")) as f:
    takes = [line.strip() for line in f if line.strip()]

settings = unreal.get_default_object(unreal.CaptureManagerEditorSettings)
lines = []
try:
    lines.append("media_directory: " + str(settings.get_editor_property("media_directory").path))
except Exception as e:  # noqa: BLE001
    lines.append(f"media_directory: unknown ({e})")
lines.append("image formats: " + ", ".join(n for n in dir(unreal.CaptureManagerImageFormat) if n.isupper()))

params = unreal.CaptureManagerConversionParams()
for take in takes:
    log(f"ingesting {take}")
    # a failed earlier attempt leaves half-made assets behind, and ingest refuses to overwrite them
    old = "/Game/CaptureManager/Imports/" + os.path.basename(take.rstrip("/\\")).split("_", 1)[1]
    if unreal.EditorAssetLibrary.does_directory_exist(old):
        unreal.EditorAssetLibrary.delete_directory(old)
        log(f"deleted leftover {old}")
    data, err = unreal.CaptureManagerIngestBlueprintLibrary.ingest_live_link_face_sync(take, params)
    if data is None:
        lines.append(f"FAILED {take}: {err}")
        log(f"failed: {err}")
        continue
    lines.append(f"OK {take} -> {data.get_path_name()}")
    for prop in ("image_sequences", "depth_sequences", "audio_tracks", "camera_calibrations"):
        try:
            for item in data.get_editor_property(prop):
                src = item.get_editor_property("file_path") if hasattr(item, "get_editor_property") else item
                lines.append(f"  {prop}: {item.get_path_name()}")
                try:
                    lines.append(f"    files: {item.get_editor_property('sequence_path').path}")
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            lines.append(f"  {prop}: ({e})")
unreal.EditorAssetLibrary.save_directory("/Game/")
with open(os.path.join(saved, "vrface_ingest_result.txt"), "w") as f:
    f.write("\n".join(lines))
log("ingest done")
