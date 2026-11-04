"""Runs INSIDE Unreal (headless is fine): dump Epic's ARKit -> MetaHuman mapping to JSON.

    UnrealEditor-Cmd.exe <uproject> -run=pythonscript -script=tools/unreal/dump_arkit_mapping.py

AS_MetaHuman_ARKit_Mapping holds one frame per ARKit pose; each frame's CTRL_expressions_*
curve values are what that ARKit shape at 1.0 drives on the MetaHuman face. The pose names
(frame order) come from the matching pose asset PA_MetaHuman_ARKit_Mapping.
-> Saved/vrface_arkit_mapping.json {"poses": [...], "curves": {name: [times, values]}, "frame_rate": fps}
"""

import json
import os

import unreal

SAVED = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir())
anim = unreal.load_asset("/Game/MetaHumans/Common/Face/ARKit/AS_MetaHuman_ARKit_Mapping")
pose = unreal.load_asset("/Game/MetaHumans/Common/Face/ARKit/PA_MetaHuman_ARKit_Mapping")
curves = {}
for c in unreal.AnimationLibrary.get_animation_curve_names(anim, unreal.RawCurveTrackTypes.RCT_FLOAT):
    times, values = unreal.AnimationLibrary.get_float_keys(anim, c)
    curves[str(c)] = [list(times), list(values)]
try:
    poses = [str(p) for p in pose.get_pose_names()]
except Exception as e:  # noqa: BLE001
    unreal.log_warning(f"pose names: {e}")
    poses = []
try:
    rate = unreal.AnimationLibrary.get_frame_rate(anim)
except Exception:  # noqa: BLE001
    rate = None
out = os.path.join(SAVED, "vrface_arkit_mapping.json")
with open(out, "w") as f:
    json.dump({"poses": poses, "curves": curves, "frame_rate": str(rate),
               "length_s": unreal.AnimationLibrary.get_sequence_length(anim)}, f)
unreal.log(f"[vrface] {len(poses)} poses, {len(curves)} curves -> {out}")
