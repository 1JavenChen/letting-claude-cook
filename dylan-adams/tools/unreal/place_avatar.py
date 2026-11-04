"""Runs INSIDE Unreal. Puts the assembled MetaHuman on the face stage and hooks up Live Link.

    python tools/unreal/make_project.py --run tools/unreal/place_avatar.py   (editor closed)

- spawns /Game/MetaHumans/Avatar/BP_Avatar at the origin facing the camera,
- re-aims the camera and lights at the MetaHuman's actual head position,
- removes the white backdrop (an empty background renders black),
- sets the Live Link subject to "VRFace" (what `live.py --unreal` sends) on
  whatever Live Link properties this MetaHuman Blueprint exposes, and logs them
  all to Saved/vrface_avatar_props.txt.
"""

import os

import unreal

BP = "/Game/MetaHumans/Avatar/BP_Avatar"
LEVEL = "/Game/VRFace/FaceStage"
SUBJECT = "VRFace"
out = []


def log(msg):
    unreal.log(f"[vrface] {msg}")
    out.append(msg)


def props_of(obj):
    """Editor-visible property names, parsed from the object's class docstring."""
    doc = type(obj).__doc__ or ""
    return [line.split("``")[1] for line in doc.splitlines() if line.strip().startswith("- ``")]


levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
levels.load_level(LEVEL)
for a in actors.get_all_level_actors():
    if a.get_actor_label() in ("VRFace_Backdrop", "VRFace_Avatar"):
        actors.destroy_actor(a)

cls = unreal.EditorAssetLibrary.load_blueprint_class(BP)
avatar = actors.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
avatar.set_actor_label("VRFace_Avatar")

# Find the head (the "head" bone) and which way the face points (head -> midpoint of the eyes)
head, eyes = None, None
for comp in avatar.get_components_by_class(unreal.SkeletalMeshComponent):
    log(f"component {comp.get_name()} mesh={comp.get_skeletal_mesh_asset().get_name() if comp.get_skeletal_mesh_asset() else None}")
    if comp.does_socket_exist("head") and (head is None or "face" in comp.get_name().lower()):
        head = comp.get_socket_location("head")
    for l_eye, r_eye in (("FACIAL_L_Eye", "FACIAL_R_Eye"), ("eye_l", "eye_r")):
        if comp.does_socket_exist(l_eye) and comp.does_socket_exist(r_eye):
            eyes = (comp.get_socket_location(l_eye) + comp.get_socket_location(r_eye)) * 0.5
if head is None:
    head = unreal.Vector(0, 0, 165)
fwd = avatar.get_actor_forward_vector()
if eyes is not None:
    d = unreal.Vector(eyes.x - head.x, eyes.y - head.y, 0)
    fwd = d * (1.0 / max(1e-3, (d.x ** 2 + d.y ** 2) ** 0.5))
    head = eyes  # aim at the eyes, not the skull centre
log(f"head/eyes at {head}, face points {fwd} ({'from eye bones' if eyes is not None else 'actor forward'})")
for a in actors.get_all_level_actors():
    label = a.get_actor_label()
    if label == "VRFace_Camera":
        # 50 mm at 1.35 m frames the whole face (~35 cm tall) on a 16:9 Super 35 filmback;
        # aim a little below the eyes so mouth and chin are in the middle of the shot.
        target = head - unreal.Vector(0, 0, 5)
        loc = target + fwd * 135.0
        a.set_actor_location(loc, False, False)
        a.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(loc, target), False)
        cc = a.get_cine_camera_component()
        cc.set_editor_property("current_focal_length", 50.0)
        focus = cc.get_editor_property("focus_settings")
        focus.set_editor_property("manual_focus_distance", 135.0)
        cc.set_editor_property("focus_settings", focus)
    elif label in ("VRFace_Key", "VRFace_Fill", "VRFace_Rim"):
        right = unreal.Vector(-fwd.y, fwd.x, 0)
        off = {"VRFace_Key": fwd * 90 + right * 70 + unreal.Vector(0, 0, 50),
               "VRFace_Fill": fwd * 80 - right * 90 + unreal.Vector(0, 0, 5),
               "VRFace_Rim": fwd * -80 - right * 40 + unreal.Vector(0, 0, 50)}[label]
        a.set_actor_location(head + off, False, False)
        a.set_actor_rotation(unreal.MathLibrary.find_look_at_rotation(head + off, head), False)

# Live Link: log every property mentioning it, set subject/enable flags where they exist
names = props_of(avatar)
for comp in avatar.get_components_by_class(unreal.ActorComponent):
    for n in props_of(comp):
        if any(k in n.lower() for k in ("live", "arkit", "subject")):
            names.append(f"{comp.get_name()}.{n}")
log("live-link-ish properties: " + ", ".join(n for n in names if any(k in n.lower() for k in ("live", "arkit", "subject"))))


def try_set(obj, name, value):
    try:
        obj.set_editor_property(name, value)
        log(f"set {obj.get_name()}.{name} = {value}")
        return True
    except Exception as e:  # noqa: BLE001
        return False


for prop in props_of(avatar):
    low = prop.lower()
    if "subject" in low and "live" in low:
        cur = avatar.get_editor_property(prop)
        log(f"{prop} currently {cur!r} ({type(cur).__name__})")
        if isinstance(cur, unreal.LiveLinkSubjectName):
            try_set(avatar, prop, unreal.LiveLinkSubjectName(SUBJECT))
        else:
            try_set(avatar, prop, SUBJECT) or try_set(avatar, prop, unreal.Name(SUBJECT))
    elif low.startswith(("use_live_link", "b_use_live_link", "use_arkit", "arkit")) or low in ("live_link", "use_livelink"):
        try_set(avatar, prop, True)

# The 5.8 MetaHuman Blueprint's own variables (found in BP_Avatar.uasset): Live Link on,
# ARKit curve mapping on (our packets are ARKit-style), subject = what live.py sends.
for name, value in (("UseLiveLink", True), ("UseARKit", True)):
    try_set(avatar, name, value)
subject_ok = False
for value in (unreal.LiveLinkSubjectName(SUBJECT), unreal.Name(SUBJECT), SUBJECT):
    if try_set(avatar, "LiveLinkSubject", value):
        subject_ok = True
        break
log(f"LiveLinkSubject now {avatar.get_editor_property('LiveLinkSubject')!r}" if subject_ok
    else "ERROR: could not set LiveLinkSubject")

levels.save_current_level()
with open(os.path.join(unreal.Paths.project_saved_dir(), "vrface_avatar_props.txt"), "w") as f:
    f.write("\n".join(out) + "\n\nALL ACTOR PROPS:\n" + "\n".join(props_of(avatar)))
log("saved FaceStage with the avatar")
