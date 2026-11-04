"""Runs INSIDE Unreal (editor Python). Builds the face stage level.

    python tools/unreal/make_project.py --run tools/unreal/setup_stage.py

Creates /Game/VRFace/FaceStage: a cine camera framing a head at the origin
(~1.6 m tall MetaHuman), a soft three-point light rig, a dark backdrop, and
fixed exposure - then saves it as the project's startup map. Also writes
Saved/vrface_probe.txt listing which Live Link / MetaHuman classes this engine
exposes to Python, so the next script can be written against what exists.
"""

import os

import unreal

LEVEL = "/Game/VRFace/FaceStage"
HEAD = unreal.Vector(0, 0, 160)  # MetaHuman eye height is ~160 cm above its root


def log(msg):
    unreal.log(f"[vrface] {msg}")


def spawn(cls, loc, rot=(0, 0, 0), label=None):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    a = actors.spawn_actor_from_class(cls, loc, unreal.Rotator(*rot))
    if label:
        a.set_actor_label(label)
    return a


def look_at(src, dst):
    return unreal.MathLibrary.find_look_at_rotation(src, dst)


def build_level():
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
        levels.load_level(LEVEL)
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for a in actors.get_all_level_actors():
            if a.get_actor_label().startswith("VRFace_"):
                actors.destroy_actor(a)
    else:
        levels.new_level(LEVEL)

    # camera: 85 mm portrait lens, 1.1 m in front of the face
    cam_loc = HEAD + unreal.Vector(110, 0, 0)
    cam = spawn(unreal.CineCameraActor, cam_loc, label="VRFace_Camera")
    r = look_at(cam_loc, HEAD)
    cam.set_actor_rotation(r, False)
    cc = cam.get_cine_camera_component()
    cc.set_editor_property("current_focal_length", 85.0)
    focus = cc.get_editor_property("focus_settings")
    focus.set_editor_property("manual_focus_distance", 110.0)
    cc.set_editor_property("focus_settings", focus)

    # three-point lighting: big soft key (front-left, above), dim fill (right), rim (behind)
    for label, offset, intensity, temp, size in [
        ("VRFace_Key", unreal.Vector(90, 70, 50), 12.0, 5600, 80),
        ("VRFace_Fill", unreal.Vector(80, -90, 5), 3.0, 6200, 100),
        ("VRFace_Rim", unreal.Vector(-80, -40, 50), 8.0, 5200, 40),
    ]:
        loc = HEAD + offset
        light = spawn(unreal.RectLight, loc, label=label)
        light.set_actor_rotation(look_at(loc, HEAD), False)
        comp = light.get_editor_property("rect_light_component")
        comp.set_editor_property("intensity", intensity)
        comp.set_editor_property("use_temperature", True)
        comp.set_editor_property("temperature", float(temp))
        comp.set_editor_property("source_width", float(size))
        comp.set_editor_property("source_height", float(size))

    sky = spawn(unreal.SkyLight, unreal.Vector(0, 0, 300), label="VRFace_Ambient")
    sky.get_editor_property("light_component").set_editor_property("intensity", 0.3)

    # dark backdrop behind the head
    wall = spawn(unreal.StaticMeshActor, HEAD + unreal.Vector(-250, 0, -60), label="VRFace_Backdrop")
    # a thin box (visible from both sides, unlike a plane): 10 cm deep, 8 m wide, 6 m tall
    wall.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube"))
    wall.set_actor_scale3d(unreal.Vector(0.1, 8, 6))
    mat = unreal.load_asset("/Engine/BasicShapes/BasicShapeMaterial")
    if mat:
        wall.static_mesh_component.set_material(0, mat)

    # fixed exposure so the face doesn't pump brighter/darker as it moves
    pp = spawn(unreal.PostProcessVolume, HEAD, label="VRFace_PostProcess")
    pp.set_editor_property("unbound", True)
    s = pp.get_editor_property("settings")
    s.set_editor_property("override_auto_exposure_method", True)
    s.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    s.set_editor_property("override_auto_exposure_bias", True)
    s.set_editor_property("auto_exposure_bias", 10.0)
    pp.set_editor_property("settings", s)

    levels.save_current_level()
    log(f"saved {LEVEL}")


def probe():
    out = [f"engine {unreal.SystemLibrary.get_engine_version()}"]
    for name in sorted(dir(unreal)):
        low = name.lower()
        if any(k in low for k in ("livelink", "metahuman", "arkit")):
            out.append(name)
    path = os.path.join(unreal.Paths.project_saved_dir(), "vrface_probe.txt")
    with open(path, "w") as f:
        f.write("\n".join(out))
    log(f"probe: {len(out) - 1} Live Link / MetaHuman / ARKit types -> {path}")


probe()
build_level()
