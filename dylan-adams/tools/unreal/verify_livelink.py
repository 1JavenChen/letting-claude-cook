"""Runs INSIDE the Unreal editor: check that Live Link data arrives and the face moves.

    python tools/unreal/run_editor.py tools/unreal/verify_livelink.py
    (with `python live.py --file <video> --unreal --no-window` running at the same time)

After the editor settles it logs the Live Link subjects it sees, takes six
screenshots from the stage camera a few seconds apart (Saved/Screenshots/), then quits.
"""

import unreal

state = {"frame": 0, "handle": None}
SHOTS = [1500 + 40 * i for i in range(6)]  # after textures stream in; ~0.5-1 s apart


def log(msg):
    unreal.log(f"[vrface] {msg}")


def camera():
    for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
        if a.get_actor_label() == "VRFace_Camera":
            return a
    return None


def tick(dt):
    state["frame"] += 1
    f = state["frame"]
    if f == 600:
        try:
            subjects = unreal.LiveLinkBlueprintLibrary.get_live_link_subjects(True, True)
            log("live link subjects: " + ", ".join(str(s.subject_name.name) for s in subjects))
        except Exception as e:  # noqa: BLE001
            log(f"could not list subjects: {e}")
    if f == 900:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_play_simulate()
        log("started Simulate (MetaHumans apply Live Link while the world is running)")
    if f in SHOTS:
        cam = camera()
        name = f"vrface_{f}.png"
        unreal.AutomationLibrary.take_high_res_screenshot(1280, 720, name, cam)
        log(f"screenshot {name} from {cam.get_actor_label() if cam else 'viewport'}")
    if f == SHOTS[-1] + 60:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()
        unreal.unregister_slate_post_tick_callback(state["handle"])
        log("verify done")
        unreal.SystemLibrary.quit_editor()


state["handle"] = unreal.register_slate_post_tick_callback(tick)
log("verify_livelink registered")
