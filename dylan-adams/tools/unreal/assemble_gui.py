"""Runs INSIDE the Unreal editor (with its window): assemble the MetaHuman, then quit.

    python tools/unreal/run_editor.py tools/unreal/assemble_gui.py

Headless (-run=pythonscript) assembly fails: the eye/skin texture bake runs
over several editor frames, and a headless script never lets frames advance.
Here every step runs from a per-frame tick callback, so the editor keeps
ticking between steps exactly as when you click "Assemble" yourself.
The character must already be auto-rigged (tools/unreal/create_metahuman.py).
"""

import unreal

CHAR = "/Game/VRFace/Avatar"
state = {"frame": 0, "step": "wait", "char": None, "handle": None, "since": 0}
sub = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)


def log(msg):
    unreal.log(f"[vrface] {msg}")


def enum(e, want):
    return next(getattr(e, n) for n in dir(e) if n.isupper() and want in n)


def finish(ok):
    unreal.EditorAssetLibrary.save_directory("/Game/")
    log(f"done, assembly {'succeeded' if ok else 'FAILED'}")
    unreal.unregister_slate_post_tick_callback(state["handle"])
    unreal.SystemLibrary.quit_editor()


def tick(dt):
    state["frame"] += 1
    s = state["step"]
    if s == "wait" and state["frame"] > 120:  # let the editor finish starting up
        ch = unreal.load_asset(CHAR)
        if not ch or not sub.try_add_object_to_edit(ch):
            log("ERROR: character missing or can't be edited - run create_metahuman.py first")
            return finish(False)
        state.update(char=ch, step="preview", since=state["frame"])
        sub.assemble_for_preview(ch)  # builds the preview incl. texture graph bakes
        log("preview assembly started")
    elif s == "preview" and state["frame"] - state["since"] > 600:  # ~10+ s of ticking for bakes
        params = unreal.MetaHumanCharacterEditorBuildParameters()
        params.set_editor_property("pipeline_type", enum(unreal.MetaHumanDefaultPipelineType, "OPTIMIZED"))
        params.set_editor_property("pipeline_quality", enum(unreal.MetaHumanQualityLevel, "HIGH"))
        log("assembling (Optimized, High)...")
        sub.build_meta_human(state["char"], params)
        state.update(step="built", since=state["frame"])
    elif s == "built" and state["frame"] - state["since"] > 300:
        assets = unreal.EditorAssetLibrary.list_assets("/Game/VRFace/", recursive=True)
        ok = any("T_EyeIrisL_N" in a for a in assets) and any("BP_Avatar" in a for a in assets)
        log(f"{len(assets)} assets under /Game/VRFace")
        sub.remove_object_to_edit(state["char"])
        finish(ok)


state["handle"] = unreal.register_slate_post_tick_callback(tick)
log("assemble_gui registered")
