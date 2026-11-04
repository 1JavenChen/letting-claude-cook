"""Runs INSIDE Unreal. Creates, auto-rigs and assembles a MetaHuman for the avatar.

    python tools/unreal/make_project.py --run tools/unreal/create_metahuman.py

Steps (the same ones MetaHuman Creator's buttons do):
  1. create /Game/VRFace/Avatar (a MetaHuman Character asset, default face),
  2. auto-rig it with face blendshapes (Epic cloud service, uses your Epic login),
  3. download its skin textures (cloud),
  4. assemble it with the real-time "Optimized" pipeline -> a Blueprint you can
     drop in a level and drive with Live Link.
Everything is logged with a [vrface] prefix; enum choices are logged too.
"""

import unreal

FOLDER, NAME = "/Game/VRFace", "Avatar"


def log(msg):
    unreal.log(f"[vrface] {msg}")


def pick(enum, *prefer):
    """First enum member whose name contains one of `prefer` (in order); logs the options."""
    names = [m for m in dir(enum) if m.isupper()]
    log(f"{enum.__name__} options: {names}")
    for want in prefer:
        for n in names:
            if want in n:
                return getattr(enum, n)
    return getattr(enum, names[0])


def main():
    sub = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    path = f"{FOLDER}/{NAME}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        ch = unreal.load_asset(path)
        log(f"using existing {path}")
    else:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        ch = tools.create_asset(NAME, FOLDER, unreal.MetaHumanCharacter, unreal.MetaHumanCharacterFactoryNew())
        log(f"created {path}")
    if not sub.try_add_object_to_edit(ch):
        log("ERROR: could not open the character for editing")
        return

    rig = unreal.MetaHumanCharacterAutoRiggingRequestParams()
    rig.set_editor_property("rig_type", pick(unreal.MetaHumanRigType, "BLEND", "JOINTS"))
    rig.set_editor_property("blocking", True)
    rig.set_editor_property("report_progress", False)
    log("requesting auto-rig (cloud)...")
    sub.request_auto_rigging(ch, rig)

    tex = unreal.MetaHumanCharacterTextureRequestParams()
    tex.set_editor_property("blocking", True)
    tex.set_editor_property("report_progress", False)
    log("requesting textures (cloud)...")
    sub.request_texture_sources(ch, tex)
    unreal.EditorAssetLibrary.save_loaded_asset(ch)

    ready = sub.can_build_meta_human(ch, True)
    log(f"ready to assemble: {ready}")
    if ready:
        params = unreal.MetaHumanCharacterEditorBuildParameters()
        params.set_editor_property("pipeline_type", pick(unreal.MetaHumanDefaultPipelineType, "OPTIMIZED"))
        params.set_editor_property("pipeline_quality", pick(unreal.MetaHumanQualityLevel, "HIGH"))
        log("assembling...")
        sub.build_meta_human(ch, params)
    sub.remove_object_to_edit(ch)
    unreal.EditorAssetLibrary.save_directory("/Game/")
    found = [a for a in unreal.EditorAssetLibrary.list_assets("/Game/", recursive=True)
             if "BP_" in a or "Avatar" in a]
    log("assets now: " + ", ".join(found[:40]))


main()
