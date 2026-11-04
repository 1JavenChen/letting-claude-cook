"""Runs INSIDE the Unreal editor (with its window): MetaHuman Animator on an ingested iPhone depth take, then quit.

    python tools/unreal/run_editor.py tools/unreal/mha_take_gui.py

Reads Saved/vrface_mha_request.json:
  {"capture": "/Game/CaptureManager/Imports/DefaultSlate_11/CD_DefaultSlate_11",
   "identity": "MHI_DefaultSlate_11", "neutral_frame": 48954,   # only used if the identity doesn't exist yet
   "start": 40000, "end": 50800, "tag": "test",                   # processing range (video frames, end excluded)
   "mode": "depth"}  # or "mono": video only (depth needs the MetaHuman Depth Processing plugin from Fab)
Steps (Epic's sample scripts in the MetaHuman plugin, chained):
  1. identity from the neutral frame (track, conform, Mesh to MetaHuman auto-rig - needs the
     Epic login - then "Prepare for Performance")
  2. performance (depth footage) over the range, exported as an AnimSequence
  3. every float curve of that AnimSequence -> Saved/vrface_mha_<tag>.json
Progress and errors go to Saved/vrface_mha_result.txt.
"""

import json
import os
import time

import unreal

SAVED = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir())
STORE = "/Game/MHA"
state = {"frame": 0, "handle": None, "stage": "start", "t0": time.time(), "req": None, "deadline": None}


def log(msg):
    unreal.log("[vrface] " + msg)
    with open(os.path.join(SAVED, "vrface_mha_result.txt"), "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def quit_editor():
    state["stage"] = "done"
    unreal.SystemLibrary.quit_editor()


def make_identity(req):
    import utility_functions_MH  # Epic's helpers, on the plugin's Python path
    capture = unreal.load_asset(req["capture"])
    ident = utility_functions_MH.create_or_recreate_asset(req["identity"], STORE, unreal.MetaHumanIdentity,
                                                          unreal.MetaHumanIdentityFactoryNew())
    face = ident.get_or_create_part_of_class(unreal.MetaHumanIdentityFace)
    if not ident.is_logged_in_to_service():
        log("not logged in to the MetaHuman service: requesting login")
        ident.log_in_to_auto_rig_service()
    pose = unreal.new_object(type=unreal.MetaHumanIdentityPose, outer=face)
    face.add_pose_of_type(unreal.IdentityPoseType.NEUTRAL, pose)
    pose.set_capture_data(capture)
    pose.fit_eyes = True
    pose.load_default_tracker()
    frame, _ = pose.add_new_promoted_frame()
    frame.is_front_view = True
    frame.set_navigation_locked(True)
    frame.frame_number = req["neutral_frame"]
    if not unreal.PromotedFrameUtils.initialize_contour_data_for_footage_frame(pose, frame):
        raise RuntimeError("couldn't initialise contour data for the neutral frame")
    cam = pose.get_editor_property("camera")
    image = unreal.PromotedFrameUtils.get_image_path_for_frame(capture, cam, req["neutral_frame"], True, pose.timecode_alignment)
    depth = unreal.PromotedFrameUtils.get_image_path_for_frame(capture, cam, req["neutral_frame"], False, pose.timecode_alignment)
    log(f"neutral frame image {image}")
    size, pixels = unreal.PromotedFrameUtils.get_promoted_frame_as_pixel_array_from_disk(image)
    ident.set_blocking_processing(True)
    ident.start_frame_tracking_pipeline(pixels, size.x, size.y, depth, pose, frame, False)
    if not unreal.MetaHumanIdentity.handle_error(face.conform(), True):
        raise RuntimeError("face conform failed")
    log("face conformed to the neutral frame")
    ident.get_or_create_part_of_class(unreal.MetaHumanIdentityBody).body_type_index = 1
    if not ident.is_logged_in_to_service():
        raise RuntimeError("not logged in to the MetaHuman service (Mesh to MetaHuman needs the Epic login)")
    ident.on_auto_rig_service_finished_dynamic_delegate.add_callable(on_autorig)
    ident.create_dna_for_identity(True)
    state["ident"] = ident
    state["stage"] = "autorig"
    state["deadline"] = time.time() + 20 * 60
    log("Mesh to MetaHuman requested (cloud auto-rig)")


def on_autorig(ok):
    ident = state["ident"]
    ident.on_auto_rig_service_finished_dynamic_delegate.remove_callable(on_autorig)
    if not ok:
        log("ERROR auto-rig service failed to return a DNA")
        return quit_editor()
    log("auto-rig done; preparing for performance")
    ident.get_or_create_part_of_class(unreal.MetaHumanIdentityFace).run_predictive_solver_training()
    unreal.EditorAssetLibrary.save_directory(STORE)  # the identity's skeletal mesh is a separate package
    log("identity ready")
    state["stage"] = "perform"


def perform(req):
    capture = unreal.load_asset(req["capture"])
    ident = None if req.get("mode") == "mono" else unreal.load_asset(f"{STORE}/{req['identity']}")
    name = f"MHP_{req['tag']}"
    path = f"{STORE}/{name}"
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        unreal.EditorAssetLibrary.delete_asset(path)
    perf = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        asset_name=name, package_path=STORE, asset_class=unreal.MetaHumanPerformance,
        factory=unreal.MetaHumanPerformanceFactoryNew())
    if req.get("mode") == "mono":  # video only: no identity, no Depth Processing plugin needed
        perf.set_editor_property("input_type", unreal.DataInputType.MONO_FOOTAGE)
    else:
        perf.set_editor_property("input_type", unreal.DataInputType.DEPTH_FOOTAGE)
        perf.set_editor_property("identity", ident)
    perf.set_editor_property("footage_capture_data", capture)
    perf.set_editor_property("start_frame_to_process", req["start"])
    perf.set_editor_property("end_frame_to_process", req["end"])
    perf.set_blocking_processing(True)
    t = time.time()
    log(f"processing frames {req['start']}-{req['end']}")
    err = perf.start_pipeline()
    if err != unreal.StartPipelineErrorType.NONE:
        raise RuntimeError(f"performance pipeline didn't start: {err}")
    n = req["end"] - req["start"]
    log(f"processed {n} frames in {time.time() - t:.0f} s ({n / max(time.time() - t, 1):.1f} frames/s)")
    unreal.EditorAssetLibrary.save_loaded_asset(perf)

    es = unreal.MetaHumanPerformanceExportAnimationSettings()
    es.show_export_dialog = False
    es.package_path = STORE
    es.asset_name = f"AS_{req['tag']}"
    es.target_skeleton_or_skeletal_mesh = unreal.load_asset(
        "/Game/MetaHumans/Common/Face/Face_Archetype_Skeleton.Face_Archetype_Skeleton")
    es.enable_head_movement = True
    es.export_range = unreal.PerformanceExportRange.PROCESSING_RANGE
    anim = unreal.MetaHumanPerformanceExportUtils.export_animation_sequence(perf, es)
    if not anim:
        raise RuntimeError("animation export failed")
    unreal.EditorAssetLibrary.save_loaded_asset(anim)

    curves = {}
    for c in unreal.AnimationLibrary.get_animation_curve_names(anim, unreal.RawCurveTrackTypes.RCT_FLOAT):
        times, values = unreal.AnimationLibrary.get_float_keys(anim, c)
        curves[str(c)] = [list(times), list(values)]
    out = os.path.join(SAVED, f"vrface_mha_{req['tag']}.json")
    with open(out, "w") as f:
        json.dump({"request": req, "curves": curves}, f)
    log(f"{len(curves)} curves -> {out}")


def tick(dt):
    state["frame"] += 1
    if state["frame"] < 120 or state["stage"] == "done":  # let the editor finish starting up
        return
    try:
        if state["stage"] == "start":
            state["stage"] = "busy"  # loading/preparing assets pumps Slate, which calls tick() again
            with open(os.path.join(SAVED, "vrface_mha_request.json")) as f:
                state["req"] = json.load(f)
            log(f"request {state['req']}")
            if state["req"].get("mode") == "mono" or unreal.EditorAssetLibrary.does_asset_exist(
                    f"{STORE}/{state['req']['identity']}"):
                log("identity exists (or not needed); skipping to the performance")
                if state["req"].get("prepare"):  # e.g. "Prepare for Performance" failed before the plugin was installed
                    ident = unreal.load_asset(f"{STORE}/{state['req']['identity']}")
                    ident.set_blocking_processing(True)
                    t = time.time()
                    ident.get_or_create_part_of_class(unreal.MetaHumanIdentityFace).run_predictive_solver_training()
                    unreal.EditorAssetLibrary.save_loaded_asset(ident)
                    log(f"prepared the identity for performance ({time.time() - t:.0f} s)")
                state["stage"] = "perform"
            else:
                make_identity(state["req"])
        elif state["stage"] == "autorig" and time.time() > state["deadline"]:
            log("ERROR auto-rig didn't answer within 20 min")
            quit_editor()
        elif state["stage"] == "perform":
            state["stage"] = "busy"
            perform(state["req"])
            log(f"DONE in {(time.time() - state['t0']) / 60:.1f} min")
            quit_editor()
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {type(e).__name__}: {e}")
        quit_editor()


state["handle"] = unreal.register_slate_post_tick_callback(tick)
unreal.log("[vrface] mha_take_gui registered")
