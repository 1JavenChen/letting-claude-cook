"""Build the mouth scene, apply an animation file, and save / render it.

Usage (headless):
    blender -b -P blender/animate.py -- --anim out/test.json --audio data/test.wav \
        --blend out/test.blend --video out/test.mp4

Omit --video to just save a .blend you can open and scrub through.
"""

import argparse
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from build_mouth import build_mouth, build_mouth_interior  # noqa: E402
from common.blendshapes import MOUTH_SHAPES, load_animation  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--anim", required=True)
    p.add_argument("--audio")
    p.add_argument("--blend")
    p.add_argument("--video")
    p.add_argument("--resolution", type=int, default=720)
    p.add_argument("--label", help="caption shown under the mouth")
    return p.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def setup_camera_and_lights(scene):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 1.6
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (0, -3, 0)
    cam.rotation_euler = (math.pi / 2, 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    for name, loc, energy in (("Key", (1.5, -3, 1.5), 40), ("Fill", (-2, -2, 0), 15)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
        light.data.energy = energy
        light.data.size = 2
        light.location = loc
        light.rotation_euler = (math.radians(70), 0, math.atan2(loc[0], -loc[1]))
        scene.collection.objects.link(light)

    world = bpy.data.worlds.new("World")
    world.color = (0.85, 0.72, 0.62)  # skin-ish backdrop
    if hasattr(world, "use_nodes"):
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg:
        bg.inputs["Color"].default_value = (0.85, 0.72, 0.62, 1)
        bg.inputs["Strength"].default_value = 1.0
    scene.world = world


def add_label(scene, text):
    curve = bpy.data.curves.new("Label", "FONT")
    curve.body = text
    curve.align_x = "CENTER"
    curve.size = 0.12
    obj = bpy.data.objects.new("Label", curve)
    obj.location = (0, -0.3, -0.55)
    obj.rotation_euler = (math.pi / 2, 0, 0)
    mat = bpy.data.materials.new("LabelInk")
    mat.diffuse_color = (0.15, 0.1, 0.1, 1)
    if mat.node_tree and mat.node_tree.nodes.get("Principled BSDF"):
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.15, 0.1, 0.1, 1)
    curve.materials.append(mat)
    scene.collection.objects.link(obj)


def apply_animation(obj, fps, names, frames, scene):
    keys = obj.data.shape_keys.key_blocks
    scene.render.fps = int(fps)
    scene.frame_start = 1
    scene.frame_end = max(1, len(frames))
    for f, row in enumerate(frames, start=1):
        for name, value in zip(names, row):
            if name in keys:
                keys[name].value = value
                keys[name].keyframe_insert("value", frame=f)
    # Linear interpolation: the source data is already per-frame.
    anim = obj.data.shape_keys.animation_data
    if anim and anim.action:
        for fc in _fcurves(anim.action):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"


def _fcurves(action):
    if hasattr(action, "fcurves"):
        return action.fcurves
    # Blender 5.x slotted actions: curves live in layers/strips/channelbags.
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                curves.extend(bag.fcurves)
    return curves


def add_audio(scene, path):
    if not scene.sequence_editor:
        scene.sequence_editor_create()
    ed = scene.sequence_editor
    strips = ed.strips if hasattr(ed, "strips") else ed.sequences
    strips.new_sound("Voice", os.path.abspath(path), channel=1, frame_start=1)


def setup_render(scene, path, resolution):
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = scene.render.resolution_y = resolution
    img = scene.render.image_settings
    if hasattr(img, "media_type"):
        img.media_type = "VIDEO"
    img.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.filepath = os.path.abspath(path)


def main():
    args = parse_args()
    scene = reset_scene()
    fps, names, frames = load_animation(args.anim)

    mouth = build_mouth(MOUTH_SHAPES)
    interior = build_mouth_interior(MOUTH_SHAPES)
    setup_camera_and_lights(scene)
    scene.view_settings.view_transform = "Standard"  # keep lip/skin colours true
    for obj in (mouth, interior):
        apply_animation(obj, fps, names, frames, scene)
    if args.audio:
        add_audio(scene, args.audio)
    if args.label:
        add_label(scene, args.label)

    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))
        print(f"Saved {args.blend}")
    if args.video:
        setup_render(scene, args.video, args.resolution)
        bpy.ops.render.render(animation=True)
        print(f"Rendered {args.video}")


if __name__ == "__main__":
    main()
