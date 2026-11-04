"""Procedurally build a 48-vertex mouth with ARKit-named shape keys.

Runs inside Blender. The mouth is three rings of 16 vertices (inner lip edge,
lip "bulge", outer lip edge) in the XZ plane, facing -Y (Blender's front view).
Character's left is +X, matching ARKit's "Left" convention.
"""

import math

import bpy

RING_VERTS = 16
HALF_WIDTH = 0.5  # mouth is 1 unit wide

# (half-width scale, upper lip height, lower lip height, forward bulge)
RINGS = {
    "inner": (0.94, 0.006, 0.006, 0.00),
    "mid": (0.98, 0.075, 0.095, -0.06),
    "outer": (1.00, 0.150, 0.180, -0.02),
}


def _vertex_params():
    """Yield (ring, x, z, y, upper_w, lower_w) for every base vertex."""
    for ring, (wscale, up_h, low_h, bulge) in RINGS.items():
        for i in range(RING_VERTS):
            t = 2 * math.pi * i / RING_VERTS
            c, s = math.cos(t), math.sin(t)
            x = HALF_WIDTH * wscale * c
            h = up_h if s >= 0 else low_h
            # Cupid's bow: slight dip at the centre of the upper outer lip.
            bow = 0.025 * math.exp(-(c / 0.15) ** 2) if (ring == "outer" and s > 0) else 0.0
            z = h * s - bow
            y = bulge * (1 - c * c) ** 0.5
            upper = 0.5 if abs(s) < 1e-6 else (1.0 if s > 0 else 0.0)
            yield ring, x, y, z, upper, 1.0 - upper, c


def _smooth_side(c, sign):
    """0..1 weight for the character's left (sign=+1) or right (sign=-1) half."""
    v = 0.5 + 0.5 * math.tanh(4 * sign * c)
    return v


def _deltas(name, ring, x, y, z, up, lo, c):
    corner = abs(c) ** 3
    centre = 1 - abs(c)
    inner = ring == "inner"
    outer = ring == "outer"
    L, R = _smooth_side(c, +1), _smooth_side(c, -1)

    if name == "jawOpen":
        return (-0.06 * x, 0.03 * lo, -0.32 * lo * (1 - 0.4 * corner))
    if name == "mouthClose":  # used together with jawOpen: lips stay sealed
        return (0.0, 0.0, 0.30 * lo * (1 - 0.4 * corner))
    if name == "mouthFunnel":
        open_amt = 0.07 if inner else 0.03
        return (-0.28 * x, -0.12, open_amt * (up - lo))
    if name == "mouthPucker":
        return (-0.42 * x, -0.16, -0.7 * z if inner else -0.25 * z)
    if name in ("mouthLeft", "mouthRight"):
        sign = 1 if name == "mouthLeft" else -1
        return (sign * 0.10 * (1 - 0.3 * corner), 0.0, 0.0)
    for side, w in (("Left", L), ("Right", R)):
        if not name.endswith(side):
            continue
        sx = 1 if side == "Left" else -1
        base = name[: -len(side)]
        if base == "mouthSmile":
            return (sx * 0.10 * corner * w, 0.04 * corner * w, 0.12 * corner * w)
        if base == "mouthFrown":
            return (0.0, 0.0, -0.12 * corner * w)
        if base == "mouthStretch":
            return (sx * 0.14 * corner * w, 0.02 * w, -0.04 * corner * w)
        if base == "mouthPress":  # thin the lips toward the lip line
            return (0.0, 0.03 * w, -0.45 * z * w if outer else -0.25 * z * w)
        if base == "mouthLowerDown":
            return (0.0, 0.0, -0.13 * lo * w * (1 - corner))
        if base == "mouthUpperUp":
            return (0.0, 0.0, 0.11 * up * w * (1 - corner))
    if name == "mouthRollLower":
        roll = {"inner": 0.0, "mid": 0.10, "outer": 0.12}[ring]
        return (0.0, roll * lo, 0.06 * lo if outer else 0.02 * lo)
    if name == "mouthRollUpper":
        roll = {"inner": 0.0, "mid": 0.10, "outer": 0.12}[ring]
        return (0.0, roll * up, -0.05 * up if outer else -0.02 * up)
    raise ValueError(f"No deformation defined for {name}")


def build_mouth(shape_names, name="Mouth"):
    params = list(_vertex_params())
    verts = [(x, y, z) for _, x, y, z, *_ in params]

    # Quads between consecutive rings (inner-mid, mid-outer).
    faces = []
    n = RING_VERTS
    for r in range(2):
        for i in range(n):
            j = (i + 1) % n
            a, b = r * n + i, r * n + j
            faces.append((a, b, b + n, a + n))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    obj.shape_key_add(name="Basis", from_mix=False)
    for shape in shape_names:
        key = obj.shape_key_add(name=shape, from_mix=False)
        for idx, (ring, x, y, z, up, lo, c) in enumerate(params):
            dx, dy, dz = _deltas(shape, ring, x, y, z, up, lo, c)
            key.data[idx].co = (x + dx, y + dy, z + dz)
        key.slider_min, key.slider_max = 0.0, 1.0

    for poly in mesh.polygons:
        poly.use_smooth = True
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 2

    mat = bpy.data.materials.new("Lips")
    mat.diffuse_color = (0.72, 0.30, 0.32, 1.0)
    if hasattr(mat, "use_nodes"):
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.30, 0.32, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45
    mesh.materials.append(mat)
    return obj


def build_mouth_interior(shape_names, name="MouthInterior"):
    """Dark cavity filling the lip opening.

    It's a single polygon through the inner lip ring (slightly enlarged and set
    back), with the same shape keys as the lips, so it tracks the opening.
    """
    grow_x, grow_z, depth = 1.0, 1.15, 0.04
    params = [p for p in _vertex_params() if p[0] == "inner"]

    def place(x, y, z):
        return (x * grow_x, y + depth, z * grow_z)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([place(x, y, z) for _, x, y, z, *_ in params], [],
                     [tuple(range(len(params)))])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    obj.shape_key_add(name="Basis", from_mix=False)
    for shape in shape_names:
        key = obj.shape_key_add(name=shape, from_mix=False)
        for idx, (ring, x, y, z, up, lo, c) in enumerate(params):
            dx, dy, dz = _deltas(shape, ring, x, y, z, up, lo, c)
            key.data[idx].co = place(x + dx, y + dy, z + dz)

    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = 2

    mat = bpy.data.materials.new("MouthInterior")
    mat.diffuse_color = (0.06, 0.01, 0.015, 1.0)
    if mat.node_tree and mat.node_tree.nodes.get("Principled BSDF"):
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.06, 0.01, 0.015, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
    mesh.materials.append(mat)
    return obj
