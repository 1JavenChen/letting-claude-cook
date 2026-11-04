"""Real-time 3D head driven by our 51 blendshape weights (ICT-FaceKit Light, MIT).

    python realtime/head3d.py                 # window: cycles through single shapes
    python realtime/head3d.py --bench         # mesh update + render fps
    python realtime/head3d.py --sheet out/head3d_check.jpg   # contact sheet of test shapes

The mesh (26,719 verts, 53 ARKit-style shapes with _L/_R) is read from
third_party/ICT-FaceKit/FaceXModel once and cached to models/ict_head.npz.
We keep face, head/neck, mouth socket, gums/tongue, teeth and eyeballs; the
eyelash strips and transparent eye overlay meshes are dropped (they need
alpha textures). Rendering is offscreen OpenGL (moderngl), so `render()`
returns a BGR image for the live preview window or a video writer.

Sides: ICT's _L is the person's left (+x, the viewer's right when facing the
camera), the same as MediaPipe's "Left" - so eyeBlinkLeft -> eyeBlink_L.
"""

import json
import os
import sys

import cv2
import moderngl
import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "ICT-FaceKit", "FaceXModel")
CACHE = os.path.join(ROOT, "models", "ict_head.npz")
KEEP = 24591  # vertices [0, 24591): face .. eyeballs; after that lacrimal/occlusion/eyelashes

# (first vertex, material) - ranges from the ICT-FaceKit README
PARTS = [(0, 0), (11248, 1), (13294, 2), (14062, 5), (17039, 3), (21451, 4)]
INNER_EYE = [(22221, 23021), (23791, 24591)]  # textured inner iris spheres: hidden, we paint the sclera instead
SCLERA = [(21451, 22221), (23021, 23791)]
# linear-space albedo: skin, mouth socket, eye socket, teeth, sclera (iris is painted on), gums/tongue
ALBEDO = np.array([[0.60, 0.48, 0.40], [0.22, 0.05, 0.05], [0.25, 0.12, 0.11],
                   [0.62, 0.59, 0.52], [0.82, 0.80, 0.78], [0.34, 0.10, 0.10]], np.float32)
IRIS = np.array([0.16, 0.10, 0.06], np.float32)
WRAP = np.array([0.45, 0.6, 0.6, 0.1, 0.1, 0.1], np.float32)  # soft "subsurface" wrap per material
SPEC = np.array([0.18, 0.25, 0.1, 0.35, 0.5, 0.6], np.float32)


def _read_obj(path, faces=False):
    with open(path) as f:
        lines = f.read().splitlines()
    v = np.array([l.split()[1:4] for l in lines if l.startswith("v ")], np.float32)
    if not faces:
        return v
    tris = []
    for l in lines:
        if l.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in l.split()[1:]]
            tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return v, np.array(tris, np.int32)


def build_cache(src=SRC, out=CACHE):
    """Neutral mesh + per-shape deltas -> npz (cropped to KEEP vertices)."""
    names = json.load(open(os.path.join(src, "vertex_indices.json")))["expressions"]
    v0, tris = _read_obj(os.path.join(src, "generic_neutral_mesh.obj"), faces=True)
    tris = tris[tris.max(1) < KEEP]
    v0 = v0[:KEEP]
    # make triangles wind counter-clockwise seen from outside (normals point out of the face)
    n = np.cross(v0[tris[:, 1]] - v0[tris[:, 0]], v0[tris[:, 2]] - v0[tris[:, 0]])
    front = tris.max(1) < 9409
    if (n[front, 2]).sum() < 0:
        tris = tris[:, [0, 2, 1]]
    deltas = np.stack([_read_obj(os.path.join(src, f"{s}.obj"))[:KEEP] - v0 for s in names])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, vertices=v0, faces=tris, deltas=deltas.astype(np.float32),
                        names=np.array(names))
    print(f"Cached {out}: {len(v0)} verts, {len(tris)} tris, {len(names)} shapes")


def ict_mapping(our_names, ict_names):
    """(S_ours, S_ict) matrix: ICT weights = our weights @ M."""
    M = np.zeros((len(our_names), len(ict_names)), np.float32)
    for i, n in enumerate(our_names):
        if n.endswith("Left"):
            targets = [n[:-4] + "_L"]
        elif n.endswith("Right"):
            targets = [n[:-5] + "_R"]
        else:
            targets = [n, n + "_L", n + "_R"]  # browInnerUp, cheekPuff are split in ICT
        for t in targets:
            if t in ict_names:
                M[i, ict_names.index(t)] = 1.0
    return M


def euler(yaw, pitch, roll):
    """Ry(yaw) Rx(pitch) Rz(roll): the inverse of the decomposition in sources.FaceTracker."""
    cy, sy, cp, sp_, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp_], [0, sp_, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def perspective(fovy_deg, aspect, near, far):
    f = 1 / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])


VERT = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
in float in_mat;
out vec3 v_pos;
out vec3 v_nrm;
out vec3 v_col;
flat out int v_mat;
void main() {
    v_pos = (model * vec4(in_pos, 1.0)).xyz;
    v_nrm = mat3(model) * in_nrm;
    v_mat = int(in_mat + 0.5);
    v_col = in_col;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 330
uniform vec3 eye;
uniform float wrap[6];
uniform float spec[6];
in vec3 v_pos;
in vec3 v_nrm;
in vec3 v_col;
flat in int v_mat;
out vec4 f_color;
const vec3 L_DIR[3] = vec3[](normalize(vec3(-0.6, 0.55, 0.75)),   // key: upper left front
                             normalize(vec3(0.8, 0.1, 0.5)),      // fill: right
                             normalize(vec3(0.3, 0.7, -0.9)));    // rim: behind, above
const vec3 L_COL[3] = vec3[](vec3(1.05, 0.98, 0.9), vec3(0.32, 0.36, 0.45), vec3(0.7, 0.72, 0.8));
void main() {
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(eye - v_pos);
    if (dot(N, V) < 0.0) N = -N;  // open mesh: light the inside too
    vec3 a = v_col;
    float w = wrap[v_mat];
    vec3 c = a * mix(vec3(0.05, 0.05, 0.07), vec3(0.16, 0.15, 0.15), 0.5 + 0.5 * N.y);  // hemisphere ambient
    float direct = (v_mat == 1 || v_mat == 2 || v_mat == 5) ? 0.5 : 1.0;  // inside the mouth: mostly shadowed
    for (int i = 0; i < 3; i++) {
        float nl = dot(N, L_DIR[i]);
        float lam = max(nl, 0.0);
        float wr = max((nl + w) / (1.0 + w), 0.0);
        vec3 sss = vec3(0.55, 0.18, 0.12) * (wr - lam);  // red-shifted light bleeding past the terminator
        vec3 H = normalize(L_DIR[i] + V);
        float s = pow(max(dot(N, H), 0.0), 48.0) * spec[v_mat] * lam;
        c += direct * L_COL[i] * (a * (lam + sss) + s);
    }
    float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
    c += 0.06 * fres * vec3(0.8, 0.85, 1.0) * float(v_mat == 0);
    c = c / (1.0 + 0.15 * c);  // gentle tone map
    f_color = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
}
"""


class Head3D:
    """update(weights, head) then render() -> BGR uint8 image (size x size)."""

    def __init__(self, names, size=480, head_scale=1.0, samples=4, ctx=None):
        if not os.path.exists(CACHE):
            build_cache()
        d = np.load(CACHE)
        self.v0, self.faces = d["vertices"], d["faces"]
        self.ict_names = list(d["names"])
        N, F = len(self.v0), len(self.faces)
        self.D = d["deltas"].reshape(len(self.ict_names), -1)
        self.M = ict_mapping(list(names), self.ict_names)
        self.names = list(names)
        mat = np.zeros(N, np.float32)
        for start, m in PARTS:
            mat[start:] = m
        # vertex normal = sum of adjacent (area-weighted) face normals, as one sparse matmul
        rows = self.faces.ravel()
        self.adj = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, np.repeat(np.arange(F), 3))),
                                 shape=(N, F))
        self.size, self.head_scale = size, head_scale
        self.pivot = np.array([0.0, -6.0, -1.0])  # roughly the top of the neck (cm)
        self.R = np.eye(3)
        self.verts = self.v0.copy()

        self.ctx = ctx or moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.prog["wrap"].write(WRAP.tobytes())
        self.prog["spec"].write(SPEC.tobytes())
        self.vbo_pos = self.ctx.buffer(reserve=N * 12)
        self.vbo_nrm = self.ctx.buffer(reserve=N * 12)
        self.vbo_mat = self.ctx.buffer(mat.tobytes())
        self.vbo_col = self.ctx.buffer(self._colors(mat).tobytes())
        hide = np.zeros(N, bool)
        for a, b in INNER_EYE:
            hide[a:b] = True
        ibo = self.ctx.buffer(self.faces[~hide[self.faces].any(1)].astype(np.int32).tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo_pos, "3f", "in_pos"), (self.vbo_nrm, "3f", "in_nrm"),
                                                     (self.vbo_col, "3f", "in_col"), (self.vbo_mat, "1f", "in_mat")],
                                         ibo)
        sz = (size, size)
        self.fbo_ms = self.ctx.framebuffer(self.ctx.renderbuffer(sz, 4, samples=samples),
                                           self.ctx.depth_renderbuffer(sz, samples=samples))
        self.fbo = self.ctx.framebuffer(self.ctx.renderbuffer(sz, 4))
        self.eye = np.array([0.0, -1.0, 72.0])  # looking straight down -z at the face (cm)
        view = np.eye(4)
        view[:3, 3] = -self.eye
        self.vp = perspective(26, 1.0, 20, 200) @ view
        self.prog["eye"].value = tuple(self.eye)
        self.update({})

    def _colors(self, mat):
        """Per-vertex albedo: iris/pupil painted onto the sclera, fake occlusion deep in the mouth."""
        v = self.v0
        col = ALBEDO[mat.astype(int)].copy()
        for a, b in SCLERA:
            s = v[a:b]
            A = np.c_[2 * s, np.ones(len(s))]
            c = np.linalg.lstsq(A, (s ** 2).sum(1), rcond=None)[0][:3]  # eyeball centre (sphere fit)
            r = np.linalg.norm(s - c, axis=1)
            gaze = s[r > np.percentile(r, 92)].mean(0) - c  # the cornea bulge
            gaze /= np.linalg.norm(gaze)
            ang = np.degrees(np.arccos(np.clip((s - c) / r[:, None] @ gaze, -1, 1)))
            iris = np.clip((30 - ang) / 4, 0, 1)[:, None]  # iris half-angle ~30 deg, soft edge
            pupil = np.clip((13 - ang) / 3, 0, 1)[:, None]
            ring = np.exp(-((ang - 28) / 3) ** 2)[:, None]  # darker limbal ring
            c_ = col[a:b] * (1 - iris) + IRIS * iris * (1 - 0.6 * ring)
            col[a:b] = c_ * (1 - pupil) + 0.01 * pupil
        inner = np.isin(mat, [1, 2, 3, 5])
        z = v[inner, 2]
        col[inner] *= (np.clip((z - 3.0) / 8.0, 0.05, 1.0) ** 1.5)[:, None]  # darker towards the throat
        return col.astype(np.float32)

    def update(self, weights, head=(0.0, 0.0, 0.0)):
        """weights: dict name -> value (or array in `names` order); head: (yaw, pitch, roll) radians."""
        if isinstance(weights, dict):
            w = np.array([weights.get(n, 0.0) for n in self.names], np.float32)
        else:
            w = np.asarray(weights, np.float32)
        wi = np.clip(w, 0, 1) @ self.M
        nz = np.flatnonzero(wi > 1e-3)
        v = self.v0.reshape(-1) + wi[nz] @ self.D[nz] if len(nz) else self.v0.reshape(-1)
        self.verts = v.reshape(-1, 3).astype(np.float32)
        f = self.faces
        a, b, c = self.verts[f[:, 0]], self.verts[f[:, 1]], self.verts[f[:, 2]]
        fn = np.cross(b - a, c - a)
        vn = self.adj @ fn
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        self.vbo_pos.write(self.verts.tobytes())
        self.vbo_nrm.write(vn.astype(np.float32).tobytes())
        self.R = euler(*(self.head_scale * np.asarray(head, np.float64)))

    def render(self, bg=(28, 28, 30)):
        model = np.eye(4)
        model[:3, :3] = self.R
        model[:3, 3] = self.pivot - self.R @ self.pivot
        self.prog["model"].write(model.T.astype(np.float32).tobytes())
        self.prog["mvp"].write((self.vp @ model).T.astype(np.float32).tobytes())
        self.fbo_ms.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(bg[2] / 255, bg[1] / 255, bg[0] / 255, 1.0)
        self.vao.render()
        self.ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        img = np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(self.size, self.size, 3)
        return cv2.cvtColor(img[::-1], cv2.COLOR_RGB2BGR)

    def show(self, title="3D head"):
        cv2.imshow(title, self.render())
        return cv2.waitKey(1) & 0xFF


SHEET = [("neutral", {}), ("jawOpen 1", {"jawOpen": 1}), ("mouthSmileLeft 1", {"mouthSmileLeft": 1}),
         ("eyeBlinkLeft 1", {"eyeBlinkLeft": 1}), ("browInnerUp 1", {"browInnerUp": 1}),
         ("mouthPucker 1", {"mouthPucker": 1}), ("mouthFunnel 1", {"mouthFunnel": 1}),
         ("mouthClose+jawOpen .5", {"mouthClose": 0.5, "jawOpen": 0.5}),
         ("mouthFrownLeft 1", {"mouthFrownLeft": 1}), ("cheekPuff 1", {"cheekPuff": 1}),
         ("yaw +0.4", {}), ("roll +0.3", {})]


def main():
    import argparse
    import time

    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true", help="re-read the OBJs into models/ict_head.npz")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--sheet", help="write a labelled contact sheet of test shapes here")
    p.add_argument("--size", type=int, default=480)
    args = p.parse_args()
    if args.rebuild or not os.path.exists(CACHE):
        build_cache()
    names = torch.load(os.path.join(ROOT, "models", "fusion.pt"), map_location="cpu")["names"]
    head = Head3D(names, size=args.size)
    print("GL renderer:", head.ctx.info["GL_RENDERER"])
    if args.bench:
        rng = np.random.default_rng(0)
        for _ in range(10):
            head.update(rng.random(len(names)) * 0.5)
            head.render()
        n, t0 = 300, time.perf_counter()
        tu = 0.0
        for i in range(n):
            t1 = time.perf_counter()
            head.update(rng.random(len(names)) * 0.5, (0.2 * np.sin(i / 20), 0.1, 0.0))
            tu += time.perf_counter() - t1
            head.render()
        dt = (time.perf_counter() - t0) / n
        print(f"update+render {dt * 1000:.2f} ms/frame = {1 / dt:.0f} fps (update {tu / n * 1000:.2f} ms)")
        return
    if args.sheet:
        tiles = []
        for label, w in SHEET:
            pose = (0.4, 0, 0) if label.startswith("yaw") else (0, 0, 0.3) if label.startswith("roll") else (0, 0, 0)
            head.update(w, pose)
            img = head.render()
            cv2.putText(img, label, (10, args.size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(img)
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        os.makedirs(os.path.dirname(os.path.abspath(args.sheet)), exist_ok=True)
        cv2.imwrite(args.sheet, np.vstack(rows))
        print(f"Saved {args.sheet}")
        return
    k = 0
    while True:  # cycle through every shape, 1 s each
        n = names[(k // 60) % len(names)]
        head.update({n: np.sin((k % 60) / 60 * np.pi)})
        img = head.render()
        cv2.putText(img, n, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("3D head (q quits)", img)
        if cv2.waitKey(16) & 0xFF == ord("q"):
            break
        k += 1


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
