"""Build MediaPipe-landmark correspondences on the FLAME and ICT heads (run with .venv-smirk).

    .venv-smirk\\Scripts\\python tools/flame_ict_corr.py   # -> models/flame_ict_corr.npz, out/corr_*.jpg

Both neutral heads are rendered from the front (and at small yaws); MediaPipe's
FaceLandmarker finds its 478 landmarks on the renders, and each landmark is mapped
back onto the mesh surface (triangle under the pixel, closest to the camera; the
3D points from the different views are averaged and snapped to the surface as
triangle + barycentric weights). SMIRK's own 105-point MediaPipe embedding on FLAME is
only a fallback (and a check of the render approach): its inner-lip points sit a few mm out
on the red of the lips, so they separate when the lips pucker/evert even though the mouth is
closed - the render-derived points lie on the actual lip contact line.

Saved: for each mesh, tri[468] and bary[468,3] plus ok[468] (irises 468-477 are skipped).
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "models", "flame_ict_corr.npz")
N_LMK = 468
YAWS = (0.0, -0.15, 0.15)
SIZE = 1024
# Render both heads with the mouth slightly open: with closed lips MediaPipe's upper and lower
# inner-lip points fall on the same seam and could land on the wrong lip. Correspondences are
# stored as triangle + barycentric weights, so they carry over to the neutral meshes.
OPEN_ICT = 0.25
OPEN_FLAME = 0.12  # rad


def landmarker():
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(ROOT, "models", "face_landmarker.task")),
        num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))


def detect(lm, rgb):
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not r.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])[:N_LMK]


def hit_surface(px, verts_local, scr, depth, faces, snap_px=8.0):
    """For each 2D point, the 3D (local) surface point under it nearest the camera (NaN if none).
    scr: [V,2] screen positions (y down), depth: [V] (larger = closer). Only front-facing
    triangles count (both meshes wind counter-clockwise seen from outside). A point that falls
    in a gap (e.g. MediaPipe's inner-lip points sit just inside the open mouth) snaps to the
    nearest front-facing vertex within snap_px."""
    a, b, c = scr[faces[:, 0]], scr[faces[:, 1]], scr[faces[:, 2]]
    area = (b - a)[:, 0] * (c - a)[:, 1] - (c - a)[:, 0] * (b - a)[:, 1]
    keep = area < 0  # counter-clockwise in y-up = clockwise on screen
    faces, a, b, c = faces[keep], a[keep], b[keep], c[keep]
    front_v = np.unique(faces)
    lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    out = np.full((len(px), 3), np.nan)
    for i, p in enumerate(px):
        cand = np.flatnonzero((lo[:, 0] <= p[0]) & (hi[:, 0] >= p[0]) & (lo[:, 1] <= p[1]) & (hi[:, 1] >= p[1]))
        inside = np.zeros(0, bool)
        if len(cand):
            A, B, C = a[cand], b[cand], c[cand]
            v0, v1, v2 = B - A, C - A, p - A
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
        if inside.any():
            cand, u, v = cand[inside], u[inside], v[inside]
            f = faces[cand]
            bary = np.c_[1 - u - v, u, v]
            z = (depth[f] * bary).sum(1)
            k = np.argmax(z)
            out[i] = (verts_local[f[k]] * bary[k][:, None]).sum(0)
        else:
            d = np.linalg.norm(scr[front_v] - p, axis=1)
            near = d < snap_px
            if near.any():  # closest-to-camera among the nearby front vertices
                j = front_v[near][np.argmax(depth[front_v[near]] - 1e-3 * d[near])]
                out[i] = verts_local[j]
    return out


def closest_on_mesh(points, verts, faces):
    """Snap 3D points to the closest triangle: returns tri index and barycentric weights."""
    cen = verts[faces].mean(1)
    tris, barys = np.zeros(len(points), np.int64), np.zeros((len(points), 3))
    for i, p in enumerate(points):
        if np.isnan(p).any():
            continue
        cand = np.argsort(((cen - p) ** 2).sum(1))[:40]
        best = (np.inf, 0, None)
        for t in cand:
            A, B, C = verts[faces[t]]
            q, bc = _closest_point_triangle(p, A, B, C)
            dd = ((q - p) ** 2).sum()
            if dd < best[0]:
                best = (dd, t, bc)
        tris[i], barys[i] = best[1], best[2]
    return tris, barys


def _closest_point_triangle(p, a, b, c):
    """Ericson, Real-Time Collision Detection 5.1.5. Returns point and barycentrics."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a, np.array([1.0, 0, 0])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b, np.array([0, 1.0, 0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1 - v, v, 0])
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c, np.array([0, 0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1 - w, 0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0, 1 - w, w])
    den = 1 / (va + vb + vc)
    v, w = vb * den, vc * den
    return a + ab * v + ac * w, np.array([1 - v - w, v, w])


def rot_y(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# --------------------------------------------------------------------------- #
def ict_points(lm, sheet):
    from realtime.head3d import Head3D
    import torch
    names = torch.load(os.path.join(ROOT, "models", "fusion.pt"), map_location="cpu")["names"]
    head = Head3D(names, size=SIZE)
    head.update({"jawOpen": OPEN_ICT})
    v_local = head.verts.astype(np.float64)  # mouth slightly open so inner lips separate
    face_tris = head.faces[(head.faces < 11248).all(1)]  # skin of the face/head only
    pts = []
    for yaw in YAWS:
        head.update({"jawOpen": OPEN_ICT}, (yaw, 0.0, 0.0))
        bgr = head.render()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        L = detect(lm, rgb)
        if L is None:
            print(f"ICT yaw {yaw}: no face detected")
            continue
        model = np.eye(4)
        model[:3, :3] = head.R
        model[:3, 3] = head.pivot - head.R @ head.pivot
        clip = (head.vp @ model @ np.c_[v_local, np.ones(len(v_local))].T).T
        ndc = clip[:, :3] / clip[:, 3:4]
        scr = np.c_[(ndc[:, 0] + 1) / 2 * SIZE, (1 - ndc[:, 1]) / 2 * SIZE]
        depth = -clip[:, 3]  # w = distance in front of the camera; larger -w = closer
        P = hit_surface(L, v_local, scr, depth, face_tris)
        pts.append(P)
        if yaw == 0.0:
            img = bgr.copy()
            for x, y in L.astype(int):
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            sheet["ict"] = (img, scr)
        print(f"ICT yaw {yaw:+.2f}: {np.isfinite(P[:, 0]).sum()} of {N_LMK} landmarks hit the face")
    return v_local, face_tris, pts


def flame_points(lm, sheet):
    import torch
    from smirk_flame import Flame, face_mask_vertices, safe_pickle_load, ASSETS
    fl = Flame()
    with torch.no_grad():
        v, _ = fl(None, torch.zeros(1, 50), torch.zeros(1, 3), torch.tensor([[OPEN_FLAME, 0, 0]]), None)
    v_local = v[0].numpy().astype(np.float64)
    faces = fl.faces.numpy()
    masks = safe_pickle_load(os.path.join(ASSETS, "FLAME_masks", "FLAME_masks.pkl"))
    eyes = np.zeros(len(v_local), bool)
    eyes[masks["left_eyeball"]] = eyes[masks["right_eyeball"]] = True
    skin_tris = faces[~eyes[faces].any(1)]
    pts = []
    for yaw in YAWS:
        R = rot_y(yaw)
        vw = v_local @ R.T
        s = SIZE * 0.8 / 0.25  # ~25 cm of head over 55% of the image
        scr = np.c_[vw[:, 0] * s + SIZE / 2, -(vw[:, 1] - 0.0) * s + SIZE / 2]
        img = render_skin(vw, scr, faces, eyes)
        L = detect(lm, img)
        if L is None:
            print(f"FLAME yaw {yaw}: no face detected")
            continue
        P = hit_surface(L, v_local, scr, vw[:, 2], skin_tris)
        pts.append(P)
        if yaw == 0.0:
            sheet["flame"] = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR), scr, L)
        print(f"FLAME yaw {yaw:+.2f}: {np.isfinite(P[:, 0]).sum()} of {N_LMK} landmarks hit the face")
    return v_local, skin_tris, pts, fl


def render_skin(vw, scr, faces, eyes):
    """Smooth-ish shaded skin render (painter's algorithm) so MediaPipe sees a face."""
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    tri = vw[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.3, 0.4, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    p = np.round(scr * 16).astype(np.int32)
    for i in np.argsort(tri[:, :, 2].mean(1)):
        if n[i, 2] <= 0:
            continue
        if eyes[faces[i]].all():
            # eyeball: white sclera, dark iris where it faces forward
            c = np.array([25, 18, 12]) if n[i, 2] > 0.93 else np.array([235, 230, 225])
        else:
            c = np.array([224, 172, 140]) * shade[i]
        cv2.fillConvexPoly(img, p[faces[i]], tuple(int(x) for x in c), lineType=cv2.LINE_AA, shift=4)
    return img


def merge(pts, verts, tris, tol):
    """Average multi-view hits; keep points seen in >=2 views that agree within tol."""
    P = np.stack(pts)
    mean = np.nanmean(P, 0)
    spread = np.nanmax(np.linalg.norm(P - mean, axis=2), 0)
    seen = np.isfinite(P[:, :, 0]).sum(0)
    ok = (seen >= 2) & (spread < tol) & np.isfinite(P[0, :, 0])  # must be seen frontally
    mean[~ok] = np.nan
    t, b = closest_on_mesh(mean, verts, tris)
    return t, b, ok


def main():
    lm = landmarker()
    sheet = {}
    iv, itris, ipts = ict_points(lm, sheet)
    fv, ftris, fpts, fl = flame_points(lm, sheet)
    it, ib, iok = merge(ipts, iv, itris, tol=0.6)       # cm
    ft, fb, fok = merge(fpts, fv, ftris, tol=0.006)     # m

    # SMIRK's embedding for its 105 points: compare, then prefer it
    emb_idx = fl.mp_indices
    emb_faces = fl.faces.numpy()[fl.mp_faces_idx.numpy()]
    emb_pts = (fv[emb_faces] * fl.mp_bary.numpy()[:, :, None]).sum(1)
    our_pts = (fv[ftris[ft]] * fb[:, :, None]).sum(1)
    sel = fok[emb_idx]
    err = np.linalg.norm(our_pts[emb_idx] - emb_pts, axis=1)[sel] * 1000
    print(f"FLAME render landmarks vs SMIRK embedding (105 pts, {sel.sum()} valid): "
          f"median {np.median(err):.1f} mm, 90% {np.percentile(err, 90):.1f} mm, max {err.max():.1f} mm")
    # store FLAME points as (vertex triplet, bary) so both sources fit one format
    f_vids = ftris[ft]
    fallback = emb_idx[~fok[emb_idx]]
    sel_fb = ~fok[emb_idx]
    f_vids[fallback] = emb_faces[sel_fb]
    fb[fallback] = fl.mp_bary.numpy()[sel_fb]
    fok[fallback] = True
    i_vids = itris[it]
    ok = iok & fok
    print(f"correspondences: {ok.sum()} of {N_LMK} (ICT ok {iok.sum()}, FLAME ok {fok.sum()})")
    np.savez(OUT, flame_vids=f_vids, flame_bary=fb.astype(np.float32), ict_vids=i_vids,
             ict_bary=ib.astype(np.float32), ok=ok, smirk_idx=emb_idx, flame_render_err_mm=err)
    print("saved", OUT)

    # overlays: green = MediaPipe on the render, red = stored surface points re-projected
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    img, scr = sheet["ict"]
    P = (iv[i_vids] * ib[:, :, None]).sum(1)
    from realtime.head3d import perspective
    view = np.eye(4)
    view[:3, 3] = -np.array([0.0, -1.0, 72.0])
    clip = (perspective(26, 1.0, 20, 200) @ view @ np.c_[P, np.ones(len(P))].T).T
    q = np.c_[(clip[:, 0] / clip[:, 3] + 1) / 2 * SIZE, (1 - clip[:, 1] / clip[:, 3]) / 2 * SIZE]
    for k, (x, y) in enumerate(q):
        if ok[k]:
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    img2, scr2, L2 = sheet["flame"]
    s = SIZE * 0.8 / 0.25
    Pf = (fv[f_vids] * fb[:, :, None]).sum(1)
    for k, (x, y) in enumerate(L2.astype(int)):
        cv2.circle(img2, (x, y), 2, (0, 255, 0), -1)
    for k in range(N_LMK):
        if ok[k]:
            cv2.circle(img2, (int(Pf[k, 0] * s + SIZE / 2), int(-Pf[k, 1] * s + SIZE / 2)), 2,
                       (0, 0, 255) if k not in set(emb_idx) else (255, 0, 255), -1)
    both = np.hstack([img, img2])
    cv2.imwrite(os.path.join(ROOT, "out", "corr_overlay.jpg"), both)
    mouth = np.hstack([img[520:820, 312:712], img2[520:820, 312:712]])
    cv2.imwrite(os.path.join(ROOT, "out", "corr_mouth.jpg"), cv2.resize(mouth, None, fx=2, fy=2))
    print("overlays: out/corr_overlay.jpg, out/corr_mouth.jpg")


if __name__ == "__main__":
    main()
