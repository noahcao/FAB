"""Static-camera fast path of world mode (scripts/fab_world.sh --camera auto | static): a camera that does not move needs no
SLAM, so neither ViPE run happens.
  check  is the camera fixed? Corners of the first frame (Shi-Tomasi, at --check_size long side) tracked with pyramidal LK to
         every --step-th frame, a RANSAC homography per frame (people and other movers are its outliers): fixed when, on
         every checked frame, the homography moves the image corners by < --max_px (in --check_size pixels), its inliers'
         median residual is < --max_res px and >= --min_inl of the corners are inliers
         -> <out>/check/<clip>.json; exit 0 fixed, 10 moving (1: an error)
  depth  the fixed camera: T_world_cam identity on every frame; K = the focal (<work>/focal/fixed_<clip>.json, or --focal),
         principal point centred; MoGe-2 metric depth (fov from the focal, --moge_level) on every --stride-th frame; a fixed camera sees
         one scene, so each frame's depth is scaled to the per-pixel median over the frames on the static pixels and its
         static pixels are replaced by that median (the person keeps its own depth: the joint step's ruler); dynamic mask =
         Mask2Former (ADE20K: people, animals, vehicles, balls) + sky, dilated
         -> a ViPE-style dir (what the world chain reads): <out>/cams/<clip>.npz (T_world_cam, fx), <out>/intrinsics/<clip>.npz
            (inds, data fx fy cx cy), <out>/dense/<clip>.npz (bench/videomimic/vipe_depth.py's format, on the depth frames);
            with --work also <work>/seg/<clip>_seg.npz (clean_render.py seg's people / sky masks from the same Mask2Former
            pass, on every depth frame), so fab_world.sh's seg stage is skipped; <out>/floor/<clip>.json: the floor
            under the person (a strip below each frame's largest person mask, the static median depth there,
            RANSAC plane) -> its up normal in the camera frame, inlier share, points
  gravity  GeoCalib's up (<work>/geocalib_st/<clip>.npz, from fab_world.sh) refined by the floor: when the floor plane
           is well supported (>= --floor_min_inl inliers, >= --floor_min_pts points) and within --floor_max_deg of
           GeoCalib's mean up, up = --floor_mode mean: the normalised mean of the floor normal and GeoCalib's mean up
           (default; on RICH val the two err independently and GT floors under the person tilt 0.2-1.5 deg from GT up),
           floor: the floor normal; a fixed camera: one up for every frame. Else GeoCalib's. The npz keeps GeoCalib's
           vectors as up_cam_geocalib
    python -m bench.world_static check --list clips.json --clip <clip> --out <work>/static
    python -m bench.world_static depth --list clips.json --clip <clip> --out <work>/static --work <work>
    python -m bench.world_static gravity --list clips.json --clip <clip> --out <work>/static --work <work>
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

M2F = "facebook/mask2former-swin-large-ade-semantic"  # as bench/scene_recon/clean_render.py
DYNAMIC = ("person", "animal", "car", "bus", "truck", "van", "bicycle", "minibike", "boat", "airplane", "ball", "sky")


def frames_of(a):
    return json.loads(Path(a.list).read_text())[a.clip]["images"]


def gray(p, size):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    s = size / max(im.shape)
    return cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)


def stage_check(a):
    imgs = frames_of(a)
    g0 = gray(imgs[0], a.check_size)
    h, w = g0.shape
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=1000, qualityLevel=0.01, minDistance=8)
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float64)
    rows = []
    for t in list(range(a.step, len(imgs), a.step)) + [len(imgs) - 1]:
        if p0 is None or len(p0) < 20:
            rows.append({"frame": t, "corner_px": float("inf"), "res_px": float("inf"), "inl": 0.0})
            break
        g = gray(imgs[t], a.check_size)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, p0, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            rows.append({"frame": t, "corner_px": float("inf"), "res_px": float("inf"), "inl": float(ok.mean())})
            continue
        H, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
        if H is None:
            rows.append({"frame": t, "corner_px": float("inf"), "res_px": float("inf"), "inl": 0.0})
            continue
        c1 = cv2.perspectiveTransform(corners[None], H)[0]
        q = cv2.perspectiveTransform(p0[ok][inl.ravel() == 1].reshape(1, -1, 2).astype(np.float64), H)[0]
        res = np.linalg.norm(q - p1[ok][inl.ravel() == 1].reshape(-1, 2), axis=1)
        rows.append({"frame": t, "corner_px": float(np.linalg.norm(c1 - corners, axis=1).max()),
                     "res_px": float(np.median(res)) if len(res) else float("inf"), "inl": float(inl.sum() / len(p0))})
    cp, rp, il = (np.array([r[k] for r in rows]) for k in ("corner_px", "res_px", "inl"))
    fixed = bool((cp < a.max_px).all() and (rp < a.max_res).all() and (il >= a.min_inl).all())
    v = {"fixed": fixed, "corner_px_max": float(cp.max()), "corner_px_p90": float(np.percentile(cp, 90)),
         "res_px_max": float(rp.max()), "inliers_min": float(il.min()), "frames_checked": len(rows), "corners": int(len(p0)) if p0 is not None else 0,
         "check_size": a.check_size, "max_px": a.max_px, "max_res": a.max_res, "min_inl": a.min_inl}
    (Path(a.out) / "check").mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "check" / f"{a.clip}.json").write_text(json.dumps(v) + "\n")
    print(f"[static] {a.clip}: {'fixed' if fixed else 'moving'} camera (corner shift max {v['corner_px_max']:.2f} px, "
          f"residual max {v['res_px_max']:.2f} px, inliers min {v['inliers_min']:.2f}; {len(rows)} frames)", flush=True)
    return fixed


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def floor_normal(ref, K, person, tol=0.02, iters=500, seed=0):
    """The floor under the person: per frame the largest person blob's box -> a strip below it (x +-25 %, rows y2 - 5 %
    ... y2 + 20 % of the box height), on the static median depth ref; RANSAC plane (tol m), refit on its inliers ->
    (up normal in OpenCV camera axes or None, inlier share, points)."""
    h, w = ref.shape
    strip = np.zeros((h, w), bool)
    for m in person:
        k, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if k < 2:
            continue
        x, y, bw, bh = st[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), :4]
        strip[int(max(0, y + .95 * bh)):int(min(h, y + 1.2 * bh)), int(max(0, x - .25 * bw)):int(min(w, x + 1.25 * bw))] = True
    m = strip & np.isfinite(ref) & (ref > 0.3)
    if m.sum() < 50:
        return None, 0.0, int(m.sum())
    vv, uu = np.nonzero(m)
    z = ref[vv, uu].astype(np.float64)
    X = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * z, (vv + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    rng, best, bn = np.random.default_rng(seed), None, 0
    for _ in range(iters):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        inl = np.abs((X - p[0]) @ unit(n)) < tol
        if inl.sum() > bn:
            best, bn = inl, int(inl.sum())
    P = X[best]
    n = np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, 0]
    return (-n if n[1] > 0 else n), bn / len(X), len(X)


def stage_gravity(a):
    gp = Path(a.work) / "geocalib_st" / f"{a.clip}.npz"
    z = dict(np.load(gp))
    fl = json.loads((Path(a.out) / "floor" / f"{a.clip}.json").read_text())
    ug = unit(unit(z.get("up_cam_geocalib", z["up_cam"]).astype(np.float64)).mean(0))
    use, d = False, None
    if fl.get("normal") is not None:
        n = unit(np.asarray(fl["normal"], np.float64))
        d = float(np.degrees(np.arccos(np.clip(n @ ug, -1, 1))))
        use = fl["inliers"] >= a.floor_min_inl and fl["points"] >= a.floor_min_pts and d <= a.floor_max_deg
    z.setdefault("up_cam_geocalib", z["up_cam"])
    if use:
        u = n if a.floor_mode == "floor" else unit(n + ug)
        z["up_cam"] = np.repeat(u[None], len(z["up_cam_geocalib"]), 0)
        z["up_floor"] = n
    z["up_source"] = (a.floor_mode if a.floor_mode == "floor" else "floor+geocalib") if use else "geocalib"
    np.savez(gp, **z)
    fl.update(used=bool(use), floor_vs_geocalib_deg=d)
    (Path(a.out) / "floor" / f"{a.clip}.json").write_text(json.dumps(fl) + "\n")
    print(f"[static] {a.clip}: gravity from {z['up_source']} (floor inliers {fl['inliers']:.2f}, "
          f"{fl['points']} points, {d if d is None else round(d, 2)} deg from GeoCalib)", flush=True)


def stage_depth(a):
    import torch
    from moge.model.v2 import MoGeModel
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    t0 = time.perf_counter()
    imgs = frames_of(a)
    n = len(imgs)
    f = a.focal or float(json.loads((Path(a.work) / "focal" / f"fixed_{a.clip}.json").read_text())[a.clip])
    H, W = cv2.imread(imgs[0]).shape[:2]
    d = a.down
    h, w = H // d, W // d
    ks = sorted(set(range(0, n, a.stride)) | {n - 1})
    fov = float(np.degrees(2 * np.arctan(W / 2 / f)))
    moge = MoGeModel.from_pretrained(a.moge).cuda().eval()
    proc = AutoImageProcessor.from_pretrained(M2F)
    seg = Mask2FormerForUniversalSegmentation.from_pretrained(M2F).cuda().eval()
    lab = {i: s.split(";")[0].strip() for i, s in seg.config.id2label.items()}
    dyn_id = [i for i, s in lab.items() if s in DYNAMIC]
    per_id, sky_id = [i for i, s in lab.items() if s == "person"], [i for i, s in lab.items() if s == "sky"]
    depth, dyn = np.zeros((len(ks), h, w), np.float32), np.zeros((len(ks), h, w), bool)
    person, sky = np.zeros((len(ks), h, w), bool), np.zeros((len(ks), h, w), bool)
    ker = np.ones((2 * a.dilate + 1, 2 * a.dilate + 1), np.uint8)
    for b in range(0, len(ks), a.batch):
        ims = [cv2.imread(imgs[k])[:, :, ::-1] for k in ks[b: b + a.batch]]
        x = torch.tensor(np.stack(ims), dtype=torch.float32, device="cuda").permute(0, 3, 1, 2) / 255
        with torch.no_grad():
            o = moge.infer(x, fov_x=fov, resolution_level=a.moge_level)
            z = torch.where(o["mask"], o["depth"], torch.zeros_like(o["depth"])).float()
            z = torch.nn.functional.interpolate(z[:, None], size=(h, w), mode="nearest")[:, 0]
            depth[b: b + len(ims)] = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
            s = 640 / max(H, W)
            sm = proc(images=[cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) for im in ims], return_tensors="pt")
            so = seg(pixel_values=sm["pixel_values"].cuda(), pixel_mask=sm["pixel_mask"].cuda())
            for i, m in enumerate(proc.post_process_semantic_segmentation(so, target_sizes=[(h, w)] * len(ims))):
                m = m.cpu().numpy()
                dyn[b + i] = cv2.dilate(np.isin(m, dyn_id).astype(np.uint8), ker) > 0
                person[b + i], sky[b + i] = np.isin(m, per_id), np.isin(m, sky_id)
    t_net = time.perf_counter() - t0
    # one scene: per frame a scale to the per-pixel median depth on the static pixels, then the static pixels = that median
    stat = ~dyn & (depth > 0)
    ref = np.nanmedian(np.where(stat, depth, np.nan), 0)
    scale = np.ones(len(ks))
    for i in range(len(ks)):
        m = stat[i] & np.isfinite(ref)
        if m.sum() >= 100:
            scale[i] = float(np.median(ref[m] / depth[i][m]))
    depth *= scale[:, None, None].astype(np.float32)
    fill = np.isfinite(ref)[None] & ~dyn
    depth = np.where(fill, np.nan_to_num(ref)[None], np.where(dyn, depth, 0.0)).astype(np.float32)
    K = np.array([[f / d, 0, W / 2 / d], [0, f / d, H / 2 / d], [0, 0, 1.0]])
    out = Path(a.out)
    for s in ("cams", "intrinsics", "dense"):
        (out / s).mkdir(parents=True, exist_ok=True)
    np.savez(out / "cams" / f"{a.clip}.npz", T_world_cam=np.tile(np.eye(4), (n, 1, 1)), fx=f, static_fast=True)
    np.savez(out / "intrinsics" / f"{a.clip}.npz", inds=np.arange(n), data=np.tile([f, f, W / 2, H / 2], (n, 1)))
    np.savez_compressed(out / "dense" / f"{a.clip}.npz", depth=depth.astype(np.float16), dyn=dyn, K=np.repeat(K[None], len(ks), 0),
                        frames=np.array(ks), down=d, full_hw=np.array([H, W]), phrases=json.dumps({}), source="static_fast",
                        frame_scale=scale, model=a.moge, moge_level=a.moge_level, seg_model=M2F)
    n_fl, inl, npts = floor_normal(np.where(np.isfinite(ref), ref, np.nan), K, person)
    (out / "floor").mkdir(parents=True, exist_ok=True)
    (out / "floor" / f"{a.clip}.json").write_text(json.dumps({"normal": None if n_fl is None else n_fl.tolist(), "inliers": inl,
                                                             "points": npts}) + "\n")
    if a.work:  # clean_render.py seg's output on the same frames (its stage is then skipped)
        (Path(a.work) / "seg").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(a.work) / "seg" / f"{a.clip}_seg.npz", used=np.arange(len(ks)), sky=sky, person=person, model=M2F)
    print(f"[static] {a.clip}: {n} frames, fixed camera, focal {f:.1f}, MoGe-2 depth on {len(ks)} frames (stride {a.stride}) "
          f"{H}x{W} -> {h}x{w}, dynamic {dyn.mean():.3f}, depth median {np.median(depth[depth > 0]):.2f} m, frame scale "
          f"p5-p95 {np.percentile(scale, 5):.3f}-{np.percentile(scale, 95):.3f}, networks {t_net:.0f} s, peak GPU "
          f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("check", "depth", "gravity"))
    p.add_argument("--list", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", default="", help="depth: the fab-world work dir (focal/fixed_<clip>.json)")
    p.add_argument("--focal", type=float, default=0.0, help="depth: the focal (px; default: the work dir's)")
    p.add_argument("--check_size", type=int, default=960)
    p.add_argument("--step", type=int, default=15)
    p.add_argument("--max_px", type=float, default=4.0, help="corner shift bound (px at --check_size): RICH val tripods outdoors move 1-2 px")
    p.add_argument("--max_res", type=float, default=1.0, help="homography residual bound (px at --check_size): a RICH val tripod in wind 0.57 px")
    p.add_argument("--min_inl", type=float, default=0.3)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--down", type=int, default=4)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--moge", default="Ruicheng/moge-2-vitl-normal")
    p.add_argument("--moge_level", type=int, default=0, help="MoGe-2 resolution_level (0-9; RICH val: 0 is 3x faster than 9 "
                   "with a lower AbsRel vs the scans)")
    p.add_argument("--floor_min_inl", type=float, default=0.5)
    p.add_argument("--floor_min_pts", type=int, default=2000)
    p.add_argument("--floor_max_deg", type=float, default=5.0)
    p.add_argument("--floor_mode", default="mean", choices=("mean", "floor"))
    a = p.parse_args()
    if a.stage == "check":
        raise SystemExit(0 if stage_check(a) else 10)
    {"depth": stage_depth, "gravity": stage_gravity}[a.stage](a)


if __name__ == "__main__":
    main()
