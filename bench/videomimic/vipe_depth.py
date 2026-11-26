"""ViPE's dense artifacts of a clip (bench/world_vipe.py --depth) -> one compact npz for scene fusion: metric depth
(float16, downsampled by --down, nearest), a dynamic mask (any ViPE instance: person / animal / vehicle / ball / sky;
dilated by --dilate pixels at the reduced size) and the per-frame intrinsics at the reduced size.
Runs in ViPE's env (reads the zipped EXR / PNG artifacts with vipe.utils.io):
    $FAB_ENVS/vipe12/bin/python -m bench.videomimic.vipe_depth --vipe_out <vipe out> --vipe $FAB_ENVS/src/vipe12 --name <clip>
-> <vipe_out>/dense/<name>.npz: depth (T, h, w) f16 (m, ViPE's own metric scale), dyn (T, h, w) bool, K (T, 3, 3),
   frames (T,), down, phrases (json)
A chunked ViPE run (bench/world_vipe.py: <name>__c<k> artifacts, cams/<name>.npz chunk_starts / chunk_owner /
chunk_scales): each frame from the chunk that owns it, its depth x that chunk's stitch scale. A strided ViPE run
(cams/<name>.npz vipe_frames): ViPE's frame k is the clip's frame vipe_frames[k].
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FAB_HOME = os.path.expanduser(os.environ.get("FAB_HOME", "~/.cache/fab"))  # as fab.paths (this runs in ViPE's env)
FAB_ENVS = os.path.expanduser(os.environ.get("FAB_ENVS", os.path.join(FAB_HOME, "envs")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vipe_out", required=True)
    p.add_argument("--name", default="", help="clip names, comma separated (default: every depth zip)")
    p.add_argument("--down", type=int, default=4)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--vipe", default=os.environ.get("VIPE_DIR", os.path.join(FAB_ENVS, "src", "vipe")))
    args = p.parse_args()
    sys.path.insert(0, args.vipe)
    import cv2
    from vipe.utils.io import read_depth_artifacts, read_instance_artifacts, read_instance_phrases

    base = Path(args.vipe_out)
    names = args.name.split(",") if args.name else sorted({f.stem.split("__c")[0] for f in (base / "depth").glob("*.zip")})
    os.makedirs(base / "dense", exist_ok=True)
    for name in names:
        out = base / "dense" / f"{name}.npz"
        cp = base / "cams" / f"{name}.npz"
        chunked = not (base / "depth" / f"{name}.zip").exists() and cp.exists() and "chunk_owner" in np.load(cp).files \
            and len(np.load(cp)["chunk_starts"]) > 1
        if out.exists() or not ((base / "depth" / f"{name}.zip").exists() or chunked):
            continue
        d = args.down
        if chunked:
            merge_chunks(base, name, d, args.dilate, out, cv2, read_depth_artifacts, read_instance_artifacts)
            continue
        dep = {i: x.numpy() for i, x in read_depth_artifacts(base / "depth" / f"{name}.zip")}
        frames = np.array(sorted(dep))
        H, W = dep[frames[0]].shape
        d = vipe_down(cp, W, d)
        h, w = H // d, W // d
        depth = np.stack([cv2.resize(dep[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in frames]).astype(np.float16)
        dyn = np.zeros((len(frames), h, w), bool)
        mz = base / "mask" / f"{name}.zip"
        phrases = {}
        if mz.exists():
            pos = {int(f): k for k, f in enumerate(frames)}
            ker = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)
            for i, m in read_instance_artifacts(mz):
                if i in pos:
                    mm = cv2.resize((m.numpy() > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    dyn[pos[i]] = cv2.dilate(mm, ker) > 0 if args.dilate else mm > 0
            pp = base / "mask" / f"{name}.txt"
            phrases = read_instance_phrases(pp) if pp.exists() else {}
        intr = np.load(base / "intrinsics" / f"{name}.npz")
        ii = {int(k): v for k, v in zip(intr["inds"], intr["data"])}
        sx, sy = dense_scale(cp, H, W, d)
        K = np.zeros((len(frames), 3, 3))
        for k, f in enumerate(frames):
            fx, fy, cx, cy = ii.get(int(f), intr["data"][0])[:4]
            K[k] = [[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]
        sel = np.load(cp)["vipe_frames"] if cp.exists() and "vipe_frames" in np.load(cp).files else None
        np.savez_compressed(out, depth=depth, dyn=dyn, K=K, frames=frames if sel is None else sel[frames], down=d,
                            full_hw=np.array([H, W]), phrases=json.dumps({str(k): v for k, v in phrases.items()}))
        print(f"[dense] {name}: {len(frames)} frames {H}x{W} -> {h}x{w}, dynamic {dyn.mean():.3f}, "
              f"depth median {np.nanmedian(depth.astype(np.float32)):.2f} m", flush=True)


def full_hw(cp, H, W):
    """The clip's full frame size (cams full_hw of a --max_side run), else the depth's own size."""
    z = np.load(cp) if Path(cp).exists() else None
    return tuple(int(x) for x in z["full_hw"]) if z is not None and "full_hw" in z.files else (H, W)


def vipe_down(cp, W, d):
    """The down factor for depth of width W: the same dense size as the full frames / d (a --max_side run: smaller)."""
    return max(1, int(round(d * W / full_hw(cp, 1, W)[1])))


def dense_scale(cp, H, W, d):
    """Full-frame intrinsics -> the dense grid (the depth's H x W / d)."""
    H0, W0 = full_hw(cp, H, W)
    if (H0, W0) == (H, W):  # ViPE at the full frames: K / d, as before
        return 1.0 / d, 1.0 / d
    return (W // d) / W0, (H // d) / H0


def merge_chunks(base, name, d, dilate, out, cv2, read_depth_artifacts, read_instance_artifacts):
    """The dense npz of a chunked run: per chunk its owned frames (global = chunk start + local), depth x its stitch scale
    (1 for the first), its dynamic mask and intrinsics."""
    z = np.load(base / "cams" / f"{name}.npz")
    starts, owner, scales = z["chunk_starts"].astype(int), z["chunk_owner"].astype(int), [1.0] + list(z["chunk_scales"])
    d0 = d
    sel = z["vipe_frames"] if "vipe_frames" in z.files else np.arange(len(z["T_world_cam"]))
    n = len(sel)  # ViPE's frames (a strided run: every stride-th)
    ker = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    D, M, K, F = [], [], [], []
    for c, s in enumerate(starts):
        part = f"{name}__c{c}"
        lo, hi = owner[c] - s, (owner[c + 1] if c + 1 < len(owner) else n) - s
        dep = {i: x.numpy() for i, x in read_depth_artifacts(base / "depth" / f"{part}.zip") if lo <= i < hi}
        fr = sorted(dep)
        H, W = dep[fr[0]].shape
        d = vipe_down(base / "cams" / f"{name}.npz", W, d0)
        h, w = H // d, W // d
        sx, sy = dense_scale(base / "cams" / f"{name}.npz", H, W, d)
        pos = {f: k for k, f in enumerate(fr)}
        dyn = np.zeros((len(fr), h, w), bool)
        mz = base / "mask" / f"{part}.zip"
        if mz.exists():
            for i, m in read_instance_artifacts(mz):
                if i in pos:
                    mm = cv2.resize((m.numpy() > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    dyn[pos[i]] = cv2.dilate(mm, ker) > 0 if dilate else mm > 0
        intr = np.load(base / "intrinsics" / f"{part}.npz")
        ii = {int(k): v for k, v in zip(intr["inds"], intr["data"])}
        for k, f in enumerate(fr):
            fx, fy, cx, cy = ii.get(int(f), intr["data"][0])[:4]
            K.append([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
            D.append((cv2.resize(dep[f], (w, h), interpolation=cv2.INTER_NEAREST) * scales[c]).astype(np.float16))
            F.append(int(sel[s + f]))
        M.append(dyn)
    depth, dyn = np.stack(D), np.concatenate(M)
    np.savez_compressed(out, depth=depth, dyn=dyn, K=np.array(K), frames=np.array(F), down=d, full_hw=np.array([H, W]),
                        phrases=json.dumps({}), chunks=len(starts), chunk_scales=np.array(scales))
    print(f"[dense] {name}: {len(F)} frames from {len(starts)} chunks (scales {np.round(scales, 3).tolist()}) {H}x{W} -> "
          f"{h}x{w}, dynamic {dyn.mean():.3f}, depth median {np.nanmedian(depth.astype(np.float32)):.2f} m", flush=True)


if __name__ == "__main__":
    main()
