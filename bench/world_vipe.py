"""World camera from NVIDIA ViPE (bench/baselines/setup_vipe.sh, setup_vipe12.sh) on a clip list, EMDB or 3DPW
(`vipe infer`: default pipeline -- GeoCalib intrinsics optimised in SLAM, dynamic-object masks, UniDepth keyframe depth,
metric scale; pose-only unless --depth): the frames are written to an mp4 (near-lossless), ViPE's run.py is called in its
env, its c2w poses are filled to every frame.
Run in any env (it only calls ViPE's python):
    python -m bench.world_vipe --list clips.json --out <dir> --python $FAB_ENVS/vipe12/bin/python --vipe $FAB_ENVS/src/vipe12 --depth
    python -m bench.world_vipe --emdb <EMDB root> --out <dir> --gpu 0 --shard 0 --num_shards 4
    (--subset dev: the calibration sequences; --threedpw <3DPW root> --split test: 3dpw_test_<seq>.npz;
     videos > --chunk frames run as overlapping chunks <name>__c<k>, Sim(3)-stitched on the overlaps (each frame from the
     chunk that owns it: the overlaps split in the middle), the chunks' intrinsics merged into <out>/intrinsics/<name>.npz
     and their depth by bench/videomimic/vipe_depth.py; --chunk_mpix: chunk by frames x megapixels instead (GPU memory);
     --keyframe_depth metric3d-large: another metric depth prior for ViPE's scale; --stride S: ViPE on every S-th frame
     (+ the last) at fps / S, its poses interpolated to every frame (slerp / linear), its depth on those frames only)
-> <out>/cams/<name>.npz: T_world_cam (n, 4, 4; camera -> world, ViPE metric), index, fx (ViPE's median focal),
   timing; ViPE's own outputs under <out>/pose, <out>/intrinsics, logs under <out>/logs
ViPE's model downloads go to $FAB_HOME/hf-cache/vipe (FAB_HOME default ~/.cache/fab).
"""

import argparse
import glob
import json
import os
import pickle
import subprocess
import tempfile
import time

import numpy as np

FAB_HOME = os.path.expanduser(os.environ.get("FAB_HOME", "~/.cache/fab"))  # as fab.paths (this runs in ViPE's env)
FAB_ENVS = os.path.expanduser(os.environ.get("FAB_ENVS", os.path.join(FAB_HOME, "envs")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emdb", default="")
    p.add_argument("--subset", default="emdb2", help="emdb1 / emdb2 / dev (EMDB in neither benchmark: calibration)")
    p.add_argument("--threedpw", default="", help="3DPW root: run on sequenceFiles/<--split> instead of EMDB")
    p.add_argument("--split", default="test")
    p.add_argument("--list", default="", help="json {name: {images, fps}} of explicit frame lists")
    p.add_argument("--only", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "vipe_work"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--python", default=os.environ.get("FAB_VIPE_PY", os.path.join(FAB_ENVS, "vipe", "bin", "python")))
    p.add_argument("--vipe", default=os.environ.get("VIPE_DIR", os.path.join(FAB_ENVS, "src", "vipe")))
    p.add_argument("--keyframe_depth", default="", help="ViPE metric depth prior, e.g. metric3d-large (default unidepth-l)")
    p.add_argument("--chunk", type=int, default=3000, help="videos longer than this run in overlapping chunks (0 = off)")
    p.add_argument("--chunk_size", type=int, default=1200,
                   help="target chunk length (0 = --chunk); very long chunks (~2,000 frames) can fail")
    p.add_argument("--overlap", type=int, default=300)
    p.add_argument("--chunk_mpix", type=float, default=0.0,
                   help="> 0: chunk when frames x megapixels > this (chunk length = this / megapixels; --chunk, --chunk_size "
                        "then unused), e.g. what fits the GPU with --depth")
    p.add_argument("--max_side", type=int, default=0,
                   help="> 0: ViPE on frames resized so the long side is at most this (px); its intrinsics rescaled to the "
                        "full frames, the depth kept at ViPE's size (cams vipe_scale; vipe_depth.py rescales K)")
    p.add_argument("--no_instance", type=int, default=0,
                   help="1: ViPE without its instance segmentation (GroundingDINO / SAM / DeAOT): no dynamic masks in SLAM "
                        "or the dense step (the world chain masks people with its own Mask2Former seg)")
    p.add_argument("--chunk_real", type=int, default=0,
                   help="1: with --stride, chunk length and overlap in real frames (as without a stride), not ViPE's frames")
    p.add_argument("--stride", type=int, default=1,
                   help="> 1: ViPE (SLAM + depth) on every stride-th frame and the last; poses interpolated to every frame, "
                        "depth only on those frames (cams/<name>.npz vipe_frames)")
    p.add_argument("--pipeline", default="default", help="ViPE pipeline config (ViPE >= 1.2: dav3 = DA3 keyframes + mvd_dav3 depth)")
    p.add_argument("--depth_align", default="", help="with --depth: ViPE's depth_align_model (default adaptive_unidepth-l_svda)")
    p.add_argument("--depth", action="store_true",
                   help="also run ViPE's metric video depth (adaptive_unidepth-l_svda) -> <out>/depth/<name>.zip (scene meshing)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--fixed_focal_json", default="",
                   help="{clip: focal px} -> ViPE with K fixed (fx = fy = focal, centred principal point; bench/vipe_fixed_k.py)")
    args = p.parse_args()
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation, Slerp

    for k in ("cams", "logs"):
        os.makedirs(os.path.join(args.out, k), exist_ok=True)
    for name, imgs, fps in videos(args)[args.shard :: args.num_shards]:
        out_path = os.path.join(args.out, "cams", f"{name}.npz")
        if os.path.exists(out_path):
            continue
        n_all, imgs_all = len(imgs), imgs
        os.makedirs(args.work, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                   HF_HOME=os.path.join(FAB_HOME, "hf-cache", "vipe"),
                   TORCH_HOME=os.path.join(FAB_HOME, "hf-cache", "vipe", "torch"))
        if args.fixed_focal_json:  # the clip's fixed focal (no entry: skipped)
            ff = json.load(open(args.fixed_focal_json)).get(name)
            if ff is None:
                print(f"[vipe] {name}: no fixed focal, skipped", flush=True)
                continue
            h0, w0 = imageio.imread(imgs_all[0]).shape[:2]
            r0 = vipe_scale(args, h0, w0)
            env["VIPE_FIXED_K"] = f"{float(ff) * r0},{float(ff) * r0},{w0 * r0 / 2},{h0 * r0 / 2}"
        # --stride S: ViPE on every S-th frame; if ViPE fails at S (e.g. its depth alignment finds too few SLAM points),
        # the whole clip again at S - 1, ..., 1 (logged; the failed attempts' time counts in vipe_s)
        t_enc, t_vipe, tried = 0.0, 0.0, []
        for st in range(max(args.stride, 1), 0, -1):
            sel = np.unique(np.r_[np.arange(0, n_all, st), n_all - 1]) if st > 1 else np.arange(n_all)
            imgs, args.fps = [imgs_all[i] for i in sel], fps / st
            n = len(imgs)
            # long videos: overlapping chunks (ViPE's SLAM runs out of GPU memory on long videos), Sim(3)-stitched
            starts, size = [0], n
            div = st if args.chunk_real else 1  # --chunk_real: chunks and overlaps of the same real duration at any stride
            ovl = max(args.overlap // div, 1)
            chunk, chunk_size = args.chunk // div, args.chunk_size // div
            if args.chunk_mpix > 0:
                h0, w0 = imageio.imread(imgs[0]).shape[:2]
                chunk = chunk_size = max(int(args.chunk_mpix / (h0 * w0 / 1e6) / div), 2 * ovl)
            if chunk and n > chunk:
                k = int(np.ceil((n - ovl) / ((chunk_size or chunk) - ovl)))
                step = int(np.ceil((n - ovl) / k))
                starts, size = [i * step for i in range(k)], step + ovl
            T, fx, scales, owner = None, [], [], [0]
            for c, s in enumerate(starts):
                part = f"{name}__c{c}" if len(starts) > 1 else name
                r = run_vipe(args, imgs[s : s + size], part, env, imageio, Rotation, Slerp)
                if r[0] is None:
                    t_enc, t_vipe = t_enc + r[2], t_vipe + r[3]
                    T = None
                    break
                Tc, f_c, te, tv = r
                fx.append(f_c)
                t_enc, t_vipe = t_enc + te, t_vipe + tv
                if T is None:
                    T = Tc
                    continue
                ov = len(T) - s
                Tc, sc = stitch(T[-ov:], Tc)
                scales.append(sc)
                owner.append(s + ov // 2)
                T = np.concatenate([T[: s + ov // 2], Tc[ov // 2 :]])
            tried.append(st)
            if T is not None:
                break
            print(f"[vipe] {name}: ViPE failed at stride {st}" + (f", falling back to stride {st - 1}" if st > 1 else ""), flush=True)
            for f in glob.glob(os.path.join(args.work, f"{name}*.mp4")):  # the next stride's frames
                os.remove(f)
        if T is None:
            continue
        if len(starts) > 1:  # the chunks' intrinsics on the frames each chunk owns -> <out>/intrinsics/<name>.npz
            inds, data = [], []
            for c, s in enumerate(starts):
                z = np.load(os.path.join(args.out, "intrinsics", f"{name}__c{c}.npz"))
                g = np.asarray(z["inds"]).astype(int) + s
                m = (g >= owner[c]) & (g < (owner[c + 1] if c + 1 < len(owner) else n))
                inds.append(sel[g[m]])
                data.append(np.asarray(z["data"])[m])
            np.savez(os.path.join(args.out, "intrinsics", f"{name}.npz"), inds=np.concatenate(inds), data=np.concatenate(data))
        if n < n_all:  # strided: ViPE's frames sel -> every frame (rotations slerped, centres linear)
            q, Tf = np.arange(n_all), np.tile(np.eye(4), (n_all, 1, 1))
            Tf[:, :3, :3] = Slerp(sel, Rotation.from_matrix(T[:, :3, :3]))(q).as_matrix()
            for k in range(3):
                Tf[:, k, 3] = np.interp(q, sel, T[:, k, 3])
            T = Tf
        np.savez(out_path, T_world_cam=T, fx=float(np.median(fx)), encode_s=t_enc, vipe_s=t_vipe,
                 chunk_starts=np.array(starts), chunk_size=size, chunk_scales=np.array(scales), chunk_owner=np.array(owner),
                 keyframe_depth=args.keyframe_depth or "unidepth-l", stride=st, stride_requested=args.stride,
                 strides_tried=np.array(tried), vipe_frames=sel, chunk_real=args.chunk_real,
                 vipe_scale=vipe_scale(args, *imageio.imread(imgs_all[0]).shape[:2]), no_instance=args.no_instance,
                 full_hw=np.array(imageio.imread(imgs_all[0]).shape[:2]))
        print(f"[vipe] {name}: {n_all} frames ({n} to ViPE, stride {st}{', fallback from ' + str(args.stride) if st != args.stride else ''}), {len(starts)} chunk(s) (stitch scales "
              f"{np.round(scales, 3).tolist()}), ViPE {t_vipe:.0f} s, fx {np.median(fx):.1f}", flush=True)


def videos(args):
    """[(name, frame paths, fps)] of --emdb --subset (emdb1 / emdb2 / dev), --threedpw --split or a --list json
    ({name: {"images": [...], "fps": f}}), filtered by --only."""
    seqs, out = [], []
    if getattr(args, "list", ""):
        with open(args.list) as f:
            clips = json.load(f)
        return [(k, v["images"], int(v.get("fps", 30))) for k, v in sorted(clips.items())
                if not args.only or any(o in k for o in args.only.split(","))]
    if args.threedpw:
        seqs = sorted(glob.glob(os.path.join(args.threedpw, "sequenceFiles", args.split, "*.pkl")))
    for pkl in [] if args.threedpw else sorted(glob.glob(os.path.join(args.emdb, "P*/*/*_data.pkl"))):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        if args.subset == "dev":  # EMDB outside EMDB-1 / EMDB-2
            if not seq.get("emdb1", False) and not seq.get("emdb2", False):
                seqs.append(pkl)
        elif seq.get(args.subset, False):
            seqs.append(pkl)
    for pkl in seqs:
        if args.threedpw:
            name = f"3dpw_{args.split}_{os.path.basename(pkl)[:-4]}"
            img_glob = os.path.join(args.threedpw, "imageFiles", os.path.basename(pkl)[:-4], "image_*.jpg")
        else:
            name = f"{args.subset}_{os.path.basename(os.path.dirname(os.path.dirname(pkl)))}_{os.path.basename(os.path.dirname(pkl))}"
            img_glob = os.path.join(os.path.dirname(pkl), "images", "*.jpg")
        if not args.only or any(o in name for o in args.only.split(",")):
            out.append((name, sorted(glob.glob(img_glob)), getattr(args, "fps", 30)))
    return out


def run_vipe(args, imgs, name, env, imageio, Rotation, Slerp):
    """One ViPE run on the frames imgs (mp4 <work>/<name>.mp4) -> (T_world_cam filled to every frame, fx, s, s);
    a failed run: (None, None, s, s)."""
    mp4 = os.path.join(args.work, f"{name}.mp4")
    t0 = time.perf_counter()
    h0, w0 = imageio.imread(imgs[0]).shape[:2]
    r = vipe_scale(args, h0, w0)
    if not os.path.exists(mp4):
        w = imageio.get_writer(mp4 + ".tmp.mp4", fps=args.fps, codec="libx264", quality=10, macro_block_size=1,
                               ffmpeg_params=["-crf", "12"])
        for f in imgs:
            a = imageio.imread(f)
            if r < 1:  # --max_side: even size (libx264), area resampling
                from PIL import Image

                a = np.asarray(Image.fromarray(a).resize((int(round(w0 * r / 2)) * 2, int(round(h0 * r / 2)) * 2), Image.BOX))
            w.append_data(a)
        w.close()
        os.replace(mp4 + ".tmp.mp4", mp4)
    t1 = time.perf_counter()
    run_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vipe_fixed_k.py") if "VIPE_FIXED_K" in env else "run.py"
    cmd = [args.python, run_py, f"pipeline={getattr(args, 'pipeline', 'default')}", "streams=raw_mp4_stream", f"streams.base_path={mp4}",
           f"streams.frame_end={len(imgs) + 1}", f"pipeline.output.path={args.out}",
           "pipeline.output.save_artifacts=true", "pipeline.output.save_viz=false"]
    if getattr(args, "pipeline", "default").startswith("pose_only"):  # poses only, no post-processing config
        pass
    elif not getattr(args, "depth", False):  # --depth: ViPE's default post-processing (metric video depth -> <out>/depth)
        cmd.append("pipeline.post.depth_align_model=null")
    elif getattr(args, "depth_align", ""):  # e.g. adaptive_moge2-l_svda (ViPE >= 1.2)
        cmd.append(f"pipeline.post.depth_align_model={args.depth_align}")
    if args.keyframe_depth:  # ViPE's metric depth prior for the SLAM scale (default unidepth-l)
        cmd.append(f"pipeline.slam.keyframe_depth={args.keyframe_depth}")
    if getattr(args, "no_instance", 0):  # no GroundingDINO / SAM / DeAOT instance masks
        cmd.append("pipeline.init.instance=null")
    if "VIPE_FIXED_K" in env:  # K fixed (bench/vipe_fixed_k.py launches run.py), not optimised in SLAM
        cmd.append("pipeline.init.intrinsics=gt")
    with open(os.path.join(args.out, "logs", f"{name}.log"), "w") as f:
        rc = subprocess.call(cmd, cwd=args.vipe, env=env, stdout=f, stderr=subprocess.STDOUT)
    t2 = time.perf_counter()
    pose_path = os.path.join(args.out, "pose", f"{name}.npz")
    if rc != 0 or not os.path.exists(pose_path):
        print(f"[vipe] {name}: failed (rc {rc}), see {args.out}/logs/{name}.log", flush=True)
        return None, None, t1 - t0, t2 - t1
    d = np.load(pose_path)
    idx, Tk = np.asarray(d["inds"]).astype(int), d["data"].astype(np.float64)
    n = len(imgs)
    T = np.tile(np.eye(4), (n, 1, 1))
    q = np.clip(np.arange(n), idx[0], idx[-1])
    T[:, :3, :3] = Slerp(idx, Rotation.from_matrix(Tk[:, :3, :3]))(q).as_matrix()
    for k in range(3):
        T[:, k, 3] = np.interp(q, idx, Tk[:, k, 3])
    ip = os.path.join(args.out, "intrinsics", f"{name}.npz")
    if r < 1:  # ViPE's intrinsics are for the resized frames: back to the full frames (fx, cx / rw; fy, cy / rh), once
        z = dict(np.load(ip))
        if "vipe_hw" not in z:
            rw, rh = (int(round(w0 * r / 2)) * 2) / w0, (int(round(h0 * r / 2)) * 2) / h0
            z["data"] = np.asarray(z["data"], np.float64).copy()
            z["data"][:, [0, 2]] /= rw
            z["data"][:, [1, 3]] /= rh
            np.savez(ip, **z, vipe_hw=np.array([int(round(h0 * r / 2)) * 2, int(round(w0 * r / 2)) * 2]), full_hw=np.array([h0, w0]))
    intr = np.load(ip)["data"]
    return T, float(np.median(intr[:, 0])), t1 - t0, t2 - t1


def vipe_scale(args, h, w):
    """The resize factor of --max_side (1: none)."""
    m = getattr(args, "max_side", 0)
    return min(1.0, m / max(h, w)) if m and m > 0 else 1.0


def stitch(T_prev, T_cur):
    """Sim(3) of a chunk T_cur onto the stitched cameras T_prev over their shared frames (T_prev = the last ov, T_cur
    starts with the same ov frames): rotation = chordal mean of R_prev R_cur^T, scale = least squares of the centred
    camera centres (1 if the overlap barely moves; clamped to 0.8-1.25), translation from the means."""
    ov = len(T_prev)
    U, _, Vt = np.linalg.svd(np.einsum("nij,nkj->ik", T_prev[:, :3, :3], T_cur[:ov, :3, :3]))
    Ra = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    cp, cc = T_prev[:, :3, 3], T_cur[:ov, :3, 3] @ Ra.T
    dp, dc = cp - cp.mean(0), cc - cc.mean(0)
    s = float((dp * dc).sum() / (dc ** 2).sum()) if np.sqrt((dc ** 2).sum(1).mean()) > 0.1 else 1.0
    s = float(np.clip(s, 0.8, 1.25))
    out = T_cur.copy()
    out[:, :3, :3] = Ra @ T_cur[:, :3, :3]
    out[:, :3, 3] = s * T_cur[:, :3, 3] @ Ra.T + (cp.mean(0) - s * cc.mean(0))
    return out, s


if __name__ == "__main__":
    main()
