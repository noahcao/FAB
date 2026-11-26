"""Optional multi-frame focal estimates for the static-camera fast path's focal rule (scripts/fab_world.sh --focal_rule nc3):
  pi4   Pi4 (github.com/yyfz/Pi4 @ 32dfbea, CC BY-NC 4.0 code and weights): its camera-ray head on --n evenly spaced
        frames (512 px long side), per frame fx / fy fitted to the rays (Huber; after FlowHMR's
        flowhmr/utils/runtime_camera_backends.py, Apache-2.0)                          (needs --pi4_dir, --pi4_weights)
  vggt  VGGT-Omega (github.com/facebookresearch/vggt-omega, FAIR Noncommercial Research License): its pose encoding on the
        same frames (512 px), encoding_to_camera intrinsics                            (needs --vggt_dir, --vggt_ckpt)
Both non-commercial: opt-in only. Per clip the median focal ((fx fy)^0.5) over the frames, written into the same json as
bench/scene_recon/focal_compare.py ({clip: {stage: {f_median, f_all, width}}}). Runs in any env with torch + their
imports (e.g. FlowHMR's), set by FAB_NC_PY in fab_world.sh.
    $FAB_NC_PY -m bench.scene_recon.focal_nc pi4 --list L --out J --pi4_dir <pi4 models dir> --pi4_weights <model.safetensors>
    $FAB_NC_PY -m bench.scene_recon.focal_nc vggt --list L --out J --vggt_dir <vggt-omega checkout> --vggt_ckpt <.pt>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def frames_of(images, n):
    return np.unique(np.linspace(0, len(images) - 1, n).round().astype(int))


def fit_ray_intrinsics(rays, pixels):
    """u = fx * ray_x + cx, v = fy * ray_y + cy (independent Huber fits) -> 3x3 K."""
    rays, pixels = np.asarray(rays, np.float64).reshape(-1, 2), pixels.reshape(-1, 2)
    ok = np.isfinite(rays).all(1)
    rays, pixels = rays[ok], pixels[ok]
    w = np.ones_like(rays)
    for it in range(6):
        rc, pc = np.average(rays, 0, w), np.average(pixels, 0, w)
        c = rays - rc
        focal = (w * c * (pixels - pc)).sum(0) / (w * c ** 2).sum(0)
        pp = pc - focal * rc
        if it < 5:
            r = pixels - (focal * rays + pp)
            mad = np.median(abs(r - np.median(r, 0)), 0)
            w = np.minimum(1.0, 1.345 * np.maximum(0.1, 1.4826 * mad) / np.maximum(abs(r), 1e-8))
    return np.array([[focal[0], 0, pp[0]], [0, focal[1], pp[1]], [0, 0, 1.0]])


def pi4_focals(imgs, pi4_dir, weights):
    from PIL import Image
    from safetensors import safe_open

    sys.path.insert(0, str(Path(pi4_dir).parent))
    pkg = Path(pi4_dir).name
    enc = __import__(f"{pkg}.encoder", fromlist=["Encoder"])
    dec = __import__(f"{pkg}.decoder", fromlist=["GeometryDecoder"])
    cam = __import__(f"{pkg}.heads.camera", fromlist=["CameraHead"])
    met = __import__(f"{pkg}.heads.metric", fromlist=["MetricHead"])
    lay = __import__(f"{pkg}.layers", fromlist=["ConvPyramid"])
    with torch.device("meta"):
        ph = torch.nn.ModuleDict({"points": lay.ConvPyramid((512, 256, 128, 64, 32), 2)})  # the two ray channels only
        model = torch.nn.ModuleDict(dict(encoder=enc.Encoder(), geometry_decoder=dec.GeometryDecoder(),
                                         camera_head=cam.CameraHead(), metric_head=met.MetricHead(), point_head=ph))
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        st = {k: (f.get_slice(k)[:2] if k.startswith("point_head.points.output.") else f.get_tensor(k)) for k in model.state_dict()}
    model.load_state_dict(st, strict=True, assign=True)
    model.eval().cuda()
    frames = [cv2.imread(p) for p in imgs]
    H, W = frames[0].shape[:2]
    r = 512 / max(H, W)
    size = tuple(max(16, round(n * r / 16) * 16) for n in (W, H))
    x = np.stack([np.asarray(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize(size, Image.Resampling.LANCZOS)) for f in frames])
    x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float().div_(255)
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    pos = lay.patch_positions(len(x), size[1] // 16, size[0] // 16, x.device)
    out = []
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            geo, _ = model.geometry_decoder(model.encoder(x), pos, 1, len(x))
        grid = np.moveaxis(np.indices(size[::-1])[::-1], 0, -1)[4:-4:4, 4:-4:4]
        pix = (grid + 0.5) * (np.array([W, H]) / size) - 0.5
        for t in geo[:, 5:].split(1):
            rays, _ = model.point_head.points(lay.feature_pyramid(t.float(), size[1], size[0], 5))
            K = fit_ray_intrinsics(rays[0, :, 4:-4:4, 4:-4:4].permute(1, 2, 0).cpu().numpy(), pix)
            out.append(float(np.sqrt(K[0, 0] * K[1, 1])))
    return out, W


def vggt_focals(imgs, vggt_dir, ckpt):
    sys.path.insert(0, str(vggt_dir))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.cuda()
    H, W = cv2.imread(imgs[0]).shape[:2]
    x = load_and_preprocess_images(list(imgs), mode="max_size", image_resolution=512).cuda()
    with torch.inference_mode():
        p = model(x)
        _, K = encoding_to_camera(p["pose_enc"], (H, W))  # K in the original frame's pixels
    K = K[0] if K.ndim == 4 else K
    K = K.float().cpu().numpy()
    return [float(np.sqrt(k[0, 0] * k[1, 1])) for k in K], W


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("pi4", "vggt"))
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--only", default="")
    p.add_argument("--pi4_dir", default=os.environ.get("FAB_PI4_DIR", ""))
    p.add_argument("--pi4_weights", default=os.environ.get("FAB_PI4_WEIGHTS", ""))
    p.add_argument("--vggt_dir", default=os.environ.get("FAB_VGGT_DIR", ""))
    p.add_argument("--vggt_ckpt", default=os.environ.get("FAB_VGGT_CKPT", ""))
    a = p.parse_args()
    clips = json.loads(Path(a.list).read_text())
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    for c in sorted(clips):
        if (a.only and not any(o in c for o in a.only.split(","))) or a.stage in res.get(c, {}):
            continue
        imgs = [clips[c]["images"][i] for i in frames_of(clips[c]["images"], a.n)]
        fs, W = pi4_focals(imgs, a.pi4_dir, a.pi4_weights) if a.stage == "pi4" else vggt_focals(imgs, a.vggt_dir, a.vggt_ckpt)
        res.setdefault(c, {})[a.stage] = {"f_median": float(np.median(fs)), "f_all": fs, "width": int(W)}
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print(a.stage, c, round(float(np.median(fs)), 1), flush=True)


if __name__ == "__main__":
    main()
