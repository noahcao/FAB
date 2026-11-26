"""FAB-like inputs for FAB-M training (train/train_world_motion.py): clean motion windows (train/world_motion_data.py) seen
by a simulated camera, with a parametric model of FAB's per-frame errors and the camera's errors. Everything runs batched
on the training device.

Error model (camera frame unless noted; NOISE holds the ranges, --noise_json overrides them, e.g. fitted on RICH val):
  depth       log-depth error = per-clip bias + coloured noise + rare spikes, plus the focal error (FAB's depth
              scales with the focal it is given)
  image       coloured pixel noise of the pelvis, back-projected with the estimated intrinsics
  rotations   coloured root-orientation and per-joint noise; betas: per-clip bias + per-frame noise
  dropouts    gaps (held at the last valid frame, valid = 0, as bench/sonic_motion.run_fab), out-of-view frames
  contacts    the GT static labels as noisy probabilities (static + any), sometimes absent
  camera      gravity tilt (the whole world), moving cameras: rotation drift, translation scale + noise
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from fab.world_motion import aa_to_mat, rz

NOISE = {
    # fitted on FAB own vs GT, RICH val (11 views): log-depth bias p10 / p50 / p90 0.008 /
    # 0.045 / 0.078, residual std 0.010 / 0.0125 / 0.017, its correlation 3.7 / 4.6 / 5.3 frames, no spikes (0.05 %);
    # pelvis pixel residual 0.3-6 % of the box height, correlated over 5-10 frames; root orientation error median 4-7.5 deg
    # (incl. SMPL vs SMPL-X frame offsets); contact head logits +0.9 on GT contact, -2.3 off; jerk 157 vs the GT's 14
    # first differences of the log-depth error: std 0.0054 (median), second differences 0.0086: a near-white part
    # (sigma ~0.004) on top of the slow one
    "depth_bias": (0.045, 0.03),         # mean, std of the per-clip log-depth bias
    "depth_sigma": (0.002, 0.008),       # range of the slow log-depth noise std
    "depth_corr": (4.0, 15.0),           # range of its correlation (frames, Gaussian kernel sigma)
    "depth_white": (0.002, 0.0055),      # range of the white log-depth noise std
    "depth_spike_p": 0.001, "depth_spike": (0.03, 0.12),
    "focal_sigma": 0.05,
    # pelvis pixel noise, fraction of the box height: slow + white (FAB's first differences 0.0021 of the box height)
    "px_sigma": (0.002, 0.006), "px_corr": (2.0, 8.0), "px_white": (0.001, 0.002),
    # rotations: fitted with the same estimators on simulated windows (raw-input WA / jitter / PA vs FAB's 47 / 157 / ~28)
    # (simulated static-camera windows: WA 53.7 / jerk 170 / PA 28.2; first / second log-depth differences 0.0047 / 0.008
    # vs FAB's 0.0054 / 0.0086: FAB's jitter is mostly its depth, its rotation noise is smooth)
    # local-joint second differences: FAB 0.0137 m vs GT 0.0046 (sim with these: 0.0141; PA 29.7, jerk 197, WA 56)
    "root_deg": (1.0, 3.0), "root_corr": (2.0, 8.0), "joint_deg": (1.8, 5.2), "rot_corr": (1.0, 2.5),
    "betas_bias": 0.2, "betas_frame": 0.05,
    "gap_p": 0.4, "gap_len": (1, 45), "gaps": (1, 3),
    "contact_logit": (1.0, 3.0), "contact_bias": -0.7, "contact_noise": 1.0, "contact_absent_p": 0.15,
    "tilt_deg": 1.5, "drift_deg": 0.03, "cam_scale_sigma": 0.08, "cam_noise": 0.01,
}
IMAGES = ((1920, 1080), (1080, 1920), (2056, 1504), (1280, 720), (1504, 2056))
MIRROR = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
MIRROR_C = [2, 3, 0, 1, 5, 4, 7, 6]


def mirror(m, on, J0):
    """Left-right mirror (the world's x and the body's x reflected; the pelvis J0 + trans mirrored) of the windows where
    on (B,) is True."""
    B, T = m["pose"].shape[:2]
    aa = m["pose"].reshape(B, T, 24, 3)[:, :, MIRROR] * torch.tensor([1.0, -1, -1], device=on.device)
    tr = (m["trans"] + J0[:, None]) * torch.tensor([-1.0, 1, 1], device=on.device) - J0[:, None]
    w = on[:, None, None]
    out = dict(m)
    out["pose"] = torch.where(w, aa.reshape(B, T, 72), m["pose"])
    out["trans"] = torch.where(w, tr, m["trans"])
    out["contact"] = torch.where(w, m["contact"][..., MIRROR_C], m["contact"])
    return out


def U(shape, lo, hi, g, dev):
    return lo + (hi - lo) * torch.rand(shape, generator=g, device=dev)


def coloured(B, T, D, sig_lo, sig_hi, g, dev):
    """(B, T, D) unit-variance noise smoothed in time with a per-sample Gaussian of sigma in [sig_lo, sig_hi] frames."""
    s = U((B,), sig_lo, sig_hi, g, dev)
    R = int(math.ceil(3 * sig_hi))
    x = torch.arange(-R, R + 1, device=dev).float()
    k = torch.exp(-0.5 * (x[None] / s[:, None]) ** 2)
    k = k / k.pow(2).sum(1, keepdim=True).sqrt()
    w = torch.randn(B, D, T + 2 * R, generator=g, device=dev)
    out = F.conv1d(w.reshape(1, B * D, -1), k.repeat_interleave(D, 0)[:, None], groups=B * D)
    return out.reshape(B, D, T).transpose(1, 2)


def look_at(c, target, roll, up=None):
    """Camera -> world rotation (OpenCV: x right, y down, z forward) looking from c at target in a z-up world."""
    z = target - c
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    upv = torch.tensor([0.0, 0, 1], device=c.device).expand_as(z)
    x = torch.cross(z, -upv, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], -1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    o, zr = torch.ones_like(cr), torch.zeros_like(cr)
    Rr = torch.stack([cr, -sr, zr, sr, cr, zr, zr, zr, o], -1).reshape(*roll.shape, 3, 3)
    return R @ Rr


def smooth_t(x, sigma):
    """Gaussian smoothing along dim 1 (B, T, D), replicate padding."""
    R = max(1, int(3 * sigma))
    t = torch.arange(-R, R + 1, device=x.device).float()
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum())[None, None].to(x.dtype)
    B, T, D = x.shape
    y = F.pad(x.transpose(1, 2).reshape(B * D, 1, T), (R, R), mode="replicate")
    return F.conv1d(y, k).reshape(B, D, T).transpose(1, 2)


def smooth_quats_t(R, sigma):
    """(B, T, J, 3, 3) -> Gaussian-smoothed in time (sign-aligned quaternions), as bench/videomimic/recon.smooth_quats."""
    from fab.world_motion import mat_to_quat

    B, T, J = R.shape[:3]
    q = mat_to_quat(R)
    sg = torch.ones(B, T, J, 1, device=R.device)
    for t in range(1, T):
        sg[:, t] = sg[:, t - 1] * torch.where((q[:, t] * q[:, t - 1]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    q = smooth_t((q * sg).reshape(B, T, J * 4), sigma).reshape(B, T, J, 4)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(B, T, J, 3, 3)


def smooth_like_export(kin, R_c, betas, trans_c, Rwc, cw, sig_rot, sig_pos):
    """Camera-frame inputs -> the same bodies after fab-world's world smoothing (clip-median betas, root + local rotations
    sigma_rot, pelvis sigma_pos in the world), back in each camera frame."""
    B, T = R_c.shape[:2]
    bm = betas.median(1, keepdim=True).values.expand(B, T, 10)
    J0 = kin.rest(betas)[0][..., 0, :]
    pel_c = trans_c + J0
    pw = (Rwc @ pel_c[..., None])[..., 0] + cw
    Rw = torch.cat([(Rwc @ R_c[:, :, 0])[:, :, None], R_c[:, :, 1:]], 2)
    Rw = torch.cat([smooth_quats_t(Rw[:, :, :1], sig_rot), smooth_quats_t(Rw[:, :, 1:], sig_rot)], 2)
    pw = smooth_t(pw, sig_pos)
    Rc = torch.cat([(Rwc.transpose(-1, -2) @ Rw[:, :, 0])[:, :, None], Rw[:, :, 1:]], 2)
    pc = (Rwc.transpose(-1, -2) @ (pw - cw)[..., None])[..., 0]
    return Rc, bm, pc - kin.rest(bm)[0][..., 0, :]


class Motions:
    """Concatenated clean clips (world_motion_data npz files) on the device; windows of W frames, shorter clips padded."""

    def __init__(self, files, device, weights=None, store=None):
        """weights: per file, its share of the sampled windows (default: by length). store: where the frames live (default
        the training device; "cpu" for corpora that do not fit next to the model, batches are gathered there and moved)."""
        P, Tr, Cn, Be, st, ln, fid = [], [], [], [], [], [], []
        off = 0
        for i, f in enumerate(files):
            z = np.load(f)
            P.append(z["pose"]), Tr.append(z["trans"]), Cn.append(z["contact"])
            Be.append(z["betas"]), st.append(z["start"] + off), ln.append(z["length"])
            fid.append(np.full(len(z["length"]), i))
            off += len(z["pose"])
        sd = store or device
        self.pose = torch.as_tensor(np.concatenate(P), device=sd)
        self.trans = torch.as_tensor(np.concatenate(Tr), device=sd)
        self.contact = torch.as_tensor(np.concatenate(Cn), device=sd)
        self.betas = torch.as_tensor(np.concatenate(Be), device=sd)
        self.start = torch.as_tensor(np.concatenate(st), device=sd)
        self.length = torch.as_tensor(np.concatenate(ln), device=sd)
        if sd == device:
            self.contact = self.contact.float()
        self.sd = sd
        self.gc = torch.Generator(device=sd).manual_seed(int(torch.randint(0, 2 ** 31, (1,)).item())) if sd != device else None
        L = self.length.float()
        if weights is None:
            self.p = L / L.sum()  # clips by length
        else:
            fid = torch.as_tensor(np.concatenate(fid), device=sd)
            w = torch.as_tensor(weights, dtype=torch.float32, device=sd)
            tot = torch.zeros(len(files), device=sd).index_add_(0, fid, L)
            self.p = L / tot[fid] * w[fid] / w.sum()
        self.dev = device

    def sample(self, B, W, g):
        g = g if self.gc is None else self.gc
        ci = torch.multinomial(self.p, B, replacement=True, generator=g)
        L = self.length[ci]
        s = self.start[ci] + (torch.rand(B, generator=g, device=self.sd) * (L - W).clamp_min(0).float()).long()
        t = torch.arange(W, device=self.sd)
        pad = t[None] >= L[:, None]
        idx = (s[:, None] + t[None].clamp_max(L[:, None] - 1))
        out = {"pose": self.pose[idx], "trans": self.trans[idx], "contact": self.contact[idx], "betas": self.betas[ci],
               "pad": pad}
        if self.sd != self.dev:
            out = {k: v.to(self.dev, non_blocking=True) for k, v in out.items()}
            out["contact"] = out["contact"].float()
        return out


def simulate(kin, m, noise=NOISE, g=None, p_static=0.5, p_pan=0.2):
    """A batch of clean windows m (Motions.sample) -> (model inputs d, targets gt), all in the estimated world."""
    dev = m["pose"].device
    B, T = m["pose"].shape[:2]
    m = mirror(m, torch.rand(B, generator=g, device=dev) < 0.5, kin.rest(m["betas"])[0][:, 0])
    pad = m["pad"]
    R = aa_to_mat(m["pose"].reshape(B, T, 24, 3))
    # world augmentation: yaw + xy shift
    A = rz(U((B,), -math.pi, math.pi, g, dev))[:, None]
    R = torch.cat([(A @ R[:, :, 0])[:, :, None], R[:, :, 1:]], 2)
    betas = m["betas"][:, None].expand(B, T, 10)
    P0, C0, _ = kin(R, betas, torch.zeros(B, T, 3, device=dev))
    J0 = kin.rest(m["betas"])[0][:, 0]
    pel_t = (A @ (m["trans"] + J0[:, None])[..., None])[..., 0]
    trans = pel_t - P0[:, :, 0]
    Pw, Cw = P0 + trans[:, :, None], C0 + trans[:, :, None]
    pel = Pw[:, :, 0]
    # camera
    kind = torch.rand(B, generator=g, device=dev)
    static, pan = kind < p_static, (kind >= p_static) & (kind < p_static + p_pan)
    centre = pel.mean(1)
    dist = torch.exp(U((B,), math.log(1.8), math.log(9.0), g, dev))
    az = U((B,), -math.pi, math.pi, g, dev)
    hgt = U((B,), 0.3, 2.6, g, dev)
    c0 = centre + torch.stack([dist * torch.cos(az), dist * torch.sin(az), torch.zeros_like(dist)], -1)
    c0[:, 2] = hgt
    pel_s = smooth_t(pel, 15.0)
    off = torch.stack([dist * torch.cos(az), dist * torch.sin(az), torch.zeros_like(dist)], -1)
    hand = coloured(B, T, 3, 4.0, 12.0, g, dev) * U((B, 1, 1), 0.01, 0.08, g, dev)
    c_follow = pel_s + off[:, None] * U((B, 1, 1), 0.4, 0.9, g, dev) + hand
    c_follow[..., 2] = hgt[:, None] + hand[..., 2]
    cam_c = torch.where(static[:, None, None] | pan[:, None, None], c0[:, None].expand(B, T, 3), c_follow)
    tgt_static = (centre + torch.randn(B, 3, generator=g, device=dev) * 0.3)[:, None].expand(B, T, 3)
    tgt_track = smooth_t(pel, 6.0) + coloured(B, T, 3, 3.0, 8.0, g, dev) * 0.05
    tgt = torch.where(static[:, None, None], tgt_static, tgt_track)
    tgt = tgt + torch.tensor([0, 0, 1.0], device=dev) * U((B, 1, 1), -0.4, 0.2, g, dev)
    roll = (torch.randn(B, 1, generator=g, device=dev) * math.radians(3)).expand(B, T) + coloured(B, T, 1, 2, 6, g, dev)[..., 0] * math.radians(0.5) * (~static[:, None])
    Rwc = look_at(cam_c, tgt, roll)
    ii = torch.randint(0, len(IMAGES), (B,), generator=g, device=dev)
    wh = torch.tensor(IMAGES, device=dev, dtype=torch.float32)[ii]
    fov = U((B,), math.radians(35), math.radians(80), g, dev)
    f = (torch.maximum(wh[:, 0], wh[:, 1]) / 2) / torch.tan(fov / 2)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = f, f, wh[:, 0] / 2, wh[:, 1] / 2, 1
    # GT in the camera, visibility, boxes
    to_c = lambda X: (Rwc.transpose(-1, -2)[:, :, None] @ (X - cam_c[:, :, None])[..., None])[..., 0]  # noqa: E731
    Pc = to_c(torch.cat([Pw, Cw], 2))
    z = Pc[..., 2].clamp_min(1e-3)
    uv = Pc[..., :2] / z[..., None] * f[:, None, None, None] + wh[:, None, None] / 2
    inside = (uv[..., 0] > 0) & (uv[..., 0] < wh[:, None, None, 0]) & (uv[..., 1] > 0) & (uv[..., 1] < wh[:, None, None, 1]) & (Pc[..., 2] > 0.2)
    vis_frac = inside[..., :24].float().mean(-1)
    lo = torch.where(inside[..., None], uv, torch.full_like(uv, 1e9)).amin(2)
    hi = torch.where(inside[..., None], uv, torch.full_like(uv, -1e9)).amax(2)
    mg = (hi - lo) * 0.1
    box = torch.cat([(lo - mg).clamp_min(0), torch.minimum(hi + mg, wh[:, None].expand(B, T, 2))], -1)
    valid = (vis_frac > 0.5) & (Pc[:, :, 0, 2] > 0.3) & ~pad
    # dropout gaps
    for _ in range(noise["gaps"][1]):
        on = torch.rand(B, generator=g, device=dev) < noise["gap_p"] / noise["gaps"][1] * 2
        s = (torch.rand(B, generator=g, device=dev) * T).long()
        ln = torch.randint(noise["gap_len"][0], noise["gap_len"][1] + 1, (B,), generator=g, device=dev)
        t = torch.arange(T, device=dev)[None]
        valid = valid & ~(on[:, None] & (t >= s[:, None]) & (t < (s + ln)[:, None]))
    ok_any = valid.any(1)
    valid[~ok_any, 0] = True  # every window keeps one frame
    # FAB's per-frame errors
    Rc_root = Rwc.transpose(-1, -2) @ R[:, :, 0]
    pel_c = Pc[:, :, 0]
    fe = 1 + torch.randn(B, generator=g, device=dev) * noise["focal_sigma"]
    e = (noise["depth_bias"][0] + noise["depth_bias"][1] * torch.randn(B, 1, generator=g, device=dev)
         + coloured(B, T, 1, *noise["depth_corr"], g, dev)[..., 0] * U((B, 1), *noise["depth_sigma"], g, dev)
         + torch.randn(B, T, generator=g, device=dev) * U((B, 1), *noise.get("depth_white", (0.0, 0.0)), g, dev)
         + torch.log(fe)[:, None])
    sp = (torch.rand(B, T, generator=g, device=dev) < noise["depth_spike_p"]).float()
    sp = smooth_t(sp[..., None], 1.0)[..., 0] * 2.5 * U((B, 1), *noise["depth_spike"], g, dev) * torch.sign(torch.randn(B, 1, generator=g, device=dev))
    e = e + sp
    bh = (box[..., 3] - box[..., 1]).clamp_min(10)
    uv_p = uv[:, :, 0] + (coloured(B, T, 2, *noise.get("px_corr", (0.5, 3.0)), g, dev) * U((B, 1, 1), *noise["px_sigma"], g, dev)
                          + torch.randn(B, T, 2, generator=g, device=dev) * U((B, 1, 1), *noise.get("px_white", (0.0, 0.0)), g, dev)) * bh[..., None]
    fk = f * fe
    zc = pel_c[..., 2] * torch.exp(e)
    pel_c_est = torch.cat([(uv_p - wh[:, None] / 2) / fk[:, None, None] * zc[..., None], zc[..., None]], -1)
    rd = math.radians
    nr = coloured(B, T, 3, *noise.get("root_corr", noise["rot_corr"]), g, dev) * rd(1) * U((B, 1, 1), *noise["root_deg"], g, dev)
    Rc_root_est = aa_to_mat(nr) @ Rc_root
    nj = coloured(B, T, 69, *noise["rot_corr"], g, dev).reshape(B, T, 23, 3) * rd(1) * U((B, 1, 1, 1), *noise["joint_deg"], g, dev)
    R_loc_est = aa_to_mat(nj) @ R[:, :, 1:]
    betas_est = betas + noise["betas_bias"] * torch.randn(B, 1, 10, generator=g, device=dev) + noise["betas_frame"] * torch.randn(B, T, 10, generator=g, device=dev)
    J0e = kin.rest(betas_est)[0][..., 0, :]
    trans_c = pel_c_est - J0e
    R_c = torch.cat([Rc_root_est[:, :, None], R_loc_est], 2)
    # contacts
    gl = m["contact"] * 2 - 1
    lg = (gl * U((B, 1, 1), *noise["contact_logit"], g, dev) + noise.get("contact_bias", 0.0)
          + coloured(B, T, 8, 1.0, 4.0, g, dev) * noise["contact_noise"])
    cprob = torch.sigmoid(torch.cat([lg, lg + 0.5], -1))
    cmask = (torch.rand(B, 1, generator=g, device=dev) > noise["contact_absent_p"]).float().expand(B, T)
    # camera errors -> the estimated camera; the world it defines
    ax = torch.randn(B, 3, generator=g, device=dev)
    ax[:, 2] = 0
    ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-6) * (torch.randn(B, 1, generator=g, device=dev) * rd(noise["tilt_deg"]))
    Rt = aa_to_mat(ax)[:, None]
    drift = torch.cumsum(torch.randn(B, T, 3, generator=g, device=dev) * rd(noise["drift_deg"]), 1) * (~static)[:, None, None]
    Rwc_e = Rt @ aa_to_mat(drift) @ Rwc
    sc = torch.exp(torch.randn(B, 1, 1, generator=g, device=dev) * noise["cam_scale_sigma"])
    ce = cam_c[:, :1] + (cam_c - cam_c[:, :1]) * sc + coloured(B, T, 3, 2.0, 6.0, g, dev) * noise["cam_noise"] * (~static)[:, None, None]
    ce = (Rt @ (ce - cam_c[:, :1])[..., None])[..., 0] + cam_c[:, :1]
    T_wc = torch.zeros(B, T, 4, 4, device=dev)
    T_wc[..., :3, :3], T_wc[..., :3, 3], T_wc[..., 3, 3] = Rwc_e, ce, 1
    # hold the last valid frame on misses (leading misses: the first valid one), as run_fab does
    idx = torch.arange(T, device=dev)[None].expand(B, T)
    last = torch.where(valid, idx, torch.full_like(idx, -1)).cummax(1).values
    first = torch.where(valid, idx, torch.full_like(idx, T)).amin(1, keepdim=True)
    src = torch.where(last < 0, first.expand(B, T), last)
    hold = lambda x: torch.gather(x, 1, src.reshape(B, T, *([1] * (x.dim() - 2))).expand_as(x))  # noqa: E731
    Ke = K.clone()
    Ke[:, 0, 0], Ke[:, 1, 1] = fk, fk
    R_c, betas_est, trans_c = hold(R_c), hold(betas_est), hold(trans_c)
    if noise.get("smooth_input"):  # the input as fab-world exports it: world rotations / pelvis smoothed (sigma 2 / 3)
        R_c, betas_est, trans_c = smooth_like_export(kin, R_c, betas_est, trans_c, Rwc_e, ce, *noise["smooth_input"])
    d = {"R_c": R_c, "betas": betas_est, "trans_c": trans_c, "T_wc": T_wc, "K": Ke, "wh": wh,
         "box": hold(box), "valid": valid.float(), "cprob": hold(cprob), "cmask": cmask * valid.float(), "pad": pad}
    # targets: the true world mapped into the estimated world by the first frame's camera (rigid)
    M_R = Rwc_e[:, 0] @ Rwc[:, 0].transpose(-1, -2)
    M_t = ce[:, 0] - (M_R @ cam_c[:, 0, :, None])[..., 0]
    mp = lambda X: (M_R[:, None, None] @ X[..., None])[..., 0] + M_t[:, None, None]  # noqa: E731
    gt = {"joints": mp(Pw), "cpts": mp(Cw), "R_root": M_R[:, None] @ R[:, :, 0], "R_loc": R[:, :, 1:],
          "contact": m["contact"], "pad": pad, "betas": betas}
    return d, gt
