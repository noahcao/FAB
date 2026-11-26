"""Physical-plausibility refinement of a fab-world export (optional post-process; scripts/fab_world.sh --refine physics):
a light kinematic step on the final world body, its heightfield and its contact probabilities. Nothing is simulated and
the image evidence is kept: the body moves only along its camera rays (depth) and its legs are refit to put the planted
feet on the ground.

  ground the walked-on ground from the body itself: the planted footprints (per static contact run of a heel / toe channel,
         the median of its lowest vertex) -> a plane z = a x + b y + c (inlier fit, slopes ridge-shrunk where the footprints
         do not spread); --terrain plane (default): the terrain is that plane; --terrain hfield: the export's heightfield
         (the scene)
  level  a footprint-plane tilt between --level_min and --level_max degrees (a gravity error of the camera's up, or a
         depth bias that grows with distance) is rotated away about the footprints' centre (cameras, body and heightfield
         together), so the walked-on ground is horizontal
  depth  per-frame body depth: b_t, a log factor of the pelvis depth along its camera ray (the body keeps its size and its
         image position), from the planted feet (static contact probability >= --p_contact, in view, within --contact_gate
         of the ground: not on furniture / a fence) with their lowest vertex on the ground, penetration of the in-view body
         (a squared hinge under the ground), foot skate over consecutive static contacts, b smooth (step and acceleration)
         and small (--sigma_b)
  lock   the planted heel / toe groups (the contact model's, plus kinematic plants: on the ground and slow, --kin_contact)
         pinned (xy at the contact piece's median; z on the ground, or at their own height
         when off the ground) and faded in / out over --lock_blend frames, as targets of a leg refit (hip / knee / ankle /
         foot rotation increments + a small smooth pelvis shift, one shape: bench/scene_recon/joint.py foot_lock_refit's
         solver); every foot-sole vertex kept above the ground
  When the footprints are not one plane (inlier share < --min_inliers) the fit is repeated with a smooth vertical drift of
  the body over time (--floor_drift): if one floor + a drift of >= --drift_min explains them, the clip is a drifting body on one
  floor (contacts gated against the local floor, corrected to the plane by depth along the rays plus a smooth vertical shift);
  otherwise (several surfaces) there is no level and no depth stage, and the lock pins xy only. The optimisation variables are per frame but enter through a Gaussian time basis (--b_smooth,
  --lock_smooth frames), so the optimiser's per-parameter steps never reach the output as jitter.
  -> the same export keys (pose, trans, joints, T_world_cam, contacts relabelled on the new ground), the input body as
     pose_in / trans_in, and refine_world (JSON report); <name>_hfield.npz (and --scene's <name>_scene.ply) levelled with it

    python -m bench.scene_recon.world_refine --world <out>/<c>.npz --hfield <out>/<c>_hfield.npz \
        --contacts <out>/work/contacts/<c>__0.contact.npz --out <dir> [--level 1 --depth 1 --lock 1]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from fab import paths

CH_FOOT = (0, 1, 2, 3)  # l_heel, l_toe, r_heel, r_toe (fab/contact_regions.CONTACT_NAMES)
LEG = (1, 2, 4, 5, 7, 8, 10, 11)
UPPER = (3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21)  # spine, collars, shoulders, elbows, wrists (--body_pen)


def load_contacts(path, n):
    """(T, 8) static and any contact probabilities (the causal video model first, then the image head)."""
    Ps, Pa = np.zeros((n, 8), np.float32), np.zeros((n, 8), np.float32)
    if not path or not Path(path).exists():
        return Ps, Pa, "none"
    z = np.load(path)
    fr = z["frame"].astype(int)
    ok = fr < n
    head = next(h for h in ("video_", "image_", "") if f"{h}static_prob" in z.files)
    Ps[fr[ok], :4], Pa[fr[ok], :4] = z[f"{head}static_prob"][ok], z[f"{head}any_prob"][ok]
    for dst, k in ((Ps, f"{head}hk_static_prob"), (Pa, f"{head}hk_any_prob")):
        if k in z.files:
            dst[fr[ok], 4:] = z[k][ok]
    return Ps, Pa, head.rstrip("_") or "any"


def smpl_verts(body, pose, betas, trans, dev, chunk=256):
    out = []
    with torch.no_grad():
        bt = torch.as_tensor(betas, dtype=torch.float32, device=dev)
        for i in range(0, len(pose), chunk):
            p = torch.as_tensor(pose[i: i + chunk], dtype=torch.float32, device=dev)
            t = torch.as_tensor(trans[i: i + chunk], dtype=torch.float32, device=dev)
            v, _ = body(p[:, :3], p[:, 3:], bt[None].expand(len(p), -1), t)
            out.append(v.cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def robust_plane(P, thr=0.03, ridge=0.3, iters=10):
    """z = a x + b y + c through the points P (k, 3): least squares on the inliers (|residual| < thr), started level at the
    median height; the slopes are ridge-shrunk with weight k ridge^2 (no tilt from footprints that do not spread) ->
    (a, b, c), inlier share, median |residual| of the inliers."""
    x, y, z = P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean(), P[:, 2]
    coef = np.array([0.0, 0.0, np.median(z)])
    A = np.stack([x, y, np.ones(len(z))], 1)
    reg = np.diag([ridge ** 2, ridge ** 2, 0.0])
    inl = np.ones(len(z), bool)
    for _ in range(iters):
        inl = np.abs(z - A @ coef) < thr
        if inl.sum() < 3:
            break
        Ai = A[inl]
        coef = np.linalg.solve(Ai.T @ Ai + reg * inl.sum(), Ai.T @ z[inl])
    res = np.abs(z - A @ coef)
    c = coef[2] - coef[0] * P[:, 0].mean() - coef[1] * P[:, 1].mean()
    return np.array([coef[0], coef[1], c]), float(inl.mean()), float(np.median(res[inl])) if inl.any() else float("nan")


def footprints_t(V, P, regions, p_contact, min_run=3):
    """footprints (joint.py's: per heel / toe channel and run of >= min_run planted frames, the median of its lowest vertex)
    with the run's middle frame -> (k, 3), (k,)."""
    pts, ts = [], []
    for ch in CH_FOOT:
        on = P[:, ch] >= p_contact
        low = V[:, regions[ch]]
        low = low[np.arange(len(V)), low[..., 2].argmin(1)]
        t = 0
        while t < len(on):
            if not on[t]:
                t += 1
                continue
            e = t
            while e < len(on) and on[e]:
                e += 1
            if e - t >= min_run:
                pts.append(np.median(low[t:e], 0))
                ts.append(0.5 * (t + e - 1))
            t = e
    return np.array(pts).reshape(-1, 3), np.array(ts)


def robust_plane_drift(P, t, knot, thr=0.03, ridge=0.3, smooth=0.02, rate=0.04, iters=10):
    """z = a x + b y + c + delta(t): robust_plane with a smooth vertical drift over time (linear between knots every `knot`
    frames, zero-mean; first / second differences shrunk with scales `rate` / `smooth` m per knot: a drift, not a step onto
    a box or a fence): one floor seen through a body whose height drifts
    (depth / pose errors), told apart from several surfaces (footprints that jump). -> (a, b, c), delta(frames) callable,
    inlier share, median |residual| of the inliers."""
    k = len(P)
    tk = np.arange(0, t.max() + knot, knot)
    nk = len(tk)
    u = np.clip(t / knot, 0, nk - 1 - 1e-9)
    i0 = np.floor(u).astype(int)
    f = u - i0
    B = np.zeros((k, nk))
    B[np.arange(k), i0] = 1 - f
    B[np.arange(k), np.minimum(i0 + 1, nk - 1)] += f
    x, y, z = P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean(), P[:, 2]
    A = np.c_[x, y, np.ones(k), B]
    D2 = np.zeros((max(nk - 2, 0), 3 + nk))
    for j in range(nk - 2):
        D2[j, 3 + j: 6 + j] = [1, -2, 1]
    D1 = np.zeros((max(nk - 1, 0), 3 + nk))
    for j in range(nk - 1):
        D1[j, 3 + j: 5 + j] = [-1, 1]
    Z = np.zeros((1, 3 + nk))
    Z[0, 3:] = 1  # sum(delta) = 0 (soft, strong)
    reg = np.diag([ridge ** 2, ridge ** 2, 0.0] + [1e-6] * nk)
    coef = np.r_[0.0, 0.0, np.median(z), np.zeros(nk)]
    inl = np.ones(k, bool)
    for _ in range(iters):
        inl = np.abs(z - A @ coef) < thr
        if inl.sum() < 3:
            break
        Ai, n = A[inl], inl.sum()
        M = Ai.T @ Ai + reg * n + (thr / smooth) ** 2 * D2.T @ D2 + (thr / rate) ** 2 * D1.T @ D1 + 1e3 * n * Z.T @ Z
        coef = np.linalg.solve(M, Ai.T @ z[inl])
    res = np.abs(z - A @ coef)
    c = coef[2] - coef[0] * P[:, 0].mean() - coef[1] * P[:, 1].mean()
    dk = coef[3:]
    delta = lambda fr: np.interp(np.asarray(fr, np.float64) / knot, np.arange(nk), dk)  # noqa: E731
    return np.array([coef[0], coef[1], c]), delta, float(inl.mean()), float(np.median(res[inl])) if inl.any() else float("nan")


class PlaneTerrain:
    """A horizontal ground at height h (the Terrain interface)."""

    def __init__(self, h):
        self.h = float(h)

    def __call__(self, xy):
        return torch.full(xy.shape[:-1], self.h, dtype=xy.dtype, device=xy.device)


def rot_between(a, b):
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def gauss_basis(u, sigma):
    """A smooth per-frame signal from free per-frame parameters u (T, ...): u Gaussian-filtered over time (sigma frames,
    edge-replicated). Adam's per-parameter steps then never reach the output as frame-to-frame noise (jitter)."""
    if sigma <= 0:
        return u
    r = int(np.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=u.device, dtype=u.dtype) / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = u.reshape(u.shape[0], -1).T[:, None]  # (C, 1, T)
    x = torch.nn.functional.pad(x, (r, r), mode="replicate")
    return torch.nn.functional.conv1d(x, k)[:, 0].T.reshape(u.shape)


def guard_jumps(trans, pose, fps, max_speed, win):
    """Root-translation jumps (a step faster than max_speed m/s, beyond the neighbouring motion: an identity switch, a
    depth glitch) spread over +-win frames with a smoothstep, so the root stays continuous; the pose is kept. -> trans, report."""
    tr = trans.copy()
    st = np.diff(tr, axis=0)
    sp = np.linalg.norm(st, axis=1) * fps
    jumps = np.flatnonzero(sp > max_speed)
    for j in jumps:
        nb = np.r_[st[max(0, j - 5): j], st[j + 1: j + 6]]
        ex = st[j] - (np.median(nb, 0) if len(nb) else 0)  # the excess step
        k = np.arange(len(tr)) - (j + 0.5)
        x = np.clip((k + win) / (2.0 * win), 0, 1)
        ramp = x * x * (3 - 2 * x)  # 0 -> 1 over [j - win, j + win]
        tr = tr - ex * ((np.arange(len(tr)) > j).astype(float) - ramp)[:, None]
    return tr, {"jumps": int(len(jumps)), "max_speed_ms": float(sp.max()) if len(sp) else 0.0}


def depth_stage(Vs, Vch, pel, cam, terr, Ps, inview, sub_inview, fps, a, dev, near=None, vertical=0.0):
    """b_t (T,) per-frame log pelvis-depth factors (+ with vertical > 0 a smooth per-frame vertical shift dz_t, prior scale
    `vertical` m: the drift-mode correction of a body whose height drifts for reasons other than depth). Vs (T, Ns, 3) world
    body sample, Vch [(T, k, 3)] foot channels, pel / cam (T, 3) world pelvis / camera centres; Ps (T, 8) static contact
    probabilities. -> b, dz (T,), report"""
    T = len(pel)
    tt = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    Vs_, P_, C_ = tt(Vs), tt(pel), tt(cam)
    Vk = [tt(x) for x in Vch]
    on = torch.as_tensor((Ps[:, :4] >= a.p_contact) & inview[:, :4] & (True if near is None else near), device=dev)
    st = on[1:] & on[:-1]
    sv = torch.as_tensor(sub_inview, device=dev).float()
    u = torch.zeros(T, device=dev, requires_grad=True)
    uz = torch.zeros(T, device=dev, requires_grad=vertical > 0)
    opt = torch.optim.Adam([u, uz] if vertical > 0 else [u], lr=a.lr)
    ray = P_ - C_
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

    def losses():
        b = gauss_basis(u, a.b_smooth)
        dz = gauss_basis(uz, a.dz_smooth) if vertical > 0 else torch.zeros_like(u)
        d = ray * (torch.exp(b) - 1)[:, None] + dz[:, None] * ez  # (T, 3) world translation of the whole body
        L = {}
        B = Vs_ + d[:, None]
        h = terr(B[..., :2])
        L["pen"] = (sv * torch.relu(h - B[..., 2]) ** 2).sum() / sv.sum().clamp(min=1) / a.sigma ** 2
        con, sk = 0.0, 0.0
        for ch, X in enumerate(Vk):
            W = X + d[:, None]
            gap = (W[..., 2] - terr(W[..., :2])).min(1)[0]
            m = on[:, ch].float()
            con = con + (m * gap ** 2).sum() / m.sum().clamp(min=1) / a.sigma ** 2
            s = st[:, ch].float()
            sk = sk + (s * ((W[1:, :, :2] - W[:-1, :, :2]) ** 2).sum(-1).mean(1)).sum() / s.sum().clamp(min=1) / a.sigma_skate ** 2
        L["contact"], L["skate"] = con, sk
        root = P_ + d
        L["acc"] = ((root[2:] - 2 * root[1:-1] + root[:-2]) ** 2).sum(-1).mean() / a.sigma_acc ** 2
        L["bstep"] = ((b[1:] - b[:-1]) ** 2).mean() / a.sigma_b_step ** 2
        L["bacc"] = ((b[2:] - 2 * b[1:-1] + b[:-2]) ** 2).mean() / a.sigma_b_acc ** 2
        L["bprior"] = (b ** 2).mean() / a.sigma_b ** 2
        if vertical > 0:
            L["dzprior"] = (dz ** 2).mean() / vertical ** 2
            L["dzstep"] = ((dz[1:] - dz[:-1]) ** 2).mean() / a.dz_step ** 2
        return L

    wts = {"pen": a.w_pen, "contact": a.w_contact, "skate": a.w_skate, "acc": a.w_acc, "bstep": 1.0, "bacc": 1.0, "bprior": 1.0,
           "dzprior": 1.0, "dzstep": 1.0}
    with torch.no_grad():
        before = {k: float(v) for k, v in losses().items()}
    for _ in range(a.iters):
        L = losses()
        loss = sum(wts[k] * v for k, v in L.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = {k: float(v) for k, v in losses().items()}
        bb = gauss_basis(u, a.b_smooth).cpu().numpy().astype(np.float64)
        zz = (gauss_basis(uz, a.dz_smooth) if vertical > 0 else torch.zeros_like(u)).cpu().numpy().astype(np.float64)
    return bb, zz, {"before": before, "after": after, "b_abs_mean": float(np.abs(bb).mean()), "b_absmax": float(np.abs(bb).max()),
                    "dz_abs_mean_m": float(np.abs(zz).mean()), "dz_absmax_m": float(np.abs(zz).max()),
                    "contact_frames": int(on.any(1).sum().item())}


def ground_offsets(F, Fg, on, blend, min_len=3, max_off=0.0, max_pin=0.0):
    """F (T, G, 3) group centres, Fg the same with each lowest vertex on the terrain, on (T, G) planted frames -> offsets O
    (T, G, 3), weights (T, G), pins: over each run of >= min_len planted frames (cut into a new pin wherever the centre has
    moved more than 2 max_off in xy from the piece's start, when > 0) the group goes to the piece's median of Fg (pinned in
    xy, on the ground in z); the offset fades in before / out after the run over `blend` frames (cosine)."""
    T, ng = on.shape
    O, wt = np.zeros_like(F), np.zeros((T, ng))
    pins = 0
    for g in range(ng):
        t = 0
        while t < T:
            if not on[t, g]:
                t += 1
                continue
            s0 = t
            while t < T and on[t, g]:
                t += 1
            if t - s0 < min_len:
                continue
            s = s0
            while s < t:
                e = t
                if max_off > 0:
                    far = np.flatnonzero(np.linalg.norm(F[s:t, g, :2] - F[s, g, :2], axis=1) > 2 * max_off)
                    e = s + int(far[0]) if len(far) else t
                    e = max(e, s + 1)
                o = np.median(Fg[s:e, g], 0) - F[s:e, g]
                if max_pin <= 0 or np.median(np.linalg.norm(o, axis=-1)) <= max_pin:  # (a far pin: a contact error)
                    pins += 1
                    O[s:e, g] = o
                    wt[s:e, g] = 1
                s = e
            for k in range(1, blend + 1):
                f = 0.5 * (1 + np.cos(np.pi * k / (blend + 1)))
                for tt, src in ((t - 1 + k, t - 1), (s0 - k, s0)):
                    if 0 <= tt < T and wt[tt, g] < f:
                        O[tt, g], wt[tt, g] = O[src, g] * f, f
    return O, wt, pins


def ground_lock_refit(body, pose, betas, trans, terr, regions, Ps, fps, a, dev, ground_z=True, near=None):
    """Heel / toe groups planted on the ground over their static contact runs (xy: the run median; z: the terrain there
    plus the group centre's height over its lowest vertex), faded over --lock_blend frames; leg rotation increments and a
    smooth pelvis shift (one shape) fit the group centres to the targets (bench/scene_recon/joint.py foot_lock_refit's
    solver), every foot-sole vertex kept above the terrain. pose (T, 72) world, trans (T, 3) -> pose, trans shift (T, 3),
    report."""
    from bench.scene_recon.joint import smpl_sub
    from fab.gn_fit import so3_exp

    T = len(pose)
    groups = [np.asarray(regions[ch]) for ch in CH_FOOT]
    idx = np.concatenate(groups)
    seg = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
    M = torch.zeros(len(groups), len(idx), device=dev)
    M[torch.as_tensor(seg, device=dev), torch.arange(len(idx), device=dev)] = 1
    M = M / M.sum(1, keepdim=True)
    Rm0 = torch.as_tensor(Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3), dtype=torch.float32,
                          device=dev)
    bt = torch.as_tensor(betas, dtype=torch.float32, device=dev)
    it = torch.as_tensor(idx, device=dev)
    tr = torch.as_tensor(trans, dtype=torch.float32, device=dev)

    def verts(Rm, dp):  # = SMPL(pose, betas, trans + dp) at the foot groups' vertices
        return smpl_sub(body, Rm, bt, it) + tr[:, None] + dp[:, None]

    with torch.no_grad():
        X0 = verts(Rm0, torch.zeros(T, 3, device=dev)).cpu().numpy().astype(np.float64)
        F = np.einsum("gk,tkc->tgc", M.cpu().numpy(), X0)  # group centres
        low = np.stack([X0[:, seg == g, 2].min(1) for g in range(len(groups))], 1)  # each group's lowest vertex height
        tz = terr(torch.as_tensor(F[..., :2], dtype=torch.float32, device=dev)).cpu().numpy().astype(np.float64)
    sp = np.r_[np.zeros((1, len(groups))), np.linalg.norm(np.diff(F[..., :2], axis=0), axis=-1)] * fps
    lp = a.lock_p if a.lock_p > 0 else a.p_contact
    on = (Ps[:T, list(CH_FOOT)] >= lp) & (sp < a.lock_gate)
    Fg = F.copy()
    if ground_z:  # z on the terrain; else the group keeps its own height (xy pin only)
        zg = tz + (F[..., 2] - low)
        Fg[..., 2] = zg if near is None else np.where(near, zg, F[..., 2])  # contacts off the ground (furniture): own height
    O, w, runs = ground_offsets(F, Fg, on, a.lock_blend, max_off=a.lock_max_off, max_pin=a.lock_max_pin)
    tgt = torch.as_tensor(F + O, dtype=torch.float32, device=dev)
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    if a.lock_rigid:  # every heel / toe vertex pinned (its group's orientation too: no roll or twist while planted)
        lift = (Fg[..., 2] - F[..., 2])[:, seg]  # each vertex lifted with its group onto the ground
        Xg = X0.copy()
        Xg[..., 2] += lift
        Ov, wv, _ = ground_offsets(X0, Xg, on[:, seg], a.lock_blend, max_pin=a.lock_max_pin)
        tgt_v = torch.as_tensor(X0 + Ov, dtype=torch.float32, device=dev)
        wt_v = torch.as_tensor(wv, dtype=torch.float32, device=dev)
    bal = np.zeros(T, bool)
    if a.balance:  # quasi-static single support: one foot planted (heel + toe), the other lifted, no hand contact, slow COM
        cs_idx = torch.arange(0, 6890, 8, device=dev)
        with torch.no_grad():
            C0 = (smpl_sub(body, Rm0, bt, cs_idx) + tr[:, None]).mean(1).cpu().numpy().astype(np.float64)
        pl = w >= 1
        L, R = pl[:, 0] & pl[:, 1], pl[:, 2] & pl[:, 3]
        lowL, lowR = low[:, :2].min(1), low[:, 2:].min(1)
        hands = (Ps[:T, 4:6] >= 0.3).any(1) if Ps.shape[1] >= 6 else np.zeros(T, bool)
        csp = np.r_[np.inf, np.linalg.norm(np.diff(C0[:, :2], axis=0), axis=1) * fps]
        bal = ((L & ~pl[:, 2] & ~pl[:, 3] & (lowR - lowL > a.bal_lift)) | (R & ~pl[:, 0] & ~pl[:, 1] & (lowL - lowR > a.bal_lift))) \
            & ~hands & (csp < a.bal_speed)
        supi = torch.as_tensor(np.where(L, 0, 2), device=dev)  # the support foot's first group (heel); its toe = +1
    balt = torch.as_tensor(bal, device=dev)
    extra = {"upper": list(UPPER), "root": [0], "spine": [0, 3, 6, 9]}[a.body_pen_dof] if a.body_pen and ground_z else []
    li = list(LEG) + extra
    if a.body_pen and ground_z:  # every non-foot body part (hands, forearms, knees, torso, head) above the ground plane
        part = body.weights.argmax(1)
        bidx = torch.nonzero(~torch.isin(part, torch.tensor([7, 8, 10, 11], device=part.device)))[:, 0][::3].to(dev)
        with torch.no_grad():
            B0 = smpl_sub(body, Rm0, bt, bidx) + tr[:, None]
            n_pen0 = int((terr(B0[..., :2]) - B0[..., 2] > 0.005).sum().item())
    ud = torch.zeros(T, len(li), 3, device=dev, requires_grad=True)
    up = torch.zeros(T, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [ud], "lr": 0.01}, {"params": [up], "lr": 0.002}])

    def rot(d):
        Rm = Rm0.clone()
        Rm[:, li] = Rm0[:, li] @ so3_exp(d)
        return Rm

    for _ in range(a.lock_iters):
        d, dp = gauss_basis(ud, a.lock_smooth), gauss_basis(up, a.lock_smooth)
        X = verts(rot(d), dp)
        C = torch.einsum("gk,tkc->tgc", M, X)
        if a.lock_rigid:
            pin = (wt_v * ((X - tgt_v) ** 2).sum(-1)).sum() / wt_v.sum().clamp(min=1) / a.lock_sigma ** 2
        else:
            pin = (wt * ((C - tgt) ** 2).sum(-1)).sum() / wt.sum().clamp(min=1) / a.lock_sigma ** 2
        pen = (torch.relu(terr(X[..., :2]) - X[..., 2]) ** 2).mean() / a.sigma ** 2 if ground_z else 0.0
        if a.body_pen and ground_z and n_pen0 > 0:
            Bv = smpl_sub(body, rot(d), bt, bidx) + tr[:, None] + dp[:, None]
            pen = pen + a.w_body_pen * (torch.relu(terr(Bv[..., :2]) - Bv[..., 2]) ** 2).sum() / n_pen0 / a.sigma ** 2
        if bal.any():  # the COM (vertex mean) over the support foot within --bal_margin
            Cm = (smpl_sub(body, rot(d)[balt], bt, cs_idx) + tr[balt][:, None] + dp[balt][:, None]).mean(1)
            sup = 0.5 * (tgt[balt, supi[balt]] + tgt[balt, supi[balt] + 1])
            pen = pen + a.w_bal * (torch.relu((Cm[:, :2] - sup[:, :2]).norm(dim=-1) - a.bal_margin) ** 2).mean() / 0.02 ** 2
        L = (pin + a.w_pen * pen + 0.01 * (d ** 2).sum(-1).mean() / a.lock_rot_prior ** 2
             + ((d[1:] - d[:-1]) ** 2).sum(-1).mean() / 0.05 ** 2 + (dp ** 2).sum(-1).mean() / a.lock_sigma_root ** 2
             + ((dp[1:] - dp[:-1]) ** 2).sum(-1).mean() / 0.005 ** 2
             + ((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).sum(-1).mean() / a.lock_acc_rot ** 2
             + ((dp[2:] - 2 * dp[1:-1] + dp[:-2]) ** 2).sum(-1).mean() / a.lock_acc_root ** 2)
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()
    with torch.no_grad():
        d, dp = gauss_basis(ud, a.lock_smooth), gauss_basis(up, a.lock_smooth)
        X = verts(rot(d), dp)
        C = torch.einsum("gk,tkc->tgc", M, X)
        res = (C - tgt).norm(dim=-1).cpu().numpy()
        Rm = rot(d).cpu().numpy()
    pose2 = Rotation.from_matrix(Rm.reshape(-1, 3, 3)).as_rotvec().reshape(T, 72)
    off = np.linalg.norm(O, axis=-1)
    pinned = w >= 1
    rep = {"lock_pins": int(runs), "ground_z": bool(ground_z), "balance_frames": int(bal.sum()),
           "body_pen_samples0": int(n_pen0) if a.body_pen and ground_z else 0, "lock_frame_share": float(pinned.any(1).mean()),
           "offset_mm_p50": float(np.median(off[pinned]) * 1000) if pinned.any() else 0.0,
           "offset_mm_max": float(off.max() * 1000), "pin_residual_mm_mean": float(res[pinned].mean() * 1000) if pinned.any() else 0.0,
           "pelvis_shift_mm_mean": float(dp.detach().norm(dim=-1).mean() * 1000),
           "leg_rot_deg_mean": float(np.degrees(d.detach().norm(dim=-1).mean().item()))}
    return pose2, dp.detach().cpu().numpy().astype(np.float64), rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", required=True, help="the fab-world export <out>/<name>.npz")
    p.add_argument("--hfield", required=True, help="its <name>_hfield.npz (missing: a floor at z = 0)")
    p.add_argument("--contacts", default="", help="<work>/contacts/<name>__0.contact.npz (none: no contact terms)")
    p.add_argument("--scene", default="", help="the export's <name>_scene.ply: written to --out in the refined world")
    p.add_argument("--out", required=True, help="output dir: <name>.npz + <name>_hfield.npz")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--terrain", default="plane", choices=("plane", "hfield"),
                   help="the ground of the contact / penetration terms: the footprint plane or the export's heightfield")
    p.add_argument("--min_footprints", type=int, default=4, help="fewer planted footprints: no plane (hfield terrain, no level)")
    p.add_argument("--plane_ridge", type=float, default=0.3, help="footprint plane: slope shrinkage spread (m)")
    p.add_argument("--floor_drift", type=int, default=0, help="1: where the footprints fail the one-plane test only because the "
                   "body's height drifts (a plane + a smooth drift over time, knots every --floor_win s, explains them), contacts are "
                   "gated against that local floor and corrected to the plane (depth along the rays + a smooth vertical shift)")
    p.add_argument("--floor_win", type=float, default=2.0, help="drift knots (s)")
    p.add_argument("--drift_rate", type=float, default=0.10, help="drift: first-difference scale (m per knot)")
    p.add_argument("--drift_smooth", type=float, default=0.05, help="drift: second-difference scale (m per knot)")
    p.add_argument("--drift_vertical", type=float, default=0.10,
                   help="drift mode: the depth stage also moves the body vertically (smooth dz_t, prior scale m; 0 off)")
    p.add_argument("--dz_smooth", type=float, default=8.0, help="dz_t Gaussian time basis (frames)")
    p.add_argument("--dz_step", type=float, default=0.003, help="dz_t step scale (m / frame)")
    p.add_argument("--drift_min", type=float, default=0.02, help="drift mode only for a drift std of at least this (m)")
    p.add_argument("--min_inliers", type=float, default=0.6,
                   help="footprint plane inlier share below which the ground is not one plane (no level, no depth stage, "
                        "the lock pins xy only)")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--level_min", type=float, default=0.3, help="level: smaller tilts are left alone (deg)")
    p.add_argument("--level_max", type=float, default=6.0, help="level: larger tilts are not a gravity error (slopes, deg)")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--lock", type=int, default=1)
    p.add_argument("--p_contact", type=float, default=0.5)
    p.add_argument("--contact_gate", type=float, default=0.10,
                   help="planted channels farther than this from the ground (m; on furniture, a fence) do not pull the depth "
                        "and keep their own height in the lock")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--sigma", type=float, default=0.03, help="contact gap / penetration scale (m)")
    p.add_argument("--sigma_skate", type=float, default=0.01, help="skate scale (m / frame)")
    p.add_argument("--sigma_acc", type=float, default=0.01, help="root acceleration scale (m / frame^2)")
    p.add_argument("--sigma_b", type=float, default=0.02, help="depth factor prior (log)")
    p.add_argument("--sigma_b_step", type=float, default=0.01)
    p.add_argument("--sigma_b_acc", type=float, default=0.0005, help="depth factor acceleration scale (log / frame^2)")
    p.add_argument("--b_smooth", type=float, default=4.0, help="depth factors: Gaussian basis over time (frames)")
    p.add_argument("--lock_smooth", type=float, default=2.0, help="lock refit increments: Gaussian basis over time (frames)")
    p.add_argument("--w_pen", type=float, default=1.0)
    p.add_argument("--w_contact", type=float, default=1.0)
    p.add_argument("--w_skate", type=float, default=0.1)
    p.add_argument("--w_acc", type=float, default=0.1)
    p.add_argument("--lock_gate", type=float, default=1.0, help="lock: no pin while the group moves faster (m/s)")
    p.add_argument("--lock_blend", type=int, default=8)
    p.add_argument("--kin_contact", type=int, default=1, help="lock: also pin heel / toe groups on the ground (|h| < --kin_h) "
                   "and slow (< --kin_v m/s) that the contact model misses")
    p.add_argument("--lock_rigid", type=int, default=1, help="lock: pin every heel / toe vertex (no foot roll while planted)")
    p.add_argument("--lock_p", type=float, default=0.0, help="lock: its own static-contact threshold (0: --p_contact)")
    p.add_argument("--body_pen", type=int, default=0, help="lock refit: every non-foot body part (hands, knees, torso) kept above "
                   "the ground plane, with spine / arm rotation increments as extra variables (prone / plank phases)")
    p.add_argument("--w_body_pen", type=float, default=1.0)
    p.add_argument("--body_pen_dof", default="upper", choices=("upper", "root", "spine"),
                   help="body_pen: the extra rotation variables (upper: spine + arms; root: the global orientation; spine: root + spine)")
    p.add_argument("--kin_h", type=float, default=0.06)
    p.add_argument("--kin_v", type=float, default=0.35)
    p.add_argument("--balance", type=int, default=0, help="lock refit: COM (vertex mean) over the support foot on quasi-static "
                   "single-support frames (one foot planted, the other lifted > --bal_lift, no hand contact, COM < --bal_speed)")
    p.add_argument("--w_bal", type=float, default=0.1)
    p.add_argument("--bal_margin", type=float, default=0.06)
    p.add_argument("--bal_lift", type=float, default=0.10)
    p.add_argument("--bal_speed", type=float, default=0.25)
    p.add_argument("--guard", type=int, default=0, help="1: spread root jumps faster than --guard_speed m/s over +-guard_win frames")
    p.add_argument("--guard_speed", type=float, default=5.0)
    p.add_argument("--guard_win", type=int, default=8)
    p.add_argument("--lock_acc_rot", type=float, default=0.003, help="lock refit: leg rotation increment acceleration scale (rad / frame^2)")
    p.add_argument("--lock_acc_root", type=float, default=0.0005, help="lock refit: pelvis shift acceleration scale (m / frame^2)")
    p.add_argument("--lock_max_off", type=float, default=0.05, help="lock: a new pin once the group drifts 2x this (m; 0 off)")
    p.add_argument("--lock_max_pin", type=float, default=0.0, help="lock: no pin whose median offset exceeds this (m; 0 off)")
    p.add_argument("--lock_rot_prior", type=float, default=0.03, help="lock refit: leg rotation increment prior (rad, weight 0.01)")
    p.add_argument("--lock_iters", type=int, default=300)
    p.add_argument("--lock_sigma", type=float, default=0.01, help="lock refit: pin target scale (m)")
    p.add_argument("--lock_sigma_root", type=float, default=0.10, help="lock refit: pelvis shift prior (m)")
    a = p.parse_args()

    from bench.body_models import SMPL
    from bench.videomimic.recon import Terrain
    from fab.contact_regions import smpl_regions
    from fab.retarget.contact import human_contact_labels

    import time

    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wp = Path(a.world)
    name = wp.stem
    r = dict(np.load(wp, allow_pickle=True))
    if Path(a.hfield).exists():
        hf = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in np.load(a.hfield).items()}
    else:  # no scene (fab-world's flat-floor fallback): a floor at z = 0 around the path
        J = np.asarray(r["joints"], np.float64)
        lo = J[:, 0, :2].min(0) - 2.0
        ny, nx = (np.ceil((J[:, 0, :2].max(0) + 2.0 - lo) / 0.04).astype(int) + 1)[::-1]
        hf = {"x0": float(lo[0]), "y0": float(lo[1]), "res": 0.04, "height": np.zeros((ny, nx), np.float32),
              "valid": np.zeros((ny, nx), bool)}
    body = SMPL(Path(a.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").to(dev).eval()
    part = body.weights.argmax(1).cpu().numpy()
    Jreg = body.J_regressor.cpu().numpy().astype(np.float64)
    regions = smpl_regions(a.bm_dir)
    pose = np.asarray(r["pose"], np.float64)
    trans = np.asarray(r["trans"], np.float64)
    betas = np.asarray(r["betas"], np.float64)
    T = np.asarray(r["T_world_cam"], np.float64).copy()
    n = len(pose)
    fps = float(r["fps"])
    with torch.no_grad():
        _, j0 = body(torch.zeros(1, 3, device=dev), torch.zeros(1, 69, device=dev),
                     torch.as_tensor(betas, dtype=torch.float32, device=dev)[None], None)
    J0 = j0[0, 0].cpu().numpy().astype(np.float64)
    rep = {"name": name}
    rep["guard"] = {"jumps": 0}
    if a.guard:  # root-jump safety net (a tracking switch): spread each jump over +-guard_win frames
        trans, rep["guard"] = guard_jumps(trans, pose, fps, a.guard_speed, a.guard_win)
    V = smpl_verts(body, pose, betas, trans, dev)
    Ps, Pa, src = load_contacts(a.contacts, n)
    rep["contacts"] = src
    from bench.scene_recon.joint import footprints

    fp, ft = footprints_t(V, Ps, regions, a.p_contact)
    rep["ground"] = {"footprints": int(len(fp))}
    plane, drift = None, None
    if len(fp) >= a.min_footprints:
        plane, share, mad = robust_plane(fp, ridge=a.plane_ridge)
        if a.floor_drift:  # one floor under a drifting body: the plane + a smooth drift over time explains the footprints
            plane_d, delta, share_d, mad_d = robust_plane_drift(fp, ft, a.floor_win * fps, ridge=a.plane_ridge,
                                                                smooth=a.drift_smooth, rate=a.drift_rate)
            dstd = float(np.std(delta(ft)))
            rep["ground"].update(drift_inlier_share=share_d, drift_std_m=dstd, plane_inlier_share=share)
            if share < a.min_inliers <= share_d and dstd >= a.drift_min:  # only where one plane fails because of the drift
                plane, share, mad, drift = plane_d, share_d, mad_d, delta
        nrm = np.array([-plane[0], -plane[1], 1.0])
        nrm /= np.linalg.norm(nrm)
        rep["ground"].update(plane=plane.tolist(), inlier_share=share, mad_m=mad, tilt_deg=float(np.degrees(np.arccos(nrm[2]))),
                             extent_m=float(np.ptp(fp[:, :2], axis=0).max()), drift_mode=drift is not None)
    rep["level"] = {"applied": False}
    if a.level and plane is not None and a.level_min <= rep["ground"]["tilt_deg"] <= a.level_max \
            and rep["ground"]["inlier_share"] >= a.min_inliers:
        Rl = rot_between(nrm, np.array([0.0, 0, 1]))
        ctr = np.r_[fp[:, :2].mean(0), plane[0] * fp[:, 0].mean() + plane[1] * fp[:, 1].mean() + plane[2]]
        T[:, :3, :3] = Rl @ T[:, :3, :3]
        T[:, :3, 3] = (T[:, :3, 3] - ctr) @ Rl.T + ctr
        trans = (trans + J0 - ctr) @ Rl.T + ctr - J0  # the SMPL root (J0 + trans) rotated about ctr
        pose[:, :3] = (Rotation.from_matrix(Rl) * Rotation.from_rotvec(pose[:, :3])).as_rotvec()
        ny, nx = hf["height"].shape
        X, Y = np.meshgrid(float(hf["x0"]) + hf["res"] * np.arange(nx), float(hf["y0"]) + hf["res"] * np.arange(ny))
        hf["height"] = (hf["height"] - (plane[0] * (X - ctr[0]) + plane[1] * (Y - ctr[1]))).astype(np.float32)
        V = smpl_verts(body, pose, betas, trans, dev)
        fp = (fp - ctr) @ Rl.T + ctr
        rep["level"] = {"applied": True, "deg": rep["ground"]["tilt_deg"], "R": Rl.tolist(), "centre": ctr.tolist()}
    pel = np.einsum("v,tvc->tc", Jreg[0], V)
    ground_ok = plane is not None and rep["ground"]["inlier_share"] >= a.min_inliers
    rep["ground"]["ok"] = bool(ground_ok)
    if a.terrain == "plane" and ground_ok:
        g0 = float(np.median(fp[:, 2] - (drift(ft) if drift is not None else 0))) if rep["level"]["applied"] else float(
            plane[2] + plane[0] * fp[:, 0].mean() + plane[1] * fp[:, 1].mean())
        terr = PlaneTerrain(g0)
        rep["ground"]["terrain"] = f"plane z = {g0:.4f}"
    else:
        terr = Terrain(hf, device=dev)
        rep["ground"]["terrain"] = "hfield"
    # in-view channels / vertices (the camera frame of each frame)
    K = np.asarray(r["K"], np.float64)
    Vc = np.einsum("tji,tvj->tvi", T[:, :3, :3], V - T[:, None, :3, 3])
    h_img, w_img = 2 * K[1, 2], 2 * K[0, 2]

    def in_img(X):
        z = np.clip(X[..., 2], 1e-3, None)
        u, v = X[..., 0] / z * K[0, 0] + K[0, 2], X[..., 1] / z * K[1, 1] + K[1, 2]
        return (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img) & (X[..., 2] > 0.1)

    inview = np.stack([in_img(Vc[:, ix]).mean(1) > 0.5 for ix in regions], 1)
    with torch.no_grad():  # planted channels on this ground (not on furniture / a fence): within --contact_gate of it
        gap0 = np.stack([(V[:, regions[c], 2] - terr(torch.as_tensor(V[:, regions[c], :2], dtype=torch.float32, device=dev))
                          .cpu().numpy()).min(1) for c in CH_FOOT], 1)
    if drift is not None:  # drifted contacts are judged against the local floor (plane + drift), corrected to the plane
        gap0 = gap0 - drift(np.arange(n))[:, None]
    near = np.abs(gap0) < a.contact_gate
    st0 = Ps[:, :4] >= a.p_contact
    rep["ground"]["planted_off_ground_share"] = float((st0 & ~near).sum() / max(st0.sum(), 1))
    sub = np.arange(0, 6890, 7)
    sub_inview = in_img(Vc[:, sub])
    rep["depth"] = {"applied": False}
    if a.depth and ground_ok and (Ps[:, :4] >= a.p_contact).any():
        bb, zz, rd = depth_stage(V[:, sub], [V[:, ix] for ix in (regions[c] for c in CH_FOOT)], pel, T[:, :3, 3], terr, Ps,
                                 inview, sub_inview, fps, a, dev, near=near, vertical=a.drift_vertical if drift is not None else 0.0)
        shift = (pel - T[:, :3, 3]) * (np.exp(bb) - 1)[:, None] + np.c_[np.zeros((n, 2)), zz]
        pel = pel + shift
        trans = trans + shift
        V = V + shift[:, None]
        rep["depth"] = {"applied": True, **rd}
    rep["lock"] = {"applied": False}
    Pl = Ps
    if a.kin_contact and ground_ok:  # + kinematic plants: a heel / toe group on the ground and slow (the model's misses)
        with torch.no_grad():
            kin = np.zeros((n, 4), bool)
            for k, c in enumerate(CH_FOOT):
                X = V[:, regions[c]]
                h = (X[..., 2] - terr(torch.as_tensor(X[..., :2], dtype=torch.float32, device=dev)).cpu().numpy()).min(1)
                sp = np.r_[np.inf, np.linalg.norm(np.diff(X[..., :2].mean(1), axis=0), axis=1) * fps]
                kin[:, k] = (np.abs(h) < a.kin_h) & (sp < a.kin_v)
        Pl = Ps.copy()
        Pl[:, :4] = np.maximum(Ps[:, :4], kin * 1.0)
        lp = a.lock_p if a.lock_p > 0 else a.p_contact
        rep["lock_kin_added_share"] = float(((Pl[:, :4] >= lp) & ~(Ps[:, :4] >= lp)).mean())
    if a.lock and (Pl[:, :4] >= (a.lock_p if a.lock_p > 0 else a.p_contact)).any():
        pose, dtr, rl = ground_lock_refit(body, pose, betas, trans, terr, regions, Pl, fps, a, dev, ground_z=ground_ok, near=near)
        trans = trans + dtr
        V = smpl_verts(body, pose, betas, trans, dev)
        rep["lock"] = {"applied": True, **rl}
    Jw = np.einsum("jv,tvc->tjc", Jreg, V)
    Vrel = V.copy()
    Vrel[..., 2] -= terr(torch.as_tensor(V[..., :2], dtype=torch.float32, device=dev)).cpu().numpy()
    labels = human_contact_labels(Vrel, part, fps, 0.05, 0.3)
    low = Vrel[:, sub, 2].min(1)
    rep["after"] = {"pen_mm_p95": float(np.percentile(np.clip(-low, 0, None), 95) * 1000),
                    "float_mm_p50": float(np.median(np.clip(low, 0, None)) * 1000)}
    rep["runtime_s"] = round(time.time() - t_start, 2)  # (after the imports: body model + inputs + refinement)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    out = dict(r)
    out.update(pose_in=r["pose"], trans_in=r["trans"], pose=pose.astype(np.float32), trans=trans.astype(np.float32),
               joints=Jw.astype(np.float32), T_world_cam=T, contacts=json.dumps({k: v.astype(int).tolist() for k, v in labels.items()}),
               refine_world=json.dumps(rep))
    np.savez(od / f"{name}.npz", **out)
    np.savez(od / f"{name}_hfield.npz", **hf)
    if a.scene and Path(a.scene).exists():  # the scene mesh in the refined (levelled) world
        import trimesh

        m = trimesh.load(a.scene, process=False)
        if rep["level"]["applied"]:
            Rl, ctr = np.asarray(rep["level"]["R"]), np.asarray(rep["level"]["centre"])
            m.vertices = (np.asarray(m.vertices) - ctr) @ Rl.T + ctr
        m.export(od / f"{name}_scene.ply")
    print(f"[world_refine] {name}: {rep['ground']['footprints']} footprints, tilt {rep['ground'].get('tilt_deg', float('nan')):.2f} deg "
          f"({'applied' if rep['level']['applied'] else 'kept'}), depth |b| {rep['depth'].get('b_abs_mean', 0):.3f} "
          f"max {rep['depth'].get('b_absmax', 0):.3f}, lock pins {rep['lock'].get('lock_pins', 0)} pelvis shift "
          f"{rep['lock'].get('pelvis_shift_mm_mean', 0):.0f} mm, after pen p95 {rep['after']['pen_mm_p95']:.0f} mm float p50 "
          f"{rep['after']['float_mm_p50']:.0f} mm, {rep['runtime_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
