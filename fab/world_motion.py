"""FAB-M: a learned world-motion stage on top of FAB's per-frame camera-space bodies and the camera (fab-world).

Per frame it reads FAB's camera-space SMPL (raw, unsmoothed), the box, the contact probabilities and the camera
(camera -> gravity-aligned z-up world); a transformer over the clip predicts a refined local pose and root orientation,
a correction of the root along the camera ray with uncertainties, a root velocity and static contacts. A closed-form
trajectory fusion then places the pelvis: tight across the ray (the image evidence), loose along it (FAB's depth
noise), plus the learned velocity and zero velocity of the planted contact points.

    python -m fab.world_motion <out>/<name>.npz [--work <out>/work] [--model world_motion.pt] [--out <dir>] [--any_camera]
        [--fps_mode auto|resample] [--pose_src export|model] [--anchor 0.02,0.02|off] [--floor 0.05,0.5|off] [--sigma_c 0.01]
        -> the export with the refined body (pose / trans / joints / contacts); the kinematic body in <work>/kin/<name>.npz

Static cameras only: an export whose notes do not say static_camera true (fab-world's camera check) passes through
unchanged (notes.world_motion.skipped); --any_camera runs it on moving cameras too. Frame rate (FAB-M was trained at
30 fps): 30 fps runs as is; other rates are resampled to 30 fps and back, from AUTO_MIN_FPS up (--fps_mode auto) or
from MIN_FPS up (--fps_mode resample); slower clips pass through.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTACT_JOINTS = (7, 10, 8, 11, 20, 21, 4, 5)  # l_heel, l_toe, r_heel, r_toe, l_hand, r_hand, l_knee, r_knee (CONTACT_NAMES)
FEET = (0, 1, 2, 3)


# ---------------------------------------------------------------- rotations
def aa_to_mat(aa):
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return torch.eye(3, dtype=aa.dtype, device=aa.device) + s * K + (1 - c) * (K @ K)


def mat_to_aa(R):
    q = mat_to_quat(R)
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    n = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, q[..., :1])
    return q[..., 1:] * torch.where(n > 1e-8, angle / n.clamp_min(1e-8), 2.0 / q[..., :1].clamp_min(1e-8))


def mat_to_quat(R):
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qs = torch.stack([1 + t, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                      1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    i = qs.argmax(-1, keepdim=True)
    w = torch.stack([1 + t, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1)
    x = torch.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0],
                     m[..., 0, 2] + m[..., 2, 0]], -1)
    y = torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                     m[..., 1, 2] + m[..., 2, 1]], -1)
    z = torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1],
                     1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1)
    cand = torch.stack([w, x, y, z], -2)
    q = torch.gather(cand, -2, i[..., None].expand(*i.shape[:-1], 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def mat_to_6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def rz(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*yaw.shape, 3, 3)


def heading(R):
    """Yaw of the SMPL body's forward axis (+z of the root) in a z-up world."""
    f = R[..., :, 2]
    return torch.atan2(f[..., 1], f[..., 0])


# ---------------------------------------------------------------- kinematics
class SMPLKin(nn.Module):
    """SMPL joints by forward kinematics plus 8 contact points (heel / toe / hand / knee region centroids, rigidly attached
    to CONTACT_JOINTS); no skinning, so it is cheap enough for training batches."""

    def __init__(self, smpl_pkl):
        super().__init__()
        from bench.body_models import load_smpl_pkl

        m = load_smpl_pkl(smpl_pkl)
        v, sd, Jr = m["v_template"].astype(np.float64), m["shapedirs"][:, :, :10].astype(np.float64), m["J_regressor"]
        part = m["weights"].argmax(1)
        J = Jr @ v
        regs = []
        for parts in ((7, 10), (8, 11)):
            ix = np.flatnonzero(np.isin(part, parts))
            y, z = v[ix, 1], v[ix, 2]
            sole = ix[y < y.min() + 0.02]
            L = z.max() - z.min()
            regs += [sole[v[sole, 2] < z.min() + 0.3 * L], sole[v[sole, 2] > z.min() + 0.7 * L]]
        for parts in ((20, 22), (21, 23)):
            regs.append(np.flatnonzero(np.isin(part, parts)))
        for j in (4, 5):
            d = np.linalg.norm(v - J[j], axis=1)
            regs.append(np.flatnonzero((d < 0.07) & (v[:, 2] > J[j, 2])))
        C = np.stack([v[r].mean(0) for r in regs])
        Cd = np.stack([sd[r].mean(0) for r in regs])  # (8, 3, 10)
        if regs[0].size and regs[2].size:  # heel / toe points at the sole (lowest vertex height of the region)
            for k in FEET:
                C[k, 1] = v[regs[k], 1].min()
        self.register_buffer("Jt", torch.tensor(J, dtype=torch.float32))
        self.register_buffer("Jd", torch.tensor(np.einsum("jv,vcl->jcl", Jr, sd), dtype=torch.float32))
        self.register_buffer("Ct", torch.tensor(C, dtype=torch.float32))
        self.register_buffer("Cd", torch.tensor(Cd, dtype=torch.float32))
        parents = m["kintree_table"][0].copy()
        parents[0] = -1
        self.parents = parents.tolist()

    def rest(self, betas):
        b = betas[..., :10]
        return self.Jt + torch.einsum("jcl,...l->...jc", self.Jd, b), self.Ct + torch.einsum("jcl,...l->...jc", self.Cd, b)

    def forward(self, R, betas, transl=None):
        """R (..., 24, 3, 3) local rotations (R[..., 0] the root's), betas (..., 10) -> joints (..., 24, 3), contact points
        (..., 8, 3), global rotations (..., 24, 3, 3); the pelvis at J0(betas) + transl."""
        Jr, Cr = self.rest(betas)
        G, P = [R[..., 0, :, :]], [Jr[..., 0, :]]
        for i in range(1, 24):
            p = self.parents[i]
            G.append(G[p] @ R[..., i, :, :])
            P.append(P[p] + (G[p] @ (Jr[..., i, :] - Jr[..., p, :])[..., None])[..., 0])
        G, P = torch.stack(G, -3), torch.stack(P, -2)
        cj = list(CONTACT_JOINTS)
        Cp = P[..., cj, :] + (G[..., cj, :, :] @ (Cr - Jr[..., cj, :])[..., None])[..., 0]
        if transl is not None:
            P, Cp = P + transl[..., None, :], Cp + transl[..., None, :]
        return P, Cp, G


# ---------------------------------------------------------------- network
class RoPEBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, cos, sin, key_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.n2(x)))


def rope_tables(T, dh, device, base=10000.0):
    f = 1.0 / base ** (torch.arange(0, dh, 2, device=device).float() / dh)
    a = torch.arange(T, device=device).float()[:, None] * f[None]
    return torch.cos(a), torch.sin(a)


def rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


N_BODY, N_CAM = 138 + 6 + 6 + 69 + 24 + 10 + 8 + 5, 3 + 6 + 3 + 3 + 3 + 1
N_CON = 17
N_OUT = 72 + 1 + 2 + 2 + 3 + 1 + 8


class WorldMotionNet(nn.Module):
    def __init__(self, d=384, layers=8, heads=6, ff=4, dropout=0.1, pose_head=True, sig_a_max=3.0):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff, dropout=dropout, pose_head=pose_head, sig_a_max=sig_a_max)
        self.body = nn.Linear(N_BODY, d)
        self.cam = nn.Linear(N_CAM, d)
        self.con = nn.Linear(N_CON, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.norm_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(RoPEBlock(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, N_OUT)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f):
        """f: build_features output -> raw head outputs (B, T, N_OUT)."""
        v = f["valid"][..., None]
        x = torch.where(v > 0, self.body(f["body"]), self.mask_tok) + self.cam(f["cam"]) + self.con(f["con"])
        x = self.norm_in(x)
        T = x.shape[1]
        cos, sin = rope_tables(T, x.shape[-1] // self.blocks[0].h, x.device)
        km = (~f["pad"])[:, None, None, :] if "pad" in f else None  # padded frames take no part as keys
        for b in self.blocks:
            x = b(x, cos, sin, km)
        return self.out(self.norm(x))


# ---------------------------------------------------------------- features
def canonical(Rw0, pw, valid):
    """Per window: yaw so that the first valid frame's heading is +x, origin at its pelvis. -> A (B, 3, 3), o (B, 3)."""
    B = pw.shape[0]
    first = valid.float().argmax(1)
    ix = torch.arange(B, device=pw.device)
    A = rz(-heading(Rw0[ix, first]))
    return A, pw[ix, first]


def build_features(kin, d, fps=30.0):
    """d: per window, torch (B, T, ...): R_c (24, 3, 3) camera-frame rotations (root in the camera), betas (10), trans_c
    (3), T_wc (4, 4) camera -> z-up world, K (B, 3, 3), wh (B, 2), box (4), valid, cprob (16), cmask.
    -> features for WorldMotionNet and the geometry the fusion and the losses need, in the window's canonical frame."""
    Rc, betas, valid = d["R_c"], d["betas"], d["valid"].float()
    Rwc, twc = d["T_wc"][..., :3, :3], d["T_wc"][..., :3, 3]
    P_c, C_c, _ = kin(Rc, betas, d["trans_c"])
    pel_c = P_c[..., 0, :]
    Rw = Rwc[..., None, :, :] @ Rc[..., :1, :, :]
    pw = (Rwc @ pel_c[..., None])[..., 0] + twc
    A, o = canonical(Rw[..., 0, :, :], pw, valid > 0)
    Ab = A[:, None]
    Rw0 = Ab @ Rw[..., 0, :, :]
    p = (Ab @ (pw - o[:, None])[..., None])[..., 0]
    cam_R = Ab @ Rwc
    cam_c = (Ab @ (twc - o[:, None])[..., None])[..., 0]
    ray = p - cam_c
    depth = ray.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    ray_dir = ray / depth
    vel = torch.cat([torch.zeros_like(p[:, :1]), p[:, 1:] - p[:, :-1]], 1) * fps
    cvel = torch.cat([torch.zeros_like(cam_c[:, :1]), cam_c[:, 1:] - cam_c[:, :-1]], 1) * fps
    dR = cam_R[:, :-1].transpose(-1, -2) @ cam_R[:, 1:]
    cang = torch.cat([torch.zeros_like(cam_c[:, :1]), mat_to_aa(dR)], 1) * fps
    # body joints / contact points relative to the pelvis, canonical world orientation
    rel = lambda X: (Ab[:, :, None] @ (Rwc[..., None, :, :] @ (X - pel_c[..., None, :])[..., None]))[..., 0]  # noqa: E731
    J = rel(P_c[..., 1:, :]).flatten(-2)
    Cp = rel(C_c).flatten(-2)
    K, wh = d["K"], d["wh"]
    box = d["box"]
    W, H = wh[:, None, 0:1], wh[:, None, 1:2]
    bn = torch.cat([(box[..., 0:1] + box[..., 2:3]) / 2 / W, (box[..., 1:2] + box[..., 3:4]) / 2 / H,
                    (box[..., 2:3] - box[..., 0:1]) / W, (box[..., 3:4] - box[..., 1:2]) / H], -1)
    tr = torch.cat([box[..., 0:1] < 3, box[..., 1:2] < 3, box[..., 2:3] > W - 3, box[..., 3:4] > H - 3], -1).float()
    pc_n = pel_c / pel_c.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    lf = torch.log(K[:, None, 0:1, 0] / H)
    body = torch.cat([mat_to_6d(Rc[..., 1:, :, :]).flatten(-2), mat_to_6d(Rw0), torch.cat([p, vel * 0.1], -1), J, Cp,
                      betas[..., :10] * 0.3, bn, tr, pc_n, torch.log(depth) * 0.5, lf.expand_as(depth)], -1)
    cam = torch.cat([cam_c * 0.3, mat_to_6d(cam_R), cvel * 0.3, cang * 0.3, ray_dir, torch.log(depth) * 0.5], -1)
    cm = d["cmask"].float()[..., None]
    con = torch.cat([torch.nan_to_num(d["cprob"]) * cm, cm], -1)
    extra = {"pad": d["pad"]} if "pad" in d else {}
    extra.update(cprob=torch.nan_to_num(d["cprob"]), cmask=d["cmask"].float())
    return {**extra, "body": body * valid[..., None], "cam": cam, "con": con * valid[..., None], "valid": valid,
            "A": A, "o": o, "p": p, "Rw0": Rw0, "cam_c": cam_c, "cam_R": cam_R, "ray_dir": ray_dir, "depth": depth,
            "R_loc": Rc[..., 1:, :, :], "betas": betas}


# ---------------------------------------------------------------- heads + fusion
SIG = {"a": (0.005, 3.0), "p": (0.002, 0.5), "v": (5e-4, 0.5)}


def bounded_sigma(x, k, hi=None):
    lo, hi0 = SIG[k]
    hi = hi0 if hi is None else hi
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(x))


def decode(kin, f, y, sigma_c=0.01, pose_head=True, sig_a_max=3.0):
    """Raw outputs -> refined local / root rotations, the fusion's observation, weights and velocity targets
    (canonical frame, per frame). pose_head False: FAB's own rotations are kept (only the trajectory and contacts are
    learned)."""
    B, T = y.shape[:2]
    dR = aa_to_mat(y[..., :72].reshape(B, T, 24, 3) * (0.5 if pose_head else 0.0))
    R_root = f["Rw0"] @ dR[..., 0, :, :]
    R_loc = f["R_loc"] @ dR[..., 1:, :, :]
    i = 72
    delta = y[..., i] * 0.2
    eps = torch.tanh(y[..., i + 1:i + 3]) * 0.1
    sa, sp = bounded_sigma(y[..., i + 3], "a", sig_a_max), bounded_sigma(y[..., i + 4] - 2.0, "p")
    vel_h = y[..., i + 5:i + 8] * 0.05
    sv = bounded_sigma(y[..., i + 8], "v")
    cm = f["cmask"][..., None]
    base = torch.logit(f["cprob"][..., :8].clamp(0.02, 0.98)) * cm - 2.0 * (1 - cm)  # FAB's static contacts (or none)
    clog = base + y[..., i + 9:i + 17]
    # observation point: the input pelvis moved along its ray by exp(delta) + a small image-plane residual
    cx, cy = f["cam_R"][..., :, 0], f["cam_R"][..., :, 1]
    obs = f["cam_c"] + torch.exp(delta)[..., None] * (f["p"] - f["cam_c"]) + eps[..., :1] * cx + eps[..., 1:] * cy
    dd = f["ray_dir"][..., :, None] * f["ray_dir"][..., None, :]
    I3 = torch.eye(3, device=y.device)
    Lam = (dd / sa[..., None, None] ** 2 + (I3 - dd) / sp[..., None, None] ** 2) * f["valid"][..., None, None]
    Lam = Lam + 1e-4 * I3
    Rfull = torch.cat([R_root[..., None, :, :], R_loc], -3)
    zero = torch.zeros(B, T, 3, device=y.device)
    P0, C0, _ = kin(Rfull, f["betas"], zero)
    off = C0 - P0[..., :1, :]  # contact points relative to the pelvis
    yaw = heading(R_root)
    p = f["p"]
    v_in = torch.cat([p[:, 1:2] - p[:, :1], (p[:, 2:] - p[:, :-2]) / 2, p[:, -1:] - p[:, -2:-1]], 1)
    v = v_in + (rz(yaw) @ vel_h[..., None])[..., 0]  # the input velocity + a residual in the heading frame
    pc = torch.sigmoid(clog)
    wc = (pc[:, :-1] * pc[:, 1:])[..., list(FEET)]  # planted feet at t and t + 1
    doff = off[:, 1:, list(FEET)] - off[:, :-1, list(FEET)]
    wv = 1.0 / sv[:, :-1] ** 2
    wcs = wc / sigma_c ** 2
    w = wv + wcs.sum(-1)
    u = (wv[..., None] * v[:, :-1] - (wcs[..., None] * doff).sum(-2)) / w[..., None]
    return {"R_root": R_root, "R_loc": R_loc, "obs": obs, "Lam": Lam, "w": w, "u": u, "contact_logit": clog,
            "sa": sa, "sp": sp, "sv": sv, "v": v, "delta": delta, "off": off}


def fuse_dense(Lam, obs, w, u):
    """argmin_r sum_t (r_t - obs_t)^T Lam_t (r_t - obs_t) + sum_t w_t |r_{t+1} - r_t - u_t|^2, batched dense Cholesky.
    Lam (B, T, 3, 3), obs (B, T, 3), w (B, T-1), u (B, T-1, 3) -> r (B, T, 3)."""
    B, T = obs.shape[:2]
    dev = obs.device
    L = torch.zeros(B, T, T, device=dev, dtype=obs.dtype)
    ar = torch.arange(T - 1, device=dev)
    L[:, ar, ar] += w
    L[:, ar + 1, ar + 1] += w
    L[:, ar, ar + 1] -= w
    L[:, ar + 1, ar] -= w
    I3 = torch.eye(3, device=dev, dtype=obs.dtype)
    A = (L[:, :, None, :, None] * I3[None, None, :, None, :]).reshape(B, 3 * T, 3 * T)
    At = torch.arange(T, device=dev)
    A = A.reshape(B, T, 3, T, 3)
    A[:, At, :, At, :] = A[:, At, :, At, :] + Lam.transpose(0, 1)
    A = A.reshape(B, 3 * T, 3 * T)
    b = (Lam @ obs[..., None])[..., 0]
    wu = w[..., None] * u
    b = b + F.pad(wu, (0, 0, 1, 0)) - F.pad(wu, (0, 0, 0, 1))
    Lc = torch.linalg.cholesky(A)
    return torch.cholesky_solve(b.reshape(B, 3 * T, 1), Lc).reshape(B, T, 3)


def fuse_thomas(Lam, obs, w, u):
    """Same as fuse_dense for one clip (numpy float64, any length): block-tridiagonal forward / back substitution."""
    T = len(obs)
    I3 = np.eye(3)
    wp = np.r_[0.0, w]
    wn = np.r_[w, 0.0]
    b = np.einsum("tij,tj->ti", Lam, obs)
    b[1:] += w[:, None] * u
    b[:-1] -= w[:, None] * u
    Cinv, dp = np.zeros((T, 3, 3)), np.zeros((T, 3))
    M = Lam[0] + (wp[0] + wn[0]) * I3
    Cinv[0] = np.linalg.inv(M)
    dp[0] = b[0]
    for t in range(1, T):
        M = Lam[t] + (wp[t] + wn[t]) * I3 - wp[t] ** 2 * Cinv[t - 1]
        Cinv[t] = np.linalg.inv(M)
        dp[t] = b[t] + wp[t] * Cinv[t - 1] @ dp[t - 1]
    r = np.zeros((T, 3))
    r[-1] = Cinv[-1] @ dp[-1]
    for t in range(T - 2, -1, -1):
        r[t] = Cinv[t] @ (dp[t] + wn[t] * r[t + 1])
    return r


def thomas_rhs(Lam, w, b):
    """Solve the fusion's block-tridiagonal system A r = b (A from Lam (T, 3, 3) and the step weights w (T-1,))."""
    T = len(b)
    I3 = np.eye(3)
    wp, wn = np.r_[0.0, w], np.r_[w, 0.0]
    Cinv, dp = np.zeros((T, 3, 3)), np.zeros((T, 3))
    Cinv[0] = np.linalg.inv(Lam[0] + (wp[0] + wn[0]) * I3)
    dp[0] = b[0]
    for t in range(1, T):
        Cinv[t] = np.linalg.inv(Lam[t] + (wp[t] + wn[t]) * I3 - wp[t] ** 2 * Cinv[t - 1])
        dp[t] = b[t] + wp[t] * Cinv[t - 1] @ dp[t - 1]
    r = np.zeros((T, 3))
    r[-1] = Cinv[-1] @ dp[-1]
    for t in range(T - 2, -1, -1):
        r[t] = Cinv[t] @ (dp[t] + wn[t] * r[t + 1])
    return r


def fuse_floor(Lam, obs, w, u, wf, offz):
    """fuse_thomas + a flat floor shared by the whole clip: sum_tk wf_tk (r_tz + offz_tk - h)^2 with h free (the planted heel /
    toe points all at one height, so the stance height cannot drift). wf (T, K) weights (1 / m^2), offz (T, K) the contact
    points' heights relative to the pelvis. -> pelvis (T, 3), floor height h."""
    T = len(obs)
    Wt = wf.sum(1)
    Lf = Lam.copy()
    Lf[:, 2, 2] += Wt
    b = np.einsum("tij,tj->ti", Lam, obs)
    b[1:] += w[:, None] * u
    b[:-1] -= w[:, None] * u
    b0 = b.copy()
    b0[:, 2] -= (wf * offz).sum(1)
    b1 = np.zeros((T, 3))
    b1[:, 2] = Wt
    rA, rB = thomas_rhs(Lf, w, b0), thomas_rhs(Lf, w, b1)
    den = (Wt * (1 - rB[:, 2])).sum()
    if Wt.sum() <= 0 or abs(den) < 1e-9:
        return thomas_rhs(Lam, w, b), float("nan")
    h = float(((wf * (rA[:, 2:3] + offz)).sum()) / den)
    return rA + h * rB, h


# ---------------------------------------------------------------- inference on a fab-world export
def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = WorldMotionNet(**ck["cfg"]).to(device).eval()
    net.load_state_dict(ck["model"])
    return net, ck


def windows(T, W, hop):
    if T <= W:
        return [(0, T)]
    s = list(range(0, T - W, hop)) + [T - W]
    return [(a, a + W) for a in s]


@torch.no_grad()
def export_anchor(kin, d, rel, perp):
    """The input pelvis (the export's, in export mode) as an extra observation with FAB's own noise: along its camera ray
    sigma = rel x depth, across it sigma = perp (m). -> (Lam (T, 3, 3), pelvis (T, 3)), world frame."""
    Jr, _ = kin.rest(torch.as_tensor(d["betas"], dtype=torch.float32))
    pel_c = d["trans_c"] + Jr[:, 0].cpu().numpy().astype(np.float64)
    R, t = d["T_wc"][:, :3, :3], d["T_wc"][:, :3, 3]
    pw = np.einsum("tij,tj->ti", R, pel_c) + t
    ray = pw - t
    dep = np.linalg.norm(ray, axis=1, keepdims=True).clip(1e-3)
    u = ray / dep
    dd = u[:, :, None] * u[:, None, :]
    Lam = dd / (rel * dep[:, :, None]) ** 2 + (np.eye(3) - dd) / perp ** 2
    return Lam * (d["valid"] > 0)[:, None, None], pw


@torch.no_grad()
def run_clip(net, kin, d, W=240, hop=180, fps=30.0, device="cuda", sigma_c=0.01, anchor=None, floor=None,
             floor_channels="feet", floor_upright=False, nopen=0, nopen_tol=0.03):
    """anchor: (rel, perp) = export_anchor's sigmas: the export kept wherever the model is not clearly more certain
    ("do no harm"); None: off. floor: (sigma m, p_min): one flat floor under the planted heel / toe points (contact
    probability >= p_min), sigma its tolerance; None: off. floor_channels: "feet" (heels / toes) or "all" (+ hands, knees);
    floor_upright: no floor term on frames whose body is within 45 deg of horizontal (prone / supine); nopen: rounds of an
    active set that pins a hand / knee point found below the floor - nopen_tol back onto it."""
    """d: one clip as numpy arrays (R_c (T, 24, 3, 3), betas, trans_c, T_wc, K, wh, box, valid, cprob, cmask) -> refined
    world motion: R_root (T, 3, 3), R_loc (T, 23, 3, 3), pelvis (T, 3), contact probabilities (T, 8)."""
    T = len(d["valid"])
    acc = {k: np.zeros((T,) + s) for k, s in (("R_root", (3, 3)), ("R_loc", (23, 3, 3)), ("obs", (3,)), ("Lam", (3, 3)),
                                               ("w", ()), ("u", (3,)), ("c", (8,)), ("offz", (8,)))}
    wsum = np.zeros(T)
    for a, b in windows(T, W, hop):
        t = {k: torch.as_tensor(np.asarray(v[a:b] if np.ndim(v) and len(v) == T else v), dtype=torch.float32,
                                device=device)[None] for k, v in d.items()}
        f = build_features(kin, t, fps)
        o = decode(kin, f, net(f), sigma_c=sigma_c, pose_head=net.cfg.get("pose_head", True),
                   sig_a_max=net.cfg.get("sig_a_max", 3.0))
        Ai, oo = f["A"][0].T.cpu().numpy().astype(np.float64), f["o"][0].cpu().numpy().astype(np.float64)
        n = b - a
        ramp = np.ones(n)
        e = min(W - hop, n // 2)
        if a > 0:
            ramp[:e] = np.linspace(0, 1, e + 2)[1:-1]
        if b < T:
            ramp[n - e:] = np.minimum(ramp[n - e:], np.linspace(1, 0, e + 2)[1:-1])
        g = lambda k: o[k][0].cpu().numpy().astype(np.float64)  # noqa: E731
        acc["R_root"][a:b] += ramp[:, None, None] * (Ai @ g("R_root"))
        acc["R_loc"][a:b] += ramp[:, None, None, None] * g("R_loc")
        acc["obs"][a:b] += ramp[:, None] * (g("obs") @ Ai.T + oo)
        acc["Lam"][a:b] += ramp[:, None, None] * (Ai @ g("Lam") @ Ai.T)
        acc["c"][a:b] += ramp[:, None] * (1 / (1 + np.exp(-g("contact_logit"))))
        acc["offz"][a:b] += ramp[:, None] * g("off")[:, :, 2]  # z is shared by the canonical and the world frame
        acc["w"][a:b - 1] += ramp[:-1] * g("w")
        acc["u"][a:b - 1] += ramp[:-1, None] * (g("u") @ Ai.T)
        wsum[a:b] += ramp
    wsum = np.maximum(wsum, 1e-9)
    out = {k: v / wsum.reshape((T,) + (1,) * (v.ndim - 1)) for k, v in acc.items()}
    for k in ("R_root", "R_loc"):  # project the blended rotations back to SO(3)
        U, _, Vt = np.linalg.svd(out[k])
        D = np.ones(U.shape[:-1])
        D[..., -1] = np.sign(np.linalg.det(U @ Vt))
        out[k] = (U * D[..., None, :]) @ Vt
    out["u"][:-1] = acc["u"][:-1] / wsum[:-1, None]
    Lam, obs = out["Lam"], out["obs"]
    if anchor is not None:
        La, pa = export_anchor(kin.cpu(), d, *anchor)
        kin.to(device)
        b = np.einsum("tij,tj->ti", Lam, obs) + np.einsum("tij,tj->ti", La, pa)
        Lam = Lam + La
        obs = np.linalg.solve(Lam, b[..., None])[..., 0]
    if floor is not None:
        sf, pmin = floor
        ch = list(FEET) if floor_channels == "feet" else list(range(8))
        cf = out["c"][:, ch]
        wf = np.where(cf >= pmin, cf, 0.0) * (d["valid"] > 0)[:, None] / sf ** 2
        if floor_upright:  # the body's up (root y axis) within 45 deg of horizontal: prone / supine, no floor term
            wf = wf * (out["R_root"][:, 2, 1] >= np.cos(np.radians(45)))[:, None]
        offz = out["offz"][:, ch]
        out["pelvis"], out["floor"] = fuse_floor(Lam, obs, out["w"][:-1], out["u"][:-1], wf, offz)
        for _ in range(nopen):  # active set: contact points below the floor are pinned onto it
            h = out["floor"]
            if not np.isfinite(h):
                break
            z = out["pelvis"][:, 2:3] + out["offz"]
            inf = np.zeros_like(z, bool)
            inf[:, ch] = wf > 0
            low = (z < h - nopen_tol) & ~inf  # only points the floor term does not already hold,
            low[:, list(FEET)] = False  # and only hands / knees (noisy FAB legs dip feet a few cm on their own)
            if not low.any():
                break
            wp = np.where(low, 1.0 / sf ** 2, 0.0)
            wa = np.concatenate([wf, wp], 1)
            oa = np.concatenate([offz, out["offz"]], 1)
            out["pelvis"], out["floor"] = fuse_floor(Lam, obs, out["w"][:-1], out["u"][:-1], wa, oa)
            out["nopen_frames"] = int(low.any(1).sum())
    else:
        out["pelvis"] = fuse_thomas(Lam, obs, out["w"][:-1], out["u"][:-1])
    return out


def export_inputs(exp, work, name, mode="raw", kin=None):
    """The per-frame inputs from a fab-world export + its work dir (scripts/fab_world.sh layout). mode raw: FAB's per-frame
    camera-frame bodies (work/ours/fab); export: the export's smoothed world body, in each camera frame."""
    raw = np.load(Path(work) / "ours" / "fab" / f"{name}_raw.npz")
    n = len(exp["T_world_cam"])
    T_wc = exp["T_world_cam"].astype(np.float64)
    pose_c = raw["pose_c"][:n].astype(np.float64)
    R_c = aa_to_mat(torch.as_tensor(pose_c.reshape(n, 24, 3))).numpy()
    valid = raw["valid"][:n].astype(bool) & exp["valid"][:n].astype(bool)
    K = exp["K"].astype(np.float64)
    wh = np.array([2 * K[0, 2], 2 * K[1, 2]])
    cprob, cmask = np.zeros((n, 16)), np.zeros(n)
    cp = Path(work) / "contacts" / f"{name}__0.contact.npz"
    if cp.exists():
        z = np.load(cp)
        fr = z["frame"].astype(int)
        ok = fr < n
        head = next(h for h in ("video_", "image_", "") if f"{h}any_prob" in z.files)
        cprob[fr[ok], 0:4], cprob[fr[ok], 8:12] = z[f"{head}static_prob"][ok], z[f"{head}any_prob"][ok]
        for k, sl in ((f"{head}hk_static_prob", slice(4, 8)), (f"{head}hk_any_prob", slice(12, 16))):
            if k in z.files:
                cprob[fr[ok], sl] = z[k][ok]
        cmask[fr[ok]] = 1
    elif "contact" in raw.files and np.isfinite(raw["contact"]).all():
        c = raw["contact"][:n]
        cprob[:, :8] = c[:, :8]
        if c.shape[1] >= 16:
            cprob[:, 8:] = c[:, 8:16]
        cmask[:] = 1
    betas, trans_c = raw["betas"][:n].astype(np.float64), raw["trans_c"][:n].astype(np.float64)
    if mode == "export":
        Rw = aa_to_mat(torch.as_tensor(exp["pose"][:n].astype(np.float64).reshape(n, 24, 3))).numpy()
        R_c = Rw.copy()
        R_c[:, 0] = np.swapaxes(T_wc[:, :3, :3], -1, -2) @ Rw[:, 0]
        betas = np.broadcast_to(exp["betas"].astype(np.float64), (n, 10)).copy()
        J0 = kin.rest(torch.as_tensor(betas[:1], dtype=torch.float32))[0][0, 0].numpy().astype(np.float64)
        pc = np.einsum("tji,tj->ti", T_wc[:, :3, :3], exp["joints"][:n, 0].astype(np.float64) - T_wc[:, :3, 3])
        trans_c = pc - J0
    return {"R_c": R_c, "betas": betas, "trans_c": trans_c,
            "T_wc": T_wc, "K": K, "wh": wh, "box": raw["box"][:n].astype(np.float64), "valid": valid.astype(np.float64),
            "cprob": cprob, "cmask": cmask * valid}


def depth_ratio(kin, d, exp):
    """The export's camera-frame pelvis depth over the raw one (the joint step's scale on the body placement)."""
    tt = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    P, _, _ = kin(tt(d["R_c"]), tt(d["betas"]), tt(d["trans_c"]))
    pc_raw = P[:, 0].numpy().astype(np.float64)
    Twc = d["T_wc"]
    pe = exp["joints"][:, 0].astype(np.float64)
    pc_exp = np.einsum("tji,tj->ti", Twc[:, :3, :3], pe - Twc[:, :3, 3])
    v = d["valid"] > 0
    r = np.linalg.norm(pc_exp[v], axis=1) / np.maximum(np.linalg.norm(pc_raw[v], axis=1), 1e-6)
    return float(np.median(r)) if len(r) else 1.0


def smooth_rotations(R, sigma):
    """Zero-phase Gaussian smoothing of rotation matrices (..., T, J, 3, 3) in time via sign-aligned quaternions."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    T, J = R.shape[:2]
    q = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(T, J, 4)
    for t in range(1, T):
        q[t][(q[t] * q[t - 1]).sum(-1) < 0] *= -1
    q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(T, J, 3, 3)


TRAIN_FPS, MIN_FPS = 30.0, 10.0  # FAB-M was trained at 30 fps; below MIN_FPS it is skipped (the export passes through)
AUTO_MIN_FPS = 24.0  # fps_mode auto (default): skipped below it (resampling to 30 passed the fps test at 24 / 25, failed at 15 / 20)


def fps_action(fps, fps_mode="auto"):
    """-> "run" (the clip's own rate is FAB-M's), "resample" (inputs to TRAIN_FPS, outputs back) or a skip reason.
    fps_mode auto: skip below AUTO_MIN_FPS; resample: skip only below MIN_FPS."""
    if abs(fps - TRAIN_FPS) <= 1e-3:
        return "run"
    lo = AUTO_MIN_FPS if fps_mode == "auto" else MIN_FPS
    if fps < lo:
        return f"fps {fps:g} < {lo:g} (fps_mode {fps_mode})"
    return "resample"


def _lin(x, t_from, t_to):
    """Linear interpolation of (T, ...) samples at times t_from onto t_to (clamped at the ends)."""
    x = np.asarray(x, np.float64)
    i = np.clip(np.searchsorted(t_from, t_to, side="right") - 1, 0, len(t_from) - 2) if len(t_from) > 1 else np.zeros(len(t_to), int)
    if len(t_from) == 1:
        return np.repeat(x[:1], len(t_to), 0)
    w = np.clip((t_to - t_from[i]) / (t_from[i + 1] - t_from[i]), 0, 1).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[i] + w * x[i + 1]


def _rot(R, t_from, t_to):
    """Rotation matrices (T, ..., 3, 3) at t_from onto t_to: sign-aligned quaternion nlerp."""
    q = mat_to_quat(torch.as_tensor(np.asarray(R, np.float64))).numpy()
    for t in range(1, len(q)):
        q[t] *= np.where((q[t] * q[t - 1]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    q = _lin(q, t_from, t_to)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                     2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def resample_inputs(d, fps, fps_to):
    """run_clip's per-frame inputs at fps -> fps_to (rotations nlerp, the rest linear; a new frame is valid only when both
    neighbours are)."""
    T = len(d["valid"])
    tf = np.arange(T) / fps
    tt = np.arange(0, tf[-1] + 1e-9, 1 / fps_to)
    out = dict(d)
    out["R_c"] = _rot(d["R_c"], tf, tt)
    Tw = np.repeat(np.eye(4)[None], len(tt), 0)
    Tw[:, :3, :3] = _rot(d["T_wc"][:, :3, :3], tf, tt)
    Tw[:, :3, 3] = _lin(d["T_wc"][:, :3, 3], tf, tt)
    out["T_wc"] = Tw
    for k in ("betas", "trans_c", "box", "cprob"):
        out[k] = _lin(d[k], tf, tt)
    i = np.clip(np.searchsorted(tf, tt, side="right") - 1, 0, T - 1)
    j = np.clip(i + 1, 0, T - 1)
    for k in ("valid", "cmask"):
        out[k] = np.minimum(np.asarray(d[k])[i], np.asarray(d[k])[j]).astype(np.float64)
    return out, tt, tf


def pick_device(device=None):
    """auto / None, or a CUDA device without CUDA -> cuda if available, else cpu."""
    if device in (None, "", "auto") or (str(device).startswith("cuda") and not torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _pass_through(exp, od, name, model, reason):
    new = dict(exp)
    notes = json.loads(str(exp.get("notes", "{}"))) if "notes" in exp else {}
    notes["world_motion"] = {"skipped": reason, "model": str(Path(model).resolve())}
    new["notes"] = json.dumps(notes)
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"{name}.npz", **new)
    return od / f"{name}.npz"


def refine_export(npz, work, model, out_dir=None, device="auto", smpl=None, cache=None, pose_src="model", smooth_rot=0.0,
                  smooth_pos=0.0, sigma_c=0.01, anchor=None, floor=None, floor_channels="feet", floor_upright=False,
                  nopen=0, static_only=True, static=None, fps_mode="auto"):
    """cache: a dict kept across calls (the loaded model and kinematics). pose_src: the body pose from the model, or the
    export's (only the pelvis trajectory from the model). smooth_rot / smooth_pos: Gaussian sigma (frames) on the output
    rotations / pelvis. static_only: pass through any export whose camera is not static: the export notes'
    static_camera, or static (bool) when given (e.g. another camera run's check of the same video). fps_mode: auto
    (30 fps as is; above 30 and from AUTO_MIN_FPS up resampled to 30 and back; below it passed through) or resample
    (resampled from MIN_FPS up)."""
    import time

    from fab import paths

    npz = Path(npz)
    name = npz.stem
    work = Path(work) if work else npz.parent / "work"
    exp = dict(np.load(npz, allow_pickle=True))
    od = Path(out_dir) if out_dir else npz.parent
    if static_only:  # FAB-M is for static cameras (moving cameras: EMDB dev W +0.75 %, EMDB-2 test +17 %)
        st = (json.loads(str(exp["notes"])) if "notes" in exp else {}).get("static_camera") if static is None else static
        if st is not True:
            return _pass_through(exp, od, name, model, f"camera not static (static_camera {st})")
    fps = float(exp.get("fps", 30.0))
    act = fps_action(fps, fps_mode)
    if act not in ("run", "resample"):  # e.g. too sparse for a 30 fps motion model: the export passes through, marked
        return _pass_through(exp, od, name, model, act)
    device = pick_device(device)
    cache = {} if cache is None else cache
    if "net" not in cache:
        cache["kin"] = SMPLKin(smpl or (paths.smpl_dir() / "SMPL_NEUTRAL.pkl"))
        cache["net"], ck = load_model(model, device)
        cache["mode"] = "export" if (ck.get("noise") or {}).get("smooth_input") else "raw"  # what the model was trained on
        an = (ck.get("args") or {}).get("anchor") or ""
        cache["anchor"] = tuple(float(x) for x in an.split(",")) if an else None  # the do-no-harm anchor it was trained with
    kin, net = cache["kin"].cpu(), cache["net"]
    d = export_inputs(exp, work, name, cache["mode"], kin)
    k = depth_ratio(kin.cpu(), d, exp) if cache["mode"] == "raw" else 1.0
    kin = kin.to(device)
    # the raw pelvis placed at the export's depth: pelvis_c x k (FK pelvis = J0(betas) + trans_c)
    Jr, _ = kin.rest(torch.as_tensor(d["betas"], dtype=torch.float32, device=device))
    J0 = Jr[:, 0].cpu().numpy().astype(np.float64)
    d["trans_c"] = (d["trans_c"] + J0) * k - J0
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
    sync()
    t0 = time.perf_counter()
    anchor = cache["anchor"] if anchor is None else (None if anchor == "off" else anchor)
    rs = act == "resample"
    if rs:  # FAB-M at its training rate: the inputs resampled to 30 fps, the outputs back to the clip's frames
        d, t30, tsrc = resample_inputs(d, fps, TRAIN_FPS)
    o = run_clip(net, kin, d, fps=TRAIN_FPS if rs else fps, device=device, sigma_c=sigma_c, anchor=anchor, floor=floor,
                 floor_channels=floor_channels, floor_upright=floor_upright, nopen=nopen)
    if rs:
        o["R_root"], o["R_loc"] = _rot(o["R_root"], t30, tsrc), _rot(o["R_loc"], t30, tsrc)
        o["pelvis"], o["c"] = _lin(o["pelvis"], t30, tsrc), _lin(o["c"], t30, tsrc)
    sync()
    secs = time.perf_counter() - t0
    betas = exp["betas"].astype(np.float64)
    if pose_src == "export":
        R = aa_to_mat(torch.as_tensor(exp["pose"].astype(np.float64).reshape(-1, 24, 3))).numpy()
    else:
        R = np.concatenate([o["R_root"][:, None], o["R_loc"]], 1)
    if smooth_rot > 0:
        R = smooth_rotations(R, smooth_rot)
    tt = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
    with torch.no_grad():
        P, _, _ = kin(tt(R), tt(np.broadcast_to(betas, (len(R), 10)).copy()), None)
    P = P.cpu().numpy().astype(np.float64)
    if smooth_pos > 0:
        from scipy.ndimage import gaussian_filter1d

        o["pelvis"] = gaussian_filter1d(o["pelvis"], smooth_pos, axis=0, mode="nearest")
    trans = o["pelvis"] - P[:, 0]
    joints = P + trans[:, None]
    pose = mat_to_aa(torch.as_tensor(R)).numpy().reshape(len(R), 72)
    labels = {g: (o["c"][:, i] > 0.5).astype(int).tolist() for i, g in enumerate(("l_heel", "l_toe", "r_heel", "r_toe",
                                                                                   "l_hand", "r_hand", "l_knee", "r_knee"))}
    if out_dir is None:  # in place: the kinematic export is kept in work/kin/ (never written into the work dir otherwise)
        kin_dir = work / "kin"
        kin_dir.mkdir(parents=True, exist_ok=True)
        if not (kin_dir / f"{name}.npz").exists():
            np.savez(kin_dir / f"{name}.npz", **exp)
    new = dict(exp)
    new.update(pose=pose.astype(np.float32), trans=trans.astype(np.float32), joints=joints.astype(np.float32))
    notes = json.loads(str(exp.get("notes", "{}"))) if "notes" in exp else {}
    notes["world_motion"] = {"model": str(Path(model).resolve()), "md5": hashlib.md5(open(model, "rb").read()).hexdigest(),
                             "depth_ratio": k, "contacts": "world_motion_contacts", "seconds": secs, "frames": len(R),
                             "pose_src": pose_src, "smooth_rot": smooth_rot, "smooth_pos": smooth_pos, "sigma_c": sigma_c,
                             "anchor": anchor, "floor": floor, "floor_h": o.get("floor"),
                             "floor_channels": floor_channels, "floor_upright": floor_upright, "nopen": nopen,
                             "nopen_frames": o.get("nopen_frames", 0),
                             "input": cache["mode"], "fps": fps, "resampled_to": TRAIN_FPS if rs else None, "fps_mode": fps_mode}
    new["notes"] = json.dumps(notes)
    new["world_motion_contacts"] = json.dumps(labels)
    new["world_motion_contact_prob"] = o["c"].astype(np.float32)
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"{name}.npz", **new)
    return od / f"{name}.npz"


def main():
    from fab import paths

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", help="fab-world <out>/<name>.npz")
    p.add_argument("--work", default="", help="its work dir (default <out>/work)")
    p.add_argument("--model", default=str(paths.shared("world_motion.pt")))
    p.add_argument("--out", default="", help="output dir (default: the export's, in place; the original goes to work/kin/)")
    p.add_argument("--device", default="auto", help="auto: cuda if available, else cpu")
    p.add_argument("--any_camera", action="store_true", help="run on moving cameras too (default: static cameras only)")
    p.add_argument("--fps_mode", default="auto", choices=("auto", "resample"),
                   help=f"auto: 30 fps as is, above 30 and from {AUTO_MIN_FPS:g} fps up resampled to 30 and back, below passed "
                        f"through; resample: resampled from {MIN_FPS:g} fps up")
    p.add_argument("--pose_src", default="export", choices=("model", "export"),
                   help="the body pose from the export (default: only the pelvis trajectory from the model) or the model")
    p.add_argument("--anchor", default="0.02,0.02", help="rel,perp: the export pelvis as an extra observation; off")
    p.add_argument("--floor", default="0.05,0.5", help="sigma,p_min: one flat floor under the planted feet; off")
    p.add_argument("--sigma_c", type=float, default=0.01, help="the fusion's contact sigma (m / frame)")
    a = p.parse_args()
    anchor = "off" if a.anchor == "off" else tuple(float(x) for x in a.anchor.split(","))
    floor = None if a.floor == "off" else tuple(float(x) for x in a.floor.split(","))
    print("[world_motion]", refine_export(a.export, a.work, a.model, a.out or None, a.device, pose_src=a.pose_src,
                                          sigma_c=a.sigma_c, anchor=anchor, floor=floor, static_only=not a.any_camera,
                                          fps_mode=a.fps_mode))


if __name__ == "__main__":
    main()
