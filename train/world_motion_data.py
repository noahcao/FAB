"""Clean world motions for FAB-M training (fab/world_motion.py): every source -> SMPL neutral, 30 fps, z up, the floor at
z = 0, with static-contact labels of the 8 contact points, one npz per source:

    pose (N, 72) axis-angle, trans (N, 3), contact (N, 8) uint8, betas (C, 10), start / length (C,), name (C,), fps

SMPL-X sources keep the 21 body-joint rotations (SMPL's hand joints 22 / 23 at rest); the SMPL betas are fitted to the
SMPL-X rest skeleton's bone vectors (linear least squares) and the pelvis follows the SMPL-X pelvis frame by frame.
The up axis is found from the data (pelvis -> head direction). Contact: the point within --h_contact of the floor and
slower than --v_contact (m/s).

    python -m train.world_motion_data bedlam --src <neutral_ground_truth_motioninfo dir> --out bedlam.npz --smpl SMPL_NEUTRAL.pkl --smplx SMPLX_NEUTRAL.npz
    python -m train.world_motion_data embody3d --src <unzipped *_smplx dir> --out embody3d_<cat>.npz ...
    python -m train.world_motion_data aist --src <motions dir> --exclude <test sequence list> --out aist.npz ...
    python -m train.world_motion_data amass --src <AMASS SMPL+H G dir> --out amass.npz ...
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

from fab.world_motion import SMPLKin, aa_to_mat, mat_to_aa

PARENTS22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


class SMPLXRest:
    def __init__(self, path, n_betas):
        z = np.load(path, allow_pickle=True)
        Jr = np.asarray(z["J_regressor"], np.float64)
        sd = np.asarray(z["shapedirs"], np.float64)[:, :, :n_betas]
        self.J = Jr @ np.asarray(z["v_template"], np.float64)
        self.Jd = np.einsum("jv,vcl->jcl", Jr, sd)
        self.n = n_betas

    def __call__(self, betas):
        return self.J + np.einsum("jcl,l->jc", self.Jd, betas[: self.n])


class BetaFit:
    """SMPL betas whose rest bone vectors (joints 1..21) best match a given 22-joint rest skeleton."""

    def __init__(self, kin, lam=1e-3):
        Jt, Jd = kin.Jt.double().numpy(), kin.Jd.double().numpy()
        p = PARENTS22
        self.Bt = np.stack([Jt[j] - Jt[p[j]] for j in range(1, 22)])
        self.Bd = np.stack([Jd[j] - Jd[p[j]] for j in range(1, 22)])
        M = self.Bd.reshape(-1, 10)
        self.solve = np.linalg.solve(M.T @ M + lam * np.eye(10), M.T)
        self.J0 = (Jt[0], Jd[0])
        self.Jt, self.Jd = Jt, Jd
        self.wb = np.array([10.0 if j in (1, 2, 4, 5, 7, 8, 10, 11) else 1.0 for j in range(1, 22)])
        Mw = (self.wb[:, None, None] * self.Bd).reshape(-1, 10)
        self.solve_w = np.linalg.solve(Mw.T @ Mw + lam * np.eye(10), Mw.T)

    def rest_joints(self, beta):
        return self.Jt + np.einsum("jcl,l->jc", self.Jd, beta)

    def __call__(self, J22):
        p = PARENTS22
        b = np.stack([J22[j] - J22[p[j]] for j in range(1, 22)]) - self.Bt
        beta = self.solve @ b.reshape(-1)
        return beta, self.J0[0] + self.J0[1] @ beta


def up_rotation(kin, pose, betas, trans, exact=False):
    """The rotation taking the data's up axis (the mean pelvis -> head direction, snapped to an axis) to +z."""
    R = aa_to_mat(torch.as_tensor(pose[::10].reshape(-1, 24, 3), dtype=torch.float32))
    P, _, _ = kin(R, torch.as_tensor(np.broadcast_to(betas, (len(R), 10)).copy(), dtype=torch.float32))
    u = (P[:, 15] - P[:, 0]).mean(0).numpy()
    if exact:  # the mean pelvis -> head direction itself (upright clips in a tilted world)
        return rot_to_z(u / np.linalg.norm(u))
    ax = int(np.argmax(np.abs(u)))
    up = np.zeros(3)
    up[ax] = np.sign(u[ax])
    z = np.array([0.0, 0, 1])
    v, c = np.cross(up, z), up @ z
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def rot_to_z(u):
    """The smallest rotation taking the unit vector u to +z."""
    z = np.array([0.0, 0, 1])
    v, c = np.cross(u, z), float(u @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1, -1])
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def finish(kin, pose, betas, pelvis, fps_in, args, A=None):
    """pose (T, 72) SMPL axis-angle, betas (10,), pelvis (T, 3) world positions -> resampled to 30 fps, z up, floor 0,
    with contact labels; None when too short."""
    step = fps_in / 30.0
    if abs(step - round(step)) < 1e-6 and step >= 1:
        idx = np.arange(0, len(pose), int(round(step)))
    else:
        idx = np.round(np.arange(0, len(pose) - 1, step)).astype(int)
    pose, pelvis = pose[idx], pelvis[idx]
    if len(pose) < args.min_len:
        return None
    J0 = kin.rest(torch.as_tensor(betas, dtype=torch.float32))[0][0].double().numpy()
    if A is None:
        A = up_rotation(kin, pose, betas, pelvis - J0, exact=args.up == "mean")
    R = aa_to_mat(torch.as_tensor(pose.reshape(-1, 24, 3), dtype=torch.float64)).numpy()
    R[:, 0] = A @ R[:, 0]
    pelvis = pelvis @ A.T
    tt = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        P, C, _ = kin(tt(R), tt(np.broadcast_to(betas, (len(R), 10)).copy()), tt(pelvis - J0))
    C = C.double().numpy()
    if args.level:  # a tilted world (3DPW): the plane through the slow lowest foot points -> horizontal
        lo = C[np.arange(len(C)), C[:, :4, 2].argmin(1)]
        sp = np.r_[np.inf, np.linalg.norm(np.diff(lo, axis=0), axis=1) * 30]
        X = lo[sp < 0.3]
        if len(X) >= 20 and np.ptp(X[:, :2], 0).max() > 0.5:
            w = np.ones(len(X))
            for _ in range(5):  # IRLS plane z = a x + b y + c
                M = np.c_[X[:, :2], np.ones(len(X))]
                coef = np.linalg.lstsq(M * w[:, None], X[:, 2] * w, rcond=None)[0]
                r = np.abs(M @ coef - X[:, 2])
                w = 1 / np.maximum(r, 0.01)
            nrm = np.array([-coef[0], -coef[1], 1.0])
            nrm /= np.linalg.norm(nrm)
            if np.degrees(np.arccos(nrm[2])) < args.level_max:
                Lv = rot_to_z(nrm)
                cen = X.mean(0)
                R[:, 0] = Lv @ R[:, 0]
                pelvis = (pelvis - cen) @ Lv.T + cen
                C = (C - cen) @ Lv.T + cen
    floor = np.percentile(C[:, :4, 2].min(1), 2)
    pelvis[:, 2] -= floor
    C[..., 2] -= floor
    v = np.zeros(C.shape[:2])
    v[1:-1] = np.linalg.norm(C[2:] - C[:-2], axis=-1) / 2 * 30
    v[0], v[-1] = v[1], v[-2]
    h = C[..., 2]
    thr = np.array([args.h_contact] * 4 + [args.h_contact + 0.02] * 4)
    contact = ((h < thr) & (v < args.v_contact)).astype(np.uint8)
    pose = mat_to_aa(torch.as_tensor(R)).numpy().reshape(-1, 72)
    return pose.astype(np.float32), (pelvis - J0).astype(np.float32), contact, A


def min_rotation(a, b):
    """The smallest rotation taking direction a to direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def kabsch(A, B):
    """The rotation R minimising sum |R a_i - b_i|^2 over rows of A, B."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


CHILDREN22 = {j: [c for c in range(22) if PARENTS22[c] == j] for j in range(22)}


def rest_alignment(Js, Jx):
    """Per joint, Q_j with Q_j (SMPL child bone) || (SMPL-X child bone) in the rest pose (Kabsch over unit child bones for
    the pelvis / spine3, identity for leaves)."""
    Q = np.tile(np.eye(3), (22, 1, 1))
    for j in range(22):
        ch = CHILDREN22[j]
        if len(ch) == 1:
            Q[j] = min_rotation(Js[ch[0]] - Js[j], Jx[ch[0]] - Jx[j])
        elif len(ch) > 1:
            A = np.stack([(Js[c] - Js[j]) / np.linalg.norm(Js[c] - Js[j]) for c in ch])
            B = np.stack([(Jx[c] - Jx[j]) / np.linalg.norm(Jx[c] - Jx[j]) for c in ch])
            Q[j] = kabsch(A, B)
    return Q


def smplx_to_smpl(rest_x, fit, poses_x, betas_x, trans_x):
    """SMPL-X (global 3 + body 63, ...) -> SMPL pose (T, 72), betas (10,), pelvis (T, 3): SMPL bone directions follow
    the SMPL-X bones (global rotations G_s = G_x Q, Q the rest-pose bone alignment), SMPL betas fitted to the bone
    lengths, the pelvis kept."""
    Jx = rest_x(betas_x)[:22]
    beta = np.zeros(10)
    Lx = np.array([np.linalg.norm(Jx[j] - Jx[PARENTS22[j]]) for j in range(1, 22)])
    for _ in range(3):  # bone lengths (legs weighted: they carry the contacts) along the current SMPL bone directions
        Bs = fit.Bt + np.einsum("bcl,l->bc", fit.Bd, beta)
        tgt = Bs / np.linalg.norm(Bs, axis=1, keepdims=True) * Lx[:, None]
        beta = fit.solve_w @ (fit.wb[:, None] * (tgt - fit.Bt)).reshape(-1)
    Js = fit.rest_joints(beta)
    Q = rest_alignment(Js, Jx)
    from scipy.spatial.transform import Rotation

    T = len(poses_x)
    Rl = Rotation.from_rotvec(poses_x[:, :66].reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    G = np.zeros_like(Rl)
    for j in range(22):
        G[:, j] = Rl[:, j] if j == 0 else G[:, PARENTS22[j]] @ Rl[:, j]
    Gs = G @ Q
    Rs = np.tile(np.eye(3), (T, 24, 1, 1))
    Rs[:, 0] = Gs[:, 0]
    for j in range(1, 22):
        Rs[:, j] = np.swapaxes(Gs[:, PARENTS22[j]], -1, -2) @ Gs[:, j]
    pose = Rotation.from_matrix(Rs.reshape(-1, 3, 3)).as_rotvec().reshape(T, 72)
    return pose, beta, trans_x + Jx[0]


def save(out, clips, fps=30):
    pose = np.concatenate([c[1] for c in clips])
    np.savez(out, pose=pose, trans=np.concatenate([c[2] for c in clips]), contact=np.concatenate([c[3] for c in clips]),
             betas=np.stack([c[4] for c in clips]).astype(np.float32), length=np.array([len(c[1]) for c in clips]),
             start=np.cumsum([0] + [len(c[1]) for c in clips])[:-1], name=np.array([c[0] for c in clips]), fps=fps)
    n = len(pose)
    print(f"[world_motion_data] {out}: {len(clips)} clips, {n} frames ({n / 30 / 3600:.2f} h), contact rate "
          f"{np.concatenate([c[3] for c in clips]).mean(0).round(3).tolist()}", flush=True)


def src_bedlam(args, kin, fit):
    rest_x = SMPLXRest(args.smplx, 11)
    for f in sorted(glob.glob(os.path.join(args.src, "*", "*", "motion_seq.npz")))[: args.limit or None]:
        z = np.load(f, allow_pickle=True)
        pose, beta, pel = smplx_to_smpl(rest_x, fit, z["poses"].astype(np.float64), z["betas"].astype(np.float64),
                                        z["trans"].astype(np.float64))
        mi = z["motion_info"]  # the AMASS source + range first: train / val split by motion, not by subject
        yield f"{mi[2]}:{mi[4]}-{mi[5]}|{Path(f).parent.relative_to(args.src)}", pose, beta, pel, float(z["mocap_frame_rate"])


def retarget_rest(fit, Jsrc22, poses66, pel):
    """Any 22-joint SMPL-family source with rest joints Jsrc22 -> SMPL neutral (smplx_to_smpl's bone alignment)."""
    from scipy.spatial.transform import Rotation

    beta = np.zeros(10)
    Lx = np.array([np.linalg.norm(Jsrc22[j] - Jsrc22[PARENTS22[j]]) for j in range(1, 22)])
    for _ in range(3):
        Bs = fit.Bt + np.einsum("bcl,l->bc", fit.Bd, beta)
        tgt = Bs / np.linalg.norm(Bs, axis=1, keepdims=True) * Lx[:, None]
        beta = fit.solve_w @ (fit.wb[:, None] * (tgt - fit.Bt)).reshape(-1)
    Q = rest_alignment(fit.rest_joints(beta), Jsrc22)
    T = len(poses66)
    Rl = Rotation.from_rotvec(poses66.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    G = np.zeros_like(Rl)
    for j in range(22):
        G[:, j] = Rl[:, j] if j == 0 else G[:, PARENTS22[j]] @ Rl[:, j]
    Gs = G @ Q
    Rs = np.tile(np.eye(3), (T, 24, 1, 1))
    Rs[:, 0] = Gs[:, 0]
    for j in range(1, 22):
        Rs[:, j] = np.swapaxes(Gs[:, PARENTS22[j]], -1, -2) @ Gs[:, j]
    return Rotation.from_matrix(Rs.reshape(-1, 3, 3)).as_rotvec().reshape(T, 72), beta, pel


def runs(good, min_len):
    """[a, b) runs of True."""
    out, a = [], None
    for i, g in enumerate(np.r_[good, False]):
        if g and a is None:
            a = i
        elif not g and a is not None:
            if i - a >= min_len:
                out.append((a, i))
            a = None
    return out


def src_rich(args, kin, fit):
    """RICH <split>_body.zip (<split>_body/<seq>/<frame>/<subject>.pkl, gendered SMPL-X) of the sequences in GVHMR's
    resource/<split>.txt; the scan world -> z up by w2az_sahmr.json; consecutive fitted frames only."""
    import json
    import pickle
    import re
    import zipfile
    from collections import defaultdict

    from scipy.spatial.transform import Rotation

    res = Path(args.rich_res)
    info = {}
    for line in (res / f"{args.split}.txt").read_text().splitlines()[1:]:
        v = line.strip().split()
        if v:
            info[v[0]] = {"scan": f"{v[1]}_{v[2]}", "subject": int(v[3]), "gender": v[5]}
    w2az = json.loads((res / "w2az_sahmr.json").read_text())
    rest = {g: SMPLXRest(Path(args.smplx).parent / f"SMPLX_{g.upper()}.npz", 10) for g in ("male", "female")}
    z = zipfile.ZipFile(args.src)
    pk = defaultdict(dict)
    for n in z.namelist():
        m = re.match(r"(?:[A-Za-z_]+/)?([^/]+)/(\d+)/(\d+)\.pkl$", n)
        if m and m.group(1) in info and int(m.group(3)) == info[m.group(1)]["subject"]:
            pk[m.group(1)][int(m.group(2))] = n
    for seq in sorted(pk)[: args.limit or None]:
        inf = info[seq]
        fr = np.array(sorted(pk[seq]))
        rows = [pickle.load(z.open(pk[seq][f]), encoding="latin1") for f in fr]
        go = np.stack([np.asarray(r["global_orient"]).reshape(3) for r in rows])
        bp = np.stack([np.asarray(r["body_pose"]).reshape(-1)[:63] for r in rows])
        tr = np.stack([np.asarray(r["transl"]).reshape(3) for r in rows])
        be = np.median(np.stack([np.asarray(r["betas"]).reshape(-1)[:10] for r in rows]), 0)
        key = inf["scan"] if inf["scan"] in w2az else next((k for k in w2az if k.startswith(inf["scan"].split("_")[0])), None)
        Tz = np.asarray(w2az[key], np.float64).reshape(4, 4)
        Jx = rest[inf["gender"]](be)[:22]
        pel = (tr + Jx[0]) @ Tz[:3, :3].T + Tz[:3, 3]
        Rg = Tz[:3, :3] @ Rotation.from_rotvec(go).as_matrix()
        poses = np.concatenate([Rotation.from_matrix(Rg).as_rotvec(), bp], 1)
        for a, b in runs(np.r_[True, np.diff(fr) == 1], args.min_len):
            pose, beta, p = retarget_rest(fit, Jx, poses[a:b], pel[a:b])
            yield f"{seq}:{fr[a]}-{fr[b - 1]}", pose, beta, p, 30.0


def src_3dpw(args, kin, fit):
    """3DPW sequenceFiles (train / validation pkl): every person's SMPL (gendered) on campose-valid frames, 30 fps."""
    import pickle

    from bench.body_models import load_smpl_pkl

    gm = {}
    for g in ("m", "f"):
        mm = load_smpl_pkl(Path(args.smpl).parent / f"SMPL_{'MALE' if g == 'm' else 'FEMALE'}.pkl")
        gm[g] = (mm["J_regressor"].astype(np.float64) @ mm["v_template"].astype(np.float64),
                 np.einsum("jv,vcl->jcl", mm["J_regressor"].astype(np.float64), mm["shapedirs"][:, :, :10].astype(np.float64)))
    for f in sorted(glob.glob(os.path.join(args.src, "*.pkl")))[: args.limit or None]:
        z = pickle.load(open(f, "rb"), encoding="latin1")
        for p in range(len(z["poses"])):
            g = str(z["genders"][p])[:1]
            Jt, Jd = gm[g if g in gm else "m"]
            Jr = Jt + np.einsum("jcl,l->jc", Jd, np.asarray(z["betas"][p][:10], np.float64))
            ok = np.asarray(z["campose_valid"][p]).astype(bool)
            poses, tr = np.asarray(z["poses"][p], np.float64), np.asarray(z["trans"][p], np.float64)
            for a, b in runs(ok, args.min_len):
                pose, beta, pel = retarget_rest(fit, Jr[:22], poses[a:b, :66], tr[a:b] + Jr[0])
                yield f"{Path(f).stem}:{p}:{a}-{b}", pose, beta, pel, 30.0


def src_moyo(args, kin, fit):
    """MOYO mosh_smpl/<split>/*_stageii_smpl.pkl (SMPL: global_orient, body_pose (T, 69), transl, betas (T, 300)), 60 fps
    (the matching mosh stageii files record mocap_frame_rate 60); the first 10 betas."""
    import pickle

    for f in sorted(glob.glob(os.path.join(args.src, "**", "mosh_smpl", args.split, "*.pkl"), recursive=True))[: args.limit or None]:
        z = pickle.load(open(f, "rb"), encoding="latin1")
        pose = np.concatenate([np.asarray(z["global_orient"], np.float64), np.asarray(z["body_pose"], np.float64)], 1)
        beta = np.median(np.asarray(z["betas"], np.float64)[:, :10], 0)
        J0 = kin.rest(torch.as_tensor(beta, dtype=torch.float32))[0][0].double().numpy()
        yield Path(f).stem, pose, beta, np.asarray(z["transl"], np.float64) + J0, 60.0


def longest_run(good):
    best, cur = (0, 0), 0
    for i, g in enumerate(np.r_[good, False]):
        if g:
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (i - cur, i)
            cur = 0
    return best


def src_embody3d(args, kin, fit):
    rest_x = SMPLXRest(args.smplx, 300)
    for d in sorted(glob.glob(os.path.join(args.src, "**", "smplx_mesh_body_pose"), recursive=True))[: args.limit or None]:
        root = Path(d).parent
        ld = lambda k: np.load(next((root / k).glob("*.npy"))).astype(np.float64)  # noqa: E731
        body, go, tr, be = ld("smplx_mesh_body_pose"), ld("smplx_mesh_global_orient"), ld("smplx_mesh_transl"), ld("smplx_mesh_betas")
        # the "missing" member is 1 on tracked frames and 0 on untracked ones (zeroed parameters)
        ok = ld("missing").reshape(-1)[: len(body)] > 0.5 if (root / "missing").exists() else np.ones(len(body), bool)
        good = ok & np.isfinite(body).all(1) & np.isfinite(tr).all(1) & (np.abs(tr).sum(1) > 0)
        if args.skip_missing and not good.all():  # the whole clip or nothing (no tracking gaps)
            continue
        a, b = longest_run(good)
        if b - a < args.min_len:
            continue
        beta_x = np.median(be[a:b], 0)
        pose, beta, pel = smplx_to_smpl(rest_x, fit, np.concatenate([go, body], 1)[a:b], beta_x, tr[a:b])
        yield str(root.relative_to(args.src)), pose, beta, pel, args.fps


def src_aist(args, kin, fit):
    import pickle

    ex = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    J0 = kin.rest(torch.zeros(10))[0][0].double().numpy()
    for f in sorted(glob.glob(os.path.join(args.src, "*.pkl")))[: args.limit or None]:
        if Path(f).stem in ex:
            continue
        z = pickle.load(open(f, "rb"))
        pose = np.asarray(z["smpl_poses"], np.float64)
        trans = np.asarray(z["smpl_trans"], np.float64) / float(np.asarray(z["smpl_scaling"]).reshape(-1)[0])
        yield Path(f).stem, pose, np.zeros(10), trans + J0, 60.0


def src_amass(args, kin, fit):
    """AMASS SMPL+H G npz (poses (T, 156), trans, betas (16,), mocap_framerate): SMPL+H shares SMPL's body and shape
    space, so its first 66 pose values and 10 betas carry over."""
    for f in sorted(glob.glob(os.path.join(args.src, "**", "*.npz"), recursive=True))[: args.limit or None]:
        z = np.load(f)
        if "poses" not in z.files:
            continue
        pose = np.zeros((len(z["poses"]), 72))
        pose[:, :66] = z["poses"][:, :66]
        beta = np.asarray(z["betas"][:10], np.float64)
        J0 = kin.rest(torch.as_tensor(beta, dtype=torch.float32))[0][0].double().numpy()
        yield str(Path(f).relative_to(args.src)), pose, beta, np.asarray(z["trans"], np.float64) + J0, float(z["mocap_framerate"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["bedlam", "embody3d", "aist", "amass", "rich", "3dpw", "moyo"])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smpl", required=True)
    p.add_argument("--smplx", default="")
    p.add_argument("--exclude", default="", help="aist: a file of sequence names to leave out (the test split)")
    p.add_argument("--rich_res", default="", help="rich: GVHMR's hmr4d/dataset/rich/resource dir ({split}.txt, w2az_sahmr.json)")
    p.add_argument("--split", default="train", help="rich / moyo: the split (never test)")
    p.add_argument("--up", default="axis", choices=("axis", "mean"),
                   help="per-clip up (--fixed_up 0): the nearest axis to the mean pelvis -> head direction, or that direction")
    p.add_argument("--level", type=int, default=0, help="1: level each clip by its footprint plane (tilted worlds: 3DPW)")
    p.add_argument("--level_max", type=float, default=40.0, help="--level: larger footprint-plane tilts are left alone (deg)")
    p.add_argument("--fps", type=float, default=30.0, help="embody3d: its frame rate")
    p.add_argument("--skip_missing", type=int, default=0, help="embody3d: 1 = drop clips with any untracked frame")
    p.add_argument("--min_len", type=int, default=60)
    p.add_argument("--h_contact", type=float, default=0.05)
    p.add_argument("--v_contact", type=float, default=0.2)
    p.add_argument("--fixed_up", type=int, default=1, help="1: the up rotation of the first clip for the whole source")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    kin = SMPLKin(a.smpl)
    fit = BetaFit(kin)
    assert not (a.source == "rich" and a.split == "test"), "RICH test is evaluation only"
    gen = {"bedlam": src_bedlam, "embody3d": src_embody3d, "aist": src_aist, "amass": src_amass, "rich": src_rich,
           "3dpw": src_3dpw, "moyo": src_moyo}[a.source](a, kin, fit)
    head, clips, A = [], [], None
    for item in gen:
        if a.fixed_up and A is None:  # the source's up axis: the majority over its first 20 clips
            head.append(item)
            if len(head) < 20:
                continue
            Rs = [up_rotation(kin, x[1], x[2], x[3]) for x in head]
            keys = [tuple(np.round(R[2]).astype(int)) for R in Rs]
            A = Rs[max(range(len(Rs)), key=lambda i: keys.count(keys[i]))]
            print("[world_motion_data] up rotation", A.round(3).tolist(), "votes", keys.count(tuple(np.round(A[2]).astype(int))), "/ 20")
            todo, head = head, []
        else:
            todo = [item]
        for name, pose, beta, pel, fps in todo:
            r = finish(kin, pose, beta, pel, fps, a, A)
            if r is not None:
                clips.append((name, r[0], r[1], r[2], beta))
    for name, pose, beta, pel, fps in head:  # fewer than 20 clips
        r = finish(kin, pose, beta, pel, fps, a, None)
        if r is not None:
            clips.append((name, r[0], r[1], r[2], beta))
    save(a.out, clips)


if __name__ == "__main__":
    main()
