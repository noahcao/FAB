"""fab-multiview --trackers (fab/multiview_trackers.py) on synthetic data: tracker files and sample cleaning, pose
interpolation, the banded smoother and its static offset, the per-view points against the fusion's own, and whole runs
of fab-multiview on a synthetic two-camera take with a stand-in predictor and SMPL fit (the fit's targets compared):
without --trackers the run is bitwise the one of the multiview-fusion code this branch starts from (git merge-base);
with a hand tracker the hand's targets follow it, and lost / frozen / drifting samples give way to the views."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fab import multiview as mv  # noqa: E402
from fab import multiview_trackers as mvt  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def look_at(c, target=(0.0, 0.0, 1.0)):
    c, target = np.asarray(c, float), np.asarray(target, float)
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0, 1])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, np.cross(z, x), z], 1), c
    return T


def wxyz(R):
    return Rotation.from_matrix(R).as_quat()[..., [3, 0, 1, 2]]


class Input(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(mvt.parse("left_hand=a.npz, right_wrist=b.csv"), [("left_wrist", Path("a.npz")), ("right_wrist", Path("b.csv"))])
        for bad in ("elbow=a.npz", "left_wrist", "left_wrist=a.npz,left_hand=b.npz"):
            with self.assertRaises(ValueError):
                mvt.parse(bad)

    def test_load_npz_and_csv(self):
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 1, 20))[::-1]  # unsorted on purpose
        xyz, q = rng.normal(size=(20, 3)), wxyz(Rotation.random(20, random_state=1).as_matrix())
        valid = rng.random(20) > 0.2
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "a.npz", t=t, xyz=xyz, quat=q, valid=valid)
            a = mvt.load(Path(d) / "a.npz")
            lines = ["t,x,y,z,qw,qx,qy,qz,valid"] + [",".join(repr(float(v)) for v in [t[i], *xyz[i], *q[i], int(valid[i])]) for i in range(20)]
            (Path(d) / "b.csv").write_text("\n".join(lines) + "\n")
            b = mvt.load(Path(d) / "b.csv")
            (Path(d) / "c.csv").write_text("t,x,y,z\n" + "\n".join(",".join(repr(float(v)) for v in [t[i], *xyz[i]]) for i in range(20)))
            c = mvt.load(Path(d) / "c.csv")
        o = np.argsort(t)
        for r in (a, b):
            self.assertTrue(np.array_equal(r["t"], t[o]) and np.array_equal(r["xyz"], xyz[o]) and np.array_equal(r["quat"], q[o]))
            self.assertTrue(np.array_equal(r["valid"], valid[o]))
        self.assertIsNone(c["quat"])
        self.assertTrue(c["valid"].all() and np.array_equal(c["xyz"], xyz[o]))

    def test_bad_samples_dropped(self):
        rng = np.random.default_rng(1)
        n = 12
        tr = {"t": np.arange(n) / 100.0, "xyz": rng.normal(size=(n, 3)), "quat": wxyz(Rotation.random(n, random_state=2).as_matrix()),
              "valid": np.ones(n, bool)}
        tr["valid"][1] = False  # flagged
        tr["xyz"][3] = np.nan  # not finite
        tr["quat"][5] = 0.0  # an unsolved (all-zero) pose
        tr["xyz"][8], tr["quat"][8] = tr["xyz"][7], tr["quat"][7]  # frozen: the pose of the sample before, twice
        tr["xyz"][9], tr["quat"][9] = tr["xyz"][7], tr["quat"][7]
        ok = mvt.good_samples(tr)
        self.assertEqual(np.flatnonzero(~ok).tolist(), [1, 3, 5, 8, 9])


class Interpolation(unittest.TestCase):
    def test_pose_at_frames(self):
        w = np.array([0.3, -0.2, 1.1])  # rad/s
        t = np.arange(0, 2, 1 / 150.0)
        Rt = Rotation.from_rotvec(t[:, None] * w).as_matrix()
        tr = {"t": t, "xyz": np.stack([np.sin(t), 0.5 * t, np.cos(2 * t)], 1), "quat": wxyz(Rt), "valid": np.ones(len(t), bool)}
        tr["valid"][(t > 1.0) & (t < 1.2)] = False  # a 0.2 s gap: more than GAP
        times = np.arange(0.01, 1.9, 1 / 30.0)
        A = mv.up_rotation([0.0, -1.0, 0.0])
        p, R, have = mvt.at_frames(tr, mvt.good_samples(tr), times, A)
        gap = (times > 1.0) & (times < 1.2)
        self.assertTrue(have[~gap].all() and not have[gap].any())
        truth = np.stack([np.sin(times), 0.5 * times, np.cos(2 * times)], 1) @ A.T
        self.assertLess(np.abs(p[have] - truth[have]).max(), 1e-4)  # linear between 150 Hz samples
        Rtrue = A @ Rotation.from_rotvec(times[:, None] * w).as_matrix()
        self.assertLess(np.abs(R[have] - Rtrue[have]).max(), 1e-9)  # slerp of a constant angular velocity is exact
        tr["quat"] = None
        p2, R2, have2 = mvt.at_frames(tr, mvt.good_samples(tr), times, A)
        self.assertIsNone(R2)
        self.assertTrue(np.array_equal(p2, p) and np.array_equal(have2, have))


def trajectory(T, fps):
    t = np.arange(T) / fps
    return np.stack([0.4 * np.sin(1.3 * t), 0.3 * np.cos(0.9 * t), 1.0 + 0.2 * np.sin(2.1 * t)], 1), t


class Smoother(unittest.TestCase):
    def test_anisotropic_views_and_prior(self):
        """Two views, each noisy along its own ray only: the information-weighted solution keeps both views' across-ray
        coordinates (the error is far below the along-ray noise); the prior adds no lag on a smooth path."""
        rng = np.random.default_rng(3)
        T, fps = 150, 30.0
        x, _ = trajectory(T, fps)
        c = np.array([[3.0, 0, 1], [0, 3.0, 1]])
        Lam, eta = np.zeros((T, 3, 3)), np.zeros((T, 3))
        for v in range(2):
            d = x - c[v]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            Q = x + d * rng.normal(size=(T, 1)) * mvt.ALONG  # along the ray: monocular depth
            info, e = mvt.vision_info(Q[None], np.ones((1, T)), c[v:v + 1], Q)
            Lam, eta = Lam + info, eta + e
        y, o = mvt.solve_point(Lam, eta, fps)
        self.assertIsNone(o)
        self.assertLess(np.linalg.norm(y - x, axis=1).mean(), 0.012)
        y0, _ = mvt.solve_point(Lam, (Lam @ x[..., None])[..., 0], fps)  # noise-free: the path itself
        self.assertLess(np.abs(y0 - x).max(), 2e-3)

    def test_tracker_offset_estimated(self):
        """A tracker on a rotating segment, 10 cm from the point: the point follows the tracker (the views are 3 cm
        noisy) and the static offset is recovered from the views."""
        rng = np.random.default_rng(4)
        T, fps = 200, 30.0
        x, t = trajectory(T, fps)
        R = Rotation.from_rotvec(np.stack([0.8 * np.sin(t), 0.5 * t, 0.3 * np.cos(t)], 1)).as_matrix()
        o_true = np.array([0.06, -0.08, 0.02])
        p = x - np.einsum("tij,j->ti", R, o_true) + rng.normal(size=(T, 3)) * 0.002
        Lam = np.tile(np.eye(3) / 0.03 ** 2, (T, 1, 1))
        eta = (Lam @ (x + rng.normal(size=(T, 3)) * 0.03)[..., None])[..., 0]
        y, o = mvt.solve_point(Lam, eta, fps, (p, R, np.full(T, 1 / mvt.TRACKER ** 2)))
        self.assertLess(np.linalg.norm(o - o_true), 0.006)
        self.assertLess(np.linalg.norm(y - x, axis=1).mean(), 0.006)
        yv, _ = mvt.solve_point(Lam, eta, fps)
        self.assertGreater(np.linalg.norm(yv - x, axis=1).mean(), 0.012)  # the views alone

    def test_segment_rotation(self):
        rng = np.random.default_rng(5)
        P = rng.normal(size=(8, 3)) * 0.05
        R = Rotation.random(10, random_state=6).as_matrix()
        xs = np.einsum("tij,pj->tpi", R, P) + rng.normal(size=(10, 1, 3))
        Rs = mvt.segment_rotation(xs, np.ones(10, bool))
        self.assertLess(np.abs(np.einsum("tij,jk->tik", Rs, R[0]) - R).max(), 1e-9)  # relative to frame 0


# ----------------------------------------------------------------------------------------------------------- synthetic take

NV, FPS, T0 = 6890, 30.0, 1000.0
SMPL_J = np.array([[0, 0, .95], [.09, 0, .88], [-.09, 0, .88], [0, 0, 1.05], [.1, 0, .5], [-.1, 0, .5], [0, 0, 1.18],
                   [.1, 0, .08], [-.1, 0, .08], [0, 0, 1.3], [.1, .1, .03], [-.1, .1, .03], [0, 0, 1.5], [.08, 0, 1.42],
                   [-.08, 0, 1.42], [0, 0, 1.62], [.18, 0, 1.42], [-.18, 0, 1.42], [.45, 0, 1.42], [-.45, 0, 1.42],
                   [.7, 0, 1.42], [-.7, 0, 1.42], [.78, 0, 1.42], [-.78, 0, 1.42]])
J14_PART = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]  # the 14 joints' SMPL parts (r_ankle ... head)
HAND = (20, 22)


class Body:
    """A stick-figure stand-in for SMPL: each vertex on one part (its skinning weight), J_regressor the part centroids."""

    def __init__(self, rng):
        import torch

        self.part = np.repeat(np.arange(24), int(np.ceil(NV / 24)))[:NV]
        v = SMPL_J[self.part] + rng.normal(size=(NV, 3)) * 0.04
        self.reg = np.stack([(self.part == j) / (self.part == j).sum() for j in range(24)])
        self.v_template = torch.as_tensor(v)
        self.J_regressor = torch.as_tensor(self.reg)
        self.weights = torch.as_tensor(np.eye(24)[self.part])
        self.j14 = self.reg[J14_PART]
        self.sidx = np.concatenate([rng.choice(np.flatnonzero(self.part == j), 20 if j != 23 else 8, replace=False) for j in range(24)])
        assert len(self.sidx) == 468
        self.kp70 = rng.choice(NV, 70, replace=False)


def poses(T):
    """The truth: the body's rigid motion, the left hand's own (about its wrist), the tracker's mount on the hand."""
    t = np.arange(T) / FPS
    Rb = Rotation.from_rotvec(np.stack([0 * t, 0 * t, 0.4 * np.sin(0.8 * t)], 1)).as_matrix()
    cb = np.stack([0.3 * np.sin(0.6 * t), 0.2 * np.cos(0.5 * t), 0 * t], 1)
    Rh = Rotation.from_rotvec(np.stack([0.9 * np.sin(1.7 * t), 0.6 * np.cos(1.1 * t), 0.5 * np.sin(0.9 * t)], 1)).as_matrix()
    ch = np.stack([0.15 * np.sin(2.0 * t), 0.2 * np.cos(1.4 * t) - 0.2, 0.1 * np.sin(2.6 * t)], 1)  # the hand's own motion
    return t, Rb, cb, Rh, ch


def world_verts(body, T):
    t, Rb, cb, Rh, ch = poses(T)
    v0 = body.v_template.numpy()
    hand = np.isin(body.part, HAND)
    w0 = SMPL_J[20]
    V = np.einsum("tij,vj->tvi", Rb, v0) + cb[:, None]
    hw = np.einsum("tij,vj->tvi", Rh, v0[hand] - w0) + w0 + ch[:, None]  # the hand about its wrist, then moved
    V[:, hand] = np.einsum("tij,tvj->tvi", Rb, hw) + cb[:, None]
    Rhw = np.einsum("tij,tjk->tik", Rb, Rh)  # the hand's world rotation
    hc = np.einsum("tij,tj->ti", Rb, w0 + ch) + cb  # the wrist's world position
    return V, Rhw, hc


MOUNT_R = Rotation.from_rotvec([0.3, -1.2, 0.5]).as_matrix()  # the tracker's frame on the hand
MOUNT_T = np.array([0.04, -0.03, 0.08])  # its origin, in the hand's frame from the wrist


def tracker_truth(body, T, rate=120.0):
    """The left-hand tracker at rate Hz on the rig clock (T0 + frame time), poses from the truth (interpolated)."""
    V, Rhw, hc = world_verts(body, T)
    ts = np.arange(-0.05, (T - 1) / FPS + 0.05, 1 / rate)
    tf = np.arange(T) / FPS
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp

    k = np.clip(ts, 0, tf[-1])
    Rh = Slerp(tf, Rotation.from_matrix(Rhw))(k).as_matrix()
    c = interp1d(tf, hc, axis=0)(k)
    R = np.einsum("tij,jk->tik", Rh, MOUNT_R)
    p = c + np.einsum("tij,j->ti", Rh, MOUNT_T)
    return {"t": T0 + ts, "xyz": p, "quat": wxyz(R), "valid": np.ones(len(ts), bool)}


def hand_error(Rhw):
    """The views' common error on the left hand (4 cm, the fusion keeps it): in the hand's frame, zero-mean over the clip
    (a part constant in the tracker's frame would go into the estimated offsets: docs/multiview.md)."""
    t = np.arange(len(Rhw)) / FPS
    e = 0.04 * np.stack([np.sin(2 * np.pi * t / 1.5), np.cos(2 * np.pi * t / 1.0), np.sin(2 * np.pi * t / 0.75)], 1)
    return np.einsum("tij,tj->ti", Rhw, e - e.mean(0))


class FakeFAB:
    """FABPredictor's contract, on the synthetic take: the frame and camera are read from the image's pixels; each
    view's body is the truth (+ the common hand error) scaled about its camera (the monocular size error)."""

    def __init__(self, body, rig_views, T):
        self.body, self.views = body, rig_views
        self.j14 = body.j14
        self.smpl_fitter_path = "fitter.pt"
        import torch

        self.model = types.SimpleNamespace(dense_head=types.SimpleNamespace(surface_idx=torch.as_tensor(body.sidx)))
        V, Rhw, _ = world_verts(body, T)
        V[:, np.isin(body.part, HAND)] += hand_error(Rhw)[:, None]
        self.V = V

    def predict_video(self, frames, K):
        for img in frames:
            k, c = int(img[0, 0, 0]), int(img[0, 1, 0])
            Tc = np.linalg.inv(self.views[c])
            Xc = (self.V[k] @ Tc[:3, :3].T + Tc[:3, 3]) * (1.06 if c == 0 else 0.95)
            uv = Xc @ mv_K().T
            uv = uv[:, :2] / uv[:, 2:]
            j14 = self.j14 @ Xc
            j2 = j14 @ mv_K().T
            j2 = j2[:, :2] / j2[:, 2:]
            hip = j14[[2, 3]].mean(0)
            kp2 = uv[self.body.kp70]
            yield [{"box": np.r_[kp2.min(0), kp2.max(0)], "keypoints2d": kp2, "joints2d_h36m14": j2,
                    "keypoints3d": Xc[self.body.kp70], "joints_h36m14": j14 - hip, "mesh_smpl": Xc,
                    "keypoint_visibility": np.full((84, 3), 0.9), "dense_surface_cam": Xc[self.body.sidx],
                    "dense_surface2d": uv[self.body.sidx], "contact": np.zeros(16), "smpl_pose": np.zeros(72),
                    "smpl_betas": np.zeros(10), "mesh_smpl_fit": Xc}]


def mv_K():
    return np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])


def make_take(d, T):
    import cv2

    Ts = [look_at([3.2, 0.4, 1.3]), look_at([0.3, 3.0, 1.2])]
    world = {"fps": FPS, "floor_z": -0.05, "cameras": {}}
    for c, Tw in enumerate(Ts):
        name = f"cam{c}"
        (d / name).mkdir(parents=True)
        for k in range(T):
            img = np.zeros((16, 16, 3), np.uint8)
            img[0, 0], img[0, 1] = k, c
            cv2.imwrite(str(d / name / f"frame_{k:06d}.png"), img)
        np.save(d / name / "frame_index.npy", np.stack([T0 + np.arange(T) / FPS + 0.002 * c, np.arange(T)], 1))
        world["cameras"][name] = {"K": mv_K().tolist(), "T_world_cam": Tw.tolist()}
    (d / "world.json").write_text(json.dumps(world))
    return Ts


def run_stubbed(module, take, out, fab, **kw):
    """module.run with the stand-in predictor and the SMPL fit / smoothing / export replaced by recorders: -> the fit's
    inputs, the export's inputs, and the fusion record."""
    rec = {}

    def fit(surf, surf_w, j14, j14_w, valid, R_fit, fitter_path, surface_idx, j14_reg, device="cuda", body=None):
        rec["fit"] = [np.array(a) for a in (surf, surf_w, j14, j14_w, valid, R_fit)]
        T = len(valid)
        return np.tile(np.eye(3), (T, 24, 1, 1)), np.zeros(10, np.float32), np.zeros((T, NV, 3))

    def world(R, verts, betas, valid, fps, body, sigma_rot=2.0, sigma_pos=3.0):
        T = len(R)
        return np.zeros((T, 72)), np.zeros((T, 3)), np.zeros((T, 24, 3)), np.zeros((T, NV, 3))

    def export(out, name, views, fps, t, body, pose, betas, trans, joints, verts, valid, floor_z, contact, surf, surface_idx, notes):
        rec["export"] = {"t": np.array(t), "valid": np.array(valid), "floor_z": floor_z, "contact": np.array(contact),
                         "surf": np.array(surf), "notes": json.dumps(notes, sort_keys=True)}
        return Path(out) / f"{name}.npz"

    with mock.patch("fab.predictor.FABPredictor", lambda *a, **k: fab), mock.patch.object(module, "fit_sequence", fit), \
            mock.patch.object(module, "world_body", world), mock.patch.object(module, "export", export):
        module.run(str(take), str(out), name="take", device="cpu", **kw)
    rec["fusion"] = dict(np.load(Path(out) / "work" / "take_fusion.npz"))
    return rec


def base_module():
    """fab/multiview.py of the multiview-fusion commit this branch starts from (None outside a git checkout)."""
    for ref in ("origin/multiview-fusion", "multiview-fusion"):
        r = subprocess.run(["git", "-C", str(REPO), "merge-base", "HEAD", ref], capture_output=True, text=True)
        if r.returncode == 0:
            src = subprocess.run(["git", "-C", str(REPO), "show", f"{r.stdout.strip()}:fab/multiview.py"], capture_output=True,
                                 text=True)
            if src.returncode == 0:
                m = importlib.util.module_from_spec(importlib.util.spec_from_loader("fab_multiview_base", loader=None))
                exec(compile(src.stdout, "multiview_base.py", "exec"), m.__dict__)
                return m
    return None


class Runs(unittest.TestCase):
    T = 90

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.body = Body(np.random.default_rng(7))
        cls.Ts = make_take(cls.d / "take", cls.T)
        cls.fab = FakeFAB(cls.body, cls.Ts, cls.T)
        V, _, _ = world_verts(cls.body, cls.T)
        cls.truth = np.concatenate([V[:, cls.body.sidx], np.einsum("jv,tvc->tjc", cls.body.j14, V)], 1)
        cls.hand = np.r_[np.flatnonzero(np.isin(cls.body.part[cls.body.sidx], HAND)), 468 + 11]
        cls.plain = run_stubbed(mv, cls.d / "take", cls.d / "plain", cls.fab)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def targets(self, rec):
        surf, _, j14, _, valid, _ = rec["fit"]
        self.assertTrue(valid.all())
        return np.concatenate([surf, j14], 1)

    def save(self, tr, name):
        f = self.d / f"{name}.npz"
        np.savez(f, **{k: v for k, v in tr.items() if v is not None})
        return f

    def test_without_trackers_bitwise_the_base(self):
        base = base_module()
        if base is None:
            self.skipTest("not a git checkout with multiview-fusion")
        ref = run_stubbed(base, self.d / "take", self.d / "base", self.fab)
        self.assertTrue(all(np.array_equal(a, b, equal_nan=True) for a, b in zip(ref["fit"], self.plain["fit"])))
        for k in ref["export"]:
            if isinstance(ref["export"][k], np.ndarray):
                self.assertTrue(np.array_equal(ref["export"][k], self.plain["export"][k], equal_nan=True), k)
            else:
                self.assertEqual(ref["export"][k], self.plain["export"][k], k)
        self.assertEqual(sorted(ref["fusion"]), sorted(self.plain["fusion"]))
        for k in ref["fusion"]:
            self.assertTrue(np.array_equal(ref["fusion"][k], self.plain["fusion"][k], equal_nan=True), k)
        self.assertNotIn("trackers", self.plain["export"]["notes"])

    def test_hand_tracker_followed(self):
        tr = tracker_truth(self.body, self.T)
        tr["xyz"] = tr["xyz"] + np.random.default_rng(8).normal(size=tr["xyz"].shape) * 0.001
        rec = run_stubbed(mv, self.d / "take", self.d / "trk", self.fab, trackers=f"left_hand={self.save(tr, 'left')}")
        X, X0 = self.targets(rec), self.targets(self.plain)
        e = np.linalg.norm(X - self.truth, axis=-1)
        e0 = np.linalg.norm(X0 - self.truth, axis=-1)
        self.assertGreater(e0[:, self.hand].mean(), 0.03)  # the views' hand error
        self.assertLess(e[:, self.hand].mean(), 0.008)  # the tracker's
        rest = np.setdiff1d(np.arange(482), self.hand)
        self.assertLess(np.abs(e[:, rest].mean() - e0[:, rest].mean()), 0.005)  # the rest: the views (smoothed)
        notes = json.loads(rec["export"]["notes"])["trackers"]
        self.assertTrue(notes["left_wrist"]["used"] and notes["left_wrist"]["accepted"] > 0.95)
        surf_w, surf_w0 = rec["fit"][1], self.plain["fit"][1]
        self.assertTrue((surf_w[:, self.hand[:-1]] >= surf_w0[:, self.hand[:-1]]).all() and (surf_w[:, self.hand[:-1]] >= 0.5).all())
        tr["quat"] = None  # position only: the segment's rotation from the views
        rec = run_stubbed(mv, self.d / "take", self.d / "trk_pos", self.fab, trackers=f"left_wrist={self.save(tr, 'left_pos')}")
        e = np.linalg.norm(self.targets(rec) - self.truth, axis=-1)
        self.assertLess(e[:, self.hand].mean(), e0[:, self.hand].mean())

    def corrupted(self, kind, a, b):
        """The tracker with samples in (a, b) s lost (removed), frozen (the last pose before, repeated) or drifting (a
        ramp to 40 cm); -> the hand's mean target error per frame, of this run and of the run without trackers."""
        tr = tracker_truth(self.body, self.T)
        t = tr["t"] - T0
        m = (t > a) & (t < b)
        if kind == "lost":
            tr = {k: v[~m] for k, v in tr.items()}
        elif kind == "frozen":
            i0 = np.flatnonzero(m)[0] - 1
            tr["xyz"][m], tr["quat"][m] = tr["xyz"][i0], tr["quat"][i0]
        else:
            tr["xyz"][m] += 0.4 * ((t[m] - a) / (b - a))[:, None] * np.array([0.6, 0.0, 0.8])
        return self.save(tr, f"{kind}_{a}")

    def hand_err(self, rec):
        return np.linalg.norm(self.targets(rec) - self.truth, axis=-1)[:, self.hand].mean(1)

    def test_lost_and_frozen_samples(self):
        """A lost window: inside, the hand's targets are the views' (within 2 cm), outside the tracker's; frozen samples
        (the same window) are dropped, so the run is the lost one, bitwise."""
        f = self.corrupted("lost", 1.0, 1.5)
        rec = run_stubbed(mv, self.d / "take", self.d / "lost", self.fab, trackers=f"left_wrist={f}")
        e, e0 = self.hand_err(rec), self.hand_err(self.plain)
        tf = np.arange(self.T) / FPS
        m = (tf > 1.0) & (tf < 1.5)
        self.assertLess(e[m].max(), e0[m].max() + 0.02)
        self.assertLess(e[(tf < 0.8) | (tf > 1.7)].mean(), 0.015)
        rec2 = run_stubbed(mv, self.d / "take", self.d / "frozen", self.fab, trackers=f"left_wrist={self.corrupted('frozen', 1.0, 1.5)}")
        self.assertTrue(all(np.array_equal(a, b, equal_nan=True) for a, b in zip(rec["fit"], rec2["fit"])))

    def test_drift_gated(self):
        """A drift ramp to 40 cm over 1 s: late in it (24-40 cm) the hand's targets have left the drifting samples for
        the views; without the gate (robust=False) they follow them."""
        f = self.corrupted("drift", 1.9, 2.9)
        rec = run_stubbed(mv, self.d / "take", self.d / "drift", self.fab, trackers=f"left_wrist={f}")
        e = self.hand_err(rec)
        tf = np.arange(self.T) / FPS
        late = (tf > 2.5) & (tf < 2.9)
        self.assertLess(e[late].mean(), 0.08)
        self.assertLess(e[tf < 1.7].mean(), 0.03)
        orig = mvt.smooth
        with mock.patch.object(mvt, "smooth", lambda *a, **k: orig(*a, robust=False, **k)):
            rec = run_stubbed(mv, self.d / "take", self.d / "drift_nogate", self.fab, trackers=f"left_wrist={f}")
        self.assertGreater(self.hand_err(rec)[late].mean(), 0.2)


class ViewPoints(unittest.TestCase):
    def test_view_points_are_the_fusions(self):
        """Three views with monocular size errors, a wrong forearm in one and a wrist whose 2D disagrees in two (not
        triangulated): the per-view points the smoother reads, fused by the fusion's robust mean, are the fusion's own
        fused points, bitwise; a triangulated joint is its triangulated position in every view."""
        rng = np.random.default_rng(9)
        Ts = [look_at([3 * np.cos(a), 3 * np.sin(a), 1.4]) for a in (0.2, 2.3, 4.4)]
        views = [mv.View(f"c{i}", mv_K(), T) for i, T in enumerate(Ts)]
        kp = rng.normal(size=(84, 3)) * [0.25, 0.15, 0.45] + [0, 0, 1]
        surf = rng.normal(size=(468, 3)) * [0.25, 0.15, 0.45] + [0, 0, 1]
        vis_idx = rng.integers(70, 84, 468)
        obs = []
        for k, v in enumerate(views):
            Tc = np.linalg.inv(v.T_world_cam)
            cam = lambda X: X @ Tc[:3, :3].T + Tc[:3, 3]  # noqa: E731
            uv = cam(kp) @ mv_K().T
            uv = uv[:, :2] / uv[:, 2:]
            if k > 0:
                uv[81] += 40.0 * k  # the left wrist's 2D: two views disagree
            s = 1.0 + 0.07 * (k - 1)
            o = {"kp2d": uv.astype(np.float32), "vis": rng.uniform(0.02, 1, 84).astype(np.float32),
                 "kp3d": (s * cam(kp)).astype(np.float32), "surf": (s * cam(surf)).astype(np.float32),
                 "contact": np.zeros(16, np.float32), "box": np.zeros(4, np.float32)}
            if k == 0:
                o["kp3d"][77] += np.float32([0.1, 0.0, 0.0])
                o["surf"][vis_idx == 77] += np.float32([0.1, 0.0, 0.0])
            obs.append(o)
        for o_ in (obs, [obs[0], obs[1], None]):
            f = mv.fuse_frame(views, o_, [None] * 3, vis_idx)
            tri = f["tri"][mv.J14]
            self.assertFalse(tri.all())
            Q, w = mvt.view_points(views, o_, f, vis_idx)
            self.assertTrue(np.array_equal(mv.fuse(Q[:, :468], w[:, :468])[0], f["surf"]))
            self.assertTrue(np.array_equal(mv.fuse(Q[:, 468:], w[:, 468:])[0][~tri], f["j14"][~tri]))
            for v in range(3):
                if o_[v] is not None:
                    self.assertTrue(np.array_equal(Q[v, 468:][tri], f["j14"][tri]))


if __name__ == "__main__":
    unittest.main()
