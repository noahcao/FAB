"""fab-multiview (fab/multiview.py) on synthetic rigs: triangulation with outlier rejection, the robust similarity and
fusion, frame sync, rig loading, subject selection and one whole fusion frame; and that the single-view release is
untouched: importing fab.multiview changes no predictor default, and (with the weights; on the CPU, where the eager path
repeats bit for bit) the predictor's outputs through fab-multiview's frame loading are bit-identical to the plain
FABPredictor.predict_video call."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fab import multiview as mv  # noqa: E402


def have_inputs():
    from fab import paths

    return paths.model_file("fab-l").exists() and paths.shared("smpl_fitter.pt").exists() and paths.sam3d_dir().exists() \
        and (paths.smpl_dir() / "SMPL_NEUTRAL.pkl").exists() and (paths.body_models_dir() / "mhr").exists()


def look_at(c, target=(0.0, 0.0, 1.0)):
    """T_world_cam of a camera at c looking at target (OpenCV axes, world z up)."""
    c, target = np.asarray(c, float), np.asarray(target, float)
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, y, z], 1), c
    return T


K0 = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])


def rig(n=3, r=3.0):
    return [mv.View(f"cam{i}", K0, look_at([r * np.cos(a), r * np.sin(a), 1.5])) for i, a in
            enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3)]


def body_points(rng, n=84, centre=(0.0, 0.0, 1.0)):
    X = rng.normal(size=(n, 3)) * [0.25, 0.15, 0.45] + centre
    return X


class Geometry(unittest.TestCase):
    def test_triangulate_exact(self):
        rng = np.random.default_rng(0)
        V = rig(3)
        X = body_points(rng)
        uv = np.stack([mv.project(v.T_world_cam, v.K, X)[0] for v in V])
        Y, ok, used = mv.triangulate(np.stack([v.T_world_cam for v in V]), np.stack([v.K for v in V]), uv, np.ones((3, len(X))))
        self.assertTrue(ok.all())
        self.assertLess(np.abs(Y - X).max(), 1e-6)

    def test_outlier_view_dropped(self):
        rng = np.random.default_rng(1)
        V = rig(3)
        X = body_points(rng)
        uv = np.stack([mv.project(v.T_world_cam, v.K, X)[0] for v in V])
        uv[2, :10] += 60.0  # view 2 wrong on the first 10 points
        Ts, Ks = np.stack([v.T_world_cam for v in V]), np.stack([v.K for v in V])
        Y, ok, used = mv.triangulate(Ts, Ks, uv, np.ones((3, len(X))))
        self.assertTrue(ok.all())
        self.assertFalse(used[2, :10].any())
        self.assertLess(np.abs(Y - X).max(), 1e-6)

    def test_two_views_that_disagree_are_not_triangulated(self):
        rng = np.random.default_rng(2)
        V = rig(2)
        X = body_points(rng)
        uv = np.stack([mv.project(v.T_world_cam, v.K, X)[0] for v in V])
        uv[1, :5] += 80.0
        Y, ok, _ = mv.triangulate(np.stack([v.T_world_cam for v in V]), np.stack([v.K for v in V]), uv, np.ones((2, len(X))))
        self.assertFalse(ok[:5].any())
        self.assertTrue(ok[5:].all())
        self.assertTrue(np.isnan(Y[:5]).all())

    def test_low_visibility_views_left_out(self):
        rng = np.random.default_rng(3)
        V = rig(3)
        X = body_points(rng)
        uv = np.stack([mv.project(v.T_world_cam, v.K, X)[0] for v in V])
        uv[0] += 200.0
        w = np.ones((3, len(X)))
        w[0] = 0.2  # P(visible) below VIS_MIN: not used, whatever its 2D says
        Y, ok, used = mv.triangulate(np.stack([v.T_world_cam for v in V]), np.stack([v.K for v in V]), uv, w, vis_min=0.5)
        self.assertTrue(ok.all() and not used[0].any())
        Y, ok, used = mv.triangulate(np.stack([v.T_world_cam for v in V]), np.stack([v.K for v in V]), uv, w)  # all views:
        self.assertTrue(ok.all() and not used[0].any())  # the wrong view is dropped by its ray gap
        self.assertLess(np.abs(Y - X).max(), 1e-6)
        self.assertLess(np.abs(Y - X).max(), 1e-6)

    def test_similarity_robust(self):
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(4)
        X = rng.normal(size=(40, 3))
        R = Rotation.from_rotvec([0.2, -0.4, 0.3]).as_matrix()
        Y = 1.07 * X @ R.T + [0.3, -0.2, 0.5]
        Y[:6] += rng.normal(size=(6, 3)) * 0.5  # outliers
        s, R_, t = mv.similarity(X, Y, np.ones(40), iters=10)
        self.assertAlmostEqual(s, 1.07, places=3)
        self.assertLess(np.abs(R_ - R).max(), 1e-3)
        self.assertLess(np.abs(t - [0.3, -0.2, 0.5]).max(), 1e-3)

    def test_fuse_weights_and_outliers(self):
        rng = np.random.default_rng(5)
        truth = rng.normal(size=(50, 3))
        P = np.stack([truth + rng.normal(size=truth.shape) * 0.002, truth + 0.3, truth + rng.normal(size=truth.shape) * 0.002])
        Y, sw = mv.fuse(P, np.array([[1.0] * 50, [0.05] * 50, [1.0] * 50]))
        self.assertLess(np.abs(Y - truth).max(), 0.01)
        Y2, _ = mv.fuse(P, np.ones((3, 50)))  # equal weights: the outlying view is rejected by the robust mean
        self.assertLess(np.abs(Y2 - truth).max(), 0.01)
        Pn = P.copy()
        Pn[:, 0] = np.nan  # a point no view has
        Y3, sw3 = mv.fuse(Pn, np.ones((3, 50)))
        self.assertTrue(np.isnan(Y3[0]).all() and sw3[0] == 0)


class RigAndSync(unittest.TestCase):
    def test_sync_nearest_within_half_period(self):
        a = mv.View("a", K0, np.eye(4), frames=[Path(f"{i}.jpg") for i in range(10)], times=np.arange(10) / 30.0)
        tb = np.arange(10) / 30.0 + 0.01
        tb = np.delete(tb, 4)  # b drops a frame
        b = mv.View("b", K0, np.eye(4), frames=[Path(f"{i}.jpg") for i in range(9)], times=tb)
        ids, t = mv.sync([a, b], 30.0)
        self.assertEqual(ids[:, 0].tolist(), list(range(10)))
        self.assertEqual(ids[:, 1].tolist(), [0, 1, 2, 3, -1, 4, 5, 6, 7, 8])

    def test_load_rig_json_and_take_folder(self):
        import cv2

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for c in ("c0", "c1"):
                (d / c).mkdir()
                for i in range(3):
                    cv2.imwrite(str(d / c / f"frame_{i:06d}.jpg"), np.zeros((8, 8, 3), np.uint8))
            T1 = look_at([2.0, 0, 1.5])
            rigj = {"fps": 30, "floor_z": -0.5, "up": [0, 0, 1], "cameras": [
                {"name": "c0", "frames": "c0", "K": [800, 800, 640, 360], "T_world_cam": np.eye(4).tolist()},
                {"name": "c1", "frames": "c1", "K": K0.tolist(), "T_world_cam": T1.tolist(), "dist": [0.1, 0, 0, 0, 0],
                 "offset": 0.02}]}
            (d / "rig.json").write_text(json.dumps(rigj))
            V, info = mv.load_rig(d / "rig.json")
            self.assertEqual([v.name for v in V], ["c0", "c1"])
            self.assertTrue(np.allclose(V[0].K, K0) and np.allclose(V[1].T_world_cam, T1))
            self.assertIsNone(V[0].dist)
            self.assertEqual(len(V[1]), 3)
            self.assertEqual(info["floor_z"], -0.5)
            rigj["up"] = [0, -1, 0]  # a y-down world: turned so that up is +z
            (d / "rig_y.json").write_text(json.dumps(rigj))
            Vy, _ = mv.load_rig(d / "rig_y.json")
            self.assertTrue(np.allclose(Vy[1].T_world_cam[:3, 3], mv.up_rotation([0, -1, 0]) @ T1[:3, 3]))
            self.assertTrue(np.allclose(mv.up_rotation([0, -1, 0]) @ [0, -1, 0], [0, 0, 1]))
            world = {"floor_z": -0.9, "cameras": {"c0": {"K": {"fx": 800, "fy": 800, "cx": 640, "cy": 360}, "T_world_cam": np.eye(4).tolist()},
                                                  "c1": {"K": K0.tolist(), "T_world_cam": T1.tolist()},
                                                  "other": {"K": K0.tolist(), "T_world_cam": None}}}
            (d / "world.json").write_text(json.dumps(world))
            np.save(d / "c1" / "frame_index.npy", np.stack([np.arange(3) / 30.0, np.arange(3)], 1))
            V2, info2 = mv.load_rig(d)
            self.assertEqual([v.name for v in V2], ["c0", "c1"])
            self.assertIsNone(V2[0].times)
            self.assertTrue(np.allclose(V2[1].times, np.arange(3) / 30.0))
            self.assertEqual(info2["floor_z"], -0.9)


def person_obs(view, X, vis=None, rng=None, noise=0.0):
    """A slim per-view observation of world keypoints X (84, 3)."""
    uv, _ = mv.project(view.T_world_cam, view.K, X)
    if rng is not None:
        uv = uv + rng.normal(size=uv.shape) * noise
    box = np.r_[uv.min(0), uv.max(0)].astype(np.float32)
    Tc = np.linalg.inv(view.T_world_cam)
    return {"kp2d": uv.astype(np.float32), "vis": (np.ones(84) if vis is None else vis).astype(np.float32), "box": box,
            "kp3d": (X @ Tc[:3, :3].T + Tc[:3, 3]).astype(np.float32), "contact": np.full(16, 0.5, np.float32)}


class FakeFAB:
    """predict_video's contract with one person per frame (the keys slim reads)."""
    j14 = np.full((14, 6890), 1.0 / 6890)

    def __init__(self):
        self.calls = 0

    def predict_video(self, frames, K):
        for img in frames:
            self.calls += 1
            yield [{"box": np.zeros(4), "keypoints2d": np.zeros((70, 2)), "joints2d_h36m14": np.zeros((14, 2)),
                    "keypoints3d": np.zeros((70, 3)), "joints_h36m14": np.zeros((14, 3)), "mesh_smpl": np.ones((6890, 3)),
                    "keypoint_visibility": np.full((84, 3), 1 / 3), "dense_surface_cam": np.zeros((468, 3)),
                    "dense_surface2d": np.zeros((468, 2)), "contact": np.zeros(16), "smpl_pose": np.zeros(72),
                    "smpl_betas": np.zeros(10), "mesh_smpl_fit": np.zeros((6890, 3)), "img_mean": float(img.mean())}]


class Views(unittest.TestCase):
    def test_run_views_cache_and_frame_mapping(self):
        import cv2

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "c0").mkdir()
            for i in range(4):
                cv2.imwrite(str(d / "c0" / f"{i:03d}.png"), np.full((8, 8, 3), i, np.uint8))
            v = mv.View("c0", K0, np.eye(4), frames=sorted((d / "c0").iterdir()))
            ids = np.array([[0], [2], [-1], [3]])
            fab = FakeFAB()
            ppl = mv.run_views(fab, [v], ids, cache=d / "cache", key="m")
            self.assertEqual(fab.calls, 3)
            self.assertEqual([len(x) for x in ppl[0]], [1, 1, 0, 1])
            o = ppl[0][1][0]
            self.assertEqual(o["kp2d"].shape, (84, 2))
            self.assertTrue(np.allclose(o["kp3d"][70:], 1.0))  # joints in the dense frame: + the mesh's J14 hips
            self.assertTrue(np.allclose(o["vis"], 1 / 3))
            mv.run_views(fab, [v], ids, cache=d / "cache", key="m")  # cached
            self.assertEqual(fab.calls, 3)
            mv.run_views(fab, [v], ids, cache=d / "cache", key="other model")  # another model: run again
            self.assertEqual(fab.calls, 6)


class Subject(unittest.TestCase):
    def test_init_picks_most_prominent_consistent_match_then_follows(self):
        rng = np.random.default_rng(6)
        V = rig(2)
        near = body_points(rng, centre=(0.0, 0.0, 1.0))
        far = body_points(rng, centre=(1.5, 1.5, 1.0)) * [0.6, 0.6, 0.6]  # a smaller, farther bystander
        other = body_points(rng, centre=(-1.0, 1.0, 1.0))  # seen in view 1 only
        cands = [[person_obs(V[0], far), person_obs(V[0], near)], [person_obs(V[1], other), person_obs(V[1], near),
                                                                   person_obs(V[1], far)]]
        self.assertEqual(mv.init_subject(V, cands), [1, 1])
        moved = near + [0.05, 0.0, 0.0]
        self.assertEqual(mv.follow(V[1], [person_obs(V[1], far), person_obs(V[1], moved)], near), 1)
        self.assertEqual(mv.follow(V[1], [person_obs(V[1], far)], near), -1)  # the subject left this view


class FuseFrame(unittest.TestCase):
    def test_monocular_depth_error_removed(self):
        """Each view's camera-frame body is the truth scaled about its camera (the monocular depth / size error) and
        slightly turned; the triangulated joints place it back, so the fused points are the truth."""
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(7)
        V = rig(3)
        kp = body_points(rng)
        surf = body_points(rng, n=468)
        vis_idx = rng.integers(70, 84, 468)
        obs = []
        for k, v in enumerate(V):
            o = person_obs(v, kp)
            Tc = np.linalg.inv(v.T_world_cam)
            Rr = Rotation.from_rotvec(rng.normal(size=3) * 0.03).as_matrix()
            s = 1.0 + 0.08 * (k - 1)  # the view's body too big / small, hence too far / near
            cam = lambda X: X @ Tc[:3, :3].T + Tc[:3, 3]  # noqa: E731
            cc = cam(kp).mean(0)
            err = lambda X: s * ((cam(X) - cc) @ Rr.T + cc)  # noqa: E731  (turned about the body, scaled about the camera)
            o["kp3d"], o["surf"] = err(kp).astype(np.float32), err(surf).astype(np.float32)
            obs.append(o)
        f = mv.fuse_frame(V, obs, [None] * 3, vis_idx)
        self.assertTrue(f["tri"].all() and all(f["own"]))
        self.assertLess(np.abs(f["surf"] - surf).max(), 1e-4)
        self.assertLess(np.abs(f["j14"] - kp[70:84]).max(), 1e-6)
        self.assertTrue(np.allclose(f["contact"], 0.5))

    def test_limb_error_carried_by_its_joint(self):
        """A view's wrist and the dense points nearest it are 10 cm off (a wrong forearm in that view): the triangulated
        wrist's residual moves those points back."""
        rng = np.random.default_rng(9)
        V = rig(2)  # two views: the robust mean alone would land half-way
        kp, surf = body_points(rng), body_points(rng, n=468)
        vis_idx = rng.integers(70, 84, 468)
        obs = []
        for k, v in enumerate(V):
            o = person_obs(v, kp)
            Tc = np.linalg.inv(v.T_world_cam)
            o["surf"] = (surf @ Tc[:3, :3].T + Tc[:3, 3]).astype(np.float32)
            if k == 0:
                o["kp3d"][81] += [0.1, 0.0, 0.0]
                o["surf"][vis_idx == 81] += np.float32([0.1, 0.0, 0.0])
            obs.append(o)
        f = mv.fuse_frame(V, obs, [None] * 2, vis_idx)
        self.assertLess(np.abs(f["surf"] - surf).max(), 2e-3)

    def test_one_view_is_that_view(self):
        rng = np.random.default_rng(8)
        V = rig(2)
        kp, surf = body_points(rng), body_points(rng, n=468)
        o = person_obs(V[0], kp)
        Tc = np.linalg.inv(V[0].T_world_cam)
        o["surf"] = (surf @ Tc[:3, :3].T + Tc[:3, 3]).astype(np.float32)
        f = mv.fuse_frame(V, [o, None], [None, None], rng.integers(70, 84, 468))
        self.assertFalse(f["tri"].any())
        self.assertLess(np.abs(f["surf"] - surf).max(), 1e-5)


class Floor(unittest.TestCase):
    def test_predicted_contacts_placed_on_the_floor(self):
        rng = np.random.default_rng(10)
        vis_idx = rng.integers(70, 84, 468)
        surf = rng.normal(size=(3, 468, 3)) * 0.1 + [0, 0, 0.5]
        knee = vis_idx - 70 == 1
        surf[:, knee, 2] = rng.uniform(0.08, 0.2, (3, knee.sum()))  # the knee's points 8-20 cm above a floor at 0
        j14 = np.zeros((3, 14, 3))
        frames = [{"contact_best": np.zeros(16)} for _ in range(3)]
        frames[0]["contact_best"][6] = 0.8  # frame 0: l_knee on the floor
        frames[2]["contact_best"][6] = 0.4  # frame 2: below the threshold
        s2, j2, n = mv.floor_contacts(frames, np.ones(3, bool), surf, j14, vis_idx, [((6,), 1, "hold")], 0.0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(s2[0, knee, 2].min(), 0.0)
        self.assertTrue(np.allclose(s2[0, ~knee], surf[0, ~knee]) and np.allclose(s2[1:], surf[1:]))
        self.assertAlmostEqual(j2[0, 1, 2], -surf[0, knee, 2].min())
        far = surf.copy()
        far[0, knee, 2] += 0.5  # 58-70 cm up: more than the cap, left as it is
        self.assertEqual(mv.floor_contacts(frames, np.ones(3, bool), far, j14, vis_idx, [((6,), 1, "hold")], 0.0)[2], 0)

    def test_feet_only_lifted(self):
        rng = np.random.default_rng(11)
        vis_idx = rng.integers(70, 84, 468)
        foot = vis_idx - 70 == 0
        surf = np.zeros((2, 468, 3))
        surf[0, foot, 2] = rng.uniform(0.04, 0.1, foot.sum())  # frame 0: 4-10 cm above the floor (soles): left alone
        surf[1, foot, 2] = rng.uniform(-0.05, 0.1, foot.sum())  # frame 1: 5 cm into it: raised onto it
        frames = [{"contact_best": np.ones(16)} for _ in range(2)]
        s2, _, n = mv.floor_contacts(frames, np.ones(2, bool), surf, np.zeros((2, 14, 3)), vis_idx, [((), 0, "lift")], 0.0)
        self.assertEqual(n, 1)
        self.assertTrue(np.allclose(s2[0], surf[0]))
        self.assertAlmostEqual(s2[1, foot, 2].min(), 0.0)


class SingleViewUnchanged(unittest.TestCase):
    def test_import_changes_no_release_default(self):
        code = ("import copy, inspect, json, sys; import fab.predictor as P; "
                "d = {'simple': copy.deepcopy(P.SIMPLE_PROFILE), 'fabh': copy.deepcopy(P.FABH_MESH_PROFILE), "
                "'init': {k: repr(v.default) for k, v in inspect.signature(P.FABPredictor.__init__).parameters.items()}, "
                "'video': {k: repr(v.default) for k, v in inspect.signature(P.FABPredictor.predict_video_smoothed).parameters.items()}}; "
                "import fab.multiview; "
                "e = {'simple': P.SIMPLE_PROFILE, 'fabh': P.FABH_MESH_PROFILE, "
                "'init': {k: repr(v.default) for k, v in inspect.signature(P.FABPredictor.__init__).parameters.items()}, "
                "'video': {k: repr(v.default) for k, v in inspect.signature(P.FABPredictor.predict_video_smoothed).parameters.items()}}; "
                "sys.exit(0 if d == e else 1)")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def test_release_modules_do_not_import_multiview(self):
        for f in list((REPO / "fab").rglob("*.py")) + list((REPO / "bench").rglob("*.py")) + list((REPO / "scripts").glob("*.py")):
            if f.name != "multiview.py":
                import re

                hit = re.search(r"fab\.multiview|from fab import multiview|from \.+ ?import multiview|from \.+multiview", f.read_text())
                self.assertIsNone(hit, str(f))

    @unittest.skipUnless(have_inputs(), "needs the weights, SAM 3D Body's config and SMPL (README, Install)")
    def test_predictor_outputs_bit_identical(self):
        """The released predictor's video outputs on a demo image (eager, on the CPU, where it repeats bit for bit at a
        fixed thread count): the plain FABPredictor.predict_video call, with fab.multiview never imported, and the same
        call through fab-multiview's frame loading (one view, no distortion) are bitwise equal, and so are fab-multiview's
        per-view records (run_views) and slim() of the plain outputs."""
        img = str(REPO / "assets" / "demo" / "skater_seated.jpg")
        head = ("import sys, pickle, cv2, torch, numpy as np; torch.set_num_threads(8); "
                "from fab.predictor import FABPredictor, default_K; "
                f"img = {img!r}; K = default_K(*cv2.imread(img).shape[:2]); f = FABPredictor('fab-l', device='cpu'); ")
        plain = head + ("fr = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)] * 2; out = list(f.predict_video(fr, K)); "
                        "assert 'fab.multiview' not in sys.modules; "
                        "from fab import multiview as mv; out = (out, [[mv.slim(pp, f.j14) for pp in ppl] for ppl in out]); ")
        multi = head + ("from pathlib import Path; from fab import multiview as mv; "
                        "v = mv.View('c', K, np.eye(4), frames=[Path(img)] * 2); out = list(f.predict_video(v.images([0, 1]), v.K)); "
                        "out = (out, mv.run_views(f, [v], np.array([[0], [1]]))[0]); ")
        save = "pickle.dump(out, open(sys.argv[1], 'wb'))"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8"}
        with tempfile.TemporaryDirectory() as d:
            res = []
            for i, code in enumerate((plain, multi)):
                f = Path(d) / f"{i}.pkl"
                r = subprocess.run([sys.executable, "-c", code + save, str(f)], cwd=REPO, env=env, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[-3000:])
                res.append(pickle.loads(f.read_bytes()))
            for part in (0, 1):  # the predictor's outputs, then the per-view records
                a, b = res[0][part], res[1][part]
                self.assertEqual([len(x) for x in a], [len(x) for x in b])
                self.assertGreater(sum(map(len, a)), 0)
                for pa, pb in zip(a, b):
                    for qa, qb in zip(pa, pb):
                        self.assertEqual(sorted(qa), sorted(qb))
                        for k in qa:
                            if isinstance(qa[k], np.ndarray):
                                self.assertTrue(np.array_equal(qa[k], qb[k]), k)
                            else:
                                self.assertEqual(qa[k], qb[k], k)


if __name__ == "__main__":
    unittest.main()
