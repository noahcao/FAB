"""fab-retarget's standalone SMPL input (fab/retarget/smpl_input.py, scripts/fab_retarget.py smpl_inputs) on synthetic files and a
stand-in SMPL model (SMPL's topology and kinematic tree, made-up shapes and weights; no SMPL download needed): parameters,
vertices, AMASS-style SMPL-H, y-up, grounding, which files are SMPL files."""
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
JOINTS = np.array([[0, .92, 0], [.06, .83, 0], [-.06, .83, 0], [0, 1.03, 0], [.1, .47, 0], [-.1, .47, 0], [0, 1.16, 0],
                   [.08, .07, -.02], [-.08, .07, -.02], [0, 1.22, 0], [.1, .02, .1], [-.1, .02, .1], [0, 1.42, 0], [.07, 1.33, 0],
                   [-.07, 1.33, 0], [0, 1.6, 0], [.17, 1.36, 0], [-.17, 1.36, 0], [.43, 1.36, 0], [-.43, 1.36, 0], [.68, 1.36, 0],
                   [-.68, 1.36, 0], [.76, 1.36, 0], [-.76, 1.36, 0]])  # y up, as SMPL's template


def stand_in_smpl(path, seed, size=1.0):
    """An SMPL-format .pkl: 6890 vertices around the 24 joints (vertex i on joint i % 24, weight 1), J_regressor = each joint's
    vertex mean, random shape / pose blend shapes."""
    rng = np.random.default_rng(seed)
    part = np.arange(6890) % 24
    v = (JOINTS[part] + rng.normal(0, 0.03, (6890, 3))) * size
    W = np.eye(24)[part]
    Jr = W.T / W.sum(0)[:, None]
    pickle.dump({"v_template": v, "shapedirs": rng.normal(0, 0.01, (6890, 3, 10)), "posedirs": rng.normal(0, 1e-3, (6890, 3, 207)),
                 "J_regressor": Jr, "weights": W, "kintree_table": np.array([[2 ** 32 - 1] + PARENTS[1:], list(range(24))]),
                 "f": np.arange(300).reshape(100, 3)}, open(path, "wb"))


def fab_retarget():
    spec = importlib.util.spec_from_file_location("fab_retarget_script", ROOT / "scripts" / "fab_retarget.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class SmplInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fab.retarget import smpl_input as SI

        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name).resolve()
        for g, seed, size in (("NEUTRAL", 0, 1.0), ("MALE", 1, 1.08), ("FEMALE", 2, 0.95)):
            stand_in_smpl(cls.dir / f"SMPL_{g}.pkl", seed, size)
        cls.env = os.environ.get("FAB_SMPL_DIR")
        os.environ["FAB_SMPL_DIR"] = str(cls.dir)
        SI._MODELS.clear()
        cls.SI = SI

    @classmethod
    def tearDownClass(cls):
        if cls.env is None:
            os.environ.pop("FAB_SMPL_DIR", None)
        else:
            os.environ["FAB_SMPL_DIR"] = cls.env
        cls.SI._MODELS.clear()
        cls.tmp.cleanup()

    def ref(self, P, B, tr, gender="NEUTRAL"):
        """bench.body_models.SMPL directly: vertices and J_regressor joints."""
        import torch

        from bench.body_models import SMPL

        m = SMPL(self.dir / f"SMPL_{gender}.pkl").double()
        t = lambda x: torch.as_tensor(np.broadcast_to(x, (len(P), x.shape[-1])).copy(), dtype=torch.float64)  # noqa: E731
        with torch.no_grad():
            V = m(t(P[:, :3]), t(P[:, 3:]), t(B), t(tr))[0].numpy()
        return V, np.einsum("jv,tvc->tjc", m.J_regressor.numpy(), V)

    def motion(self, T=9, seed=3):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.3, (T, 72)), rng.normal(0, 1, 10), rng.normal(0, 0.5, (T, 3))

    def save(self, name, **kw):
        f = self.dir / name
        np.savez(f, **kw)
        return f

    def test_params_poses(self):
        P, B, tr = self.motion()
        f = self.save("p.npz", poses=P, betas=B, trans=tr, gender=np.array("male"), mocap_framerate=120.0)
        m = self.SI.load(f)
        V, J = self.ref(P, B, tr, "MALE")
        np.testing.assert_allclose(m["verts"], V, atol=1e-10)
        np.testing.assert_allclose(m["joints"], J, atol=1e-10)
        self.assertEqual(m["fps"], 120.0)
        np.testing.assert_array_equal(m["frames"], np.arange(len(P)))
        self.assertEqual((m["info"]["kind"], m["info"]["gender"]), ("params", "male"))
        self.assertTrue(m["info"]["model"].endswith("SMPL_MALE.pkl"))

    def test_global_orient_body_pose_is_poses(self):
        P, B, tr = self.motion()
        a = self.SI.load(self.save("a.npz", poses=P, betas=B, trans=tr, fps=30.0))
        b = self.SI.load(self.save("b.npz", global_orient=P[:, :3], body_pose=P[:, 3:], betas=B, trans=tr, fps=30.0, gender=b"neutral"))
        np.testing.assert_array_equal(a["verts"], b["verts"])
        np.testing.assert_allclose(a["verts"], self.ref(P, B, tr)[0], atol=1e-10)

    def test_amass_smplh(self):
        P, B, tr = self.motion()
        rng = np.random.default_rng(5)
        H = np.concatenate([P[:, :66], rng.normal(0, 0.5, (len(P), 90))], 1)  # SMPL-H: root + 21 body joints + 2 x 15 hand joints
        B16 = np.r_[B, rng.normal(0, 1, 6)]
        m = self.SI.load(self.save("amass.npz", poses=H, betas=B16, trans=tr, gender=np.array("female"), mocap_framerate=60.0,
                                   dmpls=np.zeros((len(P), 8))))
        P72 = np.concatenate([P[:, :66], np.zeros((len(P), 6))], 1)
        np.testing.assert_allclose(m["verts"], self.ref(P72, B, tr, "FEMALE")[0], atol=1e-10)
        self.assertIn("SMPL-H", m["info"]["poses"])
        self.assertEqual(m["info"]["betas"], "16 given, the first 10 used")
        bp63 = self.SI.load(self.save("bp63.npz", global_orient=P[:, :3], body_pose=P[:, 3:66], betas=B16, trans=tr, fps=60.0,
                                      gender="female"))
        np.testing.assert_array_equal(bp63["verts"], m["verts"])

    def test_betas_default_and_per_frame(self):
        P, B, tr = self.motion()
        m = self.SI.load(self.save("nob.npz", poses=P, trans=tr, fps=30.0))
        np.testing.assert_allclose(m["verts"], self.ref(P, np.zeros(10), tr)[0], atol=1e-10)
        short = self.SI.load(self.save("b5.npz", poses=P, betas=B[:5], trans=tr, fps=30.0))
        np.testing.assert_allclose(short["verts"], self.ref(P, np.r_[B[:5], np.zeros(5)], tr)[0], atol=1e-10)
        per = self.SI.load(self.save("bt.npz", poses=P, betas=np.tile(B, (len(P), 1)), trans=tr, fps=30.0))
        np.testing.assert_allclose(per["verts"], self.ref(P, B, tr)[0], atol=1e-10)

    def test_vertices(self):
        V = self.ref(*self.motion())[0]
        m = self.SI.load(self.save("v.npz", verts=V.astype(np.float32), fps=25.0, frames=np.arange(len(V)) + 100))
        np.testing.assert_array_equal(m["verts"], V.astype(np.float32))
        Jr = self.SI._model("neutral")[0].J_regressor.numpy()
        np.testing.assert_allclose(m["joints"], np.einsum("jv,tvc->tjc", Jr, m["verts"]), atol=1e-12)
        np.testing.assert_array_equal(m["frames"], np.arange(len(V)) + 100)
        self.assertEqual((m["info"]["kind"], m["fps"]), ("verts", 25.0))

    def test_y_up(self):
        P, B, tr = self.motion()
        Vy = self.ref(P, B, tr)[0]  # SMPL's own frame is y-up: the template's head is at +y
        m = self.SI.load(self.save("y.npz", verts=Vy, fps=30.0), up="y")
        np.testing.assert_allclose(m["verts"], np.stack([Vy[..., 0], -Vy[..., 2], Vy[..., 1]], -1), atol=1e-12)
        rest = self.SI.load(self.save("rest.npz", poses=np.zeros((2, 72)), trans=np.zeros((2, 3)), fps=30.0), up="y")
        self.assertGreater(rest["joints"][0, 15, 2] - rest["joints"][0, 7, 2], 1.4)  # head above the ankle along z
        p = self.SI.load(self.save("py.npz", poses=P, betas=B, trans=tr, fps=30.0), up="y")
        np.testing.assert_allclose(p["verts"], m["verts"], atol=1e-10)

    def test_kind(self):
        K = self.SI.kind
        export = ["frames", "T_world_cam", "gravity_world", "K", "fps", "pose", "betas", "trans", "joints", "valid", "scale_fix",
                  "joint", "contacts", "notes"]
        recon = ["fps", "frames", "valid", "pose", "betas", "trans", "joints", "verts", "contacts", "T_world_cam"]
        for mode in ("points", "mesh", "skeleton"):
            self.assertIsNone(K(export, mode))  # fab-world output
            self.assertIsNone(K(recon, mode))  # a FAB reconstruction
            self.assertIsNone(K(["joints", "fps"], mode))  # skeleton file
            self.assertEqual(K(["poses", "trans", "betas", "fps"], mode), "params")
            self.assertEqual(K(["global_orient", "body_pose", "trans", "fps"], mode), "params")
            self.assertEqual(K(["verts", "fps"], mode), "verts")
        self.assertIsNone(K(["poses", "trans", "fps", "joints"], "skeleton"))  # skeleton mode: a file with joints is a skeleton file
        self.assertEqual(K(["poses", "trans", "fps", "joints"], "mesh"), "params")
        self.assertIsNone(K(["global_orient", "trans", "fps"], "mesh"))

    def test_errors(self):
        P, B, tr = self.motion()
        for name, kw in (("bad_pose", dict(poses=P[:, :60], trans=tr, fps=30.0)), ("no_fps", dict(poses=P, trans=tr)),
                         ("no_trans", dict(poses=P, fps=30.0)), ("gender", dict(poses=P, trans=tr, fps=30.0, gender="other")),
                         ("bad_verts", dict(verts=np.zeros((3, 6000, 3)), fps=30.0))):
            with self.assertRaises(AssertionError, msg=name):
                self.SI.load(self.save(f"{name}.npz", **kw))

    def test_smpl_inputs_flat_floor(self):
        P, B, tr = self.motion()
        tr[:, 2] += 0.5
        f = self.save("walk.npz", poses=P, betas=B, trans=tr, fps=30.0)
        m = self.SI.load(f)
        rd, info = fab_retarget().smpl_inputs(f, "walk", self.dir / "out_flat")
        z, h = np.load(rd / "walk.npz"), np.load(rd / "walk_hfield.npz")
        foot = np.flatnonzero(np.isin(np.arange(6890) % 24, (7, 10, 8, 11)))  # the stand-in's foot parts
        self.assertAlmostEqual(float(z["verts"][:, foot, 2].min()), 0.0, places=6)
        dz = -m["verts"][:, foot, 2].min()
        self.assertAlmostEqual(info["ground_dz_m"], dz, places=12)
        np.testing.assert_allclose(z["verts"], m["verts"] + [0, 0, dz], atol=1e-5)
        np.testing.assert_allclose(z["joints"], m["joints"] + [0, 0, dz], atol=1e-5)
        self.assertEqual((z["verts"].dtype, z["joints"].shape, float(z["fps"]), str(z["contacts"])), (np.float32, (9, 24, 3), 30.0, "{}"))
        self.assertTrue(z["valid"].all())
        self.assertTrue((h["height"] == 0).all())
        self.assertLessEqual(float(h["x0"]), z["joints"][:, 0, 0].min() - 2.0 + 1e-6)
        self.assertEqual(info["world"], "flat floor at z = 0")

    def test_smpl_inputs_given_world(self):
        P, B, tr = self.motion()
        f = self.save("stairs.npz", poses=P, betas=B, trans=tr, fps=30.0)
        hf = dict(x0=-3.0, y0=-3.0, res=0.02, height=np.full((300, 300), 0.15, np.float32))
        np.savez(self.dir / "stairs_hfield.npz", **hf)
        rd, info = fab_retarget().smpl_inputs(f, "stairs", self.dir / "out_given")
        z, h = np.load(rd / "stairs.npz"), np.load(rd / "stairs_hfield.npz")
        np.testing.assert_array_equal(z["verts"], self.SI.load(f)["verts"].astype(np.float32))  # as given
        for k, v in hf.items():
            np.testing.assert_array_equal(h[k], v)
        self.assertEqual(info["world"], str(self.dir / "stairs_hfield.npz"))

    def test_chain_on_an_smpl_file(self):
        """scene() on an SMPL file: the recon under <out>/smpl_recon, a given world (no level snap, no clean), in mesh and
        skeleton modes; --up on another input and --ground_aware on an SMPL file stop."""
        from fab.retarget import release_config as C

        P, B, tr = self.motion()
        f = self.save("dance.npz", poses=P, betas=B, trans=tr, fps=60.0)
        for mode in ("mesh", "skeleton"):
            F = fab_retarget()
            cmds = []

            def run(cmd, env=None):
                cmd = [str(x) for x in cmd]
                cmds.append(cmd)
                o = Path(cmd[cmd.index("--out") + 1]) if "--out" in cmd else Path(cmd[cmd.index("--run") + 1]) / cmd[cmd.index("--out_tag") + 1]
                o.mkdir(parents=True, exist_ok=True)
                np.savez(o / "dance_motion.npz", scale=1.0)

            F.run = run
            sys.path.insert(0, str(ROOT))
            out = self.dir / f"out_chain_{mode}"
            fo, info = F.scene(Namespace(input=str(f), mode=mode, no_graph=False, no_clean=False, ground_aware=False, up="z"), "dance", out)
            sk = ["--mode", "skeleton"] if mode == "skeleton" else []
            cfg = C.solve_cfg()
            self.assertEqual(cmds[0], [sys.executable, "-m", "bench.videomimic.scene_retarget", "--recon", str(out / "smpl_recon"), "--only",
                                       "dance", "--out", str(out / "g" / "solve_pre")] + sk + ["--sigma_in", "8.0", "--cfg", json.dumps(cfg)])
            self.assertEqual(len(cmds), 2)
            self.assertEqual(info["clean"], "off (a given world)")
            self.assertEqual(info["smpl_input"]["kind"], "params")
            self.assertEqual(fo, out / "g" / "p_pre" / "dance_motion.npz")
        F = fab_retarget()
        with self.assertRaises(SystemExit):
            F.scene(Namespace(input=str(f), mode="mesh", no_graph=False, no_clean=False, ground_aware=True, up="z"), "dance", self.dir / "o_ga")
        sk = self.save("sk.npz", joints=np.zeros((5, 24, 3)), fps=30.0)
        with self.assertRaises(SystemExit):
            F.scene(Namespace(input=str(sk), mode="skeleton", no_graph=False, no_clean=False, ground_aware=False, up="y"), "sk", self.dir / "o_up")


if __name__ == "__main__":
    unittest.main()
