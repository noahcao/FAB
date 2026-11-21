"""fab-retarget end to end on a synthetic standalone SMPL file (a 2 s knee-bend in place), without FAB weights: the default
(mesh) and skeleton modes, and points mode stopping with its message. Skipped without a CUDA GPU, MuJoCo, the G1 description
(scripts/download_g1.sh) and the SMPL body models; set FAB_RETARGET_E2E=1 to run it (a few minutes)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ready():
    try:
        import mujoco  # noqa: F401
        import torch

        from fab import paths
        from fab.retarget import g1_files
    except ImportError:
        return False
    return torch.cuda.is_available() and g1_files.urdf().exists() and (paths.smpl_dir() / "SMPL_NEUTRAL.pkl").exists()


def knee_bends(T=60, fps=30.0):
    """SMPL parameters, z up: standing, arms down, two slow knee bends (hips / knees / ankles), no translation."""
    t = np.arange(T) / fps
    a = 0.25 * (1 - np.cos(2 * np.pi * t / (T / fps / 2)))
    P = np.zeros((T, 72))
    P[:, :3] = [np.pi / 2, 0, 0]  # SMPL's y up -> z up
    for j in (1, 2):
        P[:, 3 * j] = -a  # hips flex
    for j in (4, 5):
        P[:, 3 * j] = 2 * a  # knees
    for j in (7, 8):
        P[:, 3 * j] = -a  # ankles
    P[:, 3 * 16 + 2], P[:, 3 * 17 + 2] = -1.2, 1.2  # arms down
    return P, fps


class RetargetSmplEndToEnd(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FAB_RETARGET_E2E") and ready(), "FAB_RETARGET_E2E unset or prerequisites missing")
    def test_synthetic_smpl_file(self):
        P, fps = knee_bends()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            f = d / "bend.npz"
            np.savez(f, poses=P, trans=np.zeros((len(P), 3)), betas=np.zeros(10), fps=fps)
            env = dict(os.environ, FAB_WEIGHTS=str(d / "no_weights"))  # no FAB weights
            for mode in (None, "skeleton"):
                out = d / f"out_{mode or 'default'}"
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fab_retarget.py"), str(f), "--out", str(out)]
                                   + (["--mode", mode] if mode else []), capture_output=True, text=True, env=env)
                self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
                z = np.load(out / "bend_g1.npz", allow_pickle=True)
                info = json.loads((out / "bend_g1.json").read_text())
                self.assertEqual(info["mode"], mode or "mesh")
                self.assertEqual(info["smpl_input"]["kind"], "params")
                self.assertEqual(info["clean"], "off (a given world)")
                self.assertEqual(z["g1_dof"].shape, (int(np.floor((len(P) - 1) / fps * 50.0)) + 1, 29))
                self.assertTrue(np.isfinite(z["g1_dof"]).all() and np.isfinite(z["g1_root_pos"]).all())
                gate = json.loads(str(z["pin_report"]))["gate_after"]
                self.assertLessEqual(gate["feet_pen"], 0.012)  # standing on the floor: no foot below it
                lab = json.loads(str(z["contacts"]))
                self.assertGreater(np.mean(np.asarray(lab["lfoot"]) | np.asarray(lab["rfoot"])), 0.5)  # feet planted
            if not (Path(os.environ.get("FAB_HOME", Path.home() / ".cache" / "fab")) / "extras" / "dense_corr_g1.npz").exists():
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fab_retarget.py"), str(f), "--out", str(d / "out_points"),
                                    "--mode", "points"], capture_output=True, text=True, env=env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("--mode mesh", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
