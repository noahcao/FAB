"""fab-retarget --mode skeleton: a skeleton file's inputs (scripts/fab_retarget.py) and the joint contact rule
(fab/retarget/skeleton.py, on a stand-in template)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fab.retarget import skeleton as S

ROOT = Path(__file__).resolve().parents[1]


def fab_retarget():
    spec = importlib.util.spec_from_file_location("fab_retarget_script", ROOT / "scripts" / "fab_retarget.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def flat(xy, z=None):
    return np.zeros(xy.shape[:-1])


def standing(T=30):
    """A static stand-in skeleton (z up): ankles 8 cm and toe bases 2 cm above z = 0, hip -> knee -> ankle 0.4 / 0.4 m."""
    J = np.zeros((T, 24, 3))
    for side, y in ((0, 0.1), (1, -0.1)):
        J[:, 1 + side] = [0, y, 0.88]  # hip
        J[:, 4 + side] = [0, y, 0.48]  # knee
        J[:, 7 + side] = [0, y, 0.08]  # ankle
        J[:, 10 + side] = [0.1, y, 0.02]  # toe base
        J[:, 20 + side] = [0, y * 2, 0.9]  # wrist
        J[:, 22 + side] = [0, y * 2, 0.85]  # hand
    J[:, 0] = [0, 0, 0.9]
    return J


class SkeletonModeTest(unittest.TestCase):
    def test_skeleton_file_gets_a_flat_world(self):
        J = standing(20)
        J[:, :, 0] += np.linspace(0, 1, 20)[:, None]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "walk.npz"
            np.savez(f, joints=J, fps=30.0)
            rd, info = fab_retarget().skeleton_inputs(f, "walk", Path(d) / "out")
            z, h = np.load(rd / "walk.npz"), np.load(rd / "walk_hfield.npz")
            self.assertEqual(z["joints"].shape, (20, 24, 3))
            self.assertEqual(float(z["fps"]), 30.0)
            np.testing.assert_array_equal(z["frames"], np.arange(20))
            self.assertEqual(str(z["contacts"]), "{}")
            self.assertTrue((h["height"] == 0).all())
            self.assertLessEqual(float(h["x0"]), -2.0)
            self.assertGreaterEqual(float(h["x0"]) + float(h["res"]) * (h["height"].shape[1] - 1), 3.0)
            self.assertEqual(info["world"], "flat floor at z = 0")

    def test_skeleton_file_needs_24_joints(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "short.npz"
            np.savez(f, joints=np.zeros((5, 17, 3)), fps=30.0)
            with self.assertRaises(AssertionError):
                fab_retarget().skeleton_inputs(f, "short", Path(d) / "out")

    def test_foot_contact_rule(self):
        saved = dict(S._TEMPLATE)
        try:  # a stand-in template whose sizes match standing(): soles at z = 0, the same leg length (scale k = 1)
            r = 0.05
            S._TEMPLATE.clear()
            S._TEMPLATE.update(
                groups={"lfoot": [(7, 0.08), (10, 0.02)], "rfoot": [(8, 0.08), (11, 0.02)], "lhand": [(20, r), (22, r)],
                        "rhand": [(21, r), (23, r)], "lknee": [(4, r), (7, r)], "rknee": [(5, r), (8, r)],
                        "seat": [(0, 0.1), (1, r), (4, r), (2, r), (5, r)]},
                tips={"lfoot": (7, 10, 4, 0.0, 0.0, 0.02), "rfoot": (8, 11, 5, 0.0, 0.0, 0.02)}, leg=0.8 + np.hypot(0.1, 0.06))
            J = standing(30)
            J[20:, [4, 7, 10]] += [0, 0, 0.1]  # the left foot lifted 10 cm (knee, ankle, toe base) from frame 20
            out = S.skeleton_rules(J, flat, 30.0, 0.04, 0.15)
            self.assertAlmostEqual(out["info"]["k_scale"], 1.0)
            self.assertTrue(out["contacts"]["rfoot"].all())
            self.assertTrue(out["contacts"]["lfoot"][:20].all())
            self.assertFalse(out["contacts"]["lfoot"][21:].any())
            np.testing.assert_allclose(out["foot_low"]["rfoot"], 0.0, atol=1e-9)
            J2 = standing(30)
            J2[:, [8, 11], 0] += np.arange(30)[:, None] * 0.01  # the right foot sliding at 0.3 m/s: not a contact
            self.assertFalse(S.skeleton_rules(J2, flat, 30.0, 0.04, 0.15)["contacts"]["rfoot"][1:].any())
        finally:
            S._TEMPLATE.clear()
            S._TEMPLATE.update(saved)


if __name__ == "__main__":
    unittest.main()
