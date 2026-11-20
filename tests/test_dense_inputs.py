"""fab-retarget points-mode inputs (fab/retarget/dense_inputs.py): the similarity fit, the fallback causes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fab.retarget.dense_inputs import points_unavailable, umeyama


class DenseInputsTest(unittest.TestCase):
    def test_umeyama_recovers_a_similarity(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 3))
        a = 0.7
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        s, R2, t = umeyama(X, 1.3 * X @ R.T + np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(s, 1.3, places=9)
        np.testing.assert_allclose(R2, R, atol=1e-9)
        np.testing.assert_allclose(t, [1.0, 2.0, 3.0], atol=1e-9)

    def test_points_unavailable_names_each_cause(self):
        idx = np.arange(5)
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "c_raw.npz"
            self.assertIn("no FAB output", points_unavailable(raw, idx))
            np.savez(raw, frames=np.arange(3))
            self.assertIn("no pts / surface_idx", points_unavailable(raw, idx))
            np.savez(raw, pts=np.zeros((3, 1, 3)), surface_idx=np.zeros(0, np.int64))
            self.assertIn("without the dense head", points_unavailable(raw, idx))
            np.savez(raw, pts=np.zeros((3, 5, 3)), surface_idx=idx[::-1])
            self.assertIn("surface_idx mismatch", points_unavailable(raw, idx))
            np.savez(raw, pts=np.zeros((3, 5, 3)), surface_idx=idx)
            self.assertIsNone(points_unavailable(raw, idx))


if __name__ == "__main__":
    unittest.main()
