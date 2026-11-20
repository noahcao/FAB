"""Training smoke test (scripts/train_smoke.py): skipped without a CUDA GPU, the SAM 3D Body config + MHR model and the
SMPL files; set FAB_TRAIN_SMOKE=1 to run it (a few minutes)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ready():
    try:
        import torch

        from fab import paths
    except ImportError:
        return False
    return (torch.cuda.is_available() and (paths.sam3d_dir() / "model_config.yaml").exists()
            and (paths.smpl_dir() / "SMPL_NEUTRAL.pkl").exists())


class TrainSmokeTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FAB_TRAIN_SMOKE") and ready(), "FAB_TRAIN_SMOKE unset or prerequisites missing")
    def test_body_head_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(ROOT / "scripts/train_smoke.py"), "--out", d], capture_output=True,
                               text=True, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
            self.assertIn("train_smoke OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
