"""Inference regression on the shipped demo images (assets/demo/joggers.jpg, skater_seated.jpg): FABPredictor's eager path
(fast=False, its default detector) with FAB-L, FAB-S and FAB-H gives the outputs stored for this device class, within the
tolerances below. Needs the weights, SAM 3D Body's config with the MHR model and the SMPL body models (README, Install);
skipped otherwise.
One reference per device class, because the two classes run different precisions:
- On CUDA, the body network runs under BF16 autocast (FABPredictor.predict, the released precision): tests/data/demo_reference.npz.
- On the CPU, the same code runs in FP32 (CUDA autocast does not apply there): tests/data/demo_reference_cpu.npz.
A CPU run is never compared with the CUDA reference. On the demo images they differ by up to 7.5 cm in camera-frame keypoints
(FAB-H: 12 cm), which is the BF16 / FP32 difference, not a regression.
BF16 also rounds differently on different GPU types, so on CUDA the default is a coarse check against the shipped CUDA
reference: the same people, cam_t within 1 % of the person's depth and keypoints3d within 10 cm (CROSS_GPU), which catches
real breaks (two GPU types differed by up to 6 cm in cam_t / keypoints3d for the person 14 m away, 0.4 %). The comparison with
TOL on CUDA is opt-in, against a reference written on your own GPU:
    FAB_WRITE_REFERENCE=1 FAB_DEMO_REFERENCE=my_reference.npz python -m pytest tests/test_demo_regression.py   # once
    FAB_DEMO_REFERENCE=my_reference.npz python -m pytest tests/test_demo_regression.py                         # then
smpl_pose is compared as rotation matrices: axis-angle wraps by 2 pi near a half turn.
Rewrite a shipped reference (maintainers, on the release weights, on that device class):
    FAB_WRITE_REFERENCE=1 python -m pytest tests/test_demo_regression.py                          # CUDA
    FAB_WRITE_REFERENCE=1 CUDA_VISIBLE_DEVICES= python -m pytest tests/test_demo_regression.py    # CPU"""
import os
import platform
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REFS = {"cuda": REPO / "tests" / "data" / "demo_reference.npz", "cpu": REPO / "tests" / "data" / "demo_reference_cpu.npz"}
IMAGES = ("joggers.jpg", "skater_seated.jpg")
MODELS = ("fab-l", "fab-s", "fab-h")
# key -> absolute tolerance (units of the output: px, m, rotation-matrix entries, probability, rad)
TOL = {"box": 2.0, "keypoints2d": 2.0, "keypoints3d": 0.005, "cam_t": 0.02, "smpl_pose": 0.03, "smpl_betas": 0.05,
       "contact": 0.05, "g1_q": 0.03}
# CPU class, widened only where the measured spread needs it. The spread is the max |diff| against the CPU reference
# (written at 192 threads) when the FP32 reduction order changes with the thread count. Measured on one machine (x86_64, torch
# 2.11.0) at 1 / 8 / 32 / 192 threads; worst key per model:
#   FAB-H: keypoints3d 1.15 / 6.07 / 1.80 / 0 mm (smpl_pose 0.015, smpl_betas 0.010, cam_t 0.96 mm, all within TOL);
#   FAB-L: keypoints2d 0.07 px, keypoints3d <= 0.05 mm, smpl_pose <= 6e-4;
#   FAB-S: smpl_pose 2.0e-3 / 7.4e-3 / 5.4e-3 / 0, keypoints2d 0.12 px, keypoints3d <= 0.16 mm.
# So only FAB-H's keypoints3d gets a wider tolerance on the CPU: 10 mm (1.6x the worst measured 6.07 mm).
TOL_CPU = {"fab-h": {"keypoints3d": 0.01}}
# the coarse check (CUDA default): cam_t within this share of the reference depth, keypoints3d within this many m
CROSS_GPU = {"cam_t_rel": 0.01, "keypoints3d": 0.10}
WRITE = os.environ.get("FAB_WRITE_REFERENCE") == "1"
OWN_REF = os.environ.get("FAB_DEMO_REFERENCE", "")  # your own reference (the opt-in comparison with TOL on CUDA)


def have_inputs():
    try:
        from fab import paths
    except Exception:
        return False
    return all(paths.model_file(m).exists() for m in MODELS) and paths.shared("detector.pt").exists() and paths.sam3d_dir().exists() \
        and paths.smpl_dir().exists()


def rotmats(aa):
    """(..., 3k) axis-angle -> (k, 3, 3) rotation matrices (Rodrigues)."""
    v = np.asarray(aa, np.float64).reshape(-1, 3)
    th = np.linalg.norm(v, axis=1, keepdims=True)
    k = np.where(th > 1e-12, v / np.maximum(th, 1e-12), 0.0)
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * K @ K


def diff(key, a, b):
    if key == "smpl_pose":
        return float(np.abs(rotmats(a) - rotmats(b)).max())
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def run(model, device):
    import cv2

    from fab.predictor import FABPredictor

    fab = FABPredictor(model, fast=False, device=device)
    out = {}
    for name in IMAGES:
        img = cv2.cvtColor(cv2.imread(str(REPO / "assets" / "demo" / name)), cv2.COLOR_BGR2RGB)
        people = sorted(fab.predict(img), key=lambda q: float(np.asarray(q["box"])[0]))  # left to right
        out[f"{model}/{name}/n"] = np.array(len(people))
        for i, q in enumerate(people):
            for k in TOL:
                if k in q:
                    out[f"{model}/{name}/{i}/{k}"] = np.asarray(q[k], np.float32)
    return out


@unittest.skipUnless(have_inputs(), "needs the weights, SAM 3D Body's config + MHR and the SMPL body models")
class DemoRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.ref_file = REFS[cls.device]
        cls.got = {}
        for m in MODELS:
            cls.got.update(run(m, cls.device))
        if WRITE:
            dst = Path(OWN_REF) if OWN_REF else cls.ref_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, **cls.got, _device=np.array(f"{cls.device} {platform.machine()} torch {torch.__version__}"))
            cls.written = dst
        cls.ref = dict(np.load(cls.ref_file)) if cls.ref_file.exists() else None

    def common(self, ref):
        if WRITE:
            self.skipTest(f"reference written: {self.written} ({len(self.got)} arrays, {self.device})")
        keys = sorted(k for k in ref if not k.startswith("_"))
        self.assertEqual(sorted(self.got), keys, "output set changed (people count or keys)")
        counts = [f"{k}: {int(self.got[k])} people, reference {int(ref[k])}" for k in keys
                  if k.endswith("/n") and int(self.got[k]) != int(ref[k])]
        self.assertFalse(counts, "\n".join(counts))
        return [k for k in keys if not k.endswith("/n")]

    def test_demo_outputs_match_the_reference(self):
        """The comparison with TOL: on the CPU against the shipped CPU reference; on CUDA only against FAB_DEMO_REFERENCE."""
        if OWN_REF:
            self.assertTrue(Path(OWN_REF).exists(), f"{OWN_REF} missing: write it with FAB_WRITE_REFERENCE=1 FAB_DEMO_REFERENCE={OWN_REF}")
            ref = dict(np.load(OWN_REF))
        elif self.device == "cuda":
            self.skipTest("on CUDA the comparison with TOL needs a reference written on this GPU (FAB_DEMO_REFERENCE, see the "
                          "docstring); test_demo_outputs_near_the_reference ran against the shipped one")
        else:
            self.assertIsNotNone(self.ref, f"{self.ref_file} missing")
            ref = self.ref
        over = []
        for k in self.common(ref):
            model, name = k.split("/")[0], k.rsplit("/", 1)[1]
            tol = TOL_CPU.get(model, {}).get(name, TOL[name]) if self.device == "cpu" else TOL[name]
            err = diff(name, self.got[k], ref[k])
            if err > tol:
                over.append(f"{k}: max abs difference {err:.4g} > {tol}")
        self.assertFalse(over, f"{len(over)} outputs over tolerance (reference: {ref['_device']}, now {self.device}):\n" + "\n".join(over))

    def test_demo_outputs_near_the_reference(self):
        """The coarse check against the shipped reference of this device class: the same people, cam_t within 1 % of the
        depth, keypoints3d within 10 cm."""
        self.assertIsNotNone(self.ref, f"{self.ref_file} missing")
        over = []
        for k in self.common(self.ref):
            name = k.rsplit("/", 1)[1]
            if name == "cam_t":
                ref = np.asarray(self.ref[k], np.float64).reshape(3)
                err = float(np.abs(np.asarray(self.got[k], np.float64).reshape(3) - ref).max())
                if err > CROSS_GPU["cam_t_rel"] * abs(ref[2]):
                    over.append(f"{k}: {err:.4g} m > {CROSS_GPU['cam_t_rel']:.0%} of the depth {abs(ref[2]):.2f} m")
            elif name == "keypoints3d":
                err = diff(name, self.got[k], self.ref[k])
                if err > CROSS_GPU["keypoints3d"]:
                    over.append(f"{k}: max abs difference {err:.4g} > {CROSS_GPU['keypoints3d']} m")
        self.assertFalse(over, f"{len(over)} outputs beyond the coarse bound (reference: {self.ref['_device']}, now {self.device}):\n"
                         + "\n".join(over))


if __name__ == "__main__":
    unittest.main()
