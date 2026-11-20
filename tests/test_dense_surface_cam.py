"""`dense_surface_cam`: the dense head's surface points in the camera frame (FABPredictor, eager), on the shipped demo images, for
FAB-L, FAB-S and FAB-H's points profile: each person's points lie on `mesh_smpl` (the mesh they are anchored to) at the same SMPL
vertices (rms within RMS_MAX) and have the shape of `mesh_smpl_fit` (rms after removing each centroid within FIT_SHAPE_MAX; the fit
is placed by its own 2D step, so its depth can differ from `mesh_smpl`'s by ~1 % of the distance), and `dense_surface` (the head's
raw, hip-aligned points) plus `cam_t` does not. Needs the weights, SAM 3D Body's config with the MHR model and the SMPL body models
(README, Install); skipped otherwise."""
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
IMAGES = ("joggers.jpg", "skater_seated.jpg")
RUNS = (("fab-l", {}), ("fab-s", {}), ("fab-h", {"profile": "points"}))
RMS_MAX = 0.08  # m: points vs mesh_smpl at the same vertices
FIT_SHAPE_MAX = 0.06  # m: points vs mesh_smpl_fit, each centred


def have_inputs():
    try:
        from fab import paths
    except Exception:
        return False
    return all(paths.model_file(m).exists() for m, _ in RUNS) and paths.shared("detector.pt").exists() and paths.sam3d_dir().exists() \
        and paths.smpl_dir().exists()


@unittest.skipUnless(have_inputs(), "needs the weights, SAM 3D Body's config + MHR and the SMPL body models")
class DenseSurfaceCamTests(unittest.TestCase):
    def test_points_on_the_fitted_mesh(self):
        import cv2
        import torch

        from fab.predictor import FABPredictor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for model, kw in RUNS:
            fab = FABPredictor(model, fast=False, device=device, **kw)
            sidx = fab.model.dense_head.surface_idx.cpu().numpy()
            for name in IMAGES:
                img = cv2.cvtColor(cv2.imread(str(REPO / "assets" / "demo" / name)), cv2.COLOR_BGR2RGB)
                people = fab.predict(img)
                self.assertTrue(people, (model, name))
                for i, q in enumerate(people):
                    self.assertIn("dense_surface_cam", q, (model, name))
                    D = np.asarray(q["dense_surface_cam"], np.float64)
                    V = np.asarray(q["mesh_smpl"], np.float64)[sidx]
                    rms = float(np.sqrt(((D - V) ** 2).sum(1).mean()))
                    self.assertLess(rms, RMS_MAX, f"{model} {name} person {i}: dense_surface_cam vs mesh_smpl rms {rms:.3f} m")
                    F = np.asarray(q["mesh_smpl_fit"], np.float64)[sidx]
                    shape = float(np.sqrt(((D - D.mean(0) - F + F.mean(0)) ** 2).sum(1).mean()))
                    self.assertLess(shape, FIT_SHAPE_MAX, f"{model} {name} person {i}: dense_surface_cam vs mesh_smpl_fit centred rms {shape:.3f} m")
                    gap = D.mean(0) - F.mean(0)
                    print(f"{model} {name} p{i}: rms {rms:.3f} m to mesh_smpl; to mesh_smpl_fit centred {shape:.3f} m, centroid gap {np.round(gap, 3)} m")
                    raw = np.asarray(q["dense_surface"], np.float64) + np.asarray(q["cam_t"], np.float64)
                    off = float(np.linalg.norm((raw - np.asarray(q["mesh_smpl"], np.float64)[sidx]).mean(0)))
                    print(f"{model} {name} p{i}: dense_surface + cam_t is {off:.2f} m off the mesh (raw, hip-aligned points)")


if __name__ == "__main__":
    unittest.main()
