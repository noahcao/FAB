"""The TensorRT person-detector engine is used only when it can run: FABPredictor checks, before ultralytics loads the engine,
that tensorrt imports at the version fab-build-trt recorded for it (<engines dir>/detector.engine.json). Otherwise it warns and
uses the PyTorch detector (fab.yolo, the same boxes), and nothing is installed (YOLO_AUTOINSTALL=False: ultralytics would
otherwise pip-install another tensorrt, which cannot read the engine).
The check itself needs no weights; the predictor test needs the weights, CUDA and ultralytics (skipped otherwise)."""
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

BUILT = "11.4.0.106"


@contextlib.contextmanager
def tensorrt_as(mod):
    """sys.modules["tensorrt"] = mod (None: the import fails) for the block; only that key is restored (mock.patch.dict on
    sys.modules would also drop every module imported inside the block, e.g. torchvision, which cannot be imported twice)."""
    had, old = "tensorrt" in sys.modules, sys.modules.get("tensorrt")
    sys.modules["tensorrt"] = mod
    try:
        yield
    finally:
        if had:
            sys.modules["tensorrt"] = old
        else:
            del sys.modules["tensorrt"]


def engine_dir(tmp, record=True):
    d = Path(tmp)
    (d / "detector.engine").write_bytes(b"not a real engine")
    if record:
        (d / "detector.engine.json").write_text(json.dumps({"tensorrt": BUILT}))
    return d


class EngineCheckTests(unittest.TestCase):
    def test_reasons(self):
        from fab.predictor import detector_engine_unusable

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no build record", detector_engine_unusable(engine_dir(tmp, record=False) / "detector.engine"))
        with tempfile.TemporaryDirectory() as tmp:
            e = engine_dir(tmp) / "detector.engine"
            with tensorrt_as(None):
                self.assertIn("does not import", detector_engine_unusable(e))
            with tensorrt_as(types.SimpleNamespace(__version__="11.5.0.63")):
                self.assertIn("built with " + BUILT, detector_engine_unusable(e))
            with tensorrt_as(types.SimpleNamespace(__version__=BUILT)):
                self.assertIsNone(detector_engine_unusable(e))

    def test_autoinstall_off(self):
        import fab.predictor  # noqa: F401

        if "YOLO_AUTOINSTALL" in os.environ:
            self.assertIn(os.environ["YOLO_AUTOINSTALL"].lower(), ("false", "0"))


def have_inputs():
    try:
        import torch

        from fab import paths
    except Exception:
        return False
    return torch.cuda.is_available() and importlib.util.find_spec("ultralytics") is not None and paths.model_file("fab-l").exists() \
        and paths.shared("detector.pt").exists() and paths.sam3d_dir().exists()


@unittest.skipUnless(have_inputs(), "needs CUDA, ultralytics, the weights and SAM 3D Body's config")
class PredictorFallbackTests(unittest.TestCase):
    def test_engine_without_tensorrt_falls_back_to_fab_yolo(self):
        import fab.predictor as P

        real = subprocess.Popen

        def no_pip(args, *a, **kw):
            cmd = " ".join(args) if isinstance(args, (list, tuple)) else str(args)
            if "pip" in cmd:
                raise AssertionError(f"tried to install: {cmd}")
            return real(args, *a, **kw)

        with tempfile.TemporaryDirectory() as tmp:
            d = engine_dir(tmp)
            with mock.patch.object(P.paths, "engines_dir", lambda device=None: d), \
                    tensorrt_as(None), mock.patch.object(subprocess, "Popen", no_pip), \
                    warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fab = P.FABPredictor("fab-l", fast=True, device="cuda")
        self.assertEqual(fab.detector_backend, "fab")
        self.assertEqual(type(fab.detector).__name__, "YOLODetector")
        self.assertTrue(any("detector engine" in str(x.message) and "not used" in str(x.message) for x in w), [str(x.message) for x in w])


if __name__ == "__main__":
    unittest.main()
