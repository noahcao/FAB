"""fab.yolo: the person detector without the ultralytics package (as FAH's tests/test_yolo.py and test_letterbox.py).
Always: the anchor grid / box decode, the letterbox geometry, scale_boxes and the one-class NMS. With ultralytics installed:
letterbox() equals ultralytics' rect letterbox (the input the weights were trained on) on frame sizes of every aspect, and
scale_boxes() equals ultralytics' scale_boxes. With the release weights: the detector runs with ultralytics blocked from
import, and (with ultralytics) its boxes equal ultralytics' predictor's bit for bit, the documented check of
scripts/check_detector.py (the raw network outputs only within a float tolerance: see UltralyticsParityTests)."""
import importlib.util
import os
import subprocess
import sys
import unittest

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_UL = bool(importlib.util.find_spec("ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # before any ultralytics import: it never pip-installs


def detector_weights():
    from fab import paths

    return str(paths.shared("detector.pt"))


DET = detector_weights()
HAVE = os.path.exists(DET)


def sizes():
    rng = np.random.default_rng(0)
    return [(1080, 1920), (1080, 1620), (1080, 1440), (1080, 1080), (612, 1134), (1920, 1080), (480, 640), (1456, 1920),
            (1408, 1408), (721, 1279), (720, 1280), (376, 672)] + [tuple(int(v) for v in rng.integers(64, 2200, 2)) for _ in range(60)]


class AnchorTests(unittest.TestCase):
    def test_anchor_grid_and_box_decode(self):
        from fab.yolo import dist2box, make_anchors

        feats = [torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 1, 2)]
        a, s = make_anchors(feats, torch.tensor([8.0, 16.0]))
        self.assertEqual(a.tolist(), [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [0.5, 1.5], [1.5, 1.5], [2.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
        self.assertEqual(s[:, 0].tolist(), [8.0] * 6 + [16.0] * 2)
        box = dist2box(torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]), torch.tensor([[[10.0], [20.0]]]))
        self.assertEqual(box[0, :, 0].tolist(), [11.0, 21.0, 4.0, 6.0])  # cx, cy, w, h


class LetterboxGeometryTests(unittest.TestCase):
    def test_shape_and_padding(self):
        from fab.yolo import letterbox, letterbox_shape

        rng = np.random.default_rng(1)
        for h, w in sizes()[:20]:
            h2, w2, nh, nw = letterbox_shape(h, w)
            self.assertEqual((h2 % 32, w2 % 32), (0, 0), (h, w))
            self.assertEqual(max(nh, nw), 640, (h, w))
            self.assertLess(abs(nh / nw - h / w), 2.0 / min(nh, nw) + 1e-9, (h, w))
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            x = letterbox(img)
            self.assertEqual(tuple(x.shape), (1, 3, h2, w2))
            self.assertTrue(float(x.min()) >= 0.0 and float(x.max()) <= 1.0)
            if h2 > nh:  # the padding rows are 114 / 255
                self.assertTrue(torch.allclose(x[0, :, 0], torch.full_like(x[0, :, 0], 114 / 255.0)))

    def test_scale_boxes_inverts_the_letterbox(self):
        from fab.yolo import letterbox_shape, scale_boxes

        for h, w in sizes()[:20]:
            h2, w2, nh, nw = letterbox_shape(h, w)
            top, left = round((h2 - nh) / 2 - 0.1), round((w2 - nw) / 2 - 0.1)
            full = torch.tensor([[left, top, left + nw, top + nh]], dtype=torch.float32)  # the image's own extent
            got = scale_boxes(full.clone(), (h2, w2), (h, w))
            self.assertTrue(torch.allclose(got, torch.tensor([[0.0, 0.0, w, h]]), atol=1e-3), (h, w, got))

    def test_nms_one_class(self):
        from fab.yolo import nms_one_class

        pred = torch.tensor([[10.0, 10.5, 50.0], [10.0, 10.0, 50.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [0.9, 0.8, 0.1]])
        out = nms_one_class(pred)  # two overlapping candidates -> the better one; the third is below conf
        self.assertEqual(out.shape, (1, 5))
        self.assertEqual(out[0].tolist(), [8.0, 8.0, 12.0, 12.0, 0.8999999761581421])
        with self.assertRaises(ValueError):
            nms_one_class(torch.zeros(6, 3))


@unittest.skipUnless(HAVE_UL and importlib.util.find_spec("cv2"), "needs ultralytics")
class UltralyticsLetterboxTests(unittest.TestCase):
    def test_matches_ultralytics(self):
        from ultralytics.data.augment import LetterBox

        from fab.yolo import letterbox

        rng = np.random.default_rng(0)
        for h, w in sizes():
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            ref = LetterBox(new_shape=(640, 640), auto=True, stride=32)(image=img)  # HWC uint8, BGR order kept
            got = letterbox(img)[0].permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
            self.assertEqual(got.shape, ref.shape, (h, w))
            self.assertTrue(np.array_equal(got, ref), (h, w))

    def test_scale_boxes_matches_ultralytics(self):
        from ultralytics.utils.ops import scale_boxes as ul_scale

        from fab.yolo import letterbox_shape, scale_boxes

        rng = np.random.default_rng(2)
        for h, w in sizes()[:30]:
            h2, w2, _, _ = letterbox_shape(h, w)
            b = torch.from_numpy(rng.random((5, 4), np.float32) * np.float32([w2, h2, w2, h2]))
            b[:, 2:] = torch.maximum(b[:, 2:], b[:, :2] + 1)
            ref = ul_scale((h2, w2), b.clone(), (h, w))
            got = scale_boxes(b.clone(), (h2, w2), (h, w))
            self.assertTrue(torch.allclose(got, ref, atol=1e-4), (h, w))


@unittest.skipUnless(HAVE, "needs the weights")
class NoUltralyticsTests(unittest.TestCase):
    def test_detector_runs_with_ultralytics_blocked(self):
        code = f"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("ultralytics blocked")
sys.meta_path.insert(0, Block())
import numpy as np
from fab.yolo import YOLODetector
d = YOLODetector({DET!r}, device="cpu")
b = d.detect(np.full((480, 640, 3), 128, np.uint8))
print("ok", "ultralytics" in sys.modules, b.shape[1])
"""
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES=""))
        self.assertIn("ok False 4", r.stdout, r.stderr[-2000:])


@unittest.skipUnless(HAVE and HAVE_UL, "needs the weights and ultralytics")
class UltralyticsParityTests(unittest.TestCase):
    """The documented check (scripts/check_detector.py): with the same checkpoint, confidence, NMS IoU and 640 px input,
    fab.yolo's boxes equal those of ultralytics' own predictor (YOLO.predict: its letterbox, NMS and scale_boxes)
    bit for bit, on the demo images at three scales, on the CPU (and the GPU when there is one).
    The raw network outputs (1, 5, A) are compared with a tolerance only: the two networks compute the same
    float32 convolutions with different kernels and weight layouts. On the CPU that is a ~1e-5 relative difference
    (|diff| up to ~0.02 on box values up to ~640), and it does not change the boxes after the decode and NMS."""

    def test_boxes_equal_ultralytics_predictor(self):
        import cv2
        from ultralytics import YOLO

        from fab.yolo import PREDICT_IOU, YOLODetector, letterbox

        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        images = [cv2.imread(os.path.join(REPO, "assets", "demo", f)) for f in ("joggers.jpg", "skater_seated.jpg")]
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            host = YOLO(DET)
            fab = YOLODetector(DET, device=dev)
            host.predict(images[0], classes=[0], conf=0.25, imgsz=640, device=dev, verbose=False)
            pr = host.predictor
            self.assertEqual(pr.args.iou, PREDICT_IOU)
            n = 0
            for bgr in images:
                for s in (1.0, 0.61, 1.37):
                    img = bgr if s == 1 else cv2.resize(bgr, (max(round(bgr.shape[1] * s), 1), max(round(bgr.shape[0] * s), 1)),
                                                        interpolation=cv2.INTER_AREA)
                    rgb = np.ascontiguousarray(img[..., ::-1])
                    ref = host.predict(img, classes=[0], conf=0.25, imgsz=640, device=dev, verbose=False)[0].boxes.xyxy.float().cpu()
                    got = torch.as_tensor(np.asarray(fab.detect(rgb)), dtype=torch.float32)
                    self.assertEqual(tuple(got.shape), tuple(ref.shape), (dev, s, img.shape))
                    self.assertTrue(torch.equal(got, ref), (dev, s, img.shape, float((got - ref).abs().max()) if len(ref) else 0))
                    n += len(ref)
                    with torch.no_grad():
                        im = letterbox(rgb, 640, fab.stride, fab.device)
                        a = pr.inference(im)
                        a = (a[0] if isinstance(a, (list, tuple)) else a).float()
                        b = fab.net(im)
                        b = b[0] if isinstance(b, (list, tuple)) else b
                    self.assertTrue(torch.allclose(a, b, rtol=1e-4, atol=0.05), (dev, s, float((a - b).abs().max())))
            self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
