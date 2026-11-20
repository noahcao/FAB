"""The fab command (fab/cli.py), as FAH's tests/test_cli_json.py: its flags map to FABPredictor's arguments and calls, and its
output file holds exactly the predictor's outputs plus the documented fields (stub predictor: no weights, no GPU)."""
import contextlib
import io
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from fab import cli


def fake_person(seed):
    rng = np.random.default_rng(seed)
    return {"box": rng.random(4).astype(np.float32) * 40, "keypoints2d": rng.random((70, 2)).astype(np.float32),
            "keypoints3d": rng.normal(size=(70, 3)).astype(np.float32), "smpl_pose": rng.normal(size=72).astype(np.float32),
            "contact": rng.random(16).astype(np.float32), "cam_t": rng.normal(size=3).astype(np.float32)}


class StubPredictor:
    calls, inits = [], []

    def __init__(self, ck, **kwargs):
        StubPredictor.inits.append(dict(kwargs, ck=ck))
        self.profile = {"auto": "simple"}.get(kwargs.get("profile"), kwargs.get("profile"))
        self.smpl_fitter_path = "<fitter>"

    def predict(self, img, K=None, boxes=None):
        StubPredictor.calls.append({"shape": img.shape, "K": K, "boxes": boxes})
        n = 2 if boxes is None else len(boxes)
        return [fake_person(len(StubPredictor.calls) * 10 + k) for k in range(n)]


def write_images(d, n=1, shape=(48, 64)):
    names = []
    for k in range(n):
        f = os.path.join(d, f"img{k}.png")
        cv2.imwrite(f, np.full((*shape, 3), 100 + k, np.uint8))
        names.append(f)
    return names


def run_cli(*argv):
    StubPredictor.calls, StubPredictor.inits = [], []
    with patch("fab.predictor.FABPredictor", StubPredictor), contextlib.redirect_stdout(io.StringIO()):
        cli.main(list(argv))


class CliTests(unittest.TestCase):
    def test_image_defaults(self):
        d = tempfile.mkdtemp()
        img, = write_images(d)
        out = os.path.join(d, "out")
        run_cli(img, "--out", out)
        self.assertEqual(StubPredictor.inits, [{"ck": "fab-l", "fast": True, "profile": "auto"}])
        with open(os.path.join(out, "img0_fab.pkl"), "rb") as f:
            r = pickle.load(f)
        self.assertEqual(sorted(r), ["K", "K_used", "call", "fitter", "frames", "people", "profile", "source"])
        self.assertEqual(r["frames"], ["img0.png"])
        self.assertIsNone(r["K"])
        self.assertEqual(r["K_used"].shape, (1, 3, 3))
        self.assertEqual(r["call"], 'FABPredictor("fab-l", fast=True, profile="auto").predict(image, K)')
        self.assertEqual((r["profile"], r["fitter"]), ("simple", "<fitter>"))
        self.assertEqual(len(r["people"]), 1)
        self.assertEqual(len(r["people"][0]), 2)
        for got, ref in zip(r["people"][0], [fake_person(10), fake_person(11)]):  # exactly the predictor's outputs
            self.assertEqual(sorted(got), sorted(ref))
            for k in ref:
                self.assertTrue(np.array_equal(got[k], ref[k]), k)
        self.assertEqual(StubPredictor.calls[0]["shape"], (48, 64, 3))  # RGB image of the file's size

    def test_flags_map_to_the_predictor(self):
        d = tempfile.mkdtemp()
        img, = write_images(d)
        run_cli(img, "--out", os.path.join(d, "o"), "--model", "fab-h", "--eager", "--profile", "points", "--K", "100,110,32,24")
        self.assertEqual(StubPredictor.inits, [{"ck": "fab-h", "fast": False, "profile": "points"}])
        K = StubPredictor.calls[0]["K"]
        self.assertTrue(np.array_equal(K, np.array([[100, 0, 32], [0, 110, 24], [0, 0, 1]], np.float32)))
        with open(os.path.join(d, "o", "img0_fab.pkl"), "rb") as f:
            r = pickle.load(f)
        self.assertTrue(np.array_equal(r["K"], K) and np.array_equal(r["K_used"][0], K))
        self.assertEqual(r["call"], 'FABPredictor("fab-h", fast=False, profile="points").predict(image, K)')

    def test_directory_and_given_boxes(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "frames")
        os.makedirs(src)
        write_images(src, 3)
        boxes = np.full((2, 3, 4), np.nan, np.float32)
        boxes[0, 0] = [1, 2, 30, 40]
        boxes[1, :2] = [[0, 0, 10, 10], [5, 5, 20, 20]]
        np.savez(os.path.join(d, "b.npz"), box=boxes)
        out = os.path.join(d, "o")
        run_cli(src, "--out", out, "--boxes", os.path.join(d, "b.npz"))
        self.assertEqual(StubPredictor.inits, [{"ck": "fab-l", "fast": True, "profile": "auto", "detector": ""}])
        self.assertEqual([len(c["boxes"]) for c in StubPredictor.calls], [1, 2, 0])  # NaN rows dropped; past the file: none
        self.assertTrue(np.array_equal(StubPredictor.calls[0]["boxes"], boxes[0, :1]))
        with open(os.path.join(out, "frames_fab.pkl"), "rb") as f:
            r = pickle.load(f)
        self.assertEqual(r["frames"], ["img0.png", "img1.png", "img2.png"])
        self.assertEqual([len(p) for p in r["people"]], [1, 2, 0])
        self.assertEqual(r["K_used"].shape, (3, 3, 3))
        self.assertTrue(r["call"].endswith(".predict(image, K, boxes)"))

    def test_rejects_videos(self):
        d = tempfile.mkdtemp()
        f = os.path.join(d, "clip.mp4")
        open(f, "wb").close()
        err = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
            run_cli(f)
        self.assertIn("fab-video", err.getvalue())


if __name__ == "__main__":
    unittest.main()
