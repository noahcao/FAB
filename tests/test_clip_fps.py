"""fab.clip_fps: --fps auto snaps a video's metadata rate to exactly 30 within 1 % (29.97 / 30 fps videos keep the 30 fps
defaults byte for byte) and keeps every other rate; explicit values and frame directories."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fab.clip_fps import REF_FPS, clip_fps, snap_fps  # noqa: E402


class SnapTests(unittest.TestCase):
    def test_snap(self):
        for f in (30.0, 29.97, 29.97002997002997, 30.2, 29.71):
            self.assertEqual(snap_fps(f), 30.0, f)
            self.assertIs(type(snap_fps(f)), float)
        for f in (29.7, 30.3, 25.0, 24.0, 23.976, 20.0, 60.0, 59.94, 3.0):
            self.assertEqual(snap_fps(f), f, f)

    def test_explicit_and_directory(self):
        self.assertEqual(clip_fps("anything.mp4", "3"), (3.0, "--fps"))
        self.assertEqual(clip_fps("anything.mp4", 20), (20.0, "--fps"))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(clip_fps(d, "auto")[0], REF_FPS)

    def test_video_metadata(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("needs OpenCV")
        with tempfile.TemporaryDirectory() as d:
            for rate, want in ((29.97, 30.0), (25.0, 25.0), (60.0, 60.0)):
                p = os.path.join(d, f"v{rate}.avi")
                w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), rate, (64, 48))
                for _ in range(3):
                    w.write(np.zeros((48, 64, 3), np.uint8))
                w.release()
                fps, how = clip_fps(p, "auto")
                self.assertAlmostEqual(fps, want, places=2, msg=how)
                if want == 30.0:
                    self.assertEqual(fps, 30.0)


if __name__ == "__main__":
    unittest.main()
