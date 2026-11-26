"""Frame rate of a world-mode clip (scripts/fab_world.sh --fps). An explicit value is used as given; "auto" = a video's
metadata frame rate, snapped to exactly 30 when within 1 % of it (29.97 / 30 fps videos keep the 30 fps defaults byte for
byte); a frame directory, or a video without a frame rate in its metadata, is 30 (pass --fps for image sequences)."""
import os

REF_FPS = 30.0
SNAP_TOL = 0.01


def snap_fps(fps, ref=REF_FPS, tol=SNAP_TOL):
    """ref when |fps - ref| / ref < tol, else fps."""
    fps = float(fps)
    return ref if abs(fps - ref) / ref < tol else fps


def clip_fps(src, arg="auto"):
    """-> (fps, how it was chosen)."""
    if str(arg) != "auto":
        return float(arg), "--fps"
    if os.path.isdir(src):
        return REF_FPS, "frame directory: default (pass --fps for its true rate)"
    import cv2

    meta = float(cv2.VideoCapture(src).get(cv2.CAP_PROP_FPS) or 0.0)
    if meta <= 0:
        return REF_FPS, "video without a frame rate in its metadata: default"
    fps = snap_fps(meta)
    return fps, f"video metadata {meta:g}" + (" (snapped to 30)" if fps != meta else "")
