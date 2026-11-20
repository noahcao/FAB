# Usage and outputs

Running FAB on images and videos, from the command line or Python, and what it returns. World motion: [world.md](world.md).

## Images: `fab`

```bash
fab image.jpg --out output/fab                      # one image
fab images/ --out output/fab                        # every image in a folder
fab image.jpg --boxes boxes.npz                     # your boxes: "box" (N, 4) xyxy px; no detector
fab image.jpg --model fab-s --K 1000,1000,640,360   # another model; known intrinsics fx,fy,cx,cy
fab image.jpg --eager                               # no compilation (fast=False)
```

Writes `<out>/<name>_fab.pkl` with `frames` (image names), `people[i]` (a list of per-person dicts, below), `K_used[i]`
(the intrinsics used), and `source`, `K`, `call`, `profile`, `fitter` (the run settings).

## Videos: `fab-video`

```bash
fab-video clip.mp4 -o output/clip.mp4                # tracks, smoothing, video contacts, a render
fab-video frames/ -o output/clip.mp4 --fps 30        # a folder of frames, sorted by name
fab-video clip.mp4 --boxes boxes.npz --eager         # your boxes: "box" (T, 4) or (T, N, 4) xyxy px
fab-video clip.mp4 --no-render --results output/clip.pkl
fab-video clip.mp4 -o output/clip.mp4 --load output/clip.pkl   # render saved results again
```

The detector runs every 10th frame and boxes are tracked in between; each track gets a causal smoother
(`--lag 3 --sigma 2.5` frames) and video contacts in the camera frame (for a moving camera, use `fab-world`). It writes
the render, a `.jpg` poster and a results pickle (default: the render path with `.pkl`; keys `video`, `fps`, `size`, `K`,
`model`, `names`, `call`, `frames`). Each `frames[i]["people"]` holds the outputs below plus `track`,
`<output>_smoothed` and `contact_video` (about 0.5 MB per person and frame).

Both commands also take `--model`, `--weights DIR` and `--profile` (FAB-H, below).

## Python

```python
from fab.predictor import FABPredictor

fab = FABPredictor("fab-l", fast=True)                 # "fab-l" | "fab-s" | "fab-h" | a model.pt path
people = fab.predict(image_rgb, K=None, boxes=None)    # (H, W, 3) uint8 RGB; K (3, 3) px; boxes (N, 4) xyxy px
for people in fab.predict_video(frames_rgb, K=None):   # frames of one clip, in order
    ...
for t, people in fab.predict_video_smoothed(frames_rgb, K=None):   # plus the causal smoother
    ...
```

Without `K` the focal length is the image diagonal; without `boxes` the detector runs (`detector=""` turns it off).

## Outputs per person

Camera frame (x right, y down, z forward), metres; 2D in pixels.

| key | shape | meaning |
|---|---|---|
| `box`, `cam_t` | (4,), (3,) | the crop box; the camera translation of the mesh root |
| `keypoints3d` | (70, 3) | MHR70 keypoints |
| `keypoints2d` | (70, 2) | the dense head's 2D keypoints (best for 2D tasks); `keypoints2d_projected`: `keypoints3d` projected |
| `smpl_pose`, `smpl_betas`, `mesh_smpl_fit` | (72,), (10,), (6890, 3) | neutral SMPL fitted to the dense points: the most accurate body output |
| `mesh_mhr`, `mesh_smpl` | (18439, 3), (6890, 3) | the MHR mesh, and the same mesh in SMPL topology |
| `dense_surface_cam` | (468, 3) | dense surface points in the camera frame (SMPL vertex ids: `fab.model.dense_head.surface_idx`) |
| `dense_surface`, `dense_surface2d` | (468, 3), (468, 2) | raw head outputs: 3D up to a common offset (use `dense_surface_cam`), and 2D |
| `joints_h36m14` | (14, 3) | the evaluation skeleton, root-relative |
| `keypoint_visibility` | (84, 3) | P(visible / occluded / outside the crop) per keypoint and joint, annotation-style; `keypoint_visibility_rendered`: the geometric classifier evaluated in the paper (PyTorch path only) |
| `contact` | (16,) | contact probabilities: 8 static + 8 any, each `l_heel, l_toe, r_heel, r_toe, l_hand, r_hand, l_knee, r_knee` |
| `g1_q`, `g1_root` | (29,), (3, 3) | Unitree G1 joint angles (URDF order, rad) and root rotation |
| `parts`, `parts_roi` + `roi`, `parts_crop` + `crop_affine` | | body-part maps (`fab.body_parts.PART_NAMES`) for the image, the crop region, or the 256 x 256 crop |

FAB-H has no contact, part or G1 head and fits SMPL to its MHR mesh (MHR mesh → SMPL, `profile="mesh"`);
`profile="points"` (`--profile points`) fits its dense points instead (dense points → SMPL), and `dense_outputs=True` returns the points without changing the fit.

## Options

Defaults are the released settings behind the paper's numbers.

| option | default | effect |
|---|---|---|
| `fast` | `False` (Python), `True` (`fab`) | compile the body and replay CUDA graphs; use TensorRT engines if built ([acceleration.md](acceleration.md)) |
| `smpl_fitter` | `"auto"` | the released fitter; `""` skips the SMPL fit |
| `parts` | `True` | `False` skips the part maps (nothing else changes) |
| `crop_maps` | `"auto"` (from `config.json`) | `True`: return only the 256 x 256 crop part map |
| `full_maps` | `True` | `False`: return part maps only on the crop region (`parts_roi` + `roi`), no full-image map |
| `track_every` | 10 | video: run the detector every N frames |
| `dense_25d_mix` | `"auto"` | mix of lifted (2.5D) and regressed 3D points |
| `dense_fit` | `False` | an opt-in extra refinement, not used for the paper; needs the 23-joint regressor ([training.md](training.md)) |
| `device` | `"cuda"` | `"cpu"` needs `fast=False` |

**Numerics and CPU.** The first `fast=True` call per person count compiles for a few minutes. The compiled path (BF16)
is not bit-deterministic: a person can move by up to about 1 cm between runs, while benchmark means stay within a few
hundredths of a mm. TensorRT engines repeat exactly but differ from `fast=False` by the same order. On the CPU the
model runs in FP32 (FAB-H's backbone stays BF16), so results differ from the GPU's, most for small, distant people: up
to 7 cm for FAB-L and 12 cm for FAB-H on the joggers about 14 m away, under 1 cm for the skater about 2 m away.
