# Multi-view fusion: `fab-multiview`

`fab-multiview` recovers one person's motion from several synchronised, calibrated cameras. It runs the released FAB
model on every view and fuses the views with the model's own visibility predictions. The output uses `fab-world`'s
layout in the rig's world, so [`fab-retarget`](retarget.md) runs on it unchanged.

```bash
fab-multiview rig.json --out output/mv                       # every camera in rig.json
fab-multiview take/ --out output/mv                          # a take folder (below)
fab-multiview rig.json --out output/mv --views cam0,cam2     # some of the cameras
fab-multiview rig.json --out output/mv --subject_boxes boxes/ # pick the person: boxes/<camera name>.npz
fab-retarget output/mv/<name>.npz --out output/g1            # the G1 motion, as for fab-world
```

| option | default | effect |
|---|---|---|
| `--out`, `--name` | `output/fab_multiview`, the rig file's (folder's) name | output folder, clip name |
| `--views` | all | comma-separated camera names |
| `--subject_boxes` | the rule below | a folder of `<camera name>.npz`, `box` (T, 4) xyxy per synced frame, NaN = unknown (identity only) |
| `--model` | `fab-l` | `fab-l` or `fab-s` (the fusion needs the dense head) |
| `--fast` | off | FAB's compiled path (`fast=True`); the default is the eager path |

## The rig

**`rig.json`**, paths relative to the file:

```json
{"fps": 30, "up": [0, 0, 1], "floor_z": -0.95,
 "cameras": [{"name": "cam0", "frames": "cam0/", "K": [fx, fy, cx, cy], "T_world_cam": [[...4 x 4...]],
              "dist": [k1, k2, p1, p2, k3], "timestamps": "cam0/times.npy", "offset": 0.0},
             {"name": "cam1", "video": "cam1.mp4", "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "T_world_cam": [[...]]}]}
```

- `frames` (a folder of images, sorted by name) or `video`; `K` as `[fx, fy, cx, cy]`, a 3 x 3 matrix or
  `{"fx", "fy", "cx", "cy"}`; `T_world_cam` maps camera points (OpenCV axes: x right, y down, z forward) to the world.
- `dist` (optional): OpenCV distortion; frames are undistorted to the pinhole `K` before FAB runs.
- `timestamps` (optional): a `.npy` or text file with one time per frame in seconds (or rows `t, frame number`);
  `offset` (seconds) is added to them. Without timestamps, frame i is at `i / fps + offset`.
- `up` (default `[0, 0, 1]`): the world's up direction; the output world is turned so that it is +z.
  `floor_z` (optional): the floor height along `up`; without it, the floor is the 10th percentile of the body's lowest
  point over the clip.

**A take folder** holds `world.json` (`cameras.<folder>.K` and `.T_world_cam`, optional `floor_z`) and one folder of
frames per camera, each with an optional `frame_index.npy` (rows `t, frame number`).

Frames are matched to the first camera's timeline: each camera's nearest frame within half a frame period, else none.

## What it does

1. **Per view:** `FABPredictor` with its released defaults and the camera's `K` (video mode: detector every 10 frames,
   boxes tracked in between), every person.
2. **The subject:** on the first frame, the cross-view match (one person in each of two views) whose triangulated
   keypoints reproject within 5 % of the person's height, the most prominent such match (summed 2D height). After that,
   in each view, the person nearest the projection of the subject's previous 3D keypoints (within 25 % of their height).
   A subject lost in every view is picked again by the first rule. `--subject_boxes` overrides this per view and frame.
3. **Fusion, per frame:**
   - The 70 MHR keypoints and 14 joints are triangulated from their 2D. Every view that sees the person takes part,
     weighted by the visibility head's P(visible) (at least 0.05). A view whose ray passes more than 5 cm from the point
     is dropped while more than two remain. A point with fewer than two views, or with two views that disagree, is not
     triangulated.
   - Each view's dense points and joints are placed on the triangulated joints by a robust similarity (scale,
     rotation, translation). This removes the view's monocular depth and size error. A frame with fewer than 6
     triangulated joints reuses the view's nearest-in-time placement. Then each dense point moves by the residual of its
     nearest joint on the SMPL template (triangulated minus placed), which corrects the view's limb poses.
   - Each of the 468 dense points and 14 joints is the robust weighted mean over the views (Geman-McClure, 5 cm). A
     point's weight in a view is the P(visible) of its nearest joint on the SMPL template (at least 0.05). A
     triangulated joint keeps its triangulated position.
4. **The floor** (only with the rig's `floor_z`): each support part's fused points (those nearest its joint on the SMPL
   template) are checked against the floor, within 20 cm. A knee or hand that the image contact head puts on the floor
   (static contact >= 0.5 in the view that sees its keypoint best) is moved vertically so that its lowest point is on the
   floor. A foot is only kept from going below the floor, never pulled down: shoes add their soles. The fit treats the
   moved points as any other target.
5. **SMPL:** one fit per frame to the fused points. The released one-pass fitter runs first, then two of the release's
   Gauss-Newton steps (robust, weighted by the fused confidence). The clip's median betas follow, then two more steps
   with that shape fixed. The release's world smoothing comes last (rotations sigma 2, pelvis sigma 3 frames at 30 fps, i.e. 67 / 100 ms at any frame rate).
6. **Contacts:** the release's labels on the flat floor (lowest vertex within 4 cm, slower than 0.5 m/s), and the image
   contact head's probabilities fused over the views (weights: the P(visible) of the heel, toe, wrist or knee).

With one camera there is nothing to fuse: the output is the release's own single-view result (FAB's per-frame SMPL fit,
mapped into the world by the camera's `T_world_cam`, the clip's median betas and the release's world smoothing).

## Outputs

As `fab-world` ([world.md](world.md#outputs)), in the rig's world (z up, metres):

| file | contents |
|---|---|
| `<out>/<name>.npz` | `frames`, `fps`, `pose` (T, 72), `betas` (10,), `trans` (T, 3), `joints` (T, 24, 3), `valid` (T,), `contacts` (JSON); `T_world_cam` / `K` of the first camera, `T_world_cams` (N, 4, 4), `K_cams`, `camera_names`, `timestamps`, `floor_z`, `notes` |
| `<out>/work/joint/recon/<name>.npz` | the same + `verts` (T, 6890, 3); `_hfield.npz` and `_scene.ply`: the flat floor |
| `<out>/work/ours/fab/<name>_raw.npz` | the fused dense points `pts` (T, 468, 3) (`fab-retarget`'s points mode) |
| `<out>/work/contacts/<name>__0.contact.npz` | the fused image contact probabilities (`fab-retarget --ground_aware`) |
| `<out>/work/views/<camera>.pkl`, `<out>/work/<name>_fusion.npz` | each view's FAB outputs (reused by a re-run with the same frames, model and `--fast`) and the fusion's per-frame record (subject per view, triangulated keypoints, placements) |

`valid` marks frames where the subject was seen; the body is interpolated in between.

## Getting good results

- The calibration bounds the accuracy: a camera pose off by 2 cm moves the triangulated joints by about that much.
- Views that see the person from different sides (60-120 degrees apart) triangulate best. Two cameras that both see only
  the person's back, or a joint that only one camera sees, give that view's monocular estimate for the joint.
- The rig's world should have z up and a known floor (`up`, `floor_z`) for `fab-retarget`.
- FAB's body is barefoot: with thick soles, planted feet sit at the sole's height above the floor (about 4 cm with
  sneakers in our tests). `fab-retarget --ground_aware` holds planted feet on the floor with the fused contact
  probabilities.
- With two views, a joint whose 2D is wrong along the epipolar line in one view (e.g. a knee hidden behind the hands)
  still triangulates consistently and cannot be detected; a third camera removes this.
