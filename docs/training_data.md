# Training data format

The format of the pool tars and label sidecars that `train.train_fab` reads ([training.md](training.md)). The tools
that built our labels are not part of the release.

**Pool tar `<name>.tar`:** per person crop, `<key>.jpg` (square, resized to 512) and `<key>.npz`. The PAX header
`pool.source` names the dataset (else the folder name), which `smpl_gt_off`, `smpl_shape_off` and `foot3d_sources`
select by (`aistpp_train` is world-frame 3D). Optional `wide_jpg` + `wide_off` give wider context for loose boxes.

`<key>.npz` (float32 unless noted; "crop px" = pixels of the stored crop; camera frame x right, y down, z forward, m):

| key | shape | meaning |
|---|---|---|
| `affine` | (2, 3) | original image px -> crop px |
| `cam_int` | (3, 3) | intrinsics of the original image |
| `ori_size` | (2,) | original image width, height |
| `bbox` | (4,) | person box xyxy in the original image (optional) |
| `bbox_center`, `bbox_scale` | (2,) | the crop window's centre and size in original px (SAM 3D Body's `GetBBoxCenterScale`) |
| `gt_kp2d` | (70, 3) | human 2D labels: crop px + confidence (0 = none), MHR70 keypoint order |
| `gt_kp3d` | (70, 4) | human 3D keypoints + validity; camera axes (m or cm; MHR axes with y, z negated are detected) |
| `gt_mhr_valid`, `gt_model_params`, `gt_shape` | (), (204,), (45,) | a human MHR fit, where one exists |
| `gt_smpl_pose`, `gt_smpl_betas`, `gt_smpl_trans` | (72,), (11,), (3,) | ground-truth SMPL in the camera frame (optional) |
| `gt_smpl_valid`, `gt_smpl_gender` | () | 1 where the SMPL fields are valid; 0 neutral, 1 male, 2 female |
| `gt_smpl_v`, `gt_smpl_v_valid` | (6890, 3), () | exact SMPL-topology vertices, camera frame (optional; replace the posed ones) |
| `gt_mocap_j14`, `gt_mocap_valid` | (14, 3), () | camera-frame mocap joints (optional) |
| `t_kp2d` | (70, 2) | teacher 2D, crop px |
| `t_kp3d`, `t_cam_t` | (70, 3), (3,) | teacher 3D keypoints (camera axes, root at the origin) and camera translation |
| `t_model_params`, `t_shape` | (204,), (45,) | teacher MHR parameters |
| `t_pose_raw`, `t_global_rot`, `t_body_pose`, `t_scale`, `t_hand` | (266,), (3,), (B,), (28,), (108,) | the rest of the teacher's MHR head outputs (SAM 3D Body's `mhr` dict) |

The record's `t_*` fields are SAM 3D Body's predictions for the crop.

**Teacher labels.** `teacher_labels: teacher` reads `<name>.teacher.npz` (`keys` (N,) and `t_*` fields with a leading
N) in place of the record's `t_*`; the released runs used FAB-H's labels as the only teacher (`teacher_strict`). 3D
targets are the ground-truth SMPL where `gt_smpl_valid` (except `smpl_gt_off` sources), else the teacher's mesh;
`teacher_gate` (0.08 m) skips a teacher far from a valid ground truth. Human 2D / 3D labels add their own terms.
`kp2d_labels: wholebody!` reads `<name>.wholebody.npz` (`keys`, `kp2d` (N, 70, 3)) to fill, and with `!` replace, 2D
channels; `sidecar_no_feet` keeps the record's foot channels (MHR70 15-20).

**Head label sidecars:** `<name>.<label set>.npz` with `keys` (N,) matching the tar's sample keys.

| head | config keys | sidecar fields |
|---|---|---|
| contact | `contact_labels: contact_physics_soft,contact_physics,contact` (first present per tar), `contact_knee_labels: contact`, `contact_any_labels: contact_scene,physics_any` | `c`, `valid` (N, 8) uint8 static contact and validity; `h` (N, 8) height above the local floor (m); optional `stable` (N,) (0 = keep) and `dev_mm` (N,) (masked >= 30 mm on `3dpw` tars). Any-contact sets: `c`, `valid` for surfaces other than the floor (`phys*` sets share the physics masking). Knees come from `contact_knee_labels` |
| visibility | `vis_labels: visibility`, `vis_human: true`, `kp2d_labels: wholebody!,foot2d!` | `vis` (N, 84) uint8 per keypoint + joint (0 visible, 1 occluded, 2 outside the crop, 255 unknown); `uv` (N, 84, 2) in crop units; optional `why` (N, 84) bits (8 = human annotation). The human-flag classifier also reads `vis` (N, 70) or (N, 6) from the 2D sidecars |
| G1 | `g1_labels: g1` (the body run: `g1_body`) | `q` (N, 29) G1 joint angles (URDF order, rad), `R` (N, 3, 3) robot root rotation in the camera frame |
| parts | `mask_labels: mask_sam3,mask` (priority order), `mask_gate: score=0.65,kp_vis=0.8,kp_gt=0.7` | `fg` (N, 8192) uint8: the person mask on the crop's 256 x 256 grid, bit-packed; optional `unk` (unknown pixels); gate fields `score`, `kp_t`, `kp_gt`, `out_box` (N,); `kp_vis` checks against `<name>.visibility.npz` |

The bundle replaces the body run's own G1 head with the separately trained one (you may use the same G1 labels for
both). Part labels are rendered during training from the target mesh and clipped by the person masks
(`mask_sil` / `mask_silout` drop masks that disagree with the mesh).
