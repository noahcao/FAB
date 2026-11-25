# World mode and scenes: `fab-world`

`fab-world` turns one video into the person's motion in a metric, gravity-aligned world, with the camera path, contacts
and a mesh of the static scene. It runs the paper's world pipeline.

```bash
fab-world assets/demo/steps_up.mp4 --out output/world   # a video or a folder of frames
fab-world frames/ --out output/world --fab S            # FAB-S instead of FAB-L
fab-world clip.mp4 --out output/world --focal 1450      # a calibrated camera: its focal length in px
fab-world clip.mp4 --subject_boxes boxes.npz            # pick the person: "box" (T, 4) xyxy per frame, NaN = unknown
```

| option | default | effect |
|---|---|---|
| `--out`, `--name` | `output/fab_world`, the input's name | output folder, clip name |
| `--fps` | `auto`: a video's own frame rate (29.7-30.3 counts as 30); a frame folder 30 (`--profile paper`: 30) | the frame rate; give it for image sequences. The body smoothing is set in time, so it adapts to the rate |
| `--fab` | `L` | `L` or `S` |
| `--subject_boxes` | the most prominent tracked person | which person to follow |
| `--focal` | estimated | a known focal length (not used for any paper number) |
| `--bundle`, `--fitter`, `--contact_head`, `--contact_video` | the released models | your own body model, SMPL fitter or contact models |
| `--mvs_reuse DIR` | off | reuse an existing COLMAP workspace of the same cameras |
| `--profile` | `default` | the defaults of `--camera`, `--single_pass`, `--vipe_depth`, `--vipe_mpix`, `--mvs`, `--vipe_stride`, `--vipe_chunk_real`, `--vipe_max_side` and `--fps` (each one given overrides it). `default` = `--camera auto --single_pass 1 --vipe_depth adaptive_unidepth-l --vipe_mpix 2000 --mvs 0 --vipe_stride 2 --vipe_chunk_real 1 --vipe_max_side 960 --fps auto`: the static-camera fast path, and on a moving camera ViPE on every 2nd frame at a 960 px long side, without multi-view stereo. `paper` = `--camera vipe --single_pass 0 --mvs 1 --vipe_stride 1 --vipe_max_side 0 --fps 30` with ViPE's own depth on full frames and no chunks, and the body smoothing in the clip's own frames: the pipeline of the paper's world rows. For a slower moving-camera run use `--vipe_stride 1` (ViPE on every frame), `--vipe_max_side 0` (full frames; both the same accuracy on the development sets) and / or `--mvs 1` (multi-view stereo, several times slower) |
| `--camera` | `auto` | `vipe`: ViPE for every clip; `auto`: a fixed camera (the frames do not move) takes the static-camera fast path below, any other clip ViPE; `static`: the fast path without the check |
| `--focal_rule` | `geocalib` | the fast path's focal: `geocalib`; `median` of GeoCalib and MoGe-2; `nc3` the median of GeoCalib, Pi4 and VGGT-Omega (non-commercial models, set up separately: `bench/scene_recon/focal_nc.py`) |
| `--floor_up` | 1 | the fast path's gravity: GeoCalib's up averaged with the floor plane under the person (0: GeoCalib's) |
| `--vipe_mpix N` | 2000 | ViPE in overlapping chunks when frames x megapixels > N, so long videos fit the GPU |
| `--vipe_depth MODEL` | `adaptive_unidepth-l` | ViPE's dense depth model; `adaptive_unidepth-l` skips its Video Depth Anything alignment (same cameras, much less GPU memory) |
| `--vipe_stride S`, `--vipe_chunk_real` | 2, 1 | ViPE on every S-th frame (and the last), its poses interpolated to every frame (1: every frame); `--vipe_chunk_real 1`: the chunks of `--vipe_mpix` counted in real frames, so a chunk spans the same time at any stride (0: in ViPE's frames) |
| `--vipe_max_side PX`, `--vipe_no_instance` | 960, 0 | ViPE on frames resized to a long side of at most PX (0: full frames), its intrinsics mapped back to the full frames; `--vipe_no_instance 1`: ViPE without its own instance masks (people are still masked by the Mask2Former step) |
| `--mvs` | 0 | 1: COLMAP multi-view stereo refines the scene of a moving camera (0: the scene from ViPE's depth alone). Keep `--mvs 0` with the no-VDA depth (`adaptive_unidepth-l`): that is the pair measured to cost ~1 mm W on EMDB; with ViPE's own VDA-aligned depth, dropping MVS cost more |
| `--single_pass` | 1 | 1: the contact model's image features from the FAB stage's own forward instead of a second FAB pass |
| `--det_size` | 640 | the person detector's input size; 960 finds people under ~50 px in a 640 input (slightly slower); it also finds more bystanders, which can change the subject the default tracker follows, and it boxes a subject filling most of the frame smaller |
| `--track`, `--track_kw` | `iou` | `oc`: persistent Kalman / OC-SORT ids, the detector on every frame, the subject chosen over the whole clip by FAB's keypoint extent x centrality; frames where the subject is not seen are interpolated, never given to another person (`fab/track.py`, `SUBJECT_KW` in `bench/sonic_motion.py`); with several people of similar size no automatic choice is reliable: give `--init_box` |
| `--init_box`, `--init_frame` | none | one prompt box for the subject (xyxy px on frame `--init_frame`, or an npz `box` (T, 4) whose first finite row is used; implies `--track oc`): the tracked person it overlaps is followed through the clip, else the automatic choice; recommended for any video with more than one person (EgoBody val: W-MPJPE 166.7 -> 55.1 mm vs `--track oc` alone); people in close contact (grappling) can still be lost, and long occlusion gaps can still hurt |
| `--motion` | `off` | `off`: the kinematic export; `learned`: FAB-M ([below](#learned-world-motion-fab-m---motion)) re-places the body in the world on static cameras |
| `--motion_model`, `--motion_fps` | `shared/world_motion.pt`, `auto` | FAB-M's model; `auto`: 30 fps as is, 24 fps and up resampled to 30 and back, slower clips passed through; `resample`: from 10 fps up |
| `--refine` | `none` | `physics`: the physical-plausibility post-process ([below](#physical-plausibility-refinement---refine-physics)) |

## What it does

A fixed camera takes the static-camera fast path below. Otherwise ViPE 1.2 + UniDepth estimate cameras and depth (by
default on every 2nd frame, at a 960 px long side), GeoCalib the gravity; a camera that turns out static gets its focal
length from three estimators. People, sky and moving objects are masked out. FAB runs on every frame, then the contact
models. The scene is a TSDF mesh from the depth (`--mvs 1`: refined by COLMAP multi-view stereo on moving cameras),
carved so it never covers the visible person, plus a heightfield. A final joint optimization sets one metric scale for
body and scene. `--profile paper` sends every clip through ViPE on every full frame, with multi-view stereo.

The static-camera fast path (`--camera auto | static`) runs no SLAM: identity cameras, the focal from `--focal_rule`
(or `--focal`), MoGe-2 metric depth on every 3rd frame fused into one static scene, Mask2Former masks for people and
other movers, and GeoCalib gravity refined by the floor. On a fixed camera it is about 5x faster end to end than the
ViPE path and needs about a quarter less GPU memory (RICH val).
Its camera stage runs neither ViPE's SLAM nor UniDepth (CC BY-NC 4.0): only GeoCalib (Apache-2.0, the copy in ViPE's
env) and MoGe-2 (MIT), unless `--focal_rule nc3`.

Environment: `FAB_LOAD_WORKERS` (4) threads decode the frames ahead of the FAB and contact stages; `FAB_FAB_FAST=1` runs
the FAB stage with `FABPredictor(fast=True)` (TensorRT engines from `fab-build-trt` when built, else compiled PyTorch,
which compiles once per process).

## Learned world motion: FAB-M (`--motion`)

Opt-in (`--motion learned`; default `off`). After the export, FAB-M (`fab/world_motion.py`; `shared/world_motion.pt` in the weights) re-places the body in
the world from FAB's per-frame bodies, the cameras and the contacts; the kinematic export is kept in `<out>/work/kin/`. It runs
on static cameras only (the camera check above): a moving camera's export passes through unchanged, and the export's notes say
why (`world_motion.skipped`). FAB-M was trained at 30 fps: 30 fps clips run as they are, clips from 24 fps up are resampled to
30 fps and back, slower ones pass through (`--motion_fps resample`: from 10 fps up). It uses the GPU when there is one, else
the CPU. On an existing export: `python -m fab.world_motion output/world/clip.npz`.

On the development sets (world MPJPE, with the default camera on the static cameras FAB-M runs on; the other sets,
moving cameras or scored in camera space, are unchanged):

| dataset | FAB's detector | ground-truth identity |
|---|---|---|
| RICH val | 81.1 -> 65.1 mm (-19.8 %) | 80.4 -> 64.9 mm (-19.3 %) |
| Embody 3D val | 96.1 -> 89.5 mm (-6.9 %) | 67.0 -> 64.7 mm (-3.6 %) |
| AIST++ dev | 83.0 -> 83.9 mm (+1.0 %) | 84.0 -> 84.9 mm (+1.1 %) |
| EgoBody val | 143.4 -> 145.0 mm (+1.1 %) | 56.2 -> 57.4 mm (+2.1 %) |

On EgoBody, where people often sit on sofas and chairs (support other than the floor), it is slightly worse: its floor
term assumes one flat floor under the planted feet. That is why FAB-M is not the default.

## Physical-plausibility refinement (`--refine physics`)

`--refine physics` adds a light kinematic step on the final body (`bench/scene_recon/world_refine.py`; nothing is simulated):

- **ground**: the planted footprints (static contacts of heels / toes) define the walked-on ground plane; a tilt of 0.3-6
  degrees (a gravity error) is rotated away, cameras, body and scene together;
- **depth**: each frame's body moves along its camera ray (its image position kept) so the planted feet stand on that
  ground without sliding or sinking, smoothly and by a few percent at most;
- **foot lock**: planted heels and toes are pinned in place on the ground and the legs refit to them.

When the footprints are not one plane (e.g. stepping onto furniture) only the slide-free foot lock runs. It trades a little
pose accuracy for plausibility, so it is off by default. The export keeps the unrefined body as `pose_in` / `trans_in`, and
the report as `refine_world` (JSON). It can also be run on an existing export:

```bash
python -m bench.scene_recon.world_refine --world output/world/clip.npz --hfield output/world/clip_hfield.npz \
    --contacts output/world/work/contacts/clip__0.contact.npz --scene output/world/clip_scene.ply --out output/world_refined
```

## Outputs

`<out>/<name>.npz`:

| key | meaning |
|---|---|
| `frames`, `fps` | frame indices, frame rate |
| `T_world_cam` (T, 4, 4), `K` (3, 3) | cameras (camera -> world) and intrinsics; the world is z-up and metric (`gravity_world` = (0, 0, -1)) |
| `pose` (T, 72), `betas` (10,), `trans` (T, 3), `joints` (T, 24, 3), `valid` (T,) | the world body (SMPL) |
| `contacts` | JSON (`json.loads(str(z["contacts"]))`): per-frame 0 / 1 labels for `lfoot`, `rfoot`, `lhand`, `rhand`, `lknee`, `rknee` on the scene |
| `scale_fix`, `joint`, `notes` | the joint step's scale and report; the models and camera choices used (JSON) |

Also `<out>/<name>_scene.ply` (scene mesh), `<out>/<name>_hfield.npz` (heightfield of the walkable surface) and, in
`<out>/work/contacts/<name>__0.contact.npz`, the per-frame contact probabilities.

## Install

World mode needs a CUDA GPU (the first of `CUDA_VISIBLE_DEVICES`), `git`, [uv](https://docs.astral.sh/uv/), a CUDA
toolkit whose `nvcc` matches PyTorch's, a C++ compiler and about 20 GB of disk. `fab-world` installs its tools on first
use; to install them up front:

```bash
pip install -e ".[world]"
sudo apt-get install -y libgl1 libglib2.0-0 libegl1 libusb-1.0-0   # servers without a display (Ubuntu / Debian)
bash bench/baselines/setup_vipe12.sh     # ViPE 1.2 (cameras + UniDepth depth)
bash bench/baselines/setup_vipe.sh       # ViPE (GeoCalib, fixed-focal run)
bash scripts/colmap_setup.sh             # COLMAP with CUDA
bash bench/videomimic/setup_recon.sh     # reconstruction packages
```

The tools live under `$FAB_ENVS` (default `$FAB_HOME/envs`). Override them with `FAB_VIPE12_PY`, `FAB_VIPE_PY`,
`VIPE12_DIR`, `VIPE_DIR` and `FAB_COLMAP`; CUDA builds read `TORCH_CUDA_ARCH_LIST` (default `8.6;8.9`) and `MAX_JOBS`.
ViPE, UniDepth (CC BY-NC 4.0) and COLMAP include non-commercial or copyleft parts ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md));
none of them ships with FAB.

## Getting good results

The scene is built from depth up to 8 m from the camera, and the body stands on that scene. It works best when:
- the person stays within about 5 m of the camera;
- the ground along their path is visible in some frames.

Otherwise the person can float above the reconstructed ground (2-20 cm in our tests). Passing `--focal` does not fix
this. If no static scene is found (a plain backdrop masked as sky, a far-away subject, or unusable depth), `fab-world`
prints a WARNING, writes no `_scene.ply`, and contacts and `fab-retarget` fall back to a floor at z = 0.

ViPE varies slightly from run to run, so world metrics move slightly between runs (about 0.2 mm on EMDB-2). Moving-camera
clips are slower (ViPE's SLAM; `--mvs 1` adds multi-view stereo); for long, high-resolution clips, split them or lower the
resolution.
