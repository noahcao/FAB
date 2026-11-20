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
| `--out`, `--name`, `--fps` | `output/fab_world`, the input's name, 30 | output folder, clip name, frame rate |
| `--fab` | `L` | `L` or `S` |
| `--subject_boxes` | the most prominent tracked person | which person to follow |
| `--focal` | estimated | a known focal length (not used for any paper number) |
| `--bundle`, `--fitter`, `--contact_head`, `--contact_video` | the released models | your own body model, SMPL fitter or contact models |
| `--mvs_reuse DIR` | off | reuse an existing COLMAP workspace of the same cameras |

## What it does

ViPE 1.2 + UniDepth estimate cameras and depth, GeoCalib gravity; a static camera gets its focal length from three
estimators. People, sky and moving objects are masked out. FAB runs on every frame, then the contact models. The scene
is a TSDF mesh from the depth (refined by COLMAP multi-view stereo for moving cameras), carved so it never covers the
visible person, plus a heightfield. A final joint optimization sets one metric scale for body and scene.

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
clips are slower (multi-view stereo); for long, high-resolution clips, split them or lower the resolution.
