# Troubleshooting

## Install

- **`FAB weights not found`:** download them (README, Install) to `~/.cache/fab/weights`, or point `$FAB_WEIGHTS` /
  `--weights DIR` at them; check with `sha256sum -c SHA256SUMS`.
- **`SAM 3D Body config not found`**, or a 401 / 403 from Hugging Face: accept the SAM licence on Hugging Face, run
  `hf auth login`, then `python -c "import fab; fab.download_sam3d()"`.
- **`the Unitree G1 URDF ... not found`:** run `bash scripts/download_g1.sh` (needs `git` and access to GitHub and
  Hugging Face).
- **`ImportError: libGL.so.1` / `libEGL.so.1` / `libusb-1.0.so.0`** (machines without a display): if you installed the
  `engine` extra, restore the headless OpenCV (`pip uninstall -y opencv-python && pip install --force-reinstall --no-deps
  "opencv-python-headless>=4.9"`). World mode's tools need the system libraries:
  `sudo apt-get install -y libgl1 libglib2.0-0 libegl1 libusb-1.0-0`. `fab-build-trt` shows the error under
  "could not load fab-l".

## Running

- **The first frames are slow:** `fast=True` compiles once per person count (about 4 min for a first `fab` command,
  about 1.5 min once PyTorch's compile cache is filled). Use `--eager` for a quick test.
- **Out of GPU memory on crowded videos:** `fast=True` keeps compiled graphs per person count. Pass boxes for the people
  you need (`--boxes`) or use `--eager`.
- **No people found:** pass your own boxes (`--boxes boxes.npz`, or `boxes=` in Python); `det_conf` (default 0.25) sets
  the detector threshold.
- **CPU oversubscription on many-core machines:** set `OMP_NUM_THREADS=4`.
- **Numbers differ slightly from the paper:** see [usage.md](usage.md) (Numerics and CPU); world mode adds ViPE's
  run-to-run spread, and the retargeter's physics checks can differ across CPU types.

## World and retargeting

- **`fab-world` fails in the ViPE / COLMAP steps:** see `<out>/work/logs/*.log`. Building ViPE needs `nvcc` matching
  PyTorch, `TORCH_CUDA_ARCH_LIST` for your GPU and about 20 GB of disk.
- **`fab-world`: "WARNING ... no scene":** nothing static was reconstructed (a plain backdrop masked as sky, a subject
  beyond 8 m, or unusable depth). Film closer, with the ground in view ([world.md](world.md#getting-good-results)).
- **The person floats above the ground or has no scene under the feet:** see [world.md](world.md#getting-good-results).
- **`fab-retarget` fails at the clean step:** the scene has no valid heightfield cells; the command keeps the first
  solve and notes it in `<clip>_g1.json`. A WARNING about no contact frames means the reference never touches the terrain.
- **`fab-retarget`: "WARNING ... points mode without FAB's dense points":** the input has no usable dense points (it is
  not a `fab-world` output, the model has no dense head, or the point ids differ), so the body's vertices were used. For
  another model, rebuild the anchors: `python -m fab.retarget.dense_corr --ckpt <its model.pt>`.
- **`fab-retarget` crashes with several processes on one GPU:** run one process per GPU, and open an issue if it persists.

## Harmless log messages

- `torch._dynamo hit config.recompile_limit (8)`, `Dynamo does not know how to trace the builtin ...`
- `torch.backends.cuda.preferred_linalg_library is an experimental feature`
- On a first compile: `TypedStorage is deprecated`, `torch._prims_common.check is deprecated`, and `TensorFloat32 tensor
  cores ... available but not enabled` (leave TF32 off to reproduce the paper's numbers)
- `[TRT] [W] The logger passed into createInferRuntime differs ...`
- `You are sending unauthenticated requests to the HF Hub` (set `HF_TOKEN` or ignore)
- Mask2Former's `LOAD REPORT` (`swin.layernorm` MISSING, `relative_position_index` UNEXPECTED)

## Tests

`tests/test_demo_regression.py` compares coarsely on CUDA by default (same people, `cam_t` within 1 % of the depth,
`keypoints3d` within 10 cm), because BF16 rounds differently on each GPU type. For an exact check, write a reference on
your GPU once, then compare against it:

```bash
FAB_WRITE_REFERENCE=1 FAB_DEMO_REFERENCE=my_reference.npz python -m pytest tests/test_demo_regression.py
FAB_DEMO_REFERENCE=my_reference.npz python -m pytest tests/test_demo_regression.py
```
