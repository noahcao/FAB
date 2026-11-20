# Third-party notices

This repository contains code adapted from the projects below, depends on packages that you install yourself, and
downloads some files at run time. Each keeps its own licence. This repository's own code is licensed under the Apache
License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); the files derived from SAM 3D Body and DINOv3 keep their licences
(below); all model weights, including the person detector, use [WEIGHTS_LICENSE](WEIGHTS_LICENSE).

## Code included in this repository

| component | licence | where | notice |
|---|---|---|---|
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (Meta) | SAM License | the 40 files listed below | file headers (copyright + `SPDX-License-Identifier: LicenseRef-SAM-License`); [SAM-License.txt](third_party_licenses/SAM-License.txt) |
| [DINOv3](https://github.com/facebookresearch/dinov3) (Meta) | DINOv3 License | `fab/models/backbones/dinov3_vit/` (vendored, 11 files) | file headers (`SPDX-License-Identifier: LicenseRef-DINOv3-License`); [DINOv3-License.md](third_party_licenses/DINOv3-License.md). Built with DINOv3. |
| [pymomentum](https://github.com/facebookresearch/momentum) (Meta) | MIT | `fab/models/heads/mhr_fast.py` (an eager port of the MHR rig's forward kinematics and pose correctives) | [pymomentum-MIT.txt](third_party_licenses/pymomentum-MIT.txt) |
| [MHR](https://github.com/facebookresearch/MHR) (Meta) | Apache License 2.0 | the same file (ported from the MHR TorchScript model) | [MHR-Apache-2.0.txt](third_party_licenses/MHR-Apache-2.0.txt) |

Files under the SAM License (from SAM 3D Body, modified for FAB): `fab/build_models.py`, `fab/data/__init__.py`, `fab/data/transforms/__init__.py`, `fab/data/transforms/bbox_utils.py`, `fab/data/transforms/common.py`, `fab/data/utils/io.py`, `fab/data/utils/prepare_batch.py`, `fab/metadata/__init__.py`, `fab/metadata/mhr70.py`, `fab/models/__init__.py`, `fab/models/backbones/__init__.py`, `fab/models/backbones/dinov3.py`, `fab/models/decoders/__init__.py`, `fab/models/decoders/keypoint_prompt_sampler.py`, `fab/models/decoders/prompt_encoder.py`, `fab/models/decoders/promptable_decoder.py`, `fab/models/heads/__init__.py`, `fab/models/heads/camera_head.py`, `fab/models/heads/mhr_head.py`, `fab/models/meta_arch/__init__.py`, `fab/models/meta_arch/base_lightning_module.py`, `fab/models/meta_arch/base_model.py`, `fab/models/meta_arch/sam3d_body.py`, `fab/models/modules/__init__.py`, `fab/models/modules/camera_embed.py`, `fab/models/modules/drop_path.py`, `fab/models/modules/geometry_utils.py`, `fab/models/modules/layer_scale.py`, `fab/models/modules/mhr_utils.py`, `fab/models/modules/misc.py`, `fab/models/modules/swiglu_ffn.py`, `fab/models/modules/transformer.py`, `fab/models/optim/__init__.py`, `fab/models/optim/fp16_utils.py`, `fab/sam_3d_body_estimator.py`, `fab/utils/__init__.py`, `fab/utils/checkpoint.py`, `fab/utils/config.py`, `fab/utils/dist.py`, `fab/utils/logging.py`.

Files under the DINOv3 License (vendored from DINOv3): `fab/models/backbones/dinov3_vit/__init__.py`, `fab/models/backbones/dinov3_vit/layers/__init__.py`, `fab/models/backbones/dinov3_vit/layers/attention.py`, `fab/models/backbones/dinov3_vit/layers/block.py`, `fab/models/backbones/dinov3_vit/layers/ffn_layers.py`, `fab/models/backbones/dinov3_vit/layers/layer_scale.py`, `fab/models/backbones/dinov3_vit/layers/patch_embed.py`, `fab/models/backbones/dinov3_vit/layers/rms_norm.py`, `fab/models/backbones/dinov3_vit/layers/rope_position_encoding.py`, `fab/models/backbones/dinov3_vit/utils.py`, `fab/models/backbones/dinov3_vit/vision_transformer.py`.

The SAM 3D Body files keep upstream comments on their own origins. The rest of the repository is FAB's own code.
`fab/yolo.py` implements the YOLO11 detection network itself and reads the detector checkpoint without the
Ultralytics package.

## Packages you install

Not included in this repository; installed with `pip install -e .` (requirements.txt) or an extra, each under its own
licence:

- core: PyTorch and torchvision (BSD 3-Clause), NumPy (BSD 3-Clause), OpenCV (Apache 2.0), RoMa (BSD 3-Clause), einops
  (MIT), yacs (Apache 2.0), OmegaConf (BSD 3-Clause), PyTorch Lightning (Apache 2.0), huggingface_hub (Apache 2.0),
  SciPy (BSD 3-Clause), timm (Apache 2.0), Pillow (MIT-CMU), braceexpand (MIT), PyYAML (MIT);
- **Ultralytics** (AGPL-3.0): not needed for inference (`fab/yolo.py` reads the detector checkpoint and runs the network
  without it); only the `engine` extra (the detector's optional TensorRT engine, which also runs through it) and the
  `eval` extra install it;
- extras: MuJoCo (`retarget`, Apache 2.0); Open3D (MIT), Transformers (Apache 2.0), imageio / imageio-ffmpeg (BSD 2-Clause;
  FFmpeg builds under LGPL / GPL) and trimesh (MIT) (`world`); pycocotools (BSD 2-Clause), pandas (BSD 3-Clause),
  scikit-learn (BSD 3-Clause) and **smplx** (Max Planck non-commercial licence; never vendored here) (`eval`); Weights &
  Biases (`train`, MIT client); TensorRT and NVIDIA Model Optimizer (NVIDIA licences), ONNX / ONNX Runtime (Apache 2.0 /
  MIT), Polygraphy and onnx-graphsurgeon (Apache 2.0), CuPy (MIT) (`engine`);
- separate environments for world mode ([docs/world.md](docs/world.md)): ViPE (Apache 2.0 for NVIDIA's code; it includes
  and downloads components under other licences, among them AGPL-3.0, CC BY-NC-SA 4.0 and CC BY-NC 4.0 parts),
  UniDepth (CC BY-NC 4.0), GeoCalib (Apache 2.0 code, CC BY 4.0 weights), MoGe-2 (MIT), COLMAP (BSD 3-Clause; default
  builds include LSD under AGPL-3.0, CGAL under GPL-3.0+ and SiftGPU for non-profit use), Mask2Former weights from the
  Hugging Face hub (see their model card).

## Files you download

- **SAM 3D Body** model config (gated, SAM License) from Hugging Face, and the **MHR** model (Apache 2.0) from MHR's
  release: `fab.download_sam3d()`.
- **SMPL** model files (Max Planck non-commercial licence, registration), the **H36M joint regressor** from SPIN's data
  release (BSD 3-Clause repository) and **MHR's SMPL conversion assets** (Apache 2.0): `scripts/download_body_models.sh`.
- The **Unitree G1** description (`g1_description` from Unitree's [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
  at commit `157ff950`, BSD 3-Clause, Unitree Robotics; FAB's build `scripts/build_g1.py` adds its own frames, collision
  primitives and scene), and the G1 collision-capsule model of [mjlab](https://github.com/mujocolab/mjlab) (Apache 2.0,
  commit `5e3b2e6c`, `src/mjlab/asset_zoo/robots/unitree_g1/xmls`): `scripts/download_g1.sh`.
- FAB provides the person-detector weights (`shared/detector.pt`) under [WEIGHTS_LICENSE](WEIGHTS_LICENSE).
  They use the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11 architecture and trainer;
  applicable Ultralytics terms remain in effect. Inference uses FAB's own implementation.

