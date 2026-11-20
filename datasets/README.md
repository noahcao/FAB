# Evaluation data

No dataset is redistributed. Download each official release yourself, under its own licence and terms, into
`$FAB_DATA` (default `$FAB_HOME/data`), then build the splits with the commands in
[docs/evaluation.md](../docs/evaluation.md).

| dataset | used for | where to get it | notes |
|---|---|---|---|
| 3DPW | camera space; blur / low resolution, each output on its own target, automatic detection | [virtualhumans.mpi-inf.mpg.de/3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) | `$FAB_DATA/3dpw` (sequenceFiles, imageFiles) |
| EMDB | EMDB-1; blur / low resolution, each output on its own target, automatic detection | [eth-ait.github.io/emdb](https://eth-ait.github.io/emdb/) | registration; `$FAB_DATA/emdb` (P*/ folders) |
| RICH | camera space; blur / low resolution | [rich.is.tue.mpg.de](https://rich.is.tue.mpg.de) | registration; test_body.zip, scan_calibration, JPG_images/test.tar.gz |
| EgoBody | more benchmarks | [sanweiliti.github.io/egobody](https://sanweiliti.github.io/egobody/egobody.html) | registration; annotation_egocentric_smpl_npz.zip, egocentric_color.zip |
| Harmony4D | more benchmarks | [jyuntins.github.io/harmony4d](https://jyuntins.github.io/harmony4d/) | test zips; plus SAM 3D Body's harmony4d_test annotations ([facebook/sam-3d-body-dataset](https://huggingface.co/datasets/facebook/sam-3d-body-dataset), gated) |
| SSP-3D | body shape | [github.com/akashsengupta1997/SSP-3D](https://github.com/akashsengupta1997/SSP-3D) | labels.npz + images |
| COCO val2017 | each output on its own target | [cocodataset.org](https://cocodataset.org) | val2017 images + person_keypoints_val2017.json |

Body models (registration): SMPL ([smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)) and, for RICH, the SMPL-X -> SMPL
transfer matrix ([smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de)). They are not redistributed either.
