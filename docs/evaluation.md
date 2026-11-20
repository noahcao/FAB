# Evaluation

How to score FAB on the camera-space benchmarks. `scripts/eval_paper.sh` runs predictions (`bench.eval_hmr`) and the
scorer for each table section, one shard per local GPU, resumable. No dataset is redistributed: you download each
official release ([datasets/README.md](../datasets/README.md)) and build the splits below.

```bash
export FAB_DATA=$FAB_HOME/data            # default ~/.cache/fab/data
SEC=std bash scripts/eval_paper.sh        # sections: std, hoff, degrade, auto, ssp3d, coco, parts
```

Outputs go to `$FAB_OUT` (default `$FAB_HOME/bench`): predictions in `preds/<model>/`, scores in
`scores/<section>/standard_<dataset>.json` and `standard.md`, logs in `logs/`.

## Setup

```bash
pip install -e ".[eval]"                                  # includes smplx (non-commercial licence, THIRD_PARTY_NOTICES.md)
python -c "import fab; fab.download_sam3d()"
git clone https://github.com/facebookresearch/MHR $FAB_HOME/src/MHR    # MHR's official MHR -> SMPL fit
```

- **SMPL** in `$FAB_HOME/body_models/smpl/`: `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` (registration at
  [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)) and `J_regressor_h36m.npy`. The ground truth is gendered SMPL.
- **MHR -> SMPL mapping** in `$FAB_HOME/body_models/mhr/` (README, Install).
- **SMPL-X -> SMPL transfer** for RICH: `$FAB_HOME/body_models/smplx_transfer/smplx_to_smpl.pkl` from
  [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) ("model transfer").
- `$FAB_MHR_TOOLS` overrides the MHR tool path (only needed for MHR-mesh predictions and SSP-3D).

## Protocol

- **3DPW:** 14 H36M joints, pelvis = hip midpoint (MPJPE, PA-MPJPE); PVE after the same alignment.
- **EMDB-1, RICH test, EgoBody, Harmony4D:** 24 SMPL joints, pelvis aligned; 14-joint rows too.
- **Predictions:** FAB-L / FAB-S as their fitted SMPL (`--predictor --predictor_kw FINAL`); FAB-H as MHR mesh → SMPL
  (`--predictor`, model `fab-h`) and dense points → SMPL (`--predictor_kw '{"profile":"points"}'`, model `fab-h-points`).
  SAM 3D Body's MHR mesh and FAB-H's "official" row go through MHR's official converter.
- **Boxes and cameras:** the dataset's boxes and ground-truth intrinsics. `SEC=auto` uses the released detector instead
  (every 10th frame; matched at IoU >= 0.3; recall reported).
- Every method is scored on the samples all methods share.

## Data (`$FAB_DATA`)

| folder | content | build |
|---|---|---|
| `3dpw/` | the 3DPW release | download only |
| `emdb/` | the EMDB release | download only |
| `rich_test/` | RICH test frames + index (every 30th fitted frame, all cameras, half resolution) | the three commands below |
| `egobody_test/` | EgoBody egocentric test frames + index | `python -m bench.build_egobody_test --egobody <release> --out $FAB_DATA/egobody_test` |
| `harmony4d_test/` | Harmony4D test frames + index (SAM 3D Body's annotations) | `python -m bench.build_harmony4d_test --ann <annotations>/harmony4d_test --zips "<release>/test/*.zip" --out $FAB_DATA/harmony4d_test` |
| `ssp3d/` | SSP-3D labels + images + index | `python -m bench.eval_ssp3d build --src <SSP-3D> --out $FAB_DATA/ssp3d` |
| `coco_val/` | COCO val2017 images + `annotations/person_keypoints_val2017.json` | download only |

```bash
python -m bench.build_rich_test gt --data $FAB_DATA/rich --out $FAB_DATA/rich_test --stride 30
python -m bench.build_rich_test frames --archive $FAB_DATA/rich/JPG_images/test.tar.gz --out $FAB_DATA/rich_test
python -m bench.build_rich_test index --data $FAB_DATA/rich --out $FAB_DATA/rich_test
```

## Expected numbers

The paper's values (PA-MPJPE / MPJPE, mm, unless noted); tables are named by their captions. Your means may differ by a
few hundredths of a mm (GPU kernels, compilation).

| section | paper table | sets (stride) | FAB-L | FAB-S | FAB-H, MHR mesh → SMPL | FAB-H, dense points → SMPL |
|---|---|---|---|---|---|---|
| `std` | camera space: 3DPW (14 joints) | 3dpw (4) | 32.7 / 53.1 | 37.7 / 59.8 | 33.1 / 53.2 | 33.7 / 54.8 |
| `std` | camera space: RICH test (24 joints) | rich_test (2) | 27.8 / 45.5 | 35.4 / 56.4 | 28.2 / 45.6 | 29.8 / 48.2 |
| `std` | EMDB-1 (appendix; 24 joints) | emdb1 (2) | 39.2 / 61.0 | 45.9 / 69.7 | 36.9 / 55.5 | 40.0 / 59.4 |
| `std` | more benchmarks: Harmony4D / EgoBody (24 joints) | harmony4d_test, egobody_test (1) | 33.2 / 42.1, 38.8 / 55.6 | -- | 37.8 / 54.6, 40.3 / 58.6 | 36.7 / 49.8, 40.4 / 58.7 |
| `hoff` | camera space and EMDB-1: FAB-H, MHR mesh → SMPL, official | 3dpw, rich_test, emdb1 | -- | -- | 32.4 / 52.7, 27.9 / 46.0, 36.8 / 55.3 | -- |
| `degrade` | blur / low resolution (appendix): MPJPE clean / blur / low-res | 3dpw, emdb1, rich_test | 53.1 / 61.5 / 55.4, 63.0 / 66.9 / 63.8, 47.8 / 72.7 / 51.8 | -- | 53.2 / 64.1 / 56.3, 57.4 / 64.2 / 60.7, 47.3 / 75.6 / 53.3 | 54.8 / 65.1 / 57.3, 60.1 / 66.5 / 62.9, 50.3 / 77.3 / 55.4 |
| `ssp3d` | body shape: SSP-3D PVE-T-SC | ssp3d (1) | 13.3 | -- | 10.0 | 12.9 |
| `coco` | each output on its own target (appendix): COCO AP / AP75 | coco_val | dense-head 2D 62.5 / 69.8; projected 54.2 / 60.4 | -- | -- | -- |
| `parts` | each output on its own target (appendix): part mIoU / silhouette IoU | 3dpw, emdb1 (5) | 0.446 / 0.677, 0.511 / 0.762 | -- | -- | -- |

`SEC=auto` (accuracy on the people the detector finds, on 3dpw / emdb1 and their dark versions) has no paper table.
`SEC=parts` also prints an `emdb_dev` row (your EMDB sequences outside EMDB-1); no expected value is listed.

## Baselines

SAM 3D Body, Fast SAM 3D Body and NLF-L run through the same scripts with their official code and weights, installed
by you: `--method sam3db` (needs SAM 3D Body's `model.ckpt`), `--method fast3db --fast3db_root <checkout>`,
`--method nlf --nlf_model <torchscript>`. The video baselines (TRAM, GVHMR, GEM-X, WHAM) are not in this release;
neither are the world, contact, scene and G1-tracking pipelines or the FAB-Hard index files ([reproduce.md](reproduce.md)).
