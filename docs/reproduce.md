# Reproducing the paper

What each table of the paper needs and what this release can reproduce. Expected numbers: [evaluation.md](evaluation.md).
Setup (SMPL, the MHR tools, the datasets): [evaluation.md](evaluation.md#setup).

The release holds the code, the released weights and the builders of the evaluation splits. It holds no dataset
images, stored predictions or development checkpoints: every number comes from a new run. The paper's numbers use an
`SMPL_NEUTRAL.pkl` with SHA256 `98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42`.

Status: **re-run** = run the released weights and the listed scorer; **timing** = device-dependent (the paper's fps
come from one RTX 4070 Ti SUPER with TensorRT engines); **not in this release** = reason given.

## Main text

| table | status | command |
|---|---|---|
| Camera space (3DPW, RICH test): FAB-L / FAB-S / FAB-H fitted SMPL (FAB-H: MHR mesh → SMPL and dense points → SMPL) | re-run | `SEC=std bash scripts/eval_paper.sh` |
| Camera space: FAB-H, MHR mesh → SMPL, official | re-run | `SEC=hoff bash scripts/eval_paper.sh` |
| Camera space: baselines (SAM 3D Body, Fast SAM 3D Body, NLF-L) | re-run with their official code | evaluation.md, Baselines |
| Camera space: video baselines (TRAM, GVHMR, GEM-X, WHAM) | not in this release | their harnesses are not released |
| Camera space: fps (and the appendix's speed table) | timing | `fab-build-trt`, then [acceleration.md](acceleration.md#speed) |
| World space (EMDB-2, RICH-191) | not in this release | `fab-world` gives the world body per clip; the world-space scorer and the ground-truth track builder are not released yet |
| More benchmarks (Harmony4D, EgoBody) | re-run | `SEC=std bash scripts/eval_paper.sh` (`std2` block) |
| Body shape (SSP-3D) | re-run | `SEC=ssp3d bash scripts/eval_paper.sh` |
| Foot contact F1 | not in this release | the contact label and scoring pipeline is not released |
| Scenes | not in this release | needs the scene baselines and their protocols' data |
| G1 references tracked in MuJoCo | references: re-run ([below](#g1-references)); tracking: not in this release | the tracking needs third-party policies |
| Robustness on FAB-Hard (SAM 3D Body, Fast SAM 3D Body, FAB-L) | not in this release | the FAB-Hard index files are pending a release decision |

## Appendix

| table | status | command |
|---|---|---|
| FAB-Hard | not in this release | the FAB-Hard index files are pending a release decision; the loaders are in `bench/datasets.py` (`fabhard_*`) |
| Hard poses and low light | not in this release | the low-light sensor model and the hard-pose index (5 EMDB-1 sequences) are not released yet |
| Blur / low resolution | re-run | `SEC=degrade bash scripts/eval_paper.sh` |
| EMDB-1 | re-run | `SEC=std bash scripts/eval_paper.sh` (FAB-H, MHR mesh → SMPL, official: `SEC=hoff`) |
| Each output on its own target: COCO 2D | re-run | `SEC=coco bash scripts/eval_paper.sh` |
| Each output on its own target: body parts | re-run | `SEC=parts bash scripts/eval_paper.sh` |
| Each output on its own target: ExLPose, G1 joints / bones, COCO-WholeBody feet | not in this release | their scorers are not released yet |
| Person detectors | not in this release | the detector comparison's scorer is not released |
| Scene details (per clip, focal match, PROX, VideoMimic start pose) | as Scenes | see the Scenes row |
| Automatic detection | re-run | `SEC=auto bash scripts/eval_paper.sh` |
| Legs and feet | not in this release | its sets include training-pool data |
| fps by number of people | timing | [acceleration.md](acceleration.md#speed) |
| World-space subsets and the RICH-191 cause split | not in this release | as world space |
| G1 tracking details | not in this release | as G1 references |
| FAB-Retarget's three modes (C.5) | references: re-run ([below](#g1-references)); tracking: not in this release | as G1 references |

## G1 references

The paper's G1 tables retarget each clip's world body on a flat floor (z = 0, 50 fps) with this release's solver and
stance pin, pinning the 25 clips together in sorted order (batches of up to 1,500 frames). To rerun this on your own
`fab-world` outputs of the 25 clips (`bench/sonic_clips.py` `FLAT_CLIPS`; datasets under `$FAB_DATA`):

```bash
python scripts/g1_refs_inputs.py frames <clip> work/<clip>          # the clip's frames + the dataset's subject boxes
fab-world work/<clip>/frames --name <clip> --fps <fps> --subject_boxes work/<clip>/subject_boxes.npz --out world/<clip> --profile paper
python scripts/g1_refs_inputs.py collect world/<clip> inputs <clip>  # <clip>.npz + its work/ours/fab/<clip>_raw.npz
bash scripts/reproduce_g1_refs.sh inputs refs_out                    # all 25 clips in one call -> refs_out/refs/
```

- `--profile paper` runs the world pipeline the paper used; without it, `fab-world` runs this release's faster default.
- For the 14 3DPW and EMDB clips, `frames` prints the `fab-world` command with the clip's fps.
- For the 7 RICH and 4 MOYO clips, pass your dataset copy with `--src`; `frames` writes the frames the paper used
  (`SOURCES` in `scripts/g1_refs_inputs.py`). RICH: the extracted JPEG release (`val/<seq>/cam_XX/<frame>_XX.jpeg`),
  every 3rd frame at half resolution, 10 fps. MOYO: the sequence's val image zip or folder (`Cam_<cam>_<frame>.jpg`),
  every frame, 30 fps. Run `fab-world` on them without `--subject_boxes`; the paper used calibrated subject boxes there,
  so these inputs differ more.
- Give `reproduce_g1_refs.sh` all 25 clips in one call: the stance pin depends on which clips share a batch. Results
  come out close to the paper's but not bit-exact, because FAB's fast path varies between processes.

`fab-retarget`'s default chain (for your own videos) retargets onto the reconstructed scene instead, so its references
for these clips differ: tracked by SONIC-G1 they give 6 falls and a 19.2° bone error, against the table's 4 and 18.3°
(points mode).
