# Training

How to retrain the released models: the FAB-L / FAB-S body, four heads trained on the frozen body (contact,
visibility, G1, parts), and the bundle that joins them. The tools that built our training labels are not part of the
release; you supply labels in the format below.

```bash
export FAB_DATA=/path/to/your/data          # pools, pool lists, sidecars (default: $FAB_HOME/data)
python -m train.train_fab --config configs/train/fab-l.yaml --out runs/fab-l-body          # 1. the body
for h in contact visibility g1 parts; do                                                    # 2. the heads
  python -m train.train_fab --config configs/train/heads-$h.yaml --init runs/fab-l-body/model.pt --out runs/fab-l-$h
done
python scripts/build_bundle.py --body runs/fab-l-body/model.pt --contact runs/fab-l-contact/model.pt \
  --vis runs/fab-l-visibility/model.pt --g1 runs/fab-l-g1/model.pt --parts runs/fab-l-parts/model.pt \
  --out runs/fab-l-bundle/model.pt --check_images "check_frames/*.jpg"                     # 3. one model.pt
```

- FAB-S: `fab-s.yaml` and the `-s` head configs (pass the FAB-S body as `--init`).
- `build_bundle.py` checks that each head run left the body unchanged; `--check_images` also compares outputs on real images.
- Several GPUs: `torchrun --standalone --nproc_per_node <N> -m train.train_fab --config ...`. `batch` is per process;
  the released runs used 4 processes (visibility: 8), so scale `batch` if you use another count.
- Runs resume from `<out>/last.pt`. Every config key is a `train/train_fab.py` argument (`--help`); command-line flags
  override the config. Configs may use `${FAB_DATA}` / `${FAB_HOME}`. Weights & Biases logging is off unless you pass
  `--wandb_project`.

## Requirements

- CUDA GPUs (bf16). The FAB-L body fits 24 GB per process at batch 8 (gradient checkpointing).
- `pip install -e ".[train]"`.
- SAM 3D Body's full Hugging Face folder, including `model.ckpt` (the decoder starts from it and its backbone is the
  feature teacher); default `$FAB_SAM3D`. DINOv3 weights download through timm on first use (DINOv3 License).
- `$FAB_HOME/body_models` with SMPL neutral, male and female, the H36M regressor and the MHR mapping (README, Install).
  `bm_dir: auto` points there.
- The 23-joint foot regressor (`foot_reg: auto` = `$FAB_HOME/extras/mhr23_regressor.npz`). It depends on SMPL, so
  it is not in the download; build it once (it is fitted like the released one, so numbers can differ slightly):
  ```bash
  python scripts/make_mhr23_regressor.py --root $FAB_DATA/3dpw
  ```
  `--foot_reg ''` uses SMPL foot vertices instead (not the released recipe).
- The G1 description (`scripts/download_g1.sh`) for the body run and the G1 head.

## Released configs

| config | steps | batch / process | lr (head / decoder / backbone) | trains | init |
|---|---|---|---|---|---|
| `fab-l.yaml` | 40,000 | 8 | 5e-4 / 2e-5 / 5e-5, warm-up 500, cosine, EMA 0.9998 | everything (ViT-L/16) | SAM 3D Body + DINOv3 |
| `fab-s.yaml` | 40,000 | 8 | same | everything (ViT-S+/16) | same |
| `heads-contact.yaml` | 4,000 | 64 | 1e-3 / 0 / 0, warm-up 100 | contact head (16 outputs) | the body |
| `heads-visibility.yaml` | 16,000 | 8 (8 processes) | 5e-4 / 0 / 0, warm-up 100 | visibility head (+ human-flag classifier) | the body |
| `heads-g1.yaml` | 6,000 | 16 | 3e-4 (only the G1 head trains), warm-up 200, EMA 0.999 | G1 head | the body |
| `heads-parts.yaml` | 4,000 | 32 | 1e-3 / 0 / 0, warm-up 200 | part head (6 convs) | the body |

The configs hold the released runs' arguments; only data paths and label-set names are placeholders. Augmentation:
flips 0.5, in-plane rotation 0.5 (20 % of those over the full circle), loose boxes 0.5, colour jitter and low light.
The 2.5D lift (`dense_25d: 1.0`) ramps in over the first 5,000 steps. Two settings are yours to fill:
- `pools` / `pool_list` (`"<glob>@N"` repeats a glob N times). The released body mixed public datasets with human labels
  (COCO 2017 with COCO-WholeBody 2D, MPII, AIST++, AthletePose3D, BEDLAM, 3DPW train x3, Harmony4D, EgoBody incl. its
  egocentric view, MOYO, RICH train, Embody3D and Human-Art train with their human 2D), sets with teacher labels only
  (COCO-2014 and AGORA images with detector boxes, ImageNet, frames from publicly available web videos, SA-1B), and
  hard examples mined and labelled by FAB-H (repeated 2-5x).
- `skip_takes` / `skip_pairs`: sample-key regexes / `<pool dir>|<key>` lines to skip. The released run skipped the
  benchmark test takes that also appear in training sources.

## Data format

Training reads pool tars (one crop per sample) and label sidecars next to them; the fields are in
[training_data.md](training_data.md).

## Smoke test

`scripts/train_smoke.py` writes a few synthetic samples, trains the FAB-S body for 3 steps and a contact head for 2 on
one GPU, checks the losses and checkpoints, and builds a bundle. It needs the SAM 3D Body config + MHR model, SMPL and
the G1 description, and skips the pretrained downloads (`--no_pretrained --no_teacher_weights`).

```bash
python scripts/train_smoke.py --out /tmp/fab_train_smoke
```
