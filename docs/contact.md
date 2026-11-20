# Contacts

FAB-L and FAB-S predict when the heels, toes, hands and knees touch a surface. An image contact head runs on every
person crop, and a video contact model refines it over each person's track. FAB-H has no contact head.

## Channels

Eight regions (`fab.contact_regions.CONTACT_NAMES`): `l_heel, l_toe, r_heel, r_toe, l_hand, r_hand, l_knee, r_knee`.
Each has two probabilities, and the 16 outputs are the 8 **static** ones, then the 8 **any** ones:
- **static:** touching a surface without sliding (a planted foot);
- **any:** touching a surface, moving or not.

## Where you get them

- `contact` (16,) in every person's outputs from `fab` / `FABPredictor` ([usage.md](usage.md)): the image head.
- `contact_video` (16,) from `fab-video`: the video model in the camera frame. Use it for a static camera; tracks
  shorter than 4 frames get none.
- `fab-world`: the video model in the reconstructed world, which handles a moving camera. Probabilities are in
  `<out>/work/contacts/<name>__0.contact.npz`; contact labels on the scene are in the result's `contacts` ([world.md](world.md)).

The video model (`shared/contact_video.pt`) is a causal GRU: each frame uses only that frame and earlier ones. It adds a
correction to the image head's output, from the head's features and the motion of the contact keypoints. It expects
cameras in the first camera's frame (`T_world_cam[0]` = identity, OpenCV axes), as ViPE gives them.

## Training

Both stages train on contact labels in the pool tars' sidecars ([training_data.md](training_data.md)). The
tools that built our labels are not part of this release. The sidecar's `c` gives static contact and `h` the height
above the local floor (any contact: `h` < 4 cm for feet, 5 cm for hands and knees).

Image head, on the frozen body (FAB-S: `heads-contact-s.yaml`; steps and batch: [training.md](training.md)):

```bash
python -m train.train_fab --config configs/train/heads-contact.yaml --init runs/fab-l-body/model.pt --out runs/fab-l-contact
```

Video model: extract per-track head outputs, then train the GRU.

```bash
python -m train.extract_contact_tracks --source pool:"<pool dir>/*.tar" --name <set> --fab_ckpt <model.pt> --out <track dir>
python -m train.train_contact_video --tracks "<track dir>/<set>,..." --val "<track dir>/<val set>" --out <run dir> \
    --vel_win 0.1 --w_smooth 1.0
```

Pass `--vel_win` and `--w_smooth`: they default to 0. The released model used these settings with 48-frame windows at
frame strides 1-3, batch 64, 6,000 steps and learning rate 1e-3, on RICH train, EgoBody, BEDLAM and MOYO tracks, with
EMDB-2 for validation. The script checks the camera convention above and stops if it is wrong. The extractor also reads
benchmark sets (`--source bench:<set>`) and unlabelled clip lists (`--source clips:<json>`).

The paper's contact tables use labels and a scorer that are not in this release ([reproduce.md](reproduce.md)).
