# End-effector retargeting on a flat floor: `fab-retarget-ee`

`fab-retarget-ee` runs the same solver and stance pin as [`fab-retarget`](retarget.md), plus targets for the G1's
hands and feet, so the robot reaches and touches where the person did. It works on a flat floor only; use
`fab-retarget` for scenes, slopes and stairs. Its input is a `fab-world` output (or its recon `<clip>.npz`).

```bash
fab-retarget-ee output/world/clip.npz --out output/g1_ee                          # the person's own hands and soles
fab-retarget-ee output/world/clip.npz --out output/g1_ee --mode mesh              # no dense-point term
fab-retarget-ee output/world/clip.npz --out output/g1_ee --ee_targets tracks.npz  # your own end-effector tracks
fab-retarget-ee output/world/clip.npz --out output/g1_ee --hand_windows windows.json
```

| option | default | effect |
|---|---|---|
| `--mode` | `points` | `points` or `mesh`, as in fab-retarget |
| `--ee_targets` | the person's hand joints (SMPL 22 / 23) and soles | an npz with `frames` (input frame ids) and any of `left_hand`, `right_hand`, `left_foot`, `right_foot` (T, 3), metres, input world, each with `<name>_valid` (T,) |
| `--hand_windows` | `auto` | frames where a hand is held on the floor: `auto` (lowest hand vertex within 4 cm of the floor), `off`, or a json `{"windows": {"left": [{"frames": [first, last]}], "right": [...]}}` |
| `--ee_scale` | `none` | `none`: the G1 keeps the person's horizontal positions (heights scale with the robot); `s`: scale everything, as fab-retarget |
| `--w_ee_hand`, `--w_ee_foot` | 1000, 100 | target weights (`--w_ee_foot 0`: hands only) |
| `--secondary` | 1 | factor on fab-retarget's direction and dense-point weights |
| `--ground_aware` | off | fab-retarget's ground-aware feet ([retarget.md](retarget.md#ground-aware-references---ground_aware)); planted feet outweigh the hand / foot targets |
| `--out`, `--no_graph` | | as fab-retarget |

## How it works

- **Floor.** The motion stands on one flat floor at the person's ground (the median of the lowest foot vertex). A
  scene in the input is ignored with a note; the command warns if the person's ground spans more than 25 cm.
- **Targets.** The G1's hand point is the midpoint of its thumb and pinky points; its foot point is the centre of its
  four sole points. A person's sole target is the foot's lowest point. Weights live in `fab/retarget/release_config.py` (`EE_W`).
- **Hand-floor windows.** Short gaps (under 0.1 s) are filled and short runs (under 0.2 s) dropped. In a window the
  hand is held on the floor by the contact terms instead of its position target.
- **Steps.** Solve and stance pin only; no terrain clean.

The foot weight and the hand-window rule were set on one hand-on-floor clip, separate from the evaluation below.

Outputs: `<out>/<clip>_g1.npz` (fab-retarget's keys; the floor is at s x the person's ground), `<clip>_g1.json`
(inputs, settings, s, floor height), `<clip>_ee_targets.npz` and `<clip>_hand_windows.json`.

## Measured

On the 25 clips of the paper's G1 tables (their fab-world reconstructions), both retargeters run on the same flat floor at
the person's ground (`fab-retarget`: points mode, `--no_clean`). Positions are against the person's hands and soles (pelvis
frame: relative to the root, x s); the hand-floor gap is measured in the automatic windows (19 on 10 clips); SONIC-G1 tracks
each reference in MuJoCo on a flat floor (falls of 25, bone error before the first fall).

| | fab-retarget points | fab-retarget-ee |
|---|---|---|
| hand position, horizontal / vertical (cm) | 8.7 / 7.3 | 6.1 / 3.8 |
| foot position, horizontal / vertical (cm) | 11.4 / 8.4 | 3.2 / 5.7 |
| hand / foot position, pelvis frame (cm) | 10.1 / 10.7 | 11.5 / 9.7 |
| hand-floor gap in the windows: median, windows within 3 cm, worst | 4.1 cm, 6 / 19, 16.3 cm | 0.2 cm, 15 / 19, 11.0 cm |
| bone / part direction error vs the input body (deg) | 5.1 / 18.2 | 9.8 / 21.7 |
| joint jitter (rad/s^2) / sole slip (m/s) / self-penetration (% frames) | 3.0 / 0.24 / 6.7 | 4.1 / 0.59 / 7.4 |
| reference bone error vs the real person (deg) | 13.2 | 15.2 |
| SONIC-G1 falls / bone error (deg) | 5 / 21.1 | 7 / 21.5 |

`fab-retarget-ee` improves world-frame hand and foot positions and hand-floor contact; it costs bone-direction accuracy,
pelvis-frame hand accuracy, smoothness (jitter, sole slip) and 2 more SONIC-G1 falls. Use `fab-retarget` for whole-body
motion and scenes.
