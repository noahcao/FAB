# Unitree G1 retargeting: `fab-retarget`

`fab-retarget` turns a `fab-world` output into a Unitree G1 reference motion on the reconstructed scene. It needs the G1
description (`bash scripts/download_g1.sh`, once).

```bash
fab-retarget output/world/steps_up.npz --out output/g1                        # points mode (default)
fab-retarget output/world/steps_up.npz --out output/g1_mesh --mode mesh
fab-retarget output/world/steps_up.npz --out output/g1_skeleton --mode skeleton
fab-retarget my_motion.npz --out output/g1_my_motion --mode skeleton          # a skeleton file from any source (below)
fab-retarget output/world/steps_up.npz --out output/g1_noclean --no_clean      # skip the terrain clean
```

| mode | what the solver reads |
|---|---|
| `points` (default) | the joints, the fitted mesh and FAB's dense surface points |
| `mesh` | the joints and the fitted mesh |
| `skeleton` | the joints only |

One solver and one configuration (`fab/retarget/release_config.py`) serve all three; the paper's table "FAB-Retarget's
three modes" (appendix C.5) compares them. Finished steps in `--out` are reused, so give each run its own `--out`. `--no_graph` runs
without CUDA graphs (same result).

## Steps

1. **Solve** a whole-body G1 motion on the scene's heightfield: bone directions, foot contacts (a foot's lowest point
   within 4 cm of the ground and slower than 0.15 m/s), smoothness and robot limits; points mode adds the dense-point term.
2. **Stance pin:** pin each planted foot to the terrain, with a MuJoCo validity check.
3. **Simulation package** of the scene on a 2 cm grid.
4. **Terrain clean** along the robot's path: remove spikes, slivers and pits.
5. **Solve and pin again** on the clean terrain.

**The dense-point term** gives each of FAB's dense points a fixed anchor on the G1's surface and turns each G1 part
the way its points turn, so limb twist follows the points. The anchors are built on first use into
`$FAB_HOME/extras/dense_corr_g1.npz` (by hand: `python -m fab.retarget.dense_corr`). Inputs without FAB's dense
points (e.g. motion capture) use the body's vertices instead, with a WARNING; `<clip>_g1.json` records which
(`dense.points`, `dense.no_points_cause`).

## Skeleton mode and skeleton files

Skeleton mode replaces the mesh with joint rules sized from the SMPL template. A skeleton file `<clip>.npz` holds:
- `joints`: (T, >= 24, 3), SMPL-24 order, metres, z-up (rotate y-up data first);
- `fps`: any rate (resampled to 50);
- optionally `frames`: (T,) row ids.

An optional `<clip>_hfield.npz` next to it gives the terrain (`x0`, `y0`, `res`, `height` (ny, nx) in metres; cell (i, j)
at `(x0 + j res, y0 + i res)`); without it the floor is flat at z = 0. A given terrain is used as is (no clean).

Joints are in SMPL-24 order; the solver reads the ones in the table below (others may hold any value). Other
skeletons map as follows:

| SMPL index | mocap / BVH | COCO-WholeBody / H36M-style 3D |
|---|---|---|
| 0 pelvis | Hips | mid-hip (H36M Hip) |
| 1 / 2 hips | LeftUpLeg / RightUpLeg | L / R hip |
| 3 spine1 | Spine | H36M Spine (else 0.6 pelvis + 0.4 neck) |
| 4 / 5 knees | LeftLeg / RightLeg | L / R knee |
| 7 / 8 ankles | LeftFoot / RightFoot | L / R ankle |
| 10 / 11 toe base | LeftToeBase / RightToeBase | L / R big-toe / small-toe midpoint |
| 12 neck, 15 head | Neck, Head | neck (thorax), head |
| 16 / 17 shoulders | LeftArm / RightArm | L / R shoulder |
| 18 / 19 elbows | LeftForeArm / RightForeArm | L / R elbow |
| 20 / 21 wrists | LeftHand / RightHand | L / R wrist |
| 22 / 23 hands | LeftHandMiddle1 (or = wrist) | hand middle-MCP (or = wrist) |

Non-SMPL skeletons work as well as their joints sit where SMPL's do; mocap ankle or toe markers can be a few cm off.

## Ground-aware references: `--ground_aware`

Off by default. On flat floors, it keeps the G1's feet down when the reconstructed person floats above the reconstructed
ground: `fab-retarget output/world/clip.npz --out output/g1_ground --ground_aware` (also `fab-retarget-ee`). It reads
fab-world's foot contacts (`<out>/work/contacts`) and stops if they are missing.

- A foot that FAB's video contact model calls planted (heel or toe static contact >= 0.5; short gaps are filled and short
  runs dropped, as in the stance pin) is held at the terrain under it without sliding. When no foot is planted but the model
  says a foot touches, or the person's lowest foot is under 10 cm above the terrain, the lower G1 foot is held at the terrain.
  The stance pin keeps these feet down too.
- Use it for flat floors and upright motion. On fab-world reconstructions of the paper's 25 G1 clips (headstand and cartwheel,
  whose feet really leave the ground, not counted), the time with both soles more than 2 cm above the ground drops from 178.5 s
  to 25.2 s (fab-retarget) and from 75.6 s to 8.5 s (fab-retarget-ee), and foot slip on planted frames drops by 40-50 %. On uneven
  terrain (curbs, stairs) and in lying or kneeling poses it can press the legs into the ground: frames with a link more than 1 cm
  below the terrain go from 66 to 530 (fab-retarget). Its constants were chosen on 3 of the 25 clips.
- Tracked by SONIC-G1, the references fall more often: 8 instead of 6 of 25 falls and a 22.1° instead of 19.2° bone error
  (fab-retarget-ee: 8 instead of 7, 27.8° instead of 21.5°).

## Output

`<out>/<clip>_g1.npz` at 50 fps: `g1_dof` (T, 29) joint angles (rad, URDF order), `g1_root_pos` (T, 3) and
`g1_root_quat_xyzw` (T, 4), the root in the scene's z-up metric world. `<out>/<clip>_g1.json` records the inputs, mode
and settings. A scene without a valid heightfield skips the clean step and keeps the first solve.

## Notes

- The stance pin solves clips in batches, so a reference depends slightly on its batch-mates (up to a few cm at the
  root). `fab-retarget` pins each clip alone; to reproduce a batch exactly, pin the same clips in the same order.
- The G1 files are built from Unitree's `g1_description` ([unitree_ros](https://github.com/unitreerobotics/unitree_ros),
  pinned commit, BSD 3-Clause) by `scripts/build_g1.py`; its header lists every rule. The self-collision term uses
  [mjlab](https://github.com/mujocolab/mjlab)'s G1 capsules (Apache 2.0).
