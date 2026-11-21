"""fab-retarget's chain for the inputs it took before the standalone SMPL input (a fab-world output, a skeleton file) is
unchanged: the solve / stance-pin commands scripts/fab_retarget.py runs (recorded, nothing solved) and what it records."""
import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fab_retarget():
    spec = importlib.util.spec_from_file_location("fab_retarget_script", ROOT / "scripts" / "fab_retarget.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fab_world_output(d, c="clip", T=12):
    """A stand-in fab-world output: <d>/<c>.npz (the export's keys) and its work/joint/recon/<c>.npz + _hfield.npz."""
    J = np.zeros((T, 24, 3), np.float32)
    J[:, :, 2] = 0.9
    rd = d / "work" / "joint" / "recon"
    rd.mkdir(parents=True)
    np.savez(d / f"{c}.npz", frames=np.arange(T), T_world_cam=np.tile(np.eye(4), (T, 1, 1)), gravity_world=np.array([0.0, 0, -1]),
             K=np.eye(3), fps=30.0, pose=np.zeros((T, 72), np.float32), betas=np.zeros(10, np.float32),
             trans=np.zeros((T, 3), np.float32), joints=J, valid=np.ones(T, bool), scale_fix=1.0, joint="{}", contacts="{}", notes="{}")
    np.savez(rd / f"{c}.npz", fps=30.0, frames=np.arange(T), valid=np.ones(T, bool), pose=np.zeros((T, 72), np.float32),
             betas=np.zeros(10, np.float32), trans=np.zeros((T, 3), np.float32), joints=J,
             verts=np.zeros((T, 6890, 3), np.float16), contacts="{}")
    np.savez(rd / f"{c}_hfield.npz", x0=-2.0, y0=-2.0, res=0.04, height=np.zeros((100, 100), np.float32),
             valid=np.ones((100, 100), bool))
    return d / f"{c}.npz", rd


class ChainCommandsUnchangedTest(unittest.TestCase):
    def chain(self, src, mode, out):
        """scripts/fab_retarget.py scene() with the solver / pin recorded instead of run and the package step failing (the
        reference stays step 2's) -> (commands, info)."""
        F = fab_retarget()
        cmds = []

        def run(cmd, env=None):
            cmd = [str(x) for x in cmd]
            cmds.append(cmd)
            o = Path(cmd[cmd.index("--out") + 1]) if "--out" in cmd else Path(cmd[cmd.index("--run") + 1]) / cmd[cmd.index("--out_tag") + 1]
            o.mkdir(parents=True, exist_ok=True)
            np.savez(o / f"{Path(src).stem}_motion.npz", scale=1.0)

        def inproc(cmd, env=None):
            cmds.append([str(x) for x in cmd])
            return 1

        F.run, F._inproc = run, inproc
        sys.path.insert(0, str(ROOT))
        a = Namespace(input=str(src), mode=mode, no_graph=False, no_clean=False, ground_aware=False)
        f, info = F.scene(a, Path(src).stem, out)
        return cmds, info, f

    def test_fab_world_output(self):
        from fab.retarget import release_config as C

        for mode in ("mesh", "skeleton"):
            with tempfile.TemporaryDirectory() as d:
                d = Path(d).resolve()
                src, rd = fab_world_output(d / "world")
                out = d / "out"
                cmds, info, f = self.chain(src, mode, out)
                sk = ["--mode", "skeleton"] if mode == "skeleton" else []
                self.assertEqual(cmds[0], [sys.executable, "-m", "bench.videomimic.scene_retarget", "--recon", str(rd), "--only", "clip",
                                           "--out", str(out / "g" / "solve_pre"), "--clean"] + sk
                                 + ["--sigma_in", "8.0", "--cfg", json.dumps(C.solve_cfg())])
                self.assertEqual(cmds[1], [sys.executable, "-m", "bench.stance_pin", "--run", str(out / "g"), "--tag", "solve_pre",
                                           "--out_tag", "p_pre", "--recon", str(rd), "--only", "clip", "--contacts_dir",
                                           str(out / "g" / "_no_head_files"), "--cfg", json.dumps(C.PIN), "--batch_frames", "1500", "--gate"])
                self.assertEqual(cmds[2], [sys.executable, "-m", "bench.scene_r2s_simpkg", "build", "--run", str(out / "g"), "--tags",
                                           "p_pre", "--recon", str(rd), "--clip", "clip", "--out", str(out / "stage" / "clips"),
                                           "--contacts", str(out / "g" / "_no_head_files")])
                self.assertEqual(len(cmds), 3)
                self.assertEqual(f, out / "g" / "p_pre" / "clip_motion.npz")
                self.assertEqual(sorted(info), ["cfg", "chain", "clean", "cuda_graphs", "mode", "pin", "recon", "retargeter", "sigma_in"])
                self.assertEqual(info["recon"], str(rd))
                self.assertEqual(info["clean"], "failed (rc 1): level-snapped terrain only")
                self.assertFalse((out / "smpl_recon").exists())

    def test_skeleton_file(self):
        from fab.retarget import release_config as C

        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            J = np.zeros((10, 24, 3))
            J[:, :, 2] = 0.9
            src = d / "walk.npz"
            np.savez(src, joints=J, fps=30.0)
            out = d / "out"
            cmds, info, f = self.chain(src, "skeleton", out)
            rd = out / "sk_recon"
            self.assertEqual(cmds[0], [sys.executable, "-m", "bench.videomimic.scene_retarget", "--recon", str(rd), "--only", "walk",
                                       "--out", str(out / "g" / "solve_pre"), "--mode", "skeleton", "--sigma_in", "8.0", "--cfg",
                                       json.dumps(C.solve_cfg())])
            self.assertEqual(cmds[1][:13], [sys.executable, "-m", "bench.stance_pin", "--run", str(out / "g"), "--tag", "solve_pre",
                                            "--out_tag", "p_pre", "--recon", str(rd), "--only", "walk"])
            self.assertEqual(len(cmds), 2)
            self.assertEqual(f, out / "g" / "p_pre" / "walk_motion.npz")
            self.assertEqual(info["clean"], "off (a given world)")
            self.assertEqual(info["skeleton_input"], {"skeleton_file": str(src.resolve()), "world": "flat floor at z = 0"})
            self.assertNotIn("smpl_input", info)


if __name__ == "__main__":
    unittest.main()
