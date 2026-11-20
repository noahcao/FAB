"""The fused terrain lookups (fab/retarget/req_kernel.py) give the same bits as the PyTorch expressions they replace: the stance
pin's Surface.req (values and the gradient through a penetration-like loss) and the solver's Terrain.forward, on synthetic
stepped / sloped terrains with an awkward cell size (the PyTorch scalar-division path), grid-node queries (ties, clamps) and
queries off the grids. Needs a CUDA GPU and Triton (skipped otherwise)."""
import os
import unittest

import numpy as np
import torch

from fab.retarget.req_kernel import AVAILABLE

RUN = AVAILABLE and torch.cuda.is_available()


def terrain(res, ny=180, nx=260, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    H = 0.03 * np.sin(xx / 17.0) + 0.17 * (xx // 40) + np.where((yy > 60) & (yy < 90), 0.4, 0.0) + rng.normal(0, 0.002, (ny, nx))
    H[:, :30] = 0.0  # a flat floor (ties between the package and slot grids)
    return {"x0": -1.37, "y0": -2.11, "res": res, "height": H}


@unittest.skipUnless(RUN, "needs CUDA + Triton")
class ReqKernel(unittest.TestCase):
    def _cmp(self, fn, flag):
        out = []
        for v in ("0", "1"):
            os.environ[flag] = v
            out.append(fn())
        os.environ.pop(flag, None)
        for a, b in zip(*out):
            self.assertTrue(torch.equal(a, b), float((a - b).abs().max()))

    def test_surface_req(self):
        from bench import stance_pin as SP

        for res in (0.02, 0.0313592):
            tg = terrain(res)
            root = np.stack([np.linspace(-0.5, 4.0, 300), np.linspace(-1.0, 1.0, 300), np.full(300, 0.8)], 1)
            surf = SP.Surface([SP.surfaces(tg, root[:, :2])], "cuda")
            g = torch.Generator(device="cuda").manual_seed(1)
            T, n = 300, 120
            rt = torch.as_tensor(root, dtype=torch.float32, device="cuda")
            for mode in ("rand", "nodes", "far"):
                xy = rt[:, None, :2] + (torch.rand(T, n, 2, device="cuda", generator=g) - 0.5) * (8.0 if mode == "far" else 2.0)
                if mode == "nodes":
                    xy = torch.round((xy - surf.g[0]["x0"][0]) / surf.g[0]["res"][0]) * surf.g[0]["res"][0] + surf.g[0]["x0"][0]
                z = torch.rand(T, n, device="cuda", generator=g) * 0.6
                r = torch.rand(n, device="cuda", generator=g) * 0.05 + 0.01
                cid = torch.zeros(T, 1, dtype=torch.long, device="cuda")
                const = surf.req_const(r, cid, (T, n), torch.float32, torch.device("cuda"))
                w = torch.linspace(-1, 1, n, device="cuda")

                def run():
                    W = torch.cat([xy, z[..., None]], -1).clone().requires_grad_(True)
                    q = surf.req_cols(W[..., :2], surf._slot(W[..., :2], const[1]), const)
                    q2 = surf.req(W[..., :2], r, cid)
                    ((torch.relu(0.2 - (W[..., 2] - q)) ** 2).sum(-1).mean() + (q * w).sum() * 1e-3 + (q2 * w).sum() * 1e-3).backward()
                    return q.detach(), q2.detach(), W.grad.detach()

                self._cmp(run, "PIN_REQ_FUSED")

    def test_solver_terrain(self):
        from bench.videomimic.scene_retarget import Terrain

        for res, scale in ((0.02, 1.0), (0.04, 0.7839), (0.0313592, 1.0)):
            terr = Terrain(terrain(res), scale)
            ny, nx = terr.H.shape
            g = torch.Generator(device="cuda").manual_seed(2)
            lo = torch.tensor([terr.x0, terr.y0], device="cuda")
            span = torch.tensor([nx * terr.res, ny * terr.res], device="cuda")
            for mode in ("rand", "nodes", "far"):
                xy = lo + (torch.rand(50000, 2, device="cuda", generator=g) * (1.6 if mode == "far" else 1.0) - (0.3 if mode == "far" else 0.0)) * span
                if mode == "nodes":
                    xy = torch.round((xy - lo) / terr.res) * terr.res + lo
                w = torch.linspace(-1, 1, len(xy), device="cuda")

                def run():
                    x = xy.clone().requires_grad_(True)
                    h = terr(x)
                    (h * w).sum().backward()
                    return h.detach(), x.grad.detach()

                self._cmp(run, "SOLVE_TERR_FUSED")


if __name__ == "__main__":
    unittest.main()
