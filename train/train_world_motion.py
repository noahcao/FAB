"""Train FAB-M (fab/world_motion.py) on clean motions (train/world_motion_data.py) with simulated FAB inputs
(train/world_motion_sim.py). Clips are split into train / val by a hash of their name (--val_frac).

    python -m train.train_world_motion --data bedlam.npz,embody3d_loc.npz --smpl SMPL_NEUTRAL.pkl --out runs/wm \
        [--steps 50000 --bs 64 --W 192 --lr 3e-4] [--noise_json noise.json]
    -> <out>/model.pt (the deployable checkpoint: cfg + weights), <out>/log.jsonl
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fab.world_motion import SMPLKin, WorldMotionNet, build_features, decode, fuse_dense
from train.world_motion_sim import NOISE, Motions, simulate

ANCHOR = None  # --anchor (rel, perp)
LW = {"contact": 0.1, "skate": 10.0, "acc": 30.0, "jerk": 0.0}  # --w_contact / --w_skate / --w_acc / --w_jerk


def split_files(files, out, val_frac):
    """Per input file, a train and a val npz (clips by a hash of the name before any "|": the motion, not the subject)."""
    tr, va = [], []
    for f in files:
        z = dict(np.load(f))
        h = np.array([int(hashlib.md5(str(n).split("|")[0].encode()).hexdigest()[:8], 16) / 2 ** 32 for n in z["name"]])
        for keep, dst in ((h >= val_frac, tr), (h < val_frac, va)):
            ix = np.flatnonzero(keep)
            if not len(ix):
                continue
            sl = [np.arange(z["start"][i], z["start"][i] + z["length"][i]) for i in ix]
            cat = np.concatenate(sl)
            o = Path(out) / f"{Path(f).stem}_{'train' if dst is tr else 'val'}.npz"
            np.savez(o, pose=z["pose"][cat], trans=z["trans"][cat], contact=z["contact"][cat], betas=z["betas"][ix],
                     length=z["length"][ix], start=np.cumsum([0] + [len(s) for s in sl])[:-1], name=z["name"][ix])
            dst.append(str(o))
    return tr, va


def to_can(f, X):
    return (f["A"][:, None, None] @ (X - f["o"][:, None, None])[..., None])[..., 0]


def anchor_obs(f, o, rel, perp):
    """fab/world_motion.export_anchor in the window's canonical frame: the input pelvis as an extra observation."""
    u = f["ray_dir"]
    dd = u[..., :, None] * u[..., None, :]
    I3 = torch.eye(3, device=u.device)
    La = (dd / (rel * f["depth"][..., None]) ** 2 + (I3 - dd) / perp ** 2) * f["valid"][..., None, None]
    Lam = o["Lam"] + La
    b = (o["Lam"] @ o["obs"][..., None] + La @ f["p"][..., None])[..., 0]
    return Lam, torch.linalg.solve(Lam, b[..., None])[..., 0]


def forward(net, kin, d, gt, amp=True, anchor=None):
    f = build_features(kin, d)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and d["R_c"].is_cuda):
        y = net(f)
    y = y.float()
    o = decode(kin, f, y, pose_head=net.cfg["pose_head"], sig_a_max=net.cfg["sig_a_max"])
    pad = d["pad"]
    keep = (~pad).float()
    w = o["w"] * keep[:, 1:] * keep[:, :-1]
    Lam, obs = (o["Lam"], o["obs"]) if anchor is None else anchor_obs(f, o, *anchor)
    r = fuse_dense(Lam, obs, w, o["u"])
    B, T = r.shape[:2]
    bmed = d["betas"].median(1).values[:, None].expand(B, T, 10)
    Rf = torch.cat([o["R_root"][:, :, None], o["R_loc"]], 2)
    P, C, _ = kin(Rf, bmed, torch.zeros(B, T, 3, device=r.device))
    P, C = P - P[:, :, :1] + r[:, :, None], C - P[:, :, :1] + r[:, :, None]
    return f, o, r, P, C


def losses(f, o, r, P, C, gt, kin):
    keep = (~gt["pad"]).float()
    n = keep.sum().clamp_min(1)
    Jg, Cg = to_can(f, gt["joints"]), to_can(f, gt["cpts"])
    rg = Jg[:, :, 0]
    m = lambda x: (x * keep).sum() / n  # noqa: E731
    L = {}
    L["root"] = m((r - rg).abs().sum(-1))
    k2 = keep[:, 1:] * keep[:, :-1]
    dv, dvg = r[:, 1:] - r[:, :-1], rg[:, 1:] - rg[:, :-1]
    L["vel"] = 10 * ((dv - dvg).abs().sum(-1) * k2).sum() / k2.sum().clamp_min(1)
    L["joints"] = m((P - Jg).norm(dim=-1).mean(-1)) + m((C - Cg).norm(dim=-1).mean(-1))
    L["local"] = m(((P - P[:, :, :1]) - (Jg - Jg[:, :, :1])).norm(dim=-1).mean(-1))
    Rg_root = f["A"][:, None] @ gt["R_root"]
    L["rot"] = 0.1 * m((o["R_root"] - Rg_root).pow(2).sum((-1, -2)) + (o["R_loc"] - gt["R_loc"]).pow(2).sum((-1, -2)).mean(-1))
    L["contact"] = LW["contact"] * m(F.binary_cross_entropy_with_logits(o["contact_logit"], gt["contact"], reduction="none").mean(-1))
    cg = gt["contact"][:, 1:, :4] * gt["contact"][:, :-1, :4] * k2[..., None]
    sk = (C[:, 1:, :4] - C[:, :-1, :4]).norm(dim=-1)
    L["skate"] = LW["skate"] * (sk * cg).sum() / cg.sum().clamp_min(1)
    k3 = keep[:, 2:] * keep[:, 1:-1] * keep[:, :-2]
    acc = lambda X: X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]  # noqa: E731
    L["acc"] = LW["acc"] * ((acc(P) - acc(Jg)).norm(dim=-1).mean(-1) * k3).sum() / k3.sum().clamp_min(1)
    if LW["jerk"] > 0:  # match the GT jerk (the jitter metric's third difference)
        k4 = keep[:, 3:] * keep[:, 2:-1] * keep[:, 1:-2] * keep[:, :-3]
        jk = lambda X: X[:, 3:] - 3 * X[:, 2:-1] + 3 * X[:, 1:-2] - X[:, :-3]  # noqa: E731
        L["jerk"] = LW["jerk"] * ((jk(P) - jk(Jg)).norm(dim=-1).mean(-1) * k4).sum() / k4.sum().clamp_min(1)
    # sigma calibration (Gaussian NLL of the fusion's inputs)
    e = rg - o["obs"]
    ea = (e * f["ray_dir"]).sum(-1)
    ep = (e - ea[..., None] * f["ray_dir"]).norm(dim=-1)
    v = f["valid"] * keep
    nll_o = (ea ** 2 / (2 * o["sa"] ** 2) + torch.log(o["sa"]) + ep ** 2 / (2 * o["sp"] ** 2) + 2 * torch.log(o["sp"]))
    ev = (dvg - o["v"][:, :-1]).pow(2).sum(-1)
    nll_v = ev / (2 * o["sv"][:, :-1] ** 2) + 3 * torch.log(o["sv"][:, :-1])
    L["nll"] = 0.01 * ((nll_o * v).sum() / v.sum().clamp_min(1) + (nll_v * k2).sum() / k2.sum().clamp_min(1))
    return L


def umeyama(X, Y, w):
    """Batched similarity (s, R, t) minimising sum w |s R x + t - y|^2; X, Y (B, N, 3), w (B, N)."""
    w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
    mx, my = (w[..., None] * X).sum(1), (w[..., None] * Y).sum(1)
    Xc, Yc = X - mx[:, None], Y - my[:, None]
    S = torch.einsum("bn,bni,bnj->bij", w, Yc, Xc)
    U, D, Vt = torch.linalg.svd(S)
    sgn = torch.sign(torch.linalg.det(U @ Vt))
    E = torch.diag_embed(torch.stack([torch.ones_like(sgn), torch.ones_like(sgn), sgn], -1))
    R = U @ E @ Vt
    var = (w * Xc.pow(2).sum(-1)).sum(1).clamp_min(1e-9)
    s = (D * torch.diagonal(E, dim1=-2, dim2=-1)).sum(-1) / var
    return s, R, my - s[:, None] * (R @ mx[..., None])[..., 0]


def aligned_errors(Pk, Jg, keep):
    """WA (similarity over the window) and W (similarity on the first two kept frames) MPJPE, mm."""
    B, T = Pk.shape[:2]
    w = keep[:, :, None].expand(B, T, 24).reshape(B, -1)
    s, R, t = umeyama(Pk.reshape(B, -1, 3), Jg.reshape(B, -1, 3), w)
    al = lambda X, s, R, t: s[:, None, None, None] * (R[:, None, None] @ X[..., None])[..., 0] + t[:, None, None]  # noqa: E731
    e = (al(Pk, s, R, t) - Jg).norm(dim=-1).mean(-1)
    wa = (e * keep).sum() / keep.sum()
    first = keep.cumsum(1)
    w2 = ((first <= 2) & (keep > 0)).float()[:, :, None].expand(B, T, 24).reshape(B, -1)
    s, R, t = umeyama(Pk.reshape(B, -1, 3), Jg.reshape(B, -1, 3), w2)
    e = (al(Pk, s, R, t) - Jg).norm(dim=-1).mean(-1)
    return wa * 1000, (e * keep).sum() / keep.sum() * 1000


@torch.no_grad()
def evaluate(net, kin, val, args, noise, n_batches=20):
    net.eval()
    g = torch.Generator(device=args.device).manual_seed(1234)
    acc = {}
    for _ in range(n_batches):
        m = val.sample(args.bs, args.W, g)
        d, gt = simulate(kin, m, noise, g)
        f, o, r, P, C = forward(net, kin, d, gt, anchor=ANCHOR)
        keep = (~gt["pad"]).float()
        Jg, Cg = to_can(f, gt["joints"]), to_can(f, gt["cpts"])
        # the input as FAB's world chain would place it (no smoothing / Gaussian smoothing of the pelvis, sigma 3)
        Rw = f["Rw0"]
        Pi, Ci, _ = kin(torch.cat([Rw[:, :, None], f["R_loc"]], 2), d["betas"], torch.zeros_like(r))
        Pi, Ci = Pi - Pi[:, :, :1] + f["p"][:, :, None], Ci - Pi[:, :, :1] + f["p"][:, :, None]
        from train.world_motion_sim import smooth_t

        ps = smooth_t(f["p"], 3.0)
        rows = {"model": (P, C), "input": (Pi, Ci), "input_s3": (Pi - f["p"][:, :, None] + ps[:, :, None], Ci - f["p"][:, :, None] + ps[:, :, None])}
        cg = gt["contact"][:, 1:, :4] * gt["contact"][:, :-1, :4] * (keep[:, 1:] * keep[:, :-1])[..., None]
        for k, (Pk, Ck) in rows.items():
            mp = ((Pk - Jg).norm(dim=-1).mean(-1) * keep).sum() / keep.sum()
            rt = ((Pk[:, :, 0] - Jg[:, :, 0]).norm(dim=-1) * keep).sum() / keep.sum()
            a = Pk[:, 3:] - 3 * Pk[:, 2:-1] + 3 * Pk[:, 1:-2] - Pk[:, :-3]  # jerk (GVHMR's jitter, 10 m/s^3)
            k4 = keep[:, 3:] * keep[:, 2:-1] * keep[:, 1:-2] * keep[:, :-3]
            jit = (a.norm(dim=-1).mean(-1) * k4).sum() / k4.sum() * 30 ** 3 / 10
            sk = ((Ck[:, 1:, :4] - Ck[:, :-1, :4]).norm(dim=-1) * cg).sum() / cg.sum().clamp_min(1)
            wa, w_ = aligned_errors(Pk, Jg, keep)
            for kk, vv in (("mpjpe_w", mp * 1000), ("wa", wa), ("w", w_), ("root", rt * 1000), ("jitter", jit), ("skate", sk * 1000)):
                acc.setdefault(f"{k}/{kk}", []).append(float(vv))
        pc = (torch.sigmoid(o["contact_logit"][..., :4]) > 0.5).float()
        gc = gt["contact"][..., :4]
        tp, fp, fn = (pc * gc * keep[..., None]).sum(), (pc * (1 - gc) * keep[..., None]).sum(), ((1 - pc) * gc * keep[..., None]).sum()
        acc.setdefault("model/contact_f1", []).append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    net.train()
    return {k: round(float(np.mean(v)), 3) for k, v in acc.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="comma-separated world_motion_data npz files")
    p.add_argument("--data_store", default="", help="cpu: keep the training frames in host memory (large corpora)")
    p.add_argument("--data_w", default="", help="comma-separated share of windows per --data file (default: by length)")
    p.add_argument("--smpl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val_frac", type=float, default=0.03)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--W", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--pose_head", type=int, default=1, help="0: FAB's rotations kept, only trajectory / contacts learned")
    p.add_argument("--sig_a_max", type=float, default=3.0, help="largest along-ray observation sigma (m)")
    p.add_argument("--anchor", default="", help="rel,perp: train with the export anchor in the fusion (do-no-harm)")
    p.add_argument("--w_contact", type=float, default=0.1, help="contact BCE weight (vs the GT static contacts)")
    p.add_argument("--w_skate", type=float, default=10.0, help="planted-foot speed weight on GT contacts")
    p.add_argument("--w_acc", type=float, default=30.0, help="joint-acceleration match weight")
    p.add_argument("--w_jerk", type=float, default=0.0, help="joint-jerk match weight (jitter)")
    p.add_argument("--ema", type=float, default=0.0, help="weight EMA decay (0: off); the EMA weights are saved and evaluated")
    p.add_argument("--noise_json", default="")
    p.add_argument("--smooth_input", default="", help="sigma_rot,sigma_pos: train on fab-world's smoothed export (2,3)")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_data", default="", help="extra held-out world_motion_data npz files, reported as held/* in each eval")
    p.add_argument("--keep_ckpts", type=int, default=0, help="1: also save model_s<step>.pt at every eval")
    p.add_argument("--resume_dir", default="", help="exact-resume state (model, EMA, optimizer, scheduler, RNG, step) written "
                   "here every --resume_every steps (the last two kept); a run restarted with the same dir continues")
    p.add_argument("--resume_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    global ANCHOR
    ANCHOR = tuple(float(x) for x in a.anchor.split(",")) if a.anchor else None
    LW.update(contact=a.w_contact, skate=a.w_skate, acc=a.w_acc, jerk=a.w_jerk)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)  # model init and dropout reproducible per seed
    noise = dict(NOISE)
    if a.noise_json:
        noise.update({k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(Path(a.noise_json).read_text()).items()})
    if a.smooth_input:
        noise["smooth_input"] = tuple(float(x) for x in a.smooth_input.split(","))
    tr, va = split_files(a.data.split(","), out, a.val_frac)
    kin = SMPLKin(a.smpl).to(a.device)
    dw = [float(x) for x in a.data_w.split(",")] if a.data_w else None
    if dw is not None and len(tr) != len(dw):  # a file without train clips
        dw = None
    train, val = Motions(tr, a.device, dw, store=a.data_store or None), Motions(va, a.device)
    held = Motions(a.eval_data.split(","), a.device) if a.eval_data else None
    print(f"[wm] train {int(train.length.sum())} frames / {len(train.length)} clips, val {int(val.length.sum())} / "
          f"{len(val.length)}", flush=True)
    net = WorldMotionNet(d=a.d, layers=a.layers, heads=a.heads, pose_head=bool(a.pose_head), sig_a_max=a.sig_a_max).to(a.device)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=a.device, weights_only=False)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    g = torch.Generator(device=a.device).manual_seed(a.seed)
    if train.gc is not None:  # the host-side sampler: seeded too, so a run is reproducible and resumable
        train.gc.manual_seed(a.seed + 1)
    log = open(out / "log.jsonl", "a")
    cfg = net.cfg
    ema = None
    if a.ema > 0:
        import copy

        ema = copy.deepcopy(net).eval()
        for q in ema.parameters():
            q.requires_grad_(False)
    t0, run = time.time(), {}
    start = 1
    rd = Path(a.resume_dir) if a.resume_dir else None
    if rd is not None:
        rd.mkdir(parents=True, exist_ok=True)
        cks = sorted(rd.glob("state_s*.pt"), key=lambda q: int(q.stem[7:]))
        if cks:
            st = torch.load(cks[-1], map_location="cpu", weights_only=False)  # RNG states must stay CPU byte tensors
            net.load_state_dict(st["model"])
            if ema is not None:
                ema.load_state_dict(st["ema"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            g.set_state(st["g"])
            if train.gc is not None:
                train.gc.set_state(st["gc"])
            torch.set_rng_state(st["torch_cpu"])
            if torch.cuda.is_available() and st.get("torch_cuda") is not None:
                torch.cuda.set_rng_state(st["torch_cuda"])
            start = st["step"] + 1
            print(f"[wm] resumed from {cks[-1]} at step {st['step']}", flush=True)
    for step in range(start, a.steps + 1):
        m = train.sample(a.bs, a.W, g)
        d, gt = simulate(kin, m, noise, g)
        f, o, r, P, C = forward(net, kin, d, gt, anchor=ANCHOR)
        L = losses(f, o, r, P, C, gt, kin)
        loss = sum(L.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                dk = min(a.ema, (1 + step) / (10 + step))
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(dk).add_(pn.detach(), alpha=1 - dk)
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v)
        if rd is not None and step % a.resume_every == 0:
            st = {"model": net.state_dict(), "ema": ema.state_dict() if ema is not None else None, "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "g": g.get_state(), "gc": train.gc.get_state() if train.gc is not None else None,
                  "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                  "step": step}
            tmp = rd / f"state_s{step}.pt.tmp"
            torch.save(st, tmp)
            tmp.rename(rd / f"state_s{step}.pt")
            for q in sorted(rd.glob("state_s*.pt"), key=lambda q: int(q.stem[7:]))[:-2]:
                q.unlink()
        if step % a.log_every == 0:
            rec = {"step": step, "lr": sched.get_last_lr()[0], "gn": float(gn), "s_per_step": (time.time() - t0) / step,
                   **{k: round(v / a.log_every, 5) for k, v in run.items()}}
            run = {}
            print("[wm]", json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % a.eval_every == 0 or step == a.steps:
            em = ema if ema is not None else net
            ev = evaluate(em, kin, val, a, noise)
            if held is not None:
                ev.update({f"held/{k}": v for k, v in evaluate(em, kin, held, a, noise).items()})
            print("[wm] eval", step, json.dumps(ev), flush=True)
            log.write(json.dumps({"step": step, "eval": ev}) + "\n")
            log.flush()
            ck = {"cfg": cfg, "model": em.state_dict(), "step": step, "noise": noise, "args": vars(a)}
            torch.save(ck, out / "model.pt")
            if a.keep_ckpts:
                torch.save(ck, out / f"model_s{step}.pt")


if __name__ == "__main__":
    main()
