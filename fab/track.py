"""Person tracking for video and world mode: a Kalman filter per person on its box and OC-SORT / ByteTrack-style
association, the offline tracklet linking and subject choice of world mode. Opt-in (FABPredictor(tracker="oc"),
fab-video --tracker oc, fab-world --track oc); the released default links people by box IoU only (tracker="iou").

  Tracker     the online tracker. State per person: SORT's constant-velocity Kalman filter on the box centre, area and
              aspect (x, y, s, r; s and r as SORT: the area with a velocity, the aspect without). A detector frame
              (Tracker.step(boxes, scores)) associates in three stages, every one a Hungarian assignment:
                1. the detections scoring >= high with the predicted boxes: IoU + OCM (OC-SORT's observation-centric
                   momentum: the agreement of a track's direction of motion over its last DELTA observations with the
                   direction to the detection, weighted by `inertia` and the detection's score);
                2. (BYTE) the low-scoring detections (low <= score < high) with the tracks still unmatched (IoU);
                3. (OCR) the high detections left with the unmatched tracks' last observed boxes (IoU): a person lost
                   for a few frames is found again where it was last seen.
              A track found again after a gap is re-filtered along a straight virtual path from its last observation
              (ORU, observation-centric re-update), so the gap does not leave a wrong velocity. The high detections left
              start tracks when they score >= birth; a track not observed for max_age frames ends.
              Tracker.observe({id: box}) is a frame between detector frames: each person's box comes from its own
              previous fit (FAB's keypoints); a box whose IoU with the person's predicted box is below `gate` is not
              taken (the crop may have jumped to someone else), and two people whose boxes overlap by more than `dup`
              keep only the one closer to its prediction.
  link        offline (world mode): tracklets of one person split by a long occlusion, joined when the later one starts
              where the earlier one's motion predicts it (IoU or normalised centre distance).
  subject     the target person of a clip: the identity with the largest sum over its frames of (person height /
              image height) x (centrality weight) x score; a person seen longest and largest, near the image centre
              (world mode: the height is FAB's keypoint extent, the release's "most prominent" measure).
  prompted    the target person from one prompt box (fab-world --init_box, one click as a SAM prompt): the identity
              whose box overlaps the prompt most on the first frame from the prompt's frame on where one does.
  prompt_follow   world mode's --init_box: the prompted person's own tracklets (the one under the prompt, extended by
              the new tracks that start where it was lost), without the global link.
"""
import numpy as np

DELTA = 3  # OCM: the direction of motion over this many observations


def iou_matrix(a, b):
    """(N, 4) x (M, 4) xyxy -> (N, M) IoU."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)  # noqa: E731
    u = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(u > 0, inter / np.maximum(u, 1e-9), 0.0)


def box_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)], np.float64)


def x_to_box(x):
    s, r = max(float(x[2]), 1e-6), max(float(x[3]), 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2], np.float64)


def assign(cost, valid):
    """Hungarian assignment minimising cost over the allowed pairs (valid) -> [(row, col)]."""
    from scipy.optimize import linear_sum_assignment

    if cost.size == 0 or not valid.any():
        return []
    c = np.where(valid, cost, 1e6)
    r, k = linear_sum_assignment(c)
    return [(i, j) for i, j in zip(r, k) if valid[i, j]]


class BoxFilter:
    """SORT's Kalman filter: state (x, y, s, r, vx, vy, vs), measurement (x, y, s, r); SORT's noise settings."""

    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.eye(4, 7)

    def __init__(self, box):
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

    def predict(self):
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, box):
        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(7) - Kg @ self.H) @ self.P

    def box(self):
        return x_to_box(self.x)

    def state(self):
        return self.x.copy(), self.P.copy()

    def restore(self, st):
        self.x, self.P = st[0].copy(), st[1].copy()


class Track:
    def __init__(self, tid, box, score, t):
        self.id, self.kf = tid, BoxFilter(box)
        self.obs = {t: np.asarray(box, np.float64)}  # frame -> observed box
        self.scores = {t: float(score)}  # frame -> its detection score (a box from the fit: the track's running score)
        self.last_t, self.last_box, self.score = t, np.asarray(box, np.float64), float(score)
        self.frozen = None  # the filter state at the last observation (for ORU)
        self.hits = 1

    def velocity_dir(self):
        """(unit direction of the box centre's motion from the observation about DELTA frames back to the last one,
        that earlier box's centre) for OCM, or None."""
        ts = sorted(self.obs)[-(DELTA + 1):]
        if len(ts) < 2:
            return None
        t0 = next(t for t in ts if t >= self.last_t - DELTA)
        if t0 == self.last_t:
            t0 = ts[-2]
        a, b = self.obs[t0], self.last_box
        c0 = (a[:2] + a[2:]) / 2
        d = (b[:2] + b[2:]) / 2 - c0
        n = np.linalg.norm(d)
        return (d / n, c0) if n > 1e-6 else None

    def predict(self, t):
        if self.last_t == t - 1 and self.frozen is None:
            self.frozen = self.kf.state()  # the state at the last observation
        self.kf.predict()

    def update(self, box, score, t):
        box = np.asarray(box, np.float64)
        gap = t - self.last_t
        if gap > 1 and self.frozen is not None:  # ORU: re-filter along a straight path from the last observation
            self.kf.restore(self.frozen)
            for k in range(1, gap):
                self.kf.predict()
                self.kf.update(self.last_box + (box - self.last_box) * k / gap)
            self.kf.predict()
        self.kf.update(box)
        self.frozen = None
        self.obs[t], self.scores[t] = box, float(score)
        self.last_t, self.last_box = t, box
        self.score = 0.9 * self.score + 0.1 * float(score)
        self.hits += 1


class Tracker:
    def __init__(self, high=0.25, low=0.1, birth=0.25, iou=0.3, iou_low=0.5, max_age=30, inertia=0.2, gate=0.3,
                 dup=0.7):
        self.high, self.low, self.birth, self.iou, self.iou_low = high, low, birth, iou, iou_low
        self.max_age, self.inertia, self.gate, self.dup = max_age, inertia, gate, dup
        self.tracks, self.ended, self.next_id, self.t = [], [], 0, -1

    def _advance(self):
        self.t += 1
        for tr in self.tracks:
            tr.predict(self.t)

    def _retire(self):
        keep = []
        for tr in self.tracks:
            (keep if self.t - tr.last_t <= self.max_age else self.ended).append(tr)
        self.tracks = keep

    def step(self, boxes, scores=None):
        """A detector frame: boxes (N, 4) xyxy, scores (N,) (None: all 1) -> {track id: detection index} of the
        detections taken this frame (matched or new)."""
        self._advance()
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.ones(len(boxes)) if scores is None else np.asarray(scores, np.float64).reshape(-1)
        hi = np.flatnonzero(scores >= self.high)
        lo = np.flatnonzero((scores >= self.low) & (scores < self.high))
        tracks = self.tracks
        out, free_t = {}, list(range(len(tracks)))
        # 1. high detections x predicted boxes: IoU + OCM
        if len(hi) and tracks:
            pred = np.stack([tr.kf.box() for tr in tracks])
            io = iou_matrix(boxes[hi], pred)
            ang = np.zeros_like(io)
            for j, tr in enumerate(tracks):
                vd = tr.velocity_dir()
                if vd is None:
                    continue
                v, c0 = vd
                d = (boxes[hi, :2] + boxes[hi, 2:]) / 2 - c0
                n = np.linalg.norm(d, axis=1)
                ok = n > 1e-6
                cosang = np.where(ok, (d @ v) / np.maximum(n, 1e-6), 1.0)
                diff = np.arccos(np.clip(cosang, -1, 1))
                ang[:, j] = (np.pi / 2 - np.abs(diff)) / np.pi * scores[hi]
            pairs = assign(-(io + self.inertia * ang), io >= self.iou)
            for i, j in pairs:
                out[tracks[j].id] = int(hi[i])
            used_d = {i for i, _ in pairs}
            free_t = [j for j in range(len(tracks)) if j not in {j for _, j in pairs}]
            hi_left = [int(hi[i]) for i in range(len(hi)) if i not in used_d]
        else:
            hi_left = [int(i) for i in hi]
        # 2. BYTE: low detections x the unmatched tracks' predicted boxes
        if len(lo) and free_t:
            pred = np.stack([tracks[j].kf.box() for j in free_t])
            io = iou_matrix(boxes[lo], pred)
            pairs = assign(-io, io >= self.iou_low)
            for i, j in pairs:
                out[tracks[free_t[j]].id] = int(lo[i])
            free_t = [free_t[j] for j in range(len(free_t)) if j not in {j for _, j in pairs}]
        # 3. OCR: high detections left x the unmatched tracks' last observed boxes
        if hi_left and free_t:
            last = np.stack([tracks[j].last_box for j in free_t])
            io = iou_matrix(boxes[hi_left], last)
            pairs = assign(-io, io >= self.iou)
            for i, j in pairs:
                out[tracks[free_t[j]].id] = hi_left[i]
            taken = {i for i, _ in pairs}
            hi_left = [hi_left[i] for i in range(len(hi_left)) if i not in taken]
        byid = {tr.id: tr for tr in tracks}
        for tid, i in out.items():
            byid[tid].update(boxes[i], scores[i], self.t)
        for i in hi_left:  # births
            if scores[i] >= self.birth:
                tr = Track(self.next_id, boxes[i], scores[i], self.t)
                self.tracks.append(tr)
                out[tr.id] = i
                self.next_id += 1
        self._retire()
        return out

    def observe(self, obs, scores=None):
        """A frame between detector frames: obs {track id: box} (each person's box from its own previous fit) ->
        the ids whose box was taken (IoU with the predicted box >= gate; of two overlapping by > dup, the one closer to
        its prediction)."""
        self._advance()
        byid = {tr.id: tr for tr in self.tracks}
        cand = {}
        for tid, b in obs.items():
            if tid not in byid:
                continue
            o = float(iou_matrix(b, byid[tid].kf.box())[0, 0])
            if o >= self.gate:
                cand[tid] = o
        ids = sorted(cand, key=lambda k: -cand[k])
        keep = []
        for tid in ids:  # duplicates: two crops that converged on one person
            if all(iou_matrix(obs[tid], obs[k])[0, 0] <= self.dup for k in keep):
                keep.append(tid)
        for tid in keep:
            byid[tid].update(obs[tid], (scores or {}).get(tid, byid[tid].score), self.t)
        self._retire()
        return set(keep)

    def active(self):
        """Ids of the tracks observed on the current frame."""
        return [tr.id for tr in self.tracks if tr.last_t == self.t]

    def predicted(self, tid):
        for tr in self.tracks:
            if tr.id == tid:
                return tr.kf.box()
        return None

    def all_tracks(self):
        return sorted(self.ended + self.tracks, key=lambda tr: tr.id)


def link(tracklets, max_gap=150, min_iou=0.1, max_dist=1.0):
    """Offline linking of tracklets {id: {t: box}} (time-disjoint pieces of one person) -> {id: identity}.
    A later tracklet B continues an earlier A when B starts after A ends (gap <= max_gap frames) and B's first box is
    near A's last one extrapolated with A's mean velocity over its last frames: IoU >= min_iou, or the centre distance
    over A's box height <= max_dist (x (1 + gap / 30), the reach growing with the gap). Greedy on the cost (the
    distance term minus the IoU), each tracklet continued at most once and continuing at most one other."""
    ids = sorted(tracklets, key=lambda k: min(tracklets[k]))
    span = {k: (min(v), max(v)) for k, v in tracklets.items()}

    def tail(k, n=10):
        ts = sorted(tracklets[k])[-n:]
        b = np.stack([tracklets[k][t] for t in ts])
        v = (b[-1] - b[0]) / max(ts[-1] - ts[0], 1)
        return b[-1], v

    cands = []
    for a in ids:
        last, vel = tail(a)
        for b in ids:
            if b == a or span[b][0] <= span[a][1] or span[b][0] - span[a][1] > max_gap:
                continue
            gap = span[b][0] - span[a][1]
            first = tracklets[b][span[b][0]]
            pred = last + vel * min(gap, 15)  # the velocity carried over at most half a second
            io = float(iou_matrix(pred, first)[0, 0])
            h = max(last[3] - last[1], 1.0)
            d = np.hypot((pred[0] + pred[2] - first[0] - first[2]) / 2, (pred[1] + pred[3] - first[1] - first[3]) / 2) / h
            if not (io >= min_iou or d <= max_dist * (1 + gap / 30)):
                continue
            cands.append((d - io, a, b))
    nxt, prv = {}, {}
    for _, a, b in sorted(cands):
        if a in nxt or b in prv:
            continue
        nxt[a], prv[b] = b, a
    ident = {}
    for k in ids:
        if k in prv:
            continue
        cur = k
        while cur is not None:
            ident[cur] = k
            cur = nxt.get(cur)
    return ident


def subject(tracklets, ident, hw, scores=None, center_sigma=0.5, min_frames=1, heights=None):
    """The target identity: argmax over identities of the sum over their frames of (box height / H) x centrality x score.
    centrality = exp(-d^2 / (2 center_sigma^2)), d the box centre's distance from the image centre over half the image
    diagonal; center_sigma 0 = no centrality term. scores {id: {t: score}} or None (1). heights {id: {t: px}} or None:
    per-frame person heights used instead of the box heights (e.g. FAB's keypoint extent)."""
    H, W = hw
    tot = {}
    for k, boxes in tracklets.items():
        if len(boxes) < min_frames:
            continue
        s, hk = 0.0, (heights or {}).get(k)
        for t, b in boxes.items():
            h = (hk[t] if hk is not None and t in hk else b[3] - b[1]) / H
            c = 1.0
            if center_sigma > 0:
                d = np.hypot((b[0] + b[2]) / 2 - W / 2, (b[1] + b[3]) / 2 - H / 2) / (np.hypot(W, H) / 2)
                c = np.exp(-d * d / (2 * center_sigma ** 2))
            sc = scores[k].get(t, 1.0) if scores is not None and k in scores else 1.0
            s += h * c * sc
        tot[ident.get(k, k)] = tot.get(ident.get(k, k), 0.0) + s
    return max(tot, key=tot.get) if tot else None


def prompted(tracklets, ident, box, t0=0, window=30, min_iou=0.3, min_len=30):
    """The identity a prompt box picks: on the first frame in [t0, t0 + window] where some tracklet's box has IoU >=
    min_iou with the prompt, the identity of the most overlapping one; None when no frame has one. An identity seen on
    fewer than min_len frames (a fragment, e.g. a box over two people in contact) gives way to the identity with >= min_len
    frames whose box overlaps the prompt most over [t0, t0 + window] (IoU >= min_iou), when there is one (min_len 0: off)."""
    box = np.asarray(box, np.float64)
    n = {}
    for k, b in tracklets.items():
        n[ident.get(k, k)] = n.get(ident.get(k, k), 0) + len(b)
    who = None
    for t in range(int(t0), int(t0) + int(window) + 1):
        c = [(float(iou_matrix(b[t], box)[0, 0]), k) for k, b in tracklets.items() if t in b]
        if c and max(c)[0] >= min_iou:
            who = ident.get(max(c)[1], max(c)[1])
            break
    if who is None or n[who] >= min_len:
        return who
    best = {}
    for k, b in tracklets.items():
        g = ident.get(k, k)
        if g == who or n[g] < min_len:
            continue
        for t in range(int(t0), int(t0) + int(window) + 1):
            if t in b:
                best[g] = max(best.get(g, 0.0), float(iou_matrix(b[t], box)[0, 0]))
    if best and max(best.values()) >= min_iou:
        return max(best, key=best.get)
    return who


def prompt_follow(tracklets, box, t0=0, window=30, min_iou=0.3, min_len=30, max_gap=150, max_dist=1.0, reach_iou=0.1):
    """The subject's tracklets [ids] from one prompt box, without the global link: the tracklet overlapping the prompt
    most on the first frame in [t0, t0 + window] where one has IoU >= min_iou (a tracklet of fewer than min_len frames
    gives way to the one of >= min_len overlapping the prompt most in that window), then extended forward / backward:
    after it ends, the tracklet starting later (a new track: the person found again, never one tracked alongside it)
    within max_gap frames whose first box is nearest to its last box within link's reach (IoU >= reach_iou or centre
    distance / height <= max_dist x (1 + gap / 30)), the earliest start first; [] when nothing overlaps the prompt."""
    box = np.asarray(box, np.float64)
    io = lambda a, b: float(iou_matrix(a, b)[0, 0])  # noqa: E731
    k0 = None
    for t in range(int(t0), int(t0) + int(window) + 1):
        c = [(io(b[t], box), k) for k, b in tracklets.items() if t in b]
        if c and max(c)[0] >= min_iou:
            k0 = max(c)[1]
            break
    if k0 is None:
        return []
    if len(tracklets[k0]) < min_len:
        best = {}
        for k, b in tracklets.items():
            if k != k0 and len(b) >= min_len:
                for t in range(int(t0), int(t0) + int(window) + 1):
                    if t in b:
                        best[k] = max(best.get(k, 0.0), io(b[t], box))
        if best and max(best.values()) >= min_iou:
            k0 = max(best, key=best.get)

    def near(a, b, gap):
        h = max(a[3] - a[1], 1.0)
        d = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) / h
        return io(a, b) >= reach_iou or d <= max_dist * (1 + gap / 30), d - io(a, b)

    chain = [k0]
    for fwd in (True, False):
        cur = k0
        while True:
            if fwd:
                te = max(tracklets[cur])
                end = tracklets[cur][te]
            else:
                te = min(tracklets[cur])
                end = tracklets[cur][te]
            cands = []
            for k, b in tracklets.items():
                if k in chain:
                    continue
                s_ = min(b) if fwd else max(b)
                gap = s_ - te if fwd else te - s_
                if gap <= 0 or gap > max_gap:
                    continue
                ok, cost = near(end, b[s_], gap)
                if ok:
                    cands.append((gap, cost, k))
            if not cands:
                break
            cur = min(cands)[2]
            chain.append(cur)
    return chain
