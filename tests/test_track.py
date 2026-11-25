"""fab/track.py on synthetic boxes: persistent ids through a crossing and a detection gap (ORU / OCR), BYTE's low-score
stage, gating of the boxes between detector frames, tracklet linking, the subject choice and the one-box prompt."""
import unittest

import numpy as np

from fab.track import Tracker, iou_matrix, link, prompt_follow, prompted, subject


def box(cx, cy, w=60.0, h=160.0):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class TrackerTests(unittest.TestCase):
    def test_crossing_keeps_ids(self):
        trk = Tracker()
        ids_a, ids_b = [], []
        for t in range(60):
            a, b = box(100 + 6 * t, 300), box(460 - 6 * t, 310)
            got = trk.step(np.stack([a, b]), np.array([0.9, 0.9]))
            inv = {i: k for k, i in got.items()}
            ids_a.append(inv.get(0))
            ids_b.append(inv.get(1))
        self.assertEqual(len({i for i in ids_a if i is not None}), 1)
        self.assertEqual(len({i for i in ids_b if i is not None}), 1)
        self.assertNotEqual(ids_a[0], ids_b[0])

    def test_gap_reacquired(self):
        trk = Tracker(max_age=30)
        first = None
        for t in range(50):
            dets = np.zeros((0, 4)) if 20 <= t < 32 else box(200 + 3 * t, 300)[None]
            got = trk.step(dets, np.full(len(dets), 0.9))
            if got:
                first = list(got)[0] if first is None else first
                self.assertEqual(list(got), [first], t)

    def test_byte_low_scores_extend(self):
        trk = Tracker(high=0.5, low=0.1, birth=0.5)
        trk.step(box(200, 300)[None], np.array([0.9]))
        got = trk.step(box(203, 300)[None], np.array([0.2]))  # a low score: only extends the track
        self.assertEqual(list(got), [0])
        got = trk.step(box(500, 300)[None], np.array([0.2]))  # a low score far away: no new track
        self.assertEqual(got, {})

    def test_observe_gates_jumps_and_duplicates(self):
        trk = Tracker(gate=0.3, dup=0.7)
        trk.step(np.stack([box(200, 300), box(400, 300)]), np.array([0.9, 0.9]))
        kept = trk.observe({0: box(202, 300), 1: box(204, 300)})  # 1's crop converged on person 0
        self.assertEqual(kept, {0})
        kept = trk.observe({0: box(600, 300)})  # 0's box jumped
        self.assertEqual(kept, set())

    def test_link_and_subject(self):
        tl = {0: {t: box(300 + 2 * t, 300) for t in range(0, 40)},
              1: {t: box(300 + 2 * t, 300) for t in range(55, 100)},  # the same person after an occlusion
              2: {t: box(30, 250, 40, 170) for t in range(0, 100)}}  # a bystander at the image edge
        ident = link(tl)
        self.assertEqual(ident[1], ident[0])
        self.assertNotEqual(ident[2], ident[0])
        self.assertEqual(subject(tl, ident, (600, 800)), ident[0])

    def test_prompted(self):
        tl = {0: {t: box(300 + 2 * t, 300) for t in range(0, 40)},
              1: {t: box(300 + 2 * t, 300) for t in range(55, 100)},
              2: {t: box(30, 250, 40, 170) for t in range(5, 100)}}  # the bystander, prompted; tracked from frame 5
        ident = link(tl)
        self.assertEqual(prompted(tl, ident, box(30, 250, 40, 170), 0), ident[2])
        self.assertEqual(prompted(tl, ident, box(400, 300), 50), ident[0])  # frame 55: tracklet 1, linked to 0
        self.assertIsNone(prompted(tl, ident, box(700, 100), 0))
        frag = {0: {t: box(300, 300, 120, 160) for t in range(5)},  # a 5-frame box over two people at the prompt
                1: {t: box(285, 300) for t in range(3, 100)}}  # the person, tracked from frame 3
        fi = link(frag)
        self.assertEqual(prompted(frag, fi, box(300, 300, 120, 160), 0, min_len=0), fi[0])
        self.assertEqual(prompted(frag, fi, box(300, 300, 120, 160), 0), fi[1])

    def test_init_prompt(self):
        import tempfile

        from bench.sonic_motion import SUBJECT_KW, init_prompt

        self.assertIsNone(init_prompt(dict(SUBJECT_KW)))
        f, b = init_prompt({**SUBJECT_KW, "init_box": "1,2,30,40", "init_frame": 7})
        self.assertEqual(f, 7)
        np.testing.assert_allclose(b, [1, 2, 30, 40])
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/b.npz"
            np.savez(p, box=np.array([[np.nan] * 4, [5, 6, 50, 60.0]]))
            f, b = init_prompt({**SUBJECT_KW, "init_box": p})
            self.assertEqual(f, 1)
            np.testing.assert_allclose(b, [5, 6, 50, 60])
        with self.assertRaises(ValueError):
            init_prompt({**SUBJECT_KW, "init_box": "1,2,3"})

    def test_prompt_follow(self):
        tl = {0: {t: box(300, 300) for t in range(0, 32)},  # the prompted person until frame 31
              1: {t: box(330, 300) for t in range(0, 278)},  # the other person, tracked alongside (never taken)
              2: {t: box(305, 300) for t in range(48, 295)},  # the prompted person found again (a new track)
              3: {t: box(300, 300) for t in range(92, 95)}}  # a later 3-frame fragment
        self.assertEqual(prompt_follow(tl, box(300, 300), 0), [0, 2])
        tl = {0: {t: box(300, 300, 50, 150) for t in range(0, 11)},  # a short duplicate box at the prompt
              1: {t: box(300, 300) for t in range(0, 101)},  # the person
              2: {t: box(305, 300) for t in range(136, 300)},  # the person again
              3: {t: box(700, 300) for t in range(74, 300)}}  # someone else, far away
        self.assertEqual(prompt_follow(tl, box(300, 300, 50, 150), 0), [1, 2])
        self.assertEqual(prompt_follow(tl, box(900, 100), 0), [])

    def test_iou(self):
        self.assertAlmostEqual(float(iou_matrix(box(0, 0), box(0, 0))[0, 0]), 1.0)
        self.assertEqual(float(iou_matrix(box(0, 0), box(500, 0))[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
