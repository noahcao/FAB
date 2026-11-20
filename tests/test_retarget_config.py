"""The released retargeter configuration must stay equal to its frozen copy (any change changes every G1 output)."""
import json
import unittest
from pathlib import Path

from fab.retarget import release_config as C

DATA = Path(__file__).resolve().parent / "data" / "retarget_config.json"


class RetargetConfigTest(unittest.TestCase):
    def test_frozen(self):
        ref = json.loads(DATA.read_text())
        self.assertEqual(C.SOLVE, ref["SOLVE"])
        self.assertEqual(C.PIN, ref["PIN"])
        self.assertEqual(C.SIGMA_IN, ref["SIGMA_IN"])

    def test_solve_cfg_adds_the_mesh_rule(self):
        cfg = C.solve_cfg()
        self.assertEqual((cfg["h_contact"], cfg["v_contact"]), (0.04, 0.15))
        self.assertEqual({k: v for k, v in cfg.items() if k not in ("h_contact", "v_contact")}, C.SOLVE)

    def test_dense_variant_frozen(self):
        ref = json.loads(DATA.read_text())
        self.assertEqual(C.W_PTS, ref["W_PTS"])
        self.assertEqual(C.solve_cfg_dp(), dict(C.solve_cfg(), w_pts=float(ref["W_PTS"])))

    def test_ground_aware_is_opt_in(self):
        base = C.solve_cfg_dp()
        g = C.ground_aware_cfg(base, "/contacts")
        self.assertEqual(base, C.solve_cfg_dp())
        self.assertEqual(g["contact_groups"], base["contact_groups"] + ",anyfoot")
        self.assertEqual((g["contact_key"], g["contact_mode"], g["contact_thr"], g["contact_debounce"], g["no_flight"], g["no_flight_key"]),
                         ("video_static_prob", "replace", 0.5, 1, 0.10, "video_any_prob"))
        self.assertNotIn("w_contact", g)
        self.assertEqual(C.ground_aware_cfg(dict(base, w_ee_hand=1000.0, w_ee_foot=100.0), "/contacts")["w_contact"], 1e5)
        self.assertEqual({k: g[k] for k in base if k != "contact_groups"}, {k: v for k, v in base.items() if k != "contact_groups"})
        self.assertEqual(C.ground_aware_pin(C.PIN), dict(C.PIN, w_ground=1e5))


if __name__ == "__main__":
    unittest.main()
