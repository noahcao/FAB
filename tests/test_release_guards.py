"""Release guard: the whole tree must pass scripts/release_scrub.py against the maintainers' denylist.

The denylist lives outside the repository (it lists the strings that must not ship). Point $FAB_RELEASE_DENYLIST at it;
without it the test is skipped.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseScrubTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FAB_RELEASE_DENYLIST"), "FAB_RELEASE_DENYLIST not set")
    def test_tree_has_no_denylisted_strings(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/release_scrub.py"), str(ROOT)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-2000:])

    def test_denylist_is_not_in_the_tree(self):
        path = os.environ.get("FAB_RELEASE_DENYLIST")
        if path:
            self.assertFalse(Path(path).resolve().is_relative_to(ROOT), "keep the denylist outside the release tree")


if __name__ == "__main__":
    unittest.main()
