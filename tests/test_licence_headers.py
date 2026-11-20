"""Licence markings (as FAH's tests/test_licence_headers.py): every file with a Meta copyright header carries an SPDX
identifier (SAM License or DINOv3 License), every SPDX-marked file is listed in THIRD_PARTY_NOTICES.md, the DINOv3 files are
exactly the vendored dinov3_vit/ package, and the licence texts the headers and WEIGHTS_LICENSE point to exist (the SAM
and DINOv3 licences reproduced verbatim in WEIGHTS_LICENSE)."""
import os
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}


def python_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith(".py") and not f.startswith("._"):  # (macOS resource-fork copies)
                yield Path(root, f).relative_to(REPO).as_posix()


class LicenceHeaderTests(unittest.TestCase):
    def test_meta_files_are_marked_and_listed(self):
        notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text()
        marked = {"LicenseRef-SAM-License": 0, "LicenseRef-DINOv3-License": 0}
        for f in python_files():
            head = "".join((REPO / f).read_text().splitlines(keepends=True)[:6])
            spdx = re.search(r"SPDX-License-Identifier: (\S+)", head)
            if head.startswith("# Copyright (c) Meta Platforms, Inc. and affiliates.") or "/dinov3_vit/" in f:
                self.assertIsNotNone(spdx, f)
            if spdx:
                self.assertIn(spdx.group(1), marked, f)
                marked[spdx.group(1)] += 1
                self.assertIn(f"`{f}`", notices, f)
                self.assertEqual(spdx.group(1) == "LicenseRef-DINOv3-License", "/dinov3_vit/" in f, f)
        self.assertEqual(marked, {"LicenseRef-SAM-License": 40, "LicenseRef-DINOv3-License": 11})

    def test_notices_list_only_existing_files(self):
        notices = (REPO / "THIRD_PARTY_NOTICES.md").read_text()
        for f in re.findall(r"`((?:fab|train|bench|scripts)/[^`]+\.py)`", notices):
            self.assertTrue((REPO / f).exists(), f)

    def test_licence_texts_exist(self):
        for f in ("LICENSE", "NOTICE", "WEIGHTS_LICENSE", "THIRD_PARTY_NOTICES.md", "third_party_licenses/SAM-License.txt",
                  "third_party_licenses/DINOv3-License.md", "third_party_licenses/MHR-Apache-2.0.txt",
                  "third_party_licenses/pymomentum-MIT.txt"):
            self.assertTrue((REPO / f).exists(), f)
        self.assertIn("Apache License", (REPO / "LICENSE").read_text()[:200])
        w = (REPO / "WEIGHTS_LICENSE").read_text()
        self.assertIn((REPO / "third_party_licenses/SAM-License.txt").read_text().strip(), w)  # verbatim pass-through
        self.assertIn((REPO / "third_party_licenses/DINOv3-License.md").read_text().strip(), w)
        self.assertIn("Built with DINOv3", w)
        self.assertIn("RESEARCH AND NON-COMMERCIAL USE ONLY", w)
        for term in ("weights/shared/detector.pt", "AGPL-3.0", "Ultralytics", "TensorRT", "ONNX"):
            self.assertIn(term, w)
        detector_terms = w.split("PART 4. DETECTOR PROVENANCE AND UPSTREAM TERMS", 1)[1]
        self.assertIn("credit FAB and Ultralytics", detector_terms)
        self.assertIn("Built with DINOv3", (REPO / "THIRD_PARTY_NOTICES.md").read_text())


if __name__ == "__main__":
    unittest.main()
