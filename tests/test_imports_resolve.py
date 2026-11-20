"""Every import of the repository's own packages (fab, bench, train, scripts) resolves to a file in the tree, so the
release carries every module it uses (including imports inside functions)."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN = ("fab", "bench", "train", "scripts")


def module_exists(mod):
    p = ROOT.joinpath(*mod.split("."))
    return p.with_suffix(".py").exists() or (p / "__init__.py").exists() or p.is_dir()


class ImportsResolve(unittest.TestCase):
    def test_own_imports_resolve(self):
        missing = []
        for f in ROOT.rglob("*.py"):
            if ".venv" in f.parts:
                continue
            tree = ast.parse(f.read_text())
            for n in ast.walk(tree):
                mods = []
                if isinstance(n, ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                    mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
                for m in mods:
                    if m.split(".")[0] in OWN and not module_exists(m):
                        # "from pkg import name" may import a function, not a module: accept if the parent module exists
                        parent = m.rsplit(".", 1)[0]
                        if isinstance(n, ast.ImportFrom) and m != n.module and module_exists(parent):
                            continue
                        missing.append(f"{f.relative_to(ROOT)}: {m}")
        self.assertEqual(missing, [], "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
