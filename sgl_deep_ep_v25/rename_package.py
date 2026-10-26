"""Rename the staged `deep_ep` Python package to `deep_ep_v25`.

The C++ sources and the `include/deep_ep/` header tree keep their names. The Python import
name changes, and the host extension is compiled with its `deep_ep` namespace renamed, so
`sgl-deep-ep` (`deep_ep`) can be installed and imported alongside.
"""

import re
import sys
from pathlib import Path

OLD = "deep_ep"
NEW = "deep_ep_v25"

# Python references to the package: `deep_ep._C`, `deep_ep.comm`, `import deep_ep`, ...
_PY_REFERENCE = re.compile(rf"(?<![\w/<.]){OLD}(?=\.\w|\s|$|`)", re.MULTILINE)
# setup.py paths and setuptools names; `'include/deep_ep/**/*'` is a header path and stays
_SETUP_REFERENCES = (
    (re.compile(rf"'{OLD}(\.\*|\._C)?'"), rf"'{NEW}\1'"),
    (re.compile(rf"/{OLD}/include'"), f"/{NEW}/include'"),
)
# Both extensions register C++ types such as `deep_ep::EventHandle` with pybind11, which rejects
# a second registration under the same type name. Renaming the host namespace lets one process
# import `deep_ep` and `deep_ep_v25` together. Header names in `#include <...>` and JIT-compiled
# kernels are unaffected.
_CXX_FLAGS = re.compile(r"(cxx_flags = \[)('-std=c\+\+20', )")
_CXX_FLAGS_REPLACEMENT = rf"\1\2'-D{OLD}={NEW}', "
# Anything still matching after the rewrite is reported, except header paths
_LEFTOVER = re.compile(rf"(?<![\w/<]){OLD}(?![\w/])")


def _rewrite(path: Path, rules) -> int:
    text = path.read_text()
    total = 0
    for pattern, replacement in rules:
        text, count = pattern.subn(replacement, text)
        total += count
    path.write_text(text)
    return total


def _leftovers(path: Path) -> list:
    return [f"{path}:{number}: {line.strip()}" for number, line in enumerate(path.read_text().splitlines(), 1)
            if _LEFTOVER.search(line)]


def main(staging_dir: str) -> None:
    root = Path(staging_dir)
    old_package, new_package = root / OLD, root / NEW
    if not old_package.is_dir() or new_package.exists():
        raise SystemExit(f"Expected {old_package} and no {new_package}")
    old_package.rename(new_package)

    python_files = sorted(new_package.rglob("*.py"))
    renamed = sum(_rewrite(path, [(_PY_REFERENCE, NEW)]) for path in python_files)
    if renamed == 0:
        raise SystemExit(f"No `{OLD}` references found under {new_package}")

    setup = root / "_deepep_setup.py"
    if _rewrite(setup, [(_CXX_FLAGS, _CXX_FLAGS_REPLACEMENT)]) != 1:
        raise SystemExit(f"Cannot add the namespace define to `cxx_flags` in {setup}")
    if _rewrite(setup, [*_SETUP_REFERENCES, (_PY_REFERENCE, NEW)]) == 0:
        raise SystemExit(f"No `{OLD}` references found in {setup}")

    leftovers = [line for path in [*python_files, setup] for line in _leftovers(path)]
    if leftovers:
        raise SystemExit("Unrenamed references:\n" + "\n".join(leftovers))
    print(f"Renamed {OLD} -> {NEW}: {renamed} Python references")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <staging-dir>")
    main(sys.argv[1])
