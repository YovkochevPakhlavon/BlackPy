"""Generate the figures of the user guide (``docs/images/*.png``).

The figures are drawn by the snippets of ``docs/user_guide.md`` themselves: this
script runs every snippet of the page in order, in a temporary folder, and copies
the PNG files they save into ``docs/images/``. Run it from the repository root
after changing a plot snippet or a plot function::

    .venv/bin/python docs/make_figures.py

Each figure should stay below about 200 kB (the script reports the sizes).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from doc_snippets import parse, run  # noqa: E402

IMAGES = ROOT / "docs" / "images"
LIMIT = 200_000


def main() -> int:
    page = ROOT / "docs" / "user_guide.md"
    work = Path(tempfile.mkdtemp(prefix="blackpy-figures-"))
    os.chdir(work)
    for result in run(parse(page.read_text(encoding="utf-8")), name=page.name):
        if result.error is not None:
            raise RuntimeError(f"{page}:{result.snippet.line}") from result.error
    IMAGES.mkdir(exist_ok=True)
    too_large = 0
    for png in sorted(work.glob("*.png")):
        target = IMAGES / png.name
        shutil.copyfile(png, target)
        size = target.stat().st_size
        too_large += size > LIMIT
        print(f"{target.relative_to(ROOT)}: {size / 1000:.0f} kB")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if too_large else 0


if __name__ == "__main__":
    sys.exit(main())
