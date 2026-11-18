"""Rewrite the "Output" blocks of the documentation with what the snippets print now.

Usage (from the repository root)::

    .venv/bin/python scripts/update_doc_outputs.py docs/user_guide.md [more pages]

Run it after changing a snippet, then review the diff: a changed number in an
output block is a changed result. ``tests/test_docs.py`` fails until the page and
the code agree.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from doc_snippets import update_outputs  # noqa: E402

pages = [Path(p).resolve() for p in sys.argv[1:]] or [ROOT / "docs" / "user_guide.md"]
os.chdir(tempfile.mkdtemp(prefix="blackpy-docs-"))  # snippets may write files
for page in pages:
    print(page, update_outputs(page), "output blocks")
