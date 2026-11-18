"""Every Python snippet of the user-facing documentation runs and prints what the page shows.

Conventions are described in ``tests/doc_snippets.py``. To update the outputs after
a deliberate change: ``.venv/bin/python scripts/update_doc_outputs.py <page>``.

The user guide runs several samplers and compiles most observables, so it is in
the slow subset; the README and the quickstart are quick.
"""

from pathlib import Path

import matplotlib
import pytest

from doc_snippets import check, parse, run

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "README.md": pytest.param("README.md", id="README"),
    "docs/quickstart.md": pytest.param("docs/quickstart.md", id="quickstart"),
    "docs/data.md": pytest.param("docs/data.md", id="data"),
    "docs/user_guide.md": pytest.param(
        "docs/user_guide.md", id="user_guide", marks=pytest.mark.slow
    ),
}


@pytest.mark.parametrize("page", list(PAGES.values()))
def test_documentation_snippets_run_and_print_what_the_page_shows(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # snippets may save figures and result bundles
    path = ROOT / page
    snippets = parse(path.read_text(encoding="utf-8"))
    assert snippets, f"{page} has no python snippets"
    problems = []
    for result in run(snippets, name=page):
        problems += check(result)
        if result.error is not None:
            break  # later snippets depend on this one
    assert not problems, f"{page}:\n" + "\n\n".join(problems)


def test_no_python_block_is_silently_unlabelled():
    # A block that is not meant to run must say so (python-skip or text); an
    # unlabelled fence would neither run nor be highlighted.
    for page in PAGES:
        text = (ROOT / page).read_text(encoding="utf-8")
        fences = [line.strip() for line in text.splitlines() if line.strip().startswith("```")]
        openers = fences[0::2]
        assert "```" not in openers, f"{page}: a code block without a language label"


def test_every_error_and_warning_is_explained_in_the_user_guide():
    import blackpy.errors as errors

    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    section = guide.split("## 11. Integrity guards and errors", 1)[1].split("\n## ", 1)[0]
    missing = [name for name in errors.__all__ if f"`{name}`" not in section]
    assert not missing, f"Explain these in the user guide, section 11: {missing}"


def test_every_figure_of_the_user_guide_exists_and_is_small():
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    import re

    images = re.findall(r"\]\((images/[^)]+\.png)\)", guide)
    assert images
    for image in images:
        path = ROOT / "docs" / image
        assert path.exists(), f"{image} missing: run docs/make_figures.py"
        assert path.stat().st_size < 200_000, f"{image} is larger than 200 kB"
