"""Print a Markdown checklist of every number in the data registry, for human verification.

Usage (from the repository root)::

    .venv/bin/python scripts/registry_checklist.py > checklist.md

For each dataset: every number with its path in the file, its unit, the paper
(citation and DOI/arXiv) and the location where it is printed, how the file was
read (``source_access``), then the dataset's ``todo`` items and ``open_questions``.
The list is generated from the YAML files, so it is complete by construction. The
maintainer checks each value against the paper and, when every value of a file is
confirmed, sets ``verified_by_human: true`` in that file by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "src" / "blackpy" / "data" / "registry"
NUMBER_KEYS = ("value", "err_minus", "err_plus", "lower", "upper")


def _reference(refs: dict, key) -> str:
    if not key or key not in refs:
        return (
            f"`{key}`"
            if key
            else "not recorded in the entry (see the location and the references below)"
        )
    ref = refs[key]
    ids = [
        f"doi:{ref['doi']}" if ref.get("doi") else "",
        f"arXiv:{ref['arxiv']}" if ref.get("arxiv") else "",
    ]
    ids = ", ".join(i for i in ids if i)
    return f"{ref.get('citation', key)}" + (f" ({ids})" if ids else "")


def _numbers(node: dict) -> str:
    parts = []
    for key in NUMBER_KEYS:
        if key in node:
            parts.append(f"{key} = {node[key] if node[key] is not None else 'null'}")
    unit = node.get("unit")
    return ", ".join(parts) + (f" [{unit}]" if unit else "")


def _walk(node, path: str, refs: dict, out: list[str], inherited_source=None):
    if isinstance(node, dict):
        source = node.get("source", inherited_source)
        if any(k in node for k in NUMBER_KEYS):
            location = node.get("location") or node.get("bounds_location") or "(no location)"
            out.append(
                f"- [ ] `{path}`: {_numbers(node)}. Source: {_reference(refs, source)}. "
                f"Location: {location}"
            )
        for key, value in node.items():
            if key in ("references", "todo", "open_questions", "notes", "description"):
                continue
            if key == "other_values" and isinstance(value, list):
                for k, item in enumerate(value):
                    _walk(item, f"{path}.other_values[{k}]", refs, out, source)
                continue
            if isinstance(value, (dict, list)):
                _walk(value, f"{path}.{key}" if path else key, refs, out, source)
    elif isinstance(node, list):
        for k, item in enumerate(node):
            _walk(item, f"{path}[{k}]", refs, out, inherited_source)


def checklist() -> str:
    lines = []
    for path in sorted(REGISTRY.glob("*/*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        refs = raw.get("references") or {}
        rel = path.relative_to(ROOT).as_posix()
        lines += [
            f"### `{raw['id']}` ({raw.get('object')}, {raw.get('data_type')}, "
            f"dataset version {raw.get('dataset_version')})",
            "",
            f"File: `{rel}`. verified_by_human: **{raw.get('verified_by_human')}**.",
            "",
            f"How it was read: {raw.get('source_access', '(not recorded)')}".strip(),
            "",
            "Values:",
            "",
        ]
        values: list[str] = []
        _walk(raw, "", refs, values)
        lines += values or ["- (no numbers)"]
        lines += ["", "References (check the bibliographic details too):", ""]
        lines += [f"- [ ] `{key}`: {_reference(refs, key)}" for key in refs]
        for title, key in (
            ("TODO items of the file", "todo"),
            ("Open questions", "open_questions"),
        ):
            items = raw.get(key) or []
            if items:
                lines += ["", f"{title}:", ""]
                lines += [f"- [ ] {' '.join(str(item).split())}" for item in items]
        lines += [
            "",
            f"- [ ] When every item above is confirmed: set `verified_by_human: true` in "
            f"`{rel}` by hand and bump nothing else (a changed value bumps `dataset_version`).",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.write(checklist())
