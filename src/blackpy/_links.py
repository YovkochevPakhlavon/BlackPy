"""Links to the public documentation, used in user-facing messages (private).

Messages never point to files of the source repository (design notes, verification
reports), because a ``pip install`` does not contain them.
"""

#: Root of the hosted documentation (GitHub Pages, ``site_url`` in ``mkdocs.yml``).
DOCS_URL = "https://yovkochevpakhlavon.github.io/BlackPy/"

#: How messages refer to the roadmap page (``docs/roadmap.md``).
ROADMAP = f"the 'Roadmap' page of the BlackPy documentation ({DOCS_URL}roadmap/)"
