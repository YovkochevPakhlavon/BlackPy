"""Deprecation helpers (private).

Policy (since 1.0, see ``CONTRIBUTING.md``): a public name is removed or changes
behaviour only in a major release, and only after at least one minor release in
which using it emits a :class:`DeprecationWarning` that names the replacement.
"""

from __future__ import annotations

import importlib
import warnings


def warn_deprecated(old: str, new: str, *, since: str = "1.0", stacklevel: int = 3) -> None:
    """Emit the standard :class:`DeprecationWarning` for a renamed public name."""
    warnings.warn(
        f"{old} is deprecated since BlackPy {since} and will be removed in BlackPy 2.0; "
        f"use {new} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def alias_getattr(module_name: str, aliases: dict[str, tuple[str, str, str]]):
    """Build a module ``__getattr__`` (PEP 562) that serves deprecated aliases.

    ``aliases[name] = (target_module, target_name, replacement_text)``: accessing
    ``module_name.name`` (or ``from module_name import name``) warns and returns
    ``target_module.target_name``.
    """

    def __getattr__(name: str):
        if name in aliases:
            target_module, target_name, replacement = aliases[name]
            warn_deprecated(f"{module_name}.{name}", replacement)
            return getattr(importlib.import_module(target_module), target_name)
        raise AttributeError(f"module '{module_name}' has no attribute '{name}'")

    return __getattr__
