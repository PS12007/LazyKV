"""Tiny template renderer that makes hard rule 1 mechanical.

Templates reference measured values as ``{{ phase0.feasibility.ratio.value | .1f }}``
and pre-built markdown blocks (tables) as ``{{> block_name }}``. Anything that cannot be
resolved renders as ``not measured``, so a missing experiment is visible in the doc
instead of silently becoming a stale hand-typed number.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

NOT_MEASURED = "not measured"

_BLOCK = re.compile(r"\{\{>\s*(?P<name>[A-Za-z0-9_]+)\s*\}\}")
_VALUE = re.compile(r"\{\{\s*(?P<key>[A-Za-z0-9_.\[\]\-]+)\s*(?:\|\s*(?P<fmt>[^}]*?)\s*)?\}\}")

_MISSING = object()


def lookup(ctx: Mapping[str, Any], dotted: str) -> Any:
    node: Any = ctx
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.lstrip("-").isdigit():
            idx = int(part)
            if -len(node) <= idx < len(node):
                node = node[idx]
            else:
                return _MISSING
        else:
            return _MISSING
    return node


def iec(n: float) -> str:
    """Binary-prefixed byte size, e.g. 1.5 GiB. Keeps KiB/MiB/GiB unambiguous."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for unit in units:
        if abs(v) < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(v)} B"
            return f"{v:.3g} {unit}" if v < 100 else f"{v:.0f} {unit}"
        v /= 1024
    raise AssertionError("unreachable")


def _named_formatters() -> dict[str, Callable[[Any], str]]:
    return {
        "iec": iec,
        "int": lambda v: f"{int(round(v)):,}",
        "pct": lambda v: f"{100 * v:.1f}%",
        "pct0": lambda v: f"{100 * v:.0f}%",
        "x": lambda v: f"{v:.1f}×",
        "ms": lambda v: f"{1e3 * v:.2f} ms",
        "us": lambda v: f"{1e6 * v:.1f} µs",
        "gbps": lambda v: f"{v / 1e9:.2f} GB/s",
        "str": str,
    }


def format_value(value: Any, fmt: str | None) -> str:
    if value is _MISSING or value is None:
        return NOT_MEASURED
    if not fmt:
        return str(value)
    named = _named_formatters()
    if fmt in named:
        return named[fmt](value)
    return format(value, fmt)


def render(template: str, ctx: Mapping[str, Any], blocks: Mapping[str, str] | None = None) -> str:
    blocks = blocks or {}

    def block_sub(m: re.Match[str]) -> str:
        return blocks.get(m.group("name"), f"_{NOT_MEASURED}_")

    def value_sub(m: re.Match[str]) -> str:
        return format_value(lookup(ctx, m.group("key")), m.group("fmt"))

    # Values first, then blocks, so generated table text is never re-parsed as template.
    return _BLOCK.sub(block_sub, _VALUE.sub(value_sub, template))
