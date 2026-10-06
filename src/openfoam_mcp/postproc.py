"""Readers for function-object output under postProcessing/."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_DATA_SUFFIXES = {".dat", ".xy", ".csv", ".raw", ""}


_TIME_PART = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def list_outputs(case_dir: Path, max_groups: int = 100) -> list[dict]:
    """Function-object outputs grouped across time directories.

    `postProcessing/<function>/<time>/<file>` repeated for many times becomes one entry with the
    list of times and the path of the latest one; tabular files report their columns.
    """
    root = case_dir / "postProcessing"
    if not root.is_dir():
        return []
    groups: dict[tuple[str, ...], dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(root).parts
        times = [x for x in parts[:-1] if _TIME_PART.match(x)]
        key = tuple("<time>" if _TIME_PART.match(x) else x for x in parts)
        g = groups.setdefault(key, {"pattern": "postProcessing/" + "/".join(key), "times": [], "files": []})
        g["times"].append(times[0] if times else None)
        g["files"].append(p)
    out = []
    for g in list(groups.values())[:max_groups]:
        order = sorted(range(len(g["files"])), key=lambda i: float(g["times"][i]) if g["times"][i] else 0.0)
        latest = g["files"][order[-1]]
        entry: dict = {"latest": str(latest.relative_to(case_dir)), "bytes": latest.stat().st_size}
        ts = [g["times"][i] for i in order if g["times"][i] is not None]
        if len(g["files"]) > 1:
            entry["pattern"] = g["pattern"]
            entry["times"] = ts if len(ts) <= 8 else [*ts[:3], "...", *ts[-3:]]
            entry["count"] = len(g["files"])
        if latest.suffix in _DATA_SUFFIXES and latest.stat().st_size < 200_000_000:
            cols = header_columns(latest)
            if cols:
                entry["columns"] = cols
        out.append(entry)
    return out


def header_columns(path: Path) -> list[str] | None:
    """Column names from the last comment line before the data (OpenFOAM writes `# Time  col1  col2`)."""
    last = None
    first_data = None
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                last = line[1:].strip()
            elif line.strip():
                first_data = line
                break
            if i > 200:
                break
    if last is None:
        return None
    names = [c for c in re.split(r"\t+|\s{2,}", last) if c]
    if len(names) <= 1:
        names = last.split()
    if first_data is not None:
        names = _expand_vectors(names, first_data)
    return names


def _expand_vectors(names: list[str], data_line: str) -> list[str]:
    """Expand a column whose data is a `(x y z)` vector into name_x, name_y, name_z."""
    groups = re.findall(r"\([^()]*\)|[^\s()]+", data_line.strip())
    if len(groups) != len(names):
        return names
    out = []
    for name, g in zip(names, groups, strict=True):
        if g.startswith("("):
            n = len(g[1:-1].split())
            suffixes = ["x", "y", "z"] if n == 3 else [f"{i}" for i in range(n)]
            if n == 6:
                suffixes = ["xx", "xy", "xz", "yy", "yz", "zz"]
            elif n == 9:
                suffixes = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
            out.extend(f"{name}_{s}" for s in suffixes)
        else:
            out.append(name)
    return out


def load_table(path: Path) -> tuple[list[str], np.ndarray, dict[int, list[str]]]:
    """Numeric table plus text columns (e.g. patch names), keyed by column index."""
    names = header_columns(path) or []
    rows = []
    labels: dict[int, list[str]] = {}
    width = None
    with open(path, errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.replace("(", " ").replace(")", " ").replace(",", " ").split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                vals = [_maybe_float(x) for x in parts]
            if width is None:
                width = len(vals)
            if len(vals) == width:
                for i, (v, raw) in enumerate(zip(vals, parts, strict=True)):
                    if math.isnan(v) and raw.lower() != "nan":
                        labels.setdefault(i, [""] * len(rows)).append(raw)
                    elif i in labels:
                        labels[i].append(raw)
                rows.append(vals)
    arr = np.array(rows, dtype=float) if rows else np.zeros((0, len(names) or 1))
    ncol = arr.shape[1] if arr.ndim == 2 else 0
    if len(names) != ncol:
        names = names[:ncol] + [f"col{i}" for i in range(len(names), ncol)]
    return names, arr, labels


def _maybe_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def _clean(v: float):
    v = float(v)
    return v if math.isfinite(v) else None


def summarize_table(
    names: list[str], arr: np.ndarray, columns: list[str] | None, max_rows: int, labels: dict[int, list[str]] | None = None
) -> dict:
    labels = labels or {}
    if columns:
        missing = [c for c in columns if c not in names]
        if missing:
            raise KeyError(f"Unknown columns {missing}; available: {names}")
        idx = [0] + [names.index(c) for c in columns if names.index(c) != 0]
    else:
        idx = list(range(len(names)))
    sel_names = [names[i] for i in idx]
    out: dict = {"columns": sel_names, "rows_total": int(arr.shape[0])}
    if arr.shape[0] == 0:
        return out
    stats = {}
    tail_n = max(1, arr.shape[0] // 10)
    for i in idx[1:] if len(idx) > 1 else idx:
        if i in labels:
            continue
        col = arr[:, i]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        stats[names[i]] = {
            "last": _clean(col[-1]),
            "min": _clean(finite.min()),
            "max": _clean(finite.max()),
            "mean": _clean(finite.mean()),
            "mean_last_10pct": _clean(np.nanmean(col[-tail_n:])),
            "std_last_10pct": _clean(np.nanstd(col[-tail_n:])),
        }
    out["stats"] = stats
    if max_rows > 0:
        step = max(1, math.ceil(arr.shape[0] / max_rows))
        picks = list(range(0, arr.shape[0], step))
        if picks[-1] != arr.shape[0] - 1:
            picks.append(arr.shape[0] - 1)
        out["rows"] = [
            [labels[i][r] if i in labels else _clean(arr[r, i]) for i in idx] for r in picks
        ]
        if step > 1:
            out["rows_note"] = f"every {step}th row shown (plus the last)"
    return out
