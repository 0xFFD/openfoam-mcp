"""Field file statistics without ParaView (ASCII field files)."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

from . import dictparse

_NUM_TIME = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


def time_dirs(case_dir: Path) -> list[str]:
    """Numeric time directories, sorted by value."""
    names = [p.name for p in case_dir.iterdir() if p.is_dir() and _NUM_TIME.match(p.name)]
    return sorted(names, key=float)


def processor_dirs(case_dir: Path) -> list[Path]:
    return sorted((p for p in case_dir.glob("processor*") if p.is_dir()), key=lambda p: p.name)


_COMPONENTS = {"scalar": 1, "vector": 3, "symmTensor": 6, "tensor": 9, "sphericalTensor": 1}
_BIN_LIST = re.compile(rb"List<(\w+)>\s*(\d+)\s*\(")
_BIN_REF = re.compile(r"<BIN(\d+)>")


def _read_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def _decode(data: bytes) -> tuple[str, list[np.ndarray]]:
    """Text of a field file, with binary list payloads lifted out into arrays.

    In `format binary` files each `List<T> N (` is followed by N*k raw floats and `)`. Those payloads
    are replaced by `<BINi>` placeholders so the rest can be parsed as text.
    """
    head = data[:4000].decode(errors="replace")
    if not re.search(r"format\s+binary", head):
        return data.decode(errors="replace"), []
    arch = re.search(r'arch\s+"([^"]*)"', head)
    arch = arch.group(1) if arch else "LSB;label=32;scalar=64"
    size = int(m.group(1)) // 8 if (m := re.search(r"scalar=(\d+)", arch)) else 8
    dtype = np.dtype(f"{'>' if 'MSB' in arch else '<'}f{size}")
    arrays: list[np.ndarray] = []
    parts: list[bytes] = []
    pos = 0
    for m in _BIN_LIST.finditer(data):
        if m.start() < pos or (k := _COMPONENTS.get(m.group(1).decode())) is None:
            continue
        n, start = int(m.group(2)), m.end()
        end = start + n * k * size
        if data[end : end + 1] != b")":
            continue  # not a binary payload (e.g. an ASCII list)
        arrays.append(np.frombuffer(data, dtype=dtype, count=n * k, offset=start).astype(float).reshape(n, k))
        parts += [data[pos : m.start()], f"List<{m.group(1).decode()}> {n} <BIN{len(arrays) - 1}>".encode()]
        pos = end + 1
    parts.append(data[pos:])
    return b"".join(parts).decode(errors="replace"), arrays


def field_path(case_dir: Path, time: str, field: str, region: str | None = None) -> Path | None:
    base = case_dir / time / region if region else case_dir / time
    for p in (base / field, base / f"{field}.gz"):
        if p.is_file():
            return p
    return None


def _values(text: str, arrays: list[np.ndarray] | None = None) -> np.ndarray | None:
    """Parse `uniform X` / `nonuniform List<T> N (...)` (ASCII or a binary placeholder) into an (N, k) array."""
    text = text.strip()
    if arrays and (b := _BIN_REF.search(text)):
        return arrays[int(b.group(1))]
    if text.startswith("uniform"):
        body = text[len("uniform") :].strip().strip(";").strip()
        nums = np.array(body.replace("(", " ").replace(")", " ").split(), dtype=float)
        return nums.reshape(1, -1)
    m = re.match(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text)
    if not m:
        return None
    kind, n = m.group(1), int(m.group(2))
    k = _COMPONENTS.get(kind)
    if k is None:
        return None
    body = text[m.end() :]
    body = body[: body.rfind(")")]
    arr = np.array(body.replace("(", " ").replace(")", " ").split(), dtype=float)
    if arr.size != n * k:
        return None
    return arr.reshape(n, k)


def _r(v: float) -> float:
    return float(f"{v:.6g}")


def _mmm(col: np.ndarray) -> list[float]:
    return [_r(col.min()), _r(col.max()), _r(col.mean())]


def _stats(arr: np.ndarray) -> dict:
    """[min, max, mean] per scalar / magnitude / component; a single value for uniform data."""
    if arr.shape[0] == 1:
        vals = [_r(v) for v in arr[0]]
        return {"value": vals[0] if len(vals) == 1 else vals}
    out: dict = {"count": int(arr.shape[0])}
    if arr.shape[1] == 1:
        out["min_max_mean"] = _mmm(arr[:, 0])
    else:
        out["magnitude"] = _mmm(np.linalg.norm(arr, axis=1))
        names = "xyz" if arr.shape[1] == 3 else [str(i) for i in range(arr.shape[1])]
        for i, c in enumerate(names):
            if np.ptp(arr[:, i]) > 0 or abs(arr[0, i]) > 0:
                out[c] = _mmm(arr[:, i])
    return out


def field_stats(path: Path, patches: bool = True) -> dict:
    text, arrays = _decode(_read_bytes(path))
    out: dict = {"format": "binary" if arrays else "ascii"}
    if m := re.search(r"^dimensions\s+([^;]+);", text, re.M):
        out["dimensions"] = m.group(1).strip()
    m = re.search(r"^internalField\s+(.*?);\s*$", text, re.M | re.S)
    # Restrict the match to the internalField entry (up to boundaryField).
    bf = text.find("\nboundaryField")
    if m:
        seg = text[m.start(1) : bf if bf > 0 else len(text)]
        seg = seg[: seg.rfind(";")]
        arr = _values(seg, arrays)
        out["internal"] = _stats(arr) if arr is not None else {"raw": seg[:120]}
    if patches and bf > 0:
        data = dictparse.parse(text[bf:], elide=False)
        pstats = {}
        for name, d in (data.get("boundaryField") or {}).items():
            if not isinstance(d, dict):
                continue
            entry: dict = {"type": d.get("type")}
            if isinstance(d.get("value"), str):
                arr = _values(d["value"], arrays)
                if arr is not None and arr.size:
                    entry.update(_stats(arr))
            pstats[name] = entry
        out["patches"] = pstats
    return out
