"""Mesh inspection: polyMesh boundary/size readers and a checkMesh output parser."""

from __future__ import annotations

import re
from pathlib import Path

from . import dictparse


def polymesh_dir(case_dir: Path, region: str | None = None) -> Path:
    base = case_dir / "constant"
    return base / region / "polyMesh" if region else base / "polyMesh"


def read_boundary(case_dir: Path, region: str | None = None) -> list[dict] | None:
    """Patches from constant/polyMesh/boundary as [{name, type, nFaces, ...}]."""
    path = polymesh_dir(case_dir, region) / "boundary"
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    body = re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)
    start, end = body.find("("), body.rfind(")")
    if start < 0 or end < 0:
        return None
    patches = dictparse.parse(body[start + 1 : end])
    out = []
    for name, d in patches.items():
        if not isinstance(d, dict):
            continue
        entry = {"name": name, "type": d.get("type"), "nFaces": _int(d.get("nFaces"))}
        if d.get("inGroups"):
            entry["inGroups"] = d["inGroups"]
        if d.get("neighbourPatch"):
            entry["neighbourPatch"] = d["neighbourPatch"]
        out.append(entry)
    return out


def mesh_size(case_dir: Path, region: str | None = None) -> dict | None:
    """Cell/face/point counts from the `note` in the owner file header (cheap, no full read)."""
    owner = polymesh_dir(case_dir, region) / "owner"
    if not owner.is_file():
        for gz in (owner.with_suffix(".gz"),):
            if gz.is_file():
                return {"note": "compressed mesh; run check_mesh for statistics"}
        return None
    with open(owner, "rb") as f:
        head = f.read(4096).decode(errors="replace")
    m = re.search(r'note\s+"([^"]*)"', head)
    if not m:
        return None
    return {k: int(v) for k, v in re.findall(r"(\w+):\s*(\d+)", m.group(1))}


def _int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


_NUM = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"


def parse_check_mesh(text: str) -> dict:
    """Turn checkMesh output into structured statistics, failures and advice."""
    out: dict = {}
    stats = {}
    for key in ("points", "faces", "internal faces", "cells", "boundary patches"):
        if m := re.search(rf"^\s+{key}:\s+(\d+)", text, re.M):
            stats[key.replace(" ", "_")] = int(m.group(1))
    out["stats"] = stats
    types = {}
    if m := re.search(r"Overall number of cells of each type:\n((?:\s+\w[\w ]*:\s+\d+\n)+)", text):
        for name, n in re.findall(r"^\s+([\w ]+?):\s+(\d+)", m.group(1), re.M):
            if int(n):
                types[name] = int(n)
    out["cell_types"] = types

    q: dict = {}
    if m := re.search(r"Overall domain bounding box \(([^)]*)\) \(([^)]*)\)", text):
        q["bounding_box"] = [[float(x) for x in m.group(1).split()], [float(x) for x in m.group(2).split()]]
    if m := re.search(r"Mesh has (\d) geometric \(non-empty/wedge\) directions", text):
        q["dimensions"] = int(m.group(1))
    # Passing and failing checks print values differently ("Max skewness = 0.26 OK." vs
    # "***Max skewness = 9.47, 13 highly skew faces"; "Max aspect ratio = 8.1 OK." vs
    # "***High aspect ratio cells found, Max aspect ratio: 1038.2, number of cells 20").
    if m := re.search(rf"Max aspect ratio\s*[:=]\s*{_NUM}", text):
        q["max_aspect_ratio"] = _num(m.group(1))
    if m := re.search(rf"Mesh non-orthogonality Max: {_NUM} average: {_NUM}", text):
        q["max_non_orthogonality"] = _num(m.group(1))
        q["avg_non_orthogonality"] = _num(m.group(2))
    if m := re.search(rf"Max skewness\s*=\s*{_NUM}", text):
        q["max_skewness"] = _num(m.group(1))
    if m := re.search(rf"Min volume = {_NUM}\. Max volume = {_NUM}\.\s+Total volume = {_NUM}", text):
        q["min_volume"], q["max_volume"], q["total_volume"] = (_num(x) for x in m.groups())
    if m := re.search(rf"Minimum face area = {_NUM}\. Maximum face area = {_NUM}", text):
        q["min_face_area"], q["max_face_area"] = _num(m.group(1)), _num(m.group(2))
    if m := re.search(r"Number of negative volume cells: (\d+)", text):
        q["negative_volume_cells"] = int(m.group(1))
    if m := re.search(r"Number of regions: (\d+)", text):
        q["regions"] = int(m.group(1))
    out["quality"] = q

    failures = [line.strip().lstrip("*") for line in text.splitlines() if line.strip().startswith("***")]
    if failures:
        out["failures"] = failures
    m = re.search(r"Failed (\d+) mesh checks", text)
    if m:
        out["verdict"] = f"FAILED {m.group(1)} mesh checks"
    elif "Mesh OK." in text:
        out["verdict"] = "Mesh OK"
    else:
        out["verdict"] = "unknown (checkMesh did not finish?)"

    advice = []
    no = q.get("max_non_orthogonality")
    if no is not None:
        if no > 85:
            advice.append(f"Max non-orthogonality {no:.0f} is severe (>85): fix the mesh; solvers are likely to diverge.")
        elif no > 70:
            advice.append(
                f"Max non-orthogonality {no:.0f} > 70: use nNonOrthogonalCorrectors 2-3 and limited/corrected "
                "laplacian and snGrad schemes (e.g. 'Gauss linear limited 0.5')."
            )
        elif no > 60:
            advice.append(f"Max non-orthogonality {no:.0f} > 60: consider nNonOrthogonalCorrectors 1.")
    sk = q.get("max_skewness")
    if sk is not None and sk > 4:
        advice.append(f"Max skewness {sk:.2f} > 4: expect interpolation errors; refine or smooth the mesh.")
    ar = q.get("max_aspect_ratio")
    if ar is not None and ar > 1000:
        advice.append(f"Max aspect ratio {ar:.0f} is very high; fine for boundary layers but watch convergence.")
    if q.get("negative_volume_cells"):
        advice.append("Mesh has negative-volume cells: it is invalid and must be regenerated or repaired.")
    if q.get("regions", 1) > 1:
        advice.append("Mesh has more than one disconnected region; check that this is intended.")
    if advice:
        out["advice"] = advice
    return out
