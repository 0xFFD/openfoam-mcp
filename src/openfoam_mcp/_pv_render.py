"""Render an OpenFOAM case with ParaView. Executed by a ParaView-enabled Python, not imported.

Usage: <pvpython|python3> _pv_render.py '<json params>'
Prints one line `OFMCP_RESULT <json>` with metadata (times, fields, ranges, camera).
"""

import json
import sys

from paraview.simple import (  # type: ignore[import-not-found]
    CellDatatoPointData,
    ColorBy,
    Contour,
    CreateRenderView,
    GetColorTransferFunction,
    GetScalarBar,
    OpenFOAMReader,
    Outline,
    SaveScreenshot,
    Show,
    Slice,
    StreamTracer,
    Text,
)

AXES = {"x": 0, "y": 1, "z": 2}
PRESETS = {
    "coolwarm": "Cool to Warm",
    "viridis": "Viridis (matplotlib)",
    "jet": "Jet",
    "turbo": "Turbo",
    "inferno": "Inferno (matplotlib)",
    "plasma": "Plasma (matplotlib)",
    "rainbow": "Rainbow Uniform",
    "blue-red": "Blue to Red Rainbow",
    "grayscale": "Grayscale",
}


def emit(obj):
    print("OFMCP_RESULT " + json.dumps(obj), flush=True)


def fail(msg):
    emit({"error": msg})
    sys.exit(0)


def pick_regions(available, wanted):
    chosen = []
    for w in wanted:
        if w in available:
            chosen.append(w)
            continue
        hits = [a for a in available if a.split("/", 1)[-1] == w]
        if not hits:
            fail(f"Unknown patch or group '{w}'. Available: {available}")
        chosen.extend(hits)
    return chosen


def _short(names, limit=25):
    return names if len(names) <= limit else [*names[:limit], f"... {len(names) - limit} more"]


def array_info(src, name):
    """Return ('POINTS'|'CELLS', ncomp, ranges) for an array on a source, or None."""
    for assoc, data in (("POINTS", src.PointData), ("CELLS", src.CellData)):
        if name in data.keys():
            arr = data[name]
            n = arr.GetNumberOfComponents()
            ranges = [list(arr.GetRange(i)) for i in range(n)]
            if n > 1:
                ranges.append(list(arr.GetRange(-1)))  # magnitude
            return assoc, n, ranges
    return None


def main():
    p = json.loads(sys.argv[1])
    reader = OpenFOAMReader(FileName=p["foam_file"])
    reader.SkipZeroTime = 0
    reader.CaseType = "Decomposed Case" if p.get("decomposed") else "Reconstructed Case"
    reader.Createcelltopointfiltereddata = 1
    reader.UpdatePipelineInformation()
    available = list(reader.GetProperty("MeshRegions").Available)
    tv = reader.TimestepValues
    try:
        times = [float(x) for x in tv]
    except TypeError:
        times = [float(tv)] if tv is not None else []
    fields = list(reader.GetProperty("CellArrays").Available)

    t_req = p.get("time", "latest")
    if not times:
        t = 0.0
    elif t_req in (None, "latest"):
        t = times[-1]
    elif t_req == "first":
        t = times[0]
    else:
        t = min(times, key=lambda x: abs(x - float(t_req)))

    mode = p.get("mode", "auto")
    patches = p.get("patches") or []
    if mode == "patches":
        if not patches:
            patches = [a for a in available if a.startswith("patch/")]
            patches = [a for a in patches if "frontAndBack" not in a and "empty" not in a] or patches
        reader.MeshRegions = pick_regions(available, patches)
    else:
        reader.MeshRegions = ["internalMesh"]
    reader.UpdatePipeline(t)

    info = reader.GetDataInformation()
    b = info.GetBounds()
    lo, hi = [b[0], b[2], b[4]], [b[1], b[3], b[5]]
    ext = [hi[i] - lo[i] for i in range(3)]
    center = [(lo[i] + hi[i]) / 2 for i in range(3)]
    big = max(ext) or 1.0
    # OpenFOAM marks 2-D cases with `empty` patches (one cell thick in the thin direction).
    has_empty = "group/empty" in available
    smallest = min(range(3), key=lambda i: ext[i])
    is_2d = p.get("dims") == 2 or (has_empty and ext[smallest] < 0.2 * big) or ext[smallest] < 1e-3 * big
    thin = [smallest] if is_2d else []

    # Camera/slice framing: the whole domain, or a patch/group of interest (`focus`).
    vcenter, vext = center, ext
    if p.get("focus"):
        focus = p["focus"] if isinstance(p["focus"], list) else [p["focus"]]
        fr = OpenFOAMReader(FileName=p["foam_file"])
        fr.SkipZeroTime = 0
        fr.CaseType = reader.CaseType
        fr.MeshRegions = pick_regions(available, focus)
        fr.CellArrays = []
        fr.UpdatePipeline(t)
        fb = fr.GetDataInformation().GetBounds()
        vcenter = [(fb[2 * i] + fb[2 * i + 1]) / 2 for i in range(3)]
        fsize = max(fb[2 * i + 1] - fb[2 * i] for i in range(3)) or big
        vext = [max((fb[2 * i + 1] - fb[2 * i]) * 1.6, 0.25 * fsize) for i in range(3)]

    field = p.get("field")
    if field and field not in fields:
        fail(f"Field '{field}' not found at t={t}. Available fields: {fields}")

    sig = lambda v: float(f"{v:.5g}")  # noqa: E731
    result = {
        "time": t,
        "available_times": f"{len(times)} ({times[0]:g} .. {times[-1]:g})" if times else "none",
        "fields": fields,
        "patches": _short([a.split("/", 1)[1] for a in available if a.startswith(("patch/", "group/"))]),
        "bounds": [[sig(v) for v in lo], [sig(v) for v in hi]],
        "is_2d": is_2d,
    }

    # ---------------- data selection ----------------
    if mode == "auto":
        mode = "surface" if is_2d else "slice"
    result["mode"] = mode

    src = reader
    if mode == "slice":
        normal = p.get("slice_normal") or ("z" if is_2d else min(range(3), key=lambda i: ext[i]))
        if isinstance(normal, str):
            n = [0.0, 0.0, 0.0]
            n[AXES[normal.lower()]] = 1.0
            normal = n
        if isinstance(normal, int):
            n = [0.0, 0.0, 0.0]
            n[normal] = 1.0
            normal = n
        origin = p.get("slice_origin") or vcenter
        src = Slice(Input=reader)
        src.SliceType = "Plane"
        src.SliceType.Origin = origin
        src.SliceType.Normal = normal
        src.Triangulatetheslice = 0
        result["slice"] = {"origin": [sig(v) for v in origin], "normal": [sig(v) for v in normal]}
    elif mode == "contour":
        cf = p.get("contour_field") or field
        if not cf:
            fail("contour mode needs contour_field (or field)")
        if p.get("iso_value") is None:
            fail("contour mode needs iso_value")
        pd = CellDatatoPointData(Input=reader)
        src = Contour(Input=pd)
        src.ContourBy = ["POINTS", cf]
        src.Isosurfaces = [float(p["iso_value"])]
        src.ComputeScalars = 1
    elif mode == "streamlines":
        vec = p.get("field") or "U"
        st = StreamTracer(Input=reader, SeedType="Line")
        st.Vectors = ["POINTS", vec]
        # Default: seed along the domain diagonal, which crosses inlets, wakes and recirculation zones.
        zc = (lambda v: center[2]) if is_2d else (lambda v: v)
        p1 = p.get("seed_point1") or [lo[0] + 0.001 * ext[0], lo[1] + 0.001 * ext[1], zc(lo[2])]
        p2 = p.get("seed_point2") or [hi[0] - 0.001 * ext[0], hi[1] - 0.001 * ext[1], zc(hi[2])]
        st.SeedType.Point1 = p1
        st.SeedType.Point2 = p2
        st.SeedType.Resolution = int(p.get("seed_count", 120))
        st.MaximumStreamlineLength = 3 * big
        src = st
        result["seed_line"] = [[sig(v) for v in p1], [sig(v) for v in p2]]

    # ---------------- view ----------------
    w, h = int(p.get("width", 1400)), int(p.get("height", 800))
    view = CreateRenderView()
    view.ViewSize = [w, h]
    view.ViewTime = t
    view.OrientationAxesVisibility = 1
    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    dark = p.get("background") == "dark"
    view.Background = [0.12, 0.12, 0.14] if dark else [1.0, 1.0, 1.0]
    fg = [0.95, 0.95, 0.95] if dark else [0.05, 0.05, 0.05]
    view.OrientationAxesLabelColor = fg

    if mode in ("streamlines", "contour"):
        od = Show(Outline(Input=reader), view)
        od.AmbientColor = od.DiffuseColor = [0.5, 0.5, 0.5]

    disp = Show(src, view)
    disp.Representation = "Surface With Edges" if p.get("edges") else "Surface"
    if p.get("edges"):
        disp.EdgeColor = [0.2, 0.2, 0.2] if not dark else [0.7, 0.7, 0.7]

    src.UpdatePipeline(t)
    color_field = field if mode != "contour" else (p.get("field") if p.get("field") != p.get("contour_field") else None)
    if mode == "mesh":
        disp.Representation = "Surface With Edges"
        disp.DiffuseColor = [0.85, 0.87, 0.9]
        color_field = None

    if color_field:
        ai = array_info(src, color_field)
        if ai is None:
            fail(f"Field '{color_field}' has no data on the selected geometry.")
        assoc, ncomp, ranges = ai
        comp = (p.get("component") or ("Magnitude" if ncomp > 1 else None))
        if ncomp > 1:
            ColorBy(disp, (assoc, color_field, comp if comp == "Magnitude" else comp.upper()))
        else:
            ColorBy(disp, (assoc, color_field))
        lut = GetColorTransferFunction(color_field)
        lut.ApplyPreset(PRESETS.get(p.get("colormap", "coolwarm"), p.get("colormap", "Cool to Warm")), True)
        if ncomp > 1:
            if comp == "Magnitude":
                lut.VectorMode = "Magnitude"
                rng = ranges[-1]
            else:
                lut.VectorMode = "Component"
                ci = AXES.get(comp.lower(), 0)
                lut.VectorComponent = ci
                rng = ranges[ci]
        else:
            rng = ranges[0]
        if p.get("range"):
            rng = [float(p["range"][0]), float(p["range"][1])]
        lut.RescaleTransferFunction(rng[0], rng[1])
        try:
            lut.AutomaticRescaleRangeMode = "Never"
        except AttributeError:
            pass
        disp.SetScalarBarVisibility(view, True)
        bar = GetScalarBar(lut, view)
        bar.Title = color_field + ("" if ncomp == 1 else f" ({comp})")
        bar.ComponentTitle = ""
        bar.TitleColor = fg
        bar.LabelColor = fg
        bar.Orientation = "Horizontal"
        bar.WindowLocation = "Lower Center"
        bar.ScalarBarLength = 0.5
        bar.TitleFontSize = 14
        bar.LabelFontSize = 12
        result["color"] = {
            "field": color_field,
            "component": comp,
            "range": [sig(rng[0]), sig(rng[1])],
            "data_range": [sig(v) for v in (ranges[-1] if comp == "Magnitude" or ncomp == 1 else rng)],
        }

    label = p.get("title") or f"{p.get('case_name', '')}   {color_field or mode}   t = {t:g}"
    txt = Text(Text=label)
    td = Show(txt, view)
    td.Color = fg
    td.FontSize = 14
    td.WindowLocation = "Upper Left Corner"

    # ---------------- camera ----------------
    vdir = p.get("view", "auto")
    # A direction vector [dx, dy, dz] places the camera along it (perspective), like "iso".
    cam_dir = [float(v) for v in vdir] if isinstance(vdir, list) else [1.0, -1.0, 1.0]
    if isinstance(vdir, list):
        vdir = "iso"
    if vdir == "auto":
        if mode == "slice":
            nrm = result["slice"]["normal"]
            axis = max(range(3), key=lambda i: abs(nrm[i]))
            vdir = "+" + "xyz"[axis]
        elif is_2d:
            vdir = "+" + "xyz"[thin[0] if thin else 2]
        else:
            vdir = "iso"
    cam = view.GetActiveCamera()
    focused = bool(p.get("focus"))
    ext, center = vext, vcenter
    dist = 3 * (max(ext) or big)
    if vdir == "iso":
        norm = sum(c * c for c in cam_dir) ** 0.5 or 1.0
        d = (1.25 if focused else 3.0) * (max(ext) or big) * 3 ** 0.5
        view.CameraPosition = [center[i] + d * cam_dir[i] / norm for i in range(3)]
        view.CameraViewUp = [0, 0, 1]
        view.CameraFocalPoint = center
        view.CameraParallelProjection = 0
        if not focused:
            view.ResetCamera(False)
            cam.Dolly(1.2)
    else:
        sign = -1.0 if vdir.startswith("-") else 1.0
        axis = AXES[vdir[-1].lower()]
        pos = list(center)
        pos[axis] += sign * dist  # camera sits on the +axis side looking back
        up = [0, 0, 1] if axis != 2 else [0, 1, 0]
        view.CameraPosition = pos
        view.CameraFocalPoint = center
        view.CameraViewUp = up
        view.CameraParallelProjection = 1
        if not focused:
            view.ResetCamera(False)
        # Fit the projected extent tightly, leaving room for the colour bar.
        u_axis = [i for i in range(3) if i != axis]
        up_axis = 2 if axis != 2 else 1
        horiz = [i for i in u_axis if i != up_axis][0]
        aspect = w / h
        half = max(ext[up_axis] / 2 * 1.35, ext[horiz] / 2 / aspect * 1.12)
        view.CameraParallelScale = half or 1.0
        if p.get("zoom"):
            view.CameraParallelScale = half / float(p["zoom"])
    if p.get("zoom") and vdir == "iso":
        cam.Dolly(float(p["zoom"]))
    result["view"] = vdir if vdir != "iso" else {"iso": cam_dir}

    if p.get("stats_only"):
        emit(result)
        return
    SaveScreenshot(p["out"], view, ImageResolution=[w, h], TransparentBackground=0)
    result["image"] = p["out"]
    emit(result)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # report errors as structured output
        fail(f"{type(exc).__name__}: {exc}")
