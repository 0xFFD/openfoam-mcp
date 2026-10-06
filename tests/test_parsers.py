"""Parser tests against real OpenFOAM 14 output (no OpenFOAM installation needed)."""

from pathlib import Path

import numpy as np
import pytest

from openfoam_mcp import dictparse, fields, logs, mesh, postproc

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------- dictionaries


def test_dictparse_nested_and_directives():
    text = """
    FoamFile { format ascii; class dictionary; }
    // comment
    solver incompressibleFluid;  /* block
    comment */
    SIMPLE
    {
        nNonOrthogonalCorrectors 0;
        residualControl { p 1e-2; "(k|epsilon)" 1e-3; }
    }
    #includeFunc streamlinesLine
    (
        name=streamlines,
        nPoints=10
    )
    #include "common"
    vec (1 2 3);
    code #{ int x = 1; #};
    """
    d = dictparse.parse(text)
    assert d["solver"] == "incompressibleFluid"
    assert d["SIMPLE"]["residualControl"]['"(k|epsilon)"'] == "1e-3"
    assert d["vec"] == "(1 2 3)"
    assert any(k.startswith("#includeFunc streamlinesLine(") for k in d)
    assert '#include "common"' in d
    assert d["code"].startswith("#{")


def test_dictparse_field_file():
    d = dictparse.read(DATA / "U_initial")
    assert d["internalField"] == "uniform (0 0 0)"
    assert d["boundaryField"]["inlet"]["type"] == "fixedValue"
    assert d["boundaryField"]["frontAndBack"]["type"] == "empty"


def test_elide_large_lists():
    values = " ".join(f"({i} 0 0)" for i in range(200))
    text = f"internalField nonuniform List<vector> 200 ({values});\nboundaryField {{ }}"
    d = dictparse.parse(text)
    assert d["internalField"] == "nonuniform<200_values>"


def test_boundary_file(tmp_path):
    pm = tmp_path / "constant" / "polyMesh"
    pm.mkdir(parents=True)
    (pm / "boundary").write_text((DATA / "boundary").read_text())
    names = {p["name"]: p for p in mesh.read_boundary(tmp_path)}
    assert names["inlet"]["type"] == "patch" and names["inlet"]["nFaces"] == 30
    assert names["frontAndBack"]["type"] == "empty"
    assert names["upperWall"]["type"] == "wall"


# ---------------------------------------------------------------- logs


def test_solver_log_converged():
    data = logs.LogData()
    for line in (DATA / "foamRun_steady.log").read_text().splitlines():
        data.feed(line)
    data.finish()
    rep = logs.summarize(data, end_time=2000)
    assert rep["state"] == "converged"
    assert "SIMPLE solution converged" in rep["converged"]
    assert {"Ux", "Uy", "p", "k", "epsilon"} <= set(rep["residuals"])
    assert rep["residuals"]["p"]["initial"] < 1e-2
    assert rep["time"] == 285.0  # OpenFOAM 14 prints "Time = 285s"


def test_log_cache_incremental(tmp_path):
    log = tmp_path / "log.foamRun"
    lines = (DATA / "foamRun_steady.log").read_text().splitlines(keepends=True)
    cache = logs.LogCache()
    log.write_text("".join(lines[:150]))
    first = cache.get(log)
    n1 = len(first.times)
    with open(log, "a") as f:
        f.write("".join(lines[150:]))
    second = cache.get(log)
    assert len(second.times) > n1
    assert second.converged


def test_fatal_and_divergence_detection():
    data = logs.LogData()
    text = """Time = 1s
smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.1, No Iterations 3
Time = 2s
smoothSolver:  Solving for Ux, Initial residual = nan, Final residual = nan, No Iterations 1000

--> FOAM FATAL ERROR: (openfoam-14)
Maximum number of iterations exceeded

    From function Foam::solver
FOAM exiting
"""
    for line in text.splitlines():
        data.feed(line)
    data.finish()
    rep = logs.summarize(data)
    assert rep["state"] == "failed"
    assert "Maximum number of iterations" in rep["error"]
    assert any("nan" in a for a in rep["alarms"])


def test_courant_and_deltat():
    data = logs.LogData()
    for line in [
        "Courant Number mean: 0.1 max: 0.9",
        "deltaT = 1e-05",
        "Time = 1e-05s",
        "Courant Number mean: 0.2 max: 1.4",
        "PIMPLE: Iteration 1",
        "DILUPBiCGStab:  Solving for Ux, Initial residual = 0.5, Final residual = 1e-6, No Iterations 2",
        "DILUPBiCGStab:  Solving for Ux, Initial residual = 0.01, Final residual = 1e-7, No Iterations 1",
        "ExecutionTime = 0.5 s  ClockTime = 1 s",
    ]:
        data.feed(line)
    rep = logs.summarize(data)
    assert rep["courant"]["max"] == 1.4
    assert rep["deltaT"] == 1e-05
    # only the first solve per time step is kept
    assert rep["residuals"]["Ux"]["initial"] == 0.5


# ---------------------------------------------------------------- mesh


def test_check_mesh_ok():
    rep = mesh.parse_check_mesh((DATA / "checkMesh.log").read_text())
    assert rep["verdict"] == "Mesh OK"
    assert rep["stats"]["cells"] == 12225
    assert rep["cell_types"] == {"hexahedra": 12225}
    assert rep["quality"]["dimensions"] == 2
    assert rep["quality"]["max_non_orthogonality"] == pytest.approx(5.95045)
    assert "failures" not in rep


def test_check_mesh_failures():
    text = (DATA / "checkMesh.log").read_text()
    text = text.replace("Mesh non-orthogonality Max: 5.95045", "Mesh non-orthogonality Max: 78.2")
    text = text.replace(
        "Mesh OK.",
        " ***Number of non-orthogonality errors: 12.\n  <<Writing 12 non-orthogonal faces to set nonOrthoFaces\n\nFailed 1 mesh checks.",
    )
    rep = mesh.parse_check_mesh(text)
    assert rep["verdict"] == "FAILED 1 mesh checks"
    assert rep["failures"] == ["Number of non-orthogonality errors: 12."]
    assert any("nNonOrthogonalCorrectors" in a for a in rep["advice"])


def test_check_mesh_failure_value_formats():
    """Failing checks print values with trailing punctuation or a different separator (real snappyHexMesh output)."""
    text = (DATA / "checkMesh.log").read_text()
    text = text.replace("Max aspect ratio = 8.1407 OK.", " ***High aspect ratio cells found, Max aspect ratio: 1038.27, number of cells 20")
    text = text.replace("Max skewness = 0.260575 OK.", " ***Max skewness = 9.47193, 13 highly skew faces detected which may impair the quality of the results")
    q = mesh.parse_check_mesh(text)["quality"]
    assert q["max_skewness"] == pytest.approx(9.47193)
    assert q["max_aspect_ratio"] == pytest.approx(1038.27)


# ---------------------------------------------------------------- fields & postProcessing


def test_field_stats_nonuniform(tmp_path):
    vals = "\n".join(f"({i} {-i} 0)" for i in range(10))
    p = tmp_path / "U"
    p.write_text(
        (DATA / "U_initial").read_text().replace("uniform (0 0 0);", f"nonuniform List<vector> \n10\n(\n{vals}\n)\n;", 1)
    )
    st = fields.field_stats(p)
    assert st["internal"]["count"] == 10
    assert st["internal"]["x"] == [0.0, 9.0, 4.5]
    assert st["patches"]["inlet"]["value"] == [10.0, 0.0, 0.0]


def test_time_dirs_sorted(tmp_path):
    for t in ["0", "100", "50", "1e-05", "0.orig", "constant"]:
        (tmp_path / t).mkdir()
    assert fields.time_dirs(tmp_path) == ["0", "1e-05", "50", "100"]


def test_postprocessing_text_columns():
    names, arr, labels = postproc.load_table(DATA / "yPlus.dat")
    assert names == ["Time", "patch", "min", "max", "average"]
    rep = postproc.summarize_table(names, arr, None, 10, labels)
    assert rep["rows"][0][1] == "upperWall"
    assert rep["stats"]["max"]["max"] == pytest.approx(26.51502)


def test_postprocessing_vector_columns(tmp_path):
    p = tmp_path / "forces.dat"
    p.write_text("# Force\n# Time\ttotal\tpressure\n1\t(1 2 3)\t(4 5 6)\n2\t(2 3 4)\t(5 6 7)\n")
    names, arr, _ = postproc.load_table(p)
    assert names == ["Time", "total_x", "total_y", "total_z", "pressure_x", "pressure_y", "pressure_z"]
    assert arr.shape == (2, 7)
    assert np.allclose(arr[:, 3], [3, 4])
