"""End-to-end run of a real tutorial. Skipped when OpenFOAM is not installed."""

import json

import pytest

from openfoam_mcp import server as S
from openfoam_mcp.config import Settings
from openfoam_mcp.errors import FoamError
from openfoam_mcp.foam import FoamEnv

try:
    FOAM = FoamEnv.detect()
except FoamError:
    FOAM = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(FOAM is None or not FOAM.uses_foam_run, reason="needs openfoam.org >= 11"),
]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    S.configure(Settings(workspace=tmp_path_factory.mktemp("ws"), wait_seconds=120))
    S.create_case("pitz", "incompressibleFluid/pitzDailySteady")
    S.set_dict("pitz", "system/controlDict", set={"endTime": "60", "writeInterval": "30"})
    return "pitz"


def test_mesh_and_check(case):
    r = S.run(case, "blockMesh", ["-dict", "$FOAM_TUTORIALS/resources/blockMesh/pitzDaily"])
    assert r["state"] == "succeeded", r
    chk = S.check_mesh(case)
    assert chk["verdict"] == "Mesh OK" and chk["stats"]["cells"] == 12225


def test_solve_and_monitor(case):
    r = S.run(case, "foamRun")
    assert r["state"] == "succeeded", r
    prog = S.solver_progress(case)
    assert prog["steps"] == 60 and prog["state"] == "finished"
    assert S.list_cases()[0]["latest_time"] == "60"


def test_field_stats_and_dict(case):
    st = S.field_stats(case, "U")
    assert st["time"] == "60" and st["internal"]["count"] == 12225
    assert "endTime" in S.get_dict(case, "system/controlDict", entry="endTime")


def test_summary_is_json_serialisable(case):
    json.dumps(S.case_summary(case))
