"""Probe: can this interpreter import ParaView *and* render off-screen? Executed, not imported."""

import os
import re
import sys
import tempfile

import paraview.simple as pv

view = pv.CreateRenderView()
view.ViewSize = [64, 64]
pv.Show(pv.Sphere(), view)
out = os.path.join(tempfile.gettempdir(), f"ofmcp-probe-{os.getpid()}.png")
pv.SaveScreenshot(out, view, ImageResolution=[64, 64])
ok = os.path.getsize(out) > 0
os.unlink(out)
version = ".".join(re.findall(r"\d+", str(pv.GetParaViewVersion()))) or "unknown"
print("OFMCP_OK" if ok else "OFMCP_FAIL", version)
sys.stdout.flush()
