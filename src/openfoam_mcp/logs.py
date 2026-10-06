"""Incremental parser for OpenFOAM solver logs, plus a convergence assessment."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

_TIME = re.compile(r"^Time = ([-+0-9.eE]+)\s*s?\s*$")
_RESID = re.compile(
    r"^(?:\w+:\s+)?Solving for ([\w.]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)"
)
_COURANT = re.compile(r"^(?:\w+ )?Courant Number mean: (\S+) max: (\S+)")
_CONT = re.compile(r"^time step continuity errors : sum local = ([^,]+), global = ([^,]+), cumulative = (\S+)")
_EXEC = re.compile(r"^ExecutionTime = (\S+) s\s+ClockTime = (\S+) s")
_DELTAT = re.compile(r"^deltaT = (\S+)")
_CONVERGED = re.compile(r"^(\w+) solution converged in (\S+) iterations")
_FATAL_START = re.compile(r"--> FOAM FATAL (?:IO )?ERROR")
_WARNING = re.compile(r"--> FOAM Warning")
_FPE = re.compile(r"Floating point exception|sigFpe::sigHandler|Segmentation fault|sigSegv::sigHandler")


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


@dataclass
class LogData:
    times: list[float] = field(default_factory=list)
    # field -> list of (time_index, initial_residual, final_residual, iterations); first solve per step only.
    residuals: dict[str, list[tuple[int, float, float, int]]] = field(default_factory=dict)
    courant: list[tuple[int, float, float]] = field(default_factory=list)
    continuity: list[tuple[int, float, float, float]] = field(default_factory=list)
    exec_time: list[tuple[int, float, float]] = field(default_factory=list)
    delta_t: list[tuple[int, float]] = field(default_factory=list)
    converged: str | None = None
    ended: bool = False
    fatal: str | None = None
    crash: str | None = None
    warnings: int = 0
    warning_samples: list[str] = field(default_factory=list)
    header: dict[str, str] = field(default_factory=dict)
    # parser state
    _seen_this_step: set[str] = field(default_factory=set)
    _fatal_lines: list[str] | None = None
    _warn_lines: list[str] | None = None

    def feed(self, line: str) -> None:
        line = line.rstrip("\n")
        s = line.strip()
        if self._fatal_lines is not None:
            if re.match(r"FOAM (exiting|aborting)", s) or len(self._fatal_lines) > 40:
                self.fatal = "\n".join(self._fatal_lines).strip()
                self._fatal_lines = None
            else:
                self._fatal_lines.append(line)
            return
        if self._warn_lines is not None:
            if not s or len(self._warn_lines) >= 8:
                msg = " ".join(x.strip() for x in self._warn_lines)
                if msg not in self.warning_samples and len(self.warning_samples) < 5:
                    self.warning_samples.append(msg[:400])
                self._warn_lines = None
            else:
                self._warn_lines.append(line)
            return
        if m := _TIME.match(s):
            self.times.append(_float(m.group(1)))
            self._seen_this_step = set()
            return
        idx = len(self.times) - 1
        if m := _RESID.match(s):
            fld = m.group(1)
            if fld not in self._seen_this_step:
                self._seen_this_step.add(fld)
                self.residuals.setdefault(fld, []).append(
                    (idx, _float(m.group(2)), _float(m.group(3)), int(m.group(4)))
                )
            return
        if m := _COURANT.match(s):
            if not self.courant or self.courant[-1][0] != idx:
                self.courant.append((idx, _float(m.group(1)), _float(m.group(2))))
            return
        if m := _CONT.match(s):
            self.continuity.append((idx, _float(m.group(1)), _float(m.group(2)), _float(m.group(3))))
            return
        if m := _EXEC.match(s):
            self.exec_time.append((idx, _float(m.group(1)), _float(m.group(2))))
            return
        if m := _DELTAT.match(s):
            self.delta_t.append((idx, _float(m.group(1))))
            return
        if m := _CONVERGED.match(s):
            self.converged = s
            return
        if s == "End":
            self.ended = True
            return
        if _FATAL_START.search(s):
            self._fatal_lines = [s]
            return
        if _WARNING.search(s):
            self.warnings += 1
            self._warn_lines = [s]
            return
        if _FPE.search(s) and not self.crash:
            self.crash = s
            return
        if not self.times and ":" in s and len(self.header) < 20:
            k, _, v = s.partition(":")
            k = k.strip()
            if k in ("Exec", "Case", "nProcs", "Build", "Date", "Time", "Host", "PID"):
                self.header[k] = v.strip()

    def finish(self) -> None:
        if self._fatal_lines is not None:
            self.fatal = "\n".join(self._fatal_lines).strip()


class LogCache:
    """Keeps parsed state per log file and only reads newly appended bytes on each call."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[int, int, int, LogData]] = {}
        self._lock = threading.Lock()

    def get(self, path: Path) -> LogData:
        with self._lock:
            return self._get(path)

    def _get(self, path: Path) -> LogData:
        st = path.stat()
        key = str(path)
        entry = self._entries.get(key)
        if entry and entry[0] == st.st_ino and entry[1] <= st.st_size:
            ino, offset, _, data = entry
        else:
            offset, data = 0, LogData()
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        # Only consume complete lines; keep a partial trailing line for next time.
        cut = chunk.rfind(b"\n") + 1
        for line in chunk[:cut].decode(errors="replace").splitlines():
            data.feed(line)
        self._entries[key] = (st.st_ino, offset + cut, st.st_mtime_ns, data)
        data.finish()
        return data


def _trend(values: list[float]) -> float | None:
    """Change in log10(residual) between the first and last quarter of the window (negative = falling)."""
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    if len(vals) < 8:
        return None
    q = max(2, len(vals) // 4)
    a = sum(math.log10(v) for v in vals[:q]) / q
    b = sum(math.log10(v) for v in vals[-q:]) / q
    return round(b - a, 2)


def summarize(data: LogData, end_time: float | None = None, window: int = 200) -> dict:
    """Condense parsed log data into a compact, decision-friendly report."""
    out: dict = {"steps": len(data.times)}
    if data.header.get("Exec"):
        out["exec"] = data.header["Exec"]
    if data.times:
        out["time"] = data.times[-1]
        if end_time and end_time > 0:
            start = data.times[0]
            span = end_time - start if end_time > start else end_time
            out["progress_pct"] = round(100 * (data.times[-1] - start) / span, 1) if span > 0 else None
    fields = {}
    alarms = []
    for fld, rows in data.residuals.items():
        init = [r[1] for r in rows]
        latest = init[-1]
        recent = init[-window:]
        info = {
            "initial": latest,
            "final": rows[-1][2],
            "iters": rows[-1][3],
            "min_initial": min((v for v in init if math.isfinite(v)), default=None),
            "log10_trend": _trend(recent),
        }
        fields[fld] = info
        if not math.isfinite(latest):
            alarms.append(f"{fld} residual is {latest}")
        elif info["log10_trend"] is not None and info["log10_trend"] > 1.0 and latest > 1e-3:
            alarms.append(f"{fld} residual rose {info['log10_trend']} decades over the last {len(recent)} steps")
    if fields:
        out["residuals"] = fields
    if data.courant:
        _, mean, mx = data.courant[-1]
        out["courant"] = {"mean": mean, "max": mx, "max_seen": max(c[2] for c in data.courant)}
        if mx > 50 or not math.isfinite(mx):
            alarms.append(f"Courant number max is {mx}")
    if data.continuity:
        _, local, glob, cum = data.continuity[-1]
        out["continuity"] = {"sum_local": local, "global": glob, "cumulative": cum}
        if local > 1 or not math.isfinite(local):
            alarms.append(f"continuity error sum_local={local} is large")
    if data.delta_t:
        out["deltaT"] = data.delta_t[-1][1]
    if data.exec_time:
        idx, ex, clock = data.exec_time[-1]
        out["execution_time_s"] = ex
        out["clock_time_s"] = clock
        if len(data.exec_time) > 10:
            i0, e0, _ = data.exec_time[-min(len(data.exec_time), 50)]
            if idx > i0:
                per_step = (ex - e0) / (idx - i0)
                out["seconds_per_step"] = round(per_step, 4)
                if end_time and data.times and len(data.times) > 1:
                    dt_sim = (data.times[-1] - data.times[max(0, len(data.times) - 50)]) / min(49, len(data.times) - 1)
                    if dt_sim > 0:
                        remaining = (end_time - data.times[-1]) / dt_sim * per_step
                        out["eta_s"] = round(max(0.0, remaining))
    if data.warnings:
        out["warnings"] = {"count": data.warnings, "samples": data.warning_samples}

    if data.fatal:
        state = "failed"
        out["error"] = data.fatal
    elif data.crash:
        state = "crashed"
        out["error"] = data.crash
    elif data.converged:
        state = "converged"
        out["converged"] = data.converged
    elif data.ended:
        state = "finished"
    elif alarms:
        state = "diverging"
    else:
        state = "in_progress"
    out["state"] = state
    if state != "in_progress":
        out.pop("eta_s", None)
    trends = [f["log10_trend"] for f in fields.values()]
    if state in ("in_progress", "finished") and trends and all(t is not None and abs(t) < 0.1 for t in trends):
        n = min(window, len(data.times))
        out["plateau"] = (
            f"Residuals changed by less than 0.1 decades over the last {n} steps. For a steady case this means "
            "no further convergence: stop early, check force/probe histories for a converged mean, or improve "
            "mesh/numerics; for an unsteady case it is normal."
        )
    if alarms:
        out["alarms"] = alarms
    return out


def series(data: LogData) -> dict[str, tuple[list[float], list[float]]]:
    """Residual histories as (time, initial_residual) per field, for plotting."""
    return {
        fld: ([data.times[i] if 0 <= i < len(data.times) else i for i, *_ in rows], [r[1] for r in rows])
        for fld, rows in data.residuals.items()
    }


def find_solver_log(case_dir: Path) -> Path | None:
    """Most recently modified log.* file in the case that looks like a time-stepping solver log."""
    logs = sorted(
        (p for p in case_dir.glob("log.*") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for p in logs:
        with open(p, "rb") as f:
            head = f.read(200_000)
        if re.search(rb"^Time = ", head, re.M) and b"Solving for" in head:
            return p
    return None


def tail(path: Path, lines: int = 60, max_bytes: int = 2_000_000) -> str:
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - max_bytes))
        data = f.read().decode(errors="replace")
    return "\n".join(data.splitlines()[-lines:])
