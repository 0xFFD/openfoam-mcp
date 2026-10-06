"""GPU wind tunnel: XLB lattice Boltzmann on an STL, with on-GPU rendering. Executed, not imported.

Usage: <python with xlb, jax[cuda], warp-lang> _gpu_lbm.py <params.json>

Runs inside an openfoam-mcp background job. Prints `OFMCP_PROGRESS {json}` lines for the server and
writes into the case directory:
    frames/<view>_<NNNNN>.png   rendered frames (vortex smoke projection and velocity slice)
    forces.csv                  time [s], Fx, Fy, Fz [N] on the body (momentum exchange)
    meta.json                   grid, spacing, time step, Reynolds numbers, memory
    <view>.mp4                  videos encoded at the end
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The video encoder starts ffmpeg via fork; JAX warns about forking a multithreaded process, which is
# harmless here because the child only execs ffmpeg.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=r"os.fork\(\) was called", category=RuntimeWarning)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import warp as wp  # noqa: E402
import xlb  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from xlb.compute_backend import ComputeBackend  # noqa: E402
from xlb.grid import grid_factory  # noqa: E402
from xlb.operator.boundary_condition import (  # noqa: E402
    ExtrapolationOutflowBC,
    FullwayBounceBackBC,
    HalfwayBounceBackBC,
    RegularizedBC,
)
from xlb.operator.macroscopic import Macroscopic  # noqa: E402
from xlb.operator.stepper import IncompressibleNavierStokesStepper  # noqa: E402
from xlb.precision_policy import PrecisionPolicy  # noqa: E402

wp.config.quiet = True
# Measured on an RTX 2060 (Warp backend, fp32 storage), plus margin. fp16 storage was dropped: it diverges
# at realistic speeds because rounding noise in near-equilibrium populations grows.
BYTES_PER_CELL = {"accurate": 250, "fast": 175}


def body_force(f, missing_mask, bc_mask, body_id, c, opp, exclude_ids=(), slab=48):
    """Momentum-exchange force on the body (Ladd), in lattice units.

    XLB stores post-collision populations. At a boundary node x, direction i is "missing" when
    x - c_i is solid, i.e. the link towards the wall is j = opp(i). With halfway bounce-back the
    population f_j leaving towards the wall comes straight back, transferring 2 c_j f_j to the body.

    A body boundary node can also touch the tunnel floor or walls (a body standing on the ground);
    links whose source x - c_i belongs to one of `exclude_ids` push on the tunnel, not the body, and
    are skipped. Evaluated slab by slab on views shared with Warp (no copies of the full fields).
    """
    fj, mj, bj = wp.to_jax(f), wp.to_jax(missing_mask), wp.to_jax(bc_mask)
    c_int = np.asarray(c).reshape(3, -1).astype(int)
    cj = jnp.asarray(c_int.astype(np.float32))  # (3, q)
    opp = np.asarray(opp, dtype=np.int32).reshape(-1)
    nx = fj.shape[1]
    total = jnp.zeros(3, jnp.float32)
    for x0 in range(0, nx, slab):
        a, b = max(x0 - 1, 0), min(x0 + slab + 1, nx)  # one halo cell for the neighbour lookup
        core = slice(x0 - a, x0 - a + min(slab, nx - x0))
        ids = bj[0, a:b]
        is_body = ids == body_id
        per_dir = []
        for i in range(c_int.shape[1]):
            ok = jnp.logical_and(mj[i, a:b], is_body)
            if exclude_ids:
                src = jnp.roll(ids, shift=tuple(int(v) for v in c_int[:, i]), axis=(0, 1, 2))  # value at x is ids[x - c_i]
                for e in exclude_ids:
                    ok = jnp.logical_and(ok, src != e)
            per_dir.append(jnp.sum(jnp.where(ok[core], fj[opp[i], a:b][core].astype(jnp.float32), 0.0)))
        total = total + cj[:, opp] @ jnp.stack(per_dir)
    return np.asarray(2.0 * total)


def emit(**kw):
    print("OFMCP_PROGRESS " + json.dumps(kw), flush=True)


def free_vram_mb() -> float:
    dev = wp.get_device("cuda:0")
    free = getattr(dev, "free_memory", None)
    if free:
        return free / 2**20
    import subprocess

    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return float(out.stdout.split()[0])


def rotation_to_x(direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    return trimesh.geometry.align_vectors(d, [1.0, 0.0, 0.0])


def main():
    p = json.loads(Path(sys.argv[1]).read_text())
    out = Path(p["case_dir"])
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mode = p.get("mode", "accurate")

    # ---------------- geometry ----------------
    mesh = trimesh.load_mesh(p["geometry"], process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    mesh.apply_transform(rotation_to_x(p.get("direction", [1, 0, 0])))
    lo, hi = mesh.bounds
    ext = hi - lo
    L = float(ext.max())

    ground = bool(p.get("ground", False))
    # Tunnel margins in body lengths. The side walls are no-slip (the far-field conditions in XLB 0.3.1
    # were not stable), so generous margins keep blockage and wall effects modest.
    m_up, m_down, m_side, m_top = 0.8, 2.6, 0.8, 0.8
    m_bottom = 0.0 if ground else 0.8
    dom = np.array([ext[0] + (m_up + m_down) * L, ext[1] + 2 * m_side * L, ext[2] + (m_top + m_bottom) * L])

    # Resolution: cells across the body's largest dimension; by default as fine as free VRAM allows.
    budget = float(p.get("max_cells") or (free_vram_mb() - 700) * 2**20 / BYTES_PER_CELL[mode])
    n_fit = int((budget / np.prod(dom / L)) ** (1 / 3))
    N = int(p.get("resolution") or n_fit)
    if N > n_fit * 1.02:
        raise SystemExit(f"resolution {N} needs more GPU memory than is free; the maximum now is about {n_fit}.")
    dx = L / N
    shape = tuple(int(math.ceil(v / dx)) + 2 for v in dom)
    shape = tuple(s + (s % 2) for s in shape)  # even sizes keep video encoders happy

    # XLB's mesh masker takes a triangle soup: every three consecutive vertices form one triangle.
    soup = mesh.vertices[mesh.faces].reshape(-1, 3)
    verts = (soup - lo) / dx + np.array([m_up * N + 1, m_side * N + 1, m_bottom * N + (2 if ground else 1)])

    # ---------------- physics / lattice units ----------------
    U = float(p.get("speed", 10.0))
    nu = float(p.get("nu", 1.5e-5))
    rho_phys = float(p.get("rho", 1.2))
    u_lb = float(p.get("lattice_velocity", 0.03))
    Re = U * L / nu
    # Stability is governed by the cell Reynolds number u_lb / nu_lb (measured: 100 stable, 150 not, with
    # XLB 0.3.1 KBC and no-slip tunnel walls). Physical Reynolds numbers of drones and cars (1e5 - 1e7) are far
    # beyond that at any affordable resolution, so the viscosity is floored and the simulated (effective)
    # Reynolds number - which grows with resolution - is reported next to the physical one.
    nu_lb = max(u_lb * N / Re, u_lb / float(p.get("max_cell_reynolds", 100)))
    Re_eff = u_lb * N / nu_lb
    omega = 1.0 / (3.0 * nu_lb + 0.5)
    dt = dx * u_lb / U  # physical seconds per step
    flow_throughs = float(p.get("flow_throughs", 1.5))
    steps = int(p.get("steps") or flow_throughs * shape[0] / u_lb)
    n_frames = int(p.get("frames", 120))
    frame_every = max(1, steps // n_frames)

    be = ComputeBackend.WARP
    pp = PrecisionPolicy.FP32FP32
    vs_cls = xlb.velocity_set.D3Q27 if mode == "accurate" else xlb.velocity_set.D3Q19
    vs = vs_cls(precision_policy=pp, compute_backend=be)
    xlb.init(velocity_set=vs, default_backend=be, default_precision_policy=pp)
    grid = grid_factory(shape, compute_backend=be)
    box = grid.bounding_box_indices()
    nb = grid.bounding_box_indices(remove_edges=True)
    walls = np.unique(np.array([box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(vs.d)]), axis=-1).tolist()
    # Keep XLB's wind-tunnel construction order (walls, inlet, outlet, then the body). Boundary ids are
    # assigned at construction and XLB 0.3.1 diverges within ~50 steps if the mesh body is created first.
    bcs = [
        FullwayBounceBackBC(indices=walls),
        RegularizedBC("velocity", prescribed_value=(u_lb, 0.0, 0.0), indices=nb["left"]),
        ExtrapolationOutflowBC(indices=nb["right"]),
    ]
    body = HalfwayBounceBackBC(mesh_vertices=verts)
    bcs.append(body)
    tunnel_ids = tuple(bc.id for bc in bcs[:-1])
    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=bcs, collision_type="KBC" if mode == "accurate" else "SmagorinskyLESBGK"
    )
    f0, f1, bc_mask, missing_mask = stepper.prepare_fields()
    # Sanity check: the body must not appear on the tunnel's outer boundary (other than the floor it stands on).
    ids_np = bc_mask.numpy()[0]
    solid_np = (ids_np == body.id) | (ids_np == 255)
    stray = {}
    for name, plane in (("top", solid_np[:, :, -2:]), ("front", solid_np[:, :2, :]), ("back", solid_np[:, -2:, :]),
                        ("inlet", solid_np[:2]), ("outlet", solid_np[-2:]), ("floor", solid_np[:, :, :1])):
        if plane.any() and not (name == "floor" and ground):
            stray[name] = int(plane.sum())
    if stray:
        print(f"WARNING body cells on tunnel boundary: {stray}", flush=True)
    del ids_np, solid_np
    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=pp,
        velocity_set=vs_cls(precision_policy=pp, compute_backend=ComputeBackend.JAX),
    )

    meta = {
        "grid": shape,
        "cells": int(np.prod(shape)),
        "resolution_cells_per_body": N,
        "dx_m": dx,
        "dt_s": dt,
        "steps": steps,
        "frame_every": frame_every,
        "physical_time_s": steps * dt,
        "reynolds_physical": Re,
        "reynolds_simulated": Re_eff,
        "lattice_viscosity": nu_lb,
        "omega": omega,
        "cell_reynolds": u_lb / nu_lb,
        "mode": mode,
        "collision": "KBC" if mode == "accurate" else "Smagorinsky LES",
        "blockage_ratio": None,
        "note": "Forces are indicative: the tunnel has no-slip walls (blockage raises drag) and high-Reynolds flow is "
        "under-resolved. Use it for flow visualisation and trends; use OpenFOAM for engineering-grade forces.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    emit(stage="setup", **{k: meta[k] for k in ("grid", "cells", "resolution_cells_per_body", "steps")})

    # ---------------- rendering ----------------
    view = p.get("view", "top")
    ray_axis = 2 if view == "top" else 1  # project along z (top) or y (side)
    lut = jnp.asarray(colormaps[p.get("colormap", "turbo")](np.linspace(0, 1, 256))[:, :3], dtype=jnp.float32)
    width_px = int(p.get("width", 1600))
    sil = np.zeros((shape[0], shape[1 if ray_axis == 2 else 2]), bool)
    vox = mesh.voxelized(pitch=dx)
    pts = np.floor((vox.points - lo) / dx + np.array([m_up * N + 1, m_side * N + 1, m_bottom * N + (2 if ground else 1)])).astype(int)
    pts = np.clip(pts, 0, np.array(shape) - 1)
    sil[pts[:, 0], pts[:, 1 if ray_axis == 2 else 2]] = True
    frontal_cells = len(np.unique(pts[:, 1:3], axis=0))
    meta["blockage_ratio"] = round(frontal_cells / ((shape[1] - 2) * (shape[2] - 2)), 4)
    meta["frontal_area_m2"] = frontal_cells * dx * dx
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    mid = int(np.clip(round(verts[:, ray_axis].mean()), 1, shape[ray_axis] - 2))
    w_ref = None
    slab = 64
    # Per-cell opacity so that a dense vortex region about a tenth of the ray long becomes nearly opaque.
    opacity = 25.0 / shape[ray_axis]
    # Hide the tunnel walls' own shear layers, which thicken downstream: mask a band that scales with the tunnel.
    py, pz = max(7, int(0.1 * shape[1])), max(7, int(0.1 * shape[2]))
    interior = jnp.zeros(shape, jnp.float32).at[7:-7, py:-py, pz:-pz].set(1.0)
    if ground:
        interior = jnp.zeros(shape, jnp.float32).at[7:-7, py:-py, 3:-pz].set(1.0)

    def render(step):
        nonlocal w_ref
        fj = wp.to_jax(f0)
        smoke, slice_img, samples = [], [], []
        for x0 in range(0, shape[0], slab):
            a, b = max(x0 - 1, 0), min(x0 + slab + 1, shape[0])
            _, u = macro(fj[:, a:b])
            u = u.astype(jnp.float32)
            du = [jnp.gradient(u[c], axis=k) for c in range(3) for k in range(3)]
            wx, wy, wz = du[7] - du[5], du[2] - du[6], du[3] - du[1]
            wmag = jnp.sqrt(wx**2 + wy**2 + wz**2)
            umag = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            cut = slice(x0 - a, x0 - a + min(slab, shape[0] - x0))
            wmag, umag = wmag[cut], umag[cut]
            # Ignore the tunnel walls (their no-slip shear would dominate) and the inlet/outlet planes.
            wmag = wmag * interior[x0:x0 + wmag.shape[0]]
            samples.append(wmag[::3, ::3, ::3].ravel())
            ref = w_ref if w_ref is not None else 1.0
            s = jnp.clip((wmag / ref - 0.25) / 0.75, 0.0, 1.0) ** 1.5
            alpha = 1.0 - jnp.exp(-opacity * s)
            col = lut[jnp.clip((umag / (1.4 * u_lb) * 255).astype(jnp.int32), 0, 255)]
            if ray_axis == 2:  # look down: the ray runs from the top (high z) to the ground
                alpha, col = alpha[:, :, ::-1], col[:, :, ::-1]
            # Front-to-back emission/absorption; alpha <= 1 - e^-1.2 so dividing out the own term is safe.
            trans = jnp.cumprod(1.0 - alpha, axis=ray_axis) / (1.0 - alpha)
            weight = (trans * alpha)[..., None]
            smoke.append(jnp.sum(weight * col, axis=ray_axis))
            slice_img.append(jnp.take(umag, mid, axis=ray_axis))
        allw = jnp.concatenate(samples)
        p = float(jnp.percentile(allw, 99.8))
        if p > 0:
            w_ref = p if w_ref is None else 0.8 * w_ref + 0.2 * p
        smoke_img = np.asarray(jnp.concatenate(smoke, axis=0))  # (nx, n_other, 3)
        slice_arr = np.asarray(jnp.concatenate(slice_img, axis=0))
        t_s = step * dt
        caption = f"t = {t_s:.3f} s   wind {U:g} m/s   {meta['cells'] / 1e6:.1f}M cells (GPU lattice Boltzmann)"
        _save(smoke_img, sil, frames_dir / f"vortex_{step // frame_every:05d}.png", width_px, caption, dark=True)
        sl = np.asarray(lut[np.clip((slice_arr / (1.4 * u_lb) * 255).astype(int), 0, 255)])
        _save(sl, sil, frames_dir / f"slice_{step // frame_every:05d}.png", width_px, caption, dark=False)

    # ---------------- run ----------------
    forces_csv = open(out / "forces.csv", "w")
    forces_csv.write("time_s,Fx_N,Fy_N,Fz_N\n")
    force_scale = rho_phys * dx**2 * (U / u_lb) ** 2
    t0 = time.time()
    t_last, s_last = t0, 0
    t_first = t0
    for step in range(steps + 1):
        f0, f1 = stepper(f0, f1, bc_mask, missing_mask, omega, step)
        f0, f1 = f1, f0
        if step % frame_every == 0:
            fj_probe = wp.to_jax(f0)
            if not bool(jnp.isfinite(jnp.sum(fj_probe[:, ::7, ::7, ::7]))):
                emit(stage="diverged", step=step)
                raise SystemExit("Simulation diverged. Try a lower max_cell_reynolds (e.g. 60), mode='accurate' or a coarser resolution.")
            del fj_probe
            force = body_force(f0, missing_mask, bc_mask, body.id, vs.c, vs.opp_indices, tunnel_ids) * force_scale
            if not np.all(np.isfinite(force)):
                emit(stage="diverged", step=step)
                raise SystemExit("Simulation diverged (non-finite forces). Try a lower max_cell_reynolds (e.g. 60) or mode='accurate'.")
            forces_csv.write(f"{step * dt:.6g},{force[0]:.6g},{force[1]:.6g},{force[2]:.6g}\n")
            forces_csv.flush()
            if p.get("render", True):
                render(step)
            wp.synchronize()
            now = time.time()
            mlups = np.prod(shape) * (step - s_last) / max(now - t_last, 1e-9) / 1e6
            t_last, s_last = now, step
            pct = 100.0 * step / steps
            # Time the first frame separately: it includes kernel compilation and would inflate the estimate.
            if step == 0:
                t_first = now
            eta = (now - t_first) / step * (steps - step) if step else None
            emit(stage="running", step=step, steps=steps, pct=round(pct, 1), mlups=round(mlups),
                 eta_s=round(eta) if eta is not None else None,
                 frame=step // frame_every, drag_N=round(float(force[0]), 5))
            print(f"step {step}/{steps} ({pct:.0f}%), {mlups:.0f} MLUPS, drag {force[0]:.4g} N", flush=True)
    forces_csv.close()

    # ---------------- videos ----------------
    videos = {}
    for kind in ("vortex", "slice"):
        files = sorted(frames_dir.glob(f"{kind}_*.png"))
        if files:
            target = out / f"{kind}.mp4"
            encode_mp4(files, target, int(p.get("fps", 24)))
            videos[kind] = str(target)
    meta["videos"] = videos
    meta["wall_time_s"] = round(time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    emit(stage="done", videos=videos, wall_time_s=meta["wall_time_s"])
    print("End", flush=True)


def encode_mp4(files, target, fps):
    import imageio.v2 as imageio

    with imageio.get_writer(target, fps=fps, codec="libx264", quality=8, macro_block_size=2, pixelformat="yuv420p") as w:
        for f in files:
            w.append_data(imageio.imread(f)[..., :3])


def _save(img, sil, path, width_px, caption, dark):
    """img: (nx, ny, 3) in [0,1], x along the flow. Writes a PNG with x horizontal, upscaled, with the body silhouette."""
    # Drop the outermost cell frame: those are the tunnel's bounce-back wall nodes, whose "velocity"
    # is meaningless and would show up as stray bands at the image edges.
    img, sil = np.clip(img[1:-1, 1:-1], 0, 1), sil[1:-1, 1:-1]
    body = np.array([0.82, 0.82, 0.86]) if dark else np.array([0.15, 0.15, 0.18])
    # Semi-transparent body silhouette: the flow under (or behind) it stays visible.
    img = np.where(sil[..., None], 0.45 * body + 0.55 * img, img)
    pic = (np.transpose(img, (1, 0, 2))[::-1] * 255).astype(np.uint8)  # rows = second axis, up is up
    h, w = pic.shape[:2]
    scale = width_px / w
    im = Image.fromarray(pic).resize((width_px, int(round(h * scale / 2)) * 2), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    size = max(14, width_px // 70)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    d.text((14, im.height - size - 12), caption, font=font, fill=(235, 235, 235) if dark else (15, 15, 15))
    im.save(path)


if __name__ == "__main__":
    main()
