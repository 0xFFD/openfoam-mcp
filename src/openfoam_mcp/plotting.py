"""Headless matplotlib plots returned to the client as PNG images."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def residuals(series: dict[str, tuple[list[float], list[float]]], title: str, courant=None) -> bytes:
    rows = 2 if courant else 1
    fig, axes = plt.subplots(rows, 1, figsize=(9, 4.2 * rows), squeeze=False, sharex=True)
    ax = axes[0][0]
    for fld, (t, r) in sorted(series.items()):
        t_arr, r_arr = np.asarray(t, float), np.asarray(r, float)
        mask = r_arr > 0
        if mask.any():
            ax.semilogy(t_arr[mask], r_arr[mask], label=fld, linewidth=1.1)
    ax.set_ylabel("initial residual")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if series:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    if courant:
        t, mean, mx = courant
        cax = axes[1][0]
        cax.plot(t, mx, label="Co max", linewidth=1.1)
        cax.plot(t, mean, label="Co mean", linewidth=1.1)
        cax.set_ylabel("Courant number")
        cax.grid(True, alpha=0.3)
        cax.legend(loc="upper right", fontsize=8)
    axes[-1][0].set_xlabel("time / iteration")
    return _png(fig)


def table(names: list[str], arr: np.ndarray, columns: list[str], title: str, x: str | None = None) -> bytes:
    xi = names.index(x) if x else 0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in columns:
        ax.plot(arr[:, xi], arr[:, names.index(c)], label=c, linewidth=1.1)
    ax.set_xlabel(names[xi])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _png(fig)
