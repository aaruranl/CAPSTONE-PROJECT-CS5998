"""
One shared look for every figure in the project.

Colour rules used everywhere (so a reader learns them once):
  * NORMAL   = blue   - always means "normal" groups / lines
  * ABNORMAL = orange - always means "abnormal" groups / alert lines
  * single-series charts use blue; split boundaries are thin grey lines.
Colours come from a colour-blind-checked categorical palette (slots 1 and 2).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NORMAL = "#2a78d6"
ABNORMAL = "#eb6834"
SERIES_1 = NORMAL

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT,
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": TEXT_2,
        "ytick.labelcolor": TEXT_2,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_2,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def mark_boundaries(ax, boundaries: dict, y_text: float = 0.98) -> None:
    """Draw thin vertical lines where the time-based split changes (train | val | test)."""
    for name, x in boundaries.items():
        ax.axvline(x, color=MUTED, linewidth=0.9, zorder=1)
        ax.text(x, y_text, f" {name}", transform=ax.get_xaxis_transform(), va="top",
                ha="left", fontsize=8.5, color=TEXT_2)


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
