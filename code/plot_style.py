"""Shared figure style: engine colours, markers and line styles, metric
labels, and the matplotlib settings used by every figure in the project.

    from plot_style import ENGINES, METRICS, apply_style
    plt = apply_style()
    s = ENGINES["stockfish"]
    ax.plot(x, y, color=s.colour, marker=s.marker, ls=s.linestyle, label=s.label)
    ax.set_ylabel(METRICS["escore"].axis_label)

Text in figures is Arial (Liberation Sans, which has identical metrics, is
used where Arial isn't installed). Metric symbols use matplotlib's LaTeX-style
mathtext: E_score as $E_{score}$ and Delta_Elo as $\\Delta_{Elo}$.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineStyle:
    label: str
    colour: str
    marker: str
    linestyle: str


ENGINES: dict[str, EngineStyle] = {
    "stockfish": EngineStyle("Stockfish", "#104485", "o", "-"),
    "caissa": EngineStyle("Caissa", "#f17f03", "s", "--"),
    "lc0": EngineStyle("Lc0", "#418c47", "^", ":"),
    "plentychess": EngineStyle("PlentyChess", "#b63667", "D", "-."),
}


@dataclass(frozen=True)
class MetricStyle:
    key: str          # used in file and column names
    symbol: str       # mathtext, for titles and captions
    bold_symbol: str  # mathtext in bold, for axis labels
    unit: str
    name: str         # plain-language name

    @property
    def axis_label(self) -> str:
        unit = f" ({self.unit})" if self.unit else ""
        return f"{self.bold_symbol}{unit}"


# LaTeX versions for text (manuscript, summaries): $E_{\text{score}}$ and
# $\Delta_{\text{Elo}}$.
METRICS: dict[str, MetricStyle] = {
    "escore": MetricStyle("escore", r"$E_{\mathrm{score}}$",
                          r"$\mathbf{E_{score}}$", "", "expected score"),
    "delta_elo": MetricStyle("delta_elo", r"$\Delta_{\mathrm{Elo}}$",
                             r"$\mathbf{\Delta_{Elo}}$", "", "Elo-equivalent advantage"),
}
LATEX = {"escore": r"$E_{\text{score}}$", "delta_elo": r"$\Delta_{\text{Elo}}$"}

FONTS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": FONTS,
    "font.size": 9,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 7,   # 2 pt smaller than the 9 pt base
    "axes.titleweight": "bold",
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    # Mathtext in the same font, so symbols match the surrounding text.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.default": "it",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed TrueType, so text stays editable in the PDF
}

# Shading for the practically balanced band, and secondary text.
BAND_COLOUR = "#d9d9d9"
MUTED_TEXT = "#555555"


def _available_font() -> str:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONTS if f in names), "DejaVu Sans")


def apply_style():
    """Set the project's matplotlib style (non-interactive backend) and
    return pyplot."""
    import logging
    import matplotlib
    matplotlib.use("Agg")
    # mathtext probes fonts it never uses here (cursive, cmex bold); keep quiet.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    params = dict(RC_PARAMS)
    font = _available_font()
    for key in ("mathtext.rm", "mathtext.it", "mathtext.bf", "mathtext.sf"):
        params[key] = params[key].replace("Arial", font)
    plt.rcParams.update(params)
    return plt


def bold_tick_labels(ax) -> None:
    """Bold tick labels, including any set after the axes were created."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
