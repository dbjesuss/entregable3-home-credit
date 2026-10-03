"""Estilo grafico comun y guardado de figuras."""
from __future__ import annotations

import matplotlib.pyplot as plt

from . import config

PALETA = ["#2a78d4", "#e0822f", "#2f9e6b", "#c94f7c", "#7a5cc7", "#8a6d3b", "#1a9aa5"]
COLOR_OPT = {"rejilla": "#8a8f98", "aleatoria": "#e0822f",
             "bayesiana": "#2a78d4", "genetica": "#2f9e6b"}
COLOR_BAL = {"ninguno": "#8a8f98", "smote": "#2a78d4",
             "adasyn": "#c94f7c", "class_weight": "#2f9e6b"}
COLOR_CONJ = {"entrenamiento": "#2a78d4", "validacion": "#e0822f", "prueba": "#2f9e6b"}


def estilo() -> None:
    """Aplica el estilo de todas las figuras del libro."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=PALETA),
    })


def guardar(fig, nombre: str) -> None:
    """Guarda la figura en ``figuras/<nombre>.png``."""
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.DIR_FIGURAS / f"{nombre}.png", bbox_inches="tight")
