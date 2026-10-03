"""Tecnicas de balanceo de clases (Seccion 5.1.2 de la guia).

* ``ninguno``      : linea base, el modelo ve la distribucion real (8 % positivos).
* ``smote``        : SMOTE interpola casos sinteticos entre vecinos de la clase
  minoritaria.
* ``adasyn``       : ADASYN genera mas casos sinteticos alrededor de los positivos
  dificiles (los rodeados de negativos), segun una densidad adaptativa.
* ``class_weight`` : no cambia los datos; pondera la perdida por la frecuencia
  inversa de cada clase. Cada modelo lo implementa a su manera (ver
  :func:`src.modelos.construir`).

Regla anti-fuga: el balanceo se aplica **solo** al entrenamiento de cada
particion, despues del preprocesamiento y nunca a la validacion ni a la prueba.
Las metricas se calculan siempre sobre la distribucion real.
"""
from __future__ import annotations

import numpy as np

from . import config

BALANCEOS = ("ninguno", "smote", "adasyn", "class_weight")
REMUESTREO = ("smote", "adasyn")


def aplicar(nombre: str, X: np.ndarray, y: np.ndarray, semilla: int = config.SEMILLA,
            proporcion: float = config.SMOTE_PROPORCION):
    """Devuelve el entrenamiento balanceado segun ``nombre``.

    Parameters
    ----------
    nombre : {'ninguno', 'smote', 'adasyn', 'class_weight'}
    X, y : ndarray
        Entrenamiento ya preprocesado (sin faltantes).
    semilla : int
    proporcion : float
        Razon minoritaria/mayoritaria objetivo para SMOTE y ADASYN.

    Returns
    -------
    (ndarray, ndarray)
        ``X`` e ``y`` (sin cambios para ``ninguno`` y ``class_weight``).
    """
    if nombre in ("ninguno", "class_weight"):
        return X, y
    if nombre not in REMUESTREO:
        raise ValueError(f"Balanceo desconocido: {nombre}")
    positivos = int(y.sum())
    k = max(1, min(5, positivos - 1))   # SMOTE necesita k < casos minoritarios
    if nombre == "smote":
        from imblearn.over_sampling import SMOTE
        muestreador = SMOTE(sampling_strategy=proporcion, k_neighbors=k, random_state=semilla)
    else:
        from imblearn.over_sampling import ADASYN
        muestreador = ADASYN(sampling_strategy=proporcion, n_neighbors=k, random_state=semilla)
    Xb, yb = muestreador.fit_resample(X, y)
    return np.ascontiguousarray(Xb, dtype=np.float32), np.asarray(yb, dtype=np.int8)


def pesos_balanceados(y: np.ndarray) -> np.ndarray:
    """Peso por observacion equivalente a ``class_weight='balanced'``:
    ``n / (2 * n_clase)``."""
    y = np.asarray(y)
    n, n1 = len(y), int(y.sum())
    n0 = n - n1
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(np.float64)
