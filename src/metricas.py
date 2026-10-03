"""Metricas de clasificacion, regresion y calibracion (Secciones 5.1 y 5.2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             confusion_matrix, brier_score_loss)

from . import config


def umbral_coste(y, s, coste_fn: float = config.COSTE_FN) -> float:
    """Umbral que minimiza ``coste_fn * FN + FP`` sobre ``(y, s)``.

    Es el criterio de decision de la linea base: un incumplimiento no detectado
    cuesta ``coste_fn`` veces lo que rechazar a un buen cliente. Se calcula de
    forma exacta recorriendo los puntajes ordenados (O(n log n)).
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    orden = np.argsort(-s, kind="mergesort")
    ys, ss = y[orden], s[orden]
    # si el umbral se pone justo en ss[i], se predicen positivos los i+1 primeros
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    coste = coste_fn * fn + fp
    # solo cortes donde cambia el puntaje
    valido = np.r_[ss[1:] != ss[:-1], True]
    i = int(np.argmin(np.where(valido, coste, np.inf)))
    # comparar con predecir todo negativo
    if coste_fn * ys.sum() <= coste[i]:
        return float(ss[0] + 1e-12)
    return float(ss[i])


def metricas_clasificacion(y, s, umbral: float) -> dict:
    """Metricas de la Seccion 5.1.1 a partir de puntajes y un umbral.

    Returns
    -------
    dict
        auc_roc, auc_pr, exactitud, precision, recall, especificidad, f1, umbral,
        tn, fp, fn, tp.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    yp = (s >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y, s)),
        "auc_pr": float(average_precision_score(y, s)),
        "exactitud": float(accuracy_score(y, yp)),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "especificidad": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, yp, zero_division=0)),
        "umbral": float(umbral),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def metricas_regresion(y, yhat) -> dict:
    """Metricas de la Seccion 5.2.1 (RMSE, MAE) mas R^2 y el AUC-ROC comun.

    Con un objetivo 0/1, el RMSE es la raiz del Brier score del puntaje.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
        "auc_roc": float(roc_auc_score(y.astype(int), yhat)),
        "auc_pr": float(average_precision_score(y.astype(int), yhat)),
    }


# ---------------------------------------------------------------------------
# Calibracion (Seccion 5.1.3)
# ---------------------------------------------------------------------------
def ece(y, p, n_bins: int = 15) -> float:
    """Error de Calibracion Esperado con bins de igual ancho en [0, 1].

    ``ECE = sum_b (n_b / n) * |frecuencia_observada_b - confianza_media_b|``.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    bordes = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, bordes[1:-1]), 0, n_bins - 1)
    total = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def ece_cuantiles(y, p, n_bins: int = 15) -> float:
    """ECE con bins de igual frecuencia: mas estable cuando las probabilidades se
    concentran en valores bajos (8 % de positivos)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    orden = np.argsort(p)
    total = 0.0
    for idx in np.array_split(orden, n_bins):
        if len(idx):
            total += len(idx) / len(p) * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def curva_confiabilidad(y, p, n_bins: int = 15, estrategia: str = "quantile"):
    """Puntos del diagrama de confiabilidad: (confianza media, frecuencia, n) por bin."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    if estrategia == "quantile":
        grupos = np.array_split(np.argsort(p), n_bins)
    else:
        bordes = np.linspace(0, 1, n_bins + 1)
        b = np.clip(np.digitize(p, bordes[1:-1]), 0, n_bins - 1)
        grupos = [np.where(b == k)[0] for k in range(n_bins)]
    pts = [(p[g].mean(), y[g].mean(), len(g)) for g in grupos if len(g)]
    return np.array(pts)


def resumen_calibracion(y, p) -> dict:
    """Brier, ECE (ancho igual y cuantiles) de probabilidades en [0, 1]."""
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    return {"brier": float(brier_score_loss(np.asarray(y).astype(int), p)),
            "ece": ece(y, p), "ece_cuantiles": ece_cuantiles(y, p)}


def recalibrar(s_ajuste, y_ajuste, s_aplicar, metodo: str = "platt"):
    """Ajusta un recalibrador sobre ``(s_ajuste, y_ajuste)`` y lo aplica a ``s_aplicar``.

    Parameters
    ----------
    s_ajuste, y_ajuste : array
        Puntajes y desenlaces **fuera de muestra** (aqui, las predicciones de la
        CV anidada sobre desarrollo). Nunca se usa la prueba para ajustar.
    s_aplicar : array
        Puntajes a recalibrar (prueba).
    metodo : {'platt', 'isotonica'}
        Platt = regresion logistica sobre el puntaje (sigmoide, 2 parametros);
        isotonica = funcion monotona no parametrica.
    """
    s_ajuste = np.asarray(s_ajuste, dtype=float)
    s_aplicar = np.asarray(s_aplicar, dtype=float)
    if metodo == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(s_ajuste.reshape(-1, 1), y_ajuste)
        return lr.predict_proba(s_aplicar.reshape(-1, 1))[:, 1]
    if metodo == "isotonica":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s_ajuste, y_ajuste)
        return iso.predict(s_aplicar)
    raise ValueError(metodo)
