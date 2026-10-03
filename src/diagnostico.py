"""Analisis de residuos de los modelos de regresion (Seccion 5.2.2).

Todas las pruebas se aplican a los residuos de prueba ``e = y - y_hat`` ordenados
por la fila original de ``application_train`` (orden de ``SK_ID_CURR``). Home
Credit no trae fecha de solicitud, de modo que ese es el unico orden natural
disponible; las pruebas de independencia contestan si queda estructura en ese
orden.

Advertencia de interpretacion: con un objetivo 0/1 el residuo solo puede tomar
dos valores por observacion (``-y_hat`` o ``1 - y_hat``) y su varianza condicional
es ``p(1-p)``. La heterocedasticidad y la no normalidad son consecuencia del tipo
de objetivo, no de un defecto del ajuste; el capitulo 07 lo discute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def prueba_white(residuos, predichos) -> dict:
    """Prueba de White de homocedasticidad.

    Se usa la forma especial de White (1980) con los regresores ``(1, y_hat)``: la
    regresion auxiliar de ``e^2`` sobre ``(1, y_hat, y_hat^2)``. La forma completa,
    con las ~600 columnas, sus cuadrados y productos cruzados, tendria ~180 mil
    regresores y no es estimable. H0: varianza constante.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white
    exog = sm.add_constant(np.asarray(predichos, float))
    lm, p_lm, f, p_f = het_white(np.asarray(residuos, float), exog)
    return {"lm": float(lm), "p": float(p_lm), "f": float(f), "p_f": float(p_f)}


def prueba_bds(residuos, tam_bloque: int = 5000, dimension: int = 2,
               max_bloques: int | None = None) -> dict:
    """Prueba BDS (Brock, Dechert y Scheinkman) de independencia por bloques.

    La BDS calcula integrales de correlacion sobre todos los pares de puntos:
    memoria y tiempo O(n^2). Con ~60 mil residuos serian ~3.6e9 pares, de modo que
    se aplica sobre bloques consecutivos de ``tam_bloque`` observaciones (se usan
    todos los bloques) y se combinan sus valores p con Holm (minimo ajustado) y con
    el metodo de Fisher.
    """
    from statsmodels.tsa.stattools import bds
    e = np.asarray(residuos, float)
    bloques = [e[i:i + tam_bloque] for i in range(0, len(e) - tam_bloque + 1, tam_bloque)]
    if not bloques:
        bloques = [e]
    if max_bloques:
        bloques = bloques[:max_bloques]
    ps, zs = [], []
    for b in bloques:
        z, p = bds(b, max_dim=dimension)
        zs.append(float(np.atleast_1d(z)[-1]))
        ps.append(float(np.atleast_1d(p)[-1]))
    ps = np.array(ps)
    from statsmodels.stats.multitest import multipletests
    fisher = stats.combine_pvalues(np.clip(ps, 1e-300, 1), method="fisher")[1]
    return {"bloques": len(bloques), "z_mediano": float(np.median(zs)),
            "p_holm_min": float(multipletests(ps, method="holm")[1].min()),
            "p_fisher": float(fisher), "prop_rechazo_5pct": float(np.mean(ps < 0.05)),
            "p_bloques": ps}


def prueba_ljung_box(residuos, rezagos=(10, 20)) -> pd.DataFrame:
    """Ljung-Box sobre los rezagos indicados. H0: no hay autocorrelacion serial."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    return acorr_ljungbox(np.asarray(residuos, float), lags=list(rezagos), return_df=True)


def pruebas_normalidad(residuos) -> dict:
    """Jarque-Bera y D'Agostino-Pearson (Shapiro-Wilk no es valida con n > 5000)."""
    e = np.asarray(residuos, float)
    jb = stats.jarque_bera(e)
    k2 = stats.normaltest(e)
    return {"jarque_bera": float(jb.statistic), "p_jb": float(jb.pvalue),
            "dagostino": float(k2.statistic), "p_dagostino": float(k2.pvalue),
            "asimetria": float(stats.skew(e)), "curtosis_exceso": float(stats.kurtosis(e))}


def diagnostico_completo(y, y_hat) -> dict:
    """Todas las pruebas de la Seccion 5.2.2 para un modelo."""
    y = np.asarray(y, float)
    y_hat = np.asarray(y_hat, float)
    e = y - y_hat
    lb = prueba_ljung_box(e)
    w = prueba_white(e, y_hat)
    b = prueba_bds(e)
    nrm = pruebas_normalidad(e)
    return {"p_white": w["p"], "p_bds_fisher": b["p_fisher"], "p_bds_holm": b["p_holm_min"],
            "p_ljung_box_10": float(lb["lb_pvalue"].iloc[0]),
            "p_ljung_box_20": float(lb["lb_pvalue"].iloc[-1]),
            "p_jarque_bera": nrm["p_jb"], "asimetria": nrm["asimetria"],
            "curtosis_exceso": nrm["curtosis_exceso"]}
