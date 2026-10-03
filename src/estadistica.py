"""Pruebas estadisticas de comparacion de modelos (Seccion 6 de la guia).

Clasificacion : Friedman -> Nemenyi / diagrama CD -> DeLong (+ Holm / BH)
                -> tamano del efecto (delta de Cliff) -> IC bootstrap BCa.
Regresion     : MCS -> Giacomini-White / Clark-West -> Diebold-Mariano (HLN)
                + bootstrap estacionario -> tamano del efecto (d de Cohen).
                Control de data snooping: SPA de Hansen / Reality Check de White
                y StepM de Romano-Wolf frente a un modelo de referencia.

Todas las funciones estan implementadas sobre numpy/scipy (DeLong, BCa, DM, GW,
CW) o envuelven una libreria de referencia (``arch`` para MCS y bootstrap
estacionario, ``scikit-posthocs`` para Nemenyi, ``statsmodels`` para el ajuste
de valores p).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def ajustar_p(p_crudos, metodos=("holm", "fdr_bh")) -> pd.DataFrame:
    """Ajuste por comparaciones multiples.

    Returns
    -------
    DataFrame
        Columnas ``p_crudo``, ``p_holm`` y ``p_bh`` (Holm-Bonferroni controla el
        error familiar; Benjamini-Hochberg la tasa de falsos descubrimientos).
    """
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p_crudos, dtype=float)
    out = {"p_crudo": p}
    nombres = {"holm": "p_holm", "fdr_bh": "p_bh"}
    for m in metodos:
        out[nombres.get(m, f"p_{m}")] = multipletests(p, method=m)[1]
    return pd.DataFrame(out)


def varianza_newey_west(x, rezagos: int | None = None) -> float:
    """Varianza de largo plazo de ``x`` (Newey-West, nucleo de Bartlett).

    ``rezagos`` por defecto ``floor(4 (n/100)^(2/9))``.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    L = int(math.floor(4 * (n / 100) ** (2 / 9))) if rezagos is None else rezagos
    v = x @ x / n
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        v += 2 * w * (x[k:] @ x[:-k]) / n
    return float(v)


# ---------------------------------------------------------------------------
# Friedman, Nemenyi y diagrama de Diferencia Critica
# ---------------------------------------------------------------------------
def rangos(matriz: pd.DataFrame, mayor_es_mejor: bool = True) -> pd.DataFrame:
    """Rango de cada modelo (columnas) dentro de cada bloque (filas); 1 = mejor."""
    signo = -1 if mayor_es_mejor else 1
    r = np.apply_along_axis(lambda f: stats.rankdata(signo * f), 1, matriz.to_numpy())
    return pd.DataFrame(r, index=matriz.index, columns=matriz.columns)


def friedman(matriz: pd.DataFrame, mayor_es_mejor: bool = True) -> dict:
    """Prueba omnibus de Friedman sobre bloques (filas) x modelos (columnas).

    H0: todos los modelos tienen el mismo rango medio. Se reporta tambien la
    correccion F de Iman-Davenport, menos conservadora que el chi-cuadrado.
    """
    N, k = matriz.shape
    chi2, p = stats.friedmanchisquare(*[matriz[c].to_numpy() for c in matriz.columns])
    F = (N - 1) * chi2 / max(N * (k - 1) - chi2, 1e-12)
    gl1, gl2 = k - 1, (k - 1) * (N - 1)
    return {"bloques": N, "modelos": k, "chi2": float(chi2), "gl": k - 1, "p": float(p),
            "F_iman_davenport": float(F), "gl_F": (gl1, gl2),
            "p_F": float(stats.f.sf(F, gl1, gl2)),
            "rangos_medios": rangos(matriz, mayor_es_mejor).mean().sort_values()}


def diferencia_critica(k: int, N: int, alfa: float = 0.05) -> float:
    """Diferencia critica de Nemenyi: ``q_alfa sqrt(k (k+1) / (6 N))``, con
    ``q_alfa`` el cuantil del rango studentizado (gl infinitos) dividido por sqrt(2)."""
    q = stats.studentized_range.ppf(1 - alfa, k, 1e5) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * N)))


def nemenyi(matriz: pd.DataFrame) -> pd.DataFrame:
    """Valores p de Nemenyi para todos los pares (control familiar)."""
    import scikit_posthocs as sp
    return sp.posthoc_nemenyi_friedman(matriz.to_numpy()).set_axis(
        matriz.columns, axis=0).set_axis(matriz.columns, axis=1)


def diagrama_cd(rangos_medios: pd.Series, cd: float, ax=None, titulo: str | None = None):
    """Diagrama de Diferencia Critica (Demsar, 2006).

    Ubica cada modelo en su rango medio y une con una barra horizontal los grupos
    de modelos cuya diferencia de rango es menor que ``cd`` (estadisticamente
    equivalentes segun Nemenyi).
    """
    import matplotlib.pyplot as plt
    r = rangos_medios.sort_values()
    k = len(r)
    n_izq = int(math.ceil(k / 2))
    alto = 1.2 + 0.28 * n_izq
    if ax is None:
        _, ax = plt.subplots(figsize=(11, alto + 0.6))
    lo, hi = 1, max(k, int(math.ceil(r.max())))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-alto, 1.0)
    ax.axis("off")
    ax.hlines(0, lo, hi, color="black", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.08, color="black", lw=1)
        ax.text(t, 0.15, str(t), ha="center", va="bottom", fontsize=9)
    # barra de CD
    ax.hlines(0.75, lo, lo + cd, color="black", lw=2)
    ax.text(lo + cd / 2, 0.82, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    # etiquetas
    for i, (nombre, v) in enumerate(r.items()):
        izquierda = i < n_izq
        fila = i if izquierda else k - 1 - i
        y = -0.35 - 0.28 * fila
        x_txt = lo - 0.4 if izquierda else hi + 0.4
        ax.plot([v, v, x_txt], [0, y, y], color="0.35", lw=0.8)
        ax.text(x_txt + (-0.05 if izquierda else 0.05), y, f"{nombre} ({v:.2f})",
                ha="right" if izquierda else "left", va="center", fontsize=8.5)
    # grupos no significativos (cliques maximales de rangos consecutivos)
    vals = r.to_numpy()
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in grupos):
            grupos.append((i, j))
    for g, (a, b) in enumerate(grupos):
        y = -0.12 - 0.07 * g
        ax.hlines(y, vals[a] - 0.03, vals[b] + 0.03, color="#c0392b", lw=3)
    if titulo:
        ax.set_title(titulo, fontsize=11)
    return ax


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------
def _delong_componentes(y, S):
    """AUC y matriz de covarianza de DeLong para k puntajes (Sun y Xu, 2014)."""
    y = np.asarray(y).astype(bool)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    pos, neg = S[:, y], S[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = stats.rankdata(pos, axis=1)
    ty = stats.rankdata(neg, axis=1)
    tz = stats.rankdata(np.hstack([pos, neg]), axis=1)
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong(y, s1, s2) -> dict:
    """Prueba de DeLong para dos AUC correlacionadas (mismas observaciones).

    Returns
    -------
    dict
        auc_1, auc_2, diferencia (1 - 2), error estandar, z, p (bilateral) e IC
        95 % asintotico de la diferencia.
    """
    aucs, cov = _delong_componentes(y, np.vstack([s1, s2]))
    dif = aucs[0] - aucs[1]
    ee = math.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-300))
    z = dif / ee
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diferencia": float(dif),
            "ee": ee, "z": float(z), "p": float(2 * stats.norm.sf(abs(z))),
            "ic95": (float(dif - 1.96 * ee), float(dif + 1.96 * ee))}


def delong_ic(y, s, nivel: float = 0.95) -> tuple:
    """IC asintotico de DeLong para una AUC."""
    aucs, cov = _delong_componentes(y, s)
    ee = math.sqrt(cov[0, 0])
    z = stats.norm.ppf(0.5 + nivel / 2)
    return float(aucs[0] - z * ee), float(aucs[0] + z * ee)


# ---------------------------------------------------------------------------
# Bootstrap BCa
# ---------------------------------------------------------------------------
def bootstrap_bca(estadistico, *arrays, n_boot: int = 2000, alfa: float = 0.05,
                  estratos=None, n_grupos_jack: int = 200,
                  semilla: int = config.SEMILLA) -> dict:
    """Intervalo bootstrap corregido por sesgo y acelerado (BCa; Efron, 1987).

    Parameters
    ----------
    estadistico : callable
        ``f(*arrays) -> float`` (p. ej. el AUC o una diferencia de AUC).
    *arrays : array
        Arrays alineados por observacion.
    estratos : array, optional
        Si se da (p. ej. ``y``), el remuestreo se hace dentro de cada estrato, de
        modo que cada replica conserva el numero de positivos.
    n_grupos_jack : int
        La aceleracion ``a`` se estima con un jackknife por grupos (se elimina un
        grupo de n/G observaciones cada vez). El jackknife completo exigiria n
        recalculos del estadistico (~60 mil AUC); por grupos el estimador es
        consistente y cuesta G recalculos.

    Returns
    -------
    dict
        estimacion, ic_inf, ic_sup, z0 (sesgo), a (aceleracion), replicas.
    """
    rng = np.random.default_rng(semilla)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    theta = float(estadistico(*arrays))
    if estratos is not None:
        grupos = [np.where(np.asarray(estratos) == v)[0] for v in np.unique(estratos)]
    else:
        grupos = [np.arange(n)]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in grupos])
        boots[b] = estadistico(*[a[idx] for a in arrays])
    prop = np.mean(boots < theta) + 0.5 * np.mean(boots == theta)
    z0 = stats.norm.ppf(np.clip(prop, 1e-6, 1 - 1e-6))
    perm = rng.permutation(n)
    jack = []
    for g in np.array_split(perm, n_grupos_jack):
        mask = np.ones(n, bool)
        mask[g] = False
        jack.append(estadistico(*[a[mask] for a in arrays]))
    jack = np.asarray(jack)
    d = jack.mean() - jack
    a = float((d ** 3).sum() / (6 * max((d ** 2).sum(), 1e-300) ** 1.5))
    za = stats.norm.ppf([alfa / 2, 1 - alfa / 2])
    q = stats.norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
    lo, hi = np.quantile(boots, q)
    return {"estimacion": theta, "ic_inf": float(lo), "ic_sup": float(hi),
            "z0": float(z0), "a": a, "replicas": n_boot}


# ---------------------------------------------------------------------------
# Tamanos del efecto
# ---------------------------------------------------------------------------
def delta_cliff(a, b) -> tuple:
    """Delta de Cliff: P(A > B) - P(A < B). Umbrales de Romano et al. (2006):
    |d| < 0.147 insignificante, < 0.33 pequeno, < 0.474 mediano, resto grande."""
    a = np.asarray(a, dtype=float)
    b = np.sort(np.asarray(b, dtype=float))
    mayores = np.searchsorted(b, a, side="left").sum()
    menores = (len(b) - np.searchsorted(b, a, side="right")).sum()
    d = float((mayores - menores) / (len(a) * len(b)))
    m = abs(d)
    etiqueta = ("insignificante" if m < 0.147 else "pequeno" if m < 0.33
                else "mediano" if m < 0.474 else "grande")
    return d, etiqueta


def d_cohen_pareado(diferencias) -> tuple:
    """d de Cohen para diferencias pareadas (d_z = media / desviacion). Umbrales
    de Cohen: 0.2 pequeno, 0.5 mediano, 0.8 grande."""
    x = np.asarray(diferencias, dtype=float)
    if np.allclose(x, 0):
        return 0.0, "insignificante"
    d = float(x.mean() / max(x.std(ddof=1), 1e-300))
    m = abs(d)
    etiqueta = ("insignificante" if m < 0.2 else "pequeno" if m < 0.5
                else "mediano" if m < 0.8 else "grande")
    return d, etiqueta


# ---------------------------------------------------------------------------
# Comparacion de pronosticos (regresion)
# ---------------------------------------------------------------------------
def diebold_mariano(e1, e2, h: int = 1, potencia: int = 2) -> dict:
    """Diebold-Mariano (1995) con la correccion de Harvey, Leybourne y Newbold (1997).

    ``d_t = |e1_t|^p - |e2_t|^p``; H0: E[d] = 0. d > 0 significa que el modelo 2
    tiene menor perdida. El estadistico HLN se compara con una t de n-1 gl.
    """
    d = np.abs(np.asarray(e1, float)) ** potencia - np.abs(np.asarray(e2, float)) ** potencia
    n = len(d)
    if np.allclose(d, 0):
        return {"media_dif_perdida": 0.0, "dm": 0.0, "dm_hln": 0.0, "p": 1.0}
    dc = d - d.mean()
    gam = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    dm = d.mean() / math.sqrt(max(var, 1e-300))
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm
    return {"media_dif_perdida": float(d.mean()), "dm": float(dm), "dm_hln": float(hln),
            "p": float(2 * stats.t.sf(abs(hln), df=n - 1))}


def giacomini_white(perdida1, perdida2) -> dict:
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006), h=1.

    Instrumentos ``h_{t-1} = (1, d_{t-1})``. H0: ``E[d_t | h_{t-1}] = 0``;
    estadistico ``n Zbar' Omega^{-1} Zbar ~ chi2(2)`` con ``Z_t = h_{t-1} d_t``.
    Rechazar indica que un modelo es predictivamente superior al menos en algun
    estado (no solo en promedio).
    """
    d = np.asarray(perdida1, float) - np.asarray(perdida2, float)
    if np.allclose(d, 0):
        return {"estadistico": 0.0, "gl": 2, "p": 1.0, "media_dif_perdida": 0.0,
                "mejor": "empate (perdidas identicas)"}
    H = np.column_stack([np.ones(len(d) - 1), d[:-1]])
    Z = H * d[1:, None]
    n = len(Z)
    zbar = Z.mean(0)
    omega = (Z.T @ Z) / n
    est = float(n * zbar @ np.linalg.pinv(omega) @ zbar)
    return {"estadistico": est, "gl": 2, "p": float(stats.chi2.sf(est, 2)),
            "media_dif_perdida": float(d.mean()),
            "mejor": "modelo 2" if d.mean() > 0 else "modelo 1"}


def clark_west(y, pred_restringido, pred_general) -> dict:
    """Clark y West (2007) para modelos anidados.

    Ajusta el MSPE del modelo general por el ruido de estimar parametros que bajo
    H0 son nulos: ``f_t = e1^2 - (e2^2 - (y1 - y2)^2)``. Prueba unilateral:
    H1 = el modelo general predice mejor. Varianza de Newey-West.
    """
    y = np.asarray(y, float)
    p1, p2 = np.asarray(pred_restringido, float), np.asarray(pred_general, float)
    f = (y - p1) ** 2 - ((y - p2) ** 2 - (p1 - p2) ** 2)
    if np.allclose(f, 0):
        return {"media_ajustada": 0.0, "t": 0.0, "p_unilateral": 0.5}
    ee = math.sqrt(varianza_newey_west(f) / len(f))
    t = float(f.mean() / max(ee, 1e-300))
    return {"media_ajustada": float(f.mean()), "t": t, "p_unilateral": float(stats.norm.sf(t))}


def model_confidence_set(perdidas: pd.DataFrame, alfa: float = 0.10, reps: int = 1000,
                         agrupar: int = 1, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Model Confidence Set de Hansen, Lunde y Nason (2011) con bootstrap estacionario.

    Parameters
    ----------
    perdidas : DataFrame
        Observaciones (filas, en su orden natural) x modelos (columnas).
    agrupar : int
        Si > 1, promedia las perdidas en bloques consecutivos de ese tamano antes
        del bootstrap. Conserva la perdida media de cada modelo y reduce el coste
        (con ~250 mil observaciones y 28 modelos cada replica es costosa).

    Returns
    -------
    DataFrame
        Perdida media, rango, valor p del MCS y pertenencia al conjunto.
    """
    from arch.bootstrap import MCS
    L = perdidas
    if agrupar > 1:
        g = np.arange(len(L)) // agrupar
        L = L.groupby(g).mean()
    # Modelos con perdidas identicas (p. ej. dos optimizadores que llegan a la misma
    # configuracion) hacen singular la varianza de sus diferencias. Se evalua un
    # representante por grupo de identicos y el resultado se asigna a todo el grupo.
    representante = {}
    reps_cols = []
    for c in L.columns:
        igual = next((r for r in reps_cols if np.allclose(L[c].to_numpy(), L[r].to_numpy(),
                                                          rtol=0, atol=1e-12)), None)
        if igual is None:
            reps_cols.append(c)
            representante[c] = c
        else:
            representante[c] = igual
    if len(reps_cols) == 1:
        pv = pd.Series(1.0, index=reps_cols)
        incluidos = set(reps_cols)
    else:
        mcs = MCS(L[reps_cols], size=alfa, reps=reps, method="R", bootstrap="stationary",
                  seed=semilla)
        mcs.compute()
        pv = mcs.pvalues.iloc[:, 0]
        incluidos = set(mcs.included)
    out = pd.DataFrame({"perdida_media": perdidas.mean(),
                        "rango_perdida": perdidas.mean().rank(),
                        "p_mcs": [pv.get(representante[c], np.nan) for c in perdidas.columns],
                        "identico_a": [representante[c] if representante[c] != c else ""
                                       for c in perdidas.columns]},
                       index=perdidas.columns)
    out["en_conjunto"] = [representante[c] in incluidos for c in out.index]
    return out.sort_values("perdida_media")


def ic_bootstrap_estacionario(d, reps: int = 1000, nivel: float = 0.95,
                              semilla: int = config.SEMILLA) -> dict:
    """IC de la media de una serie dependiente con el bootstrap estacionario de
    Politis y Romano (1994); longitud media de bloque optima (Politis y White, 2004)."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length
    d = np.asarray(d, float)
    bloque = float(optimal_block_length(d)["stationary"].iloc[0])
    bloque = max(1.0, bloque)
    bs = StationaryBootstrap(bloque, d, seed=semilla)
    medias = np.array([x[0][0].mean() for x in bs.bootstrap(reps)])
    a = (1 - nivel) / 2
    return {"media": float(d.mean()), "ic_inf": float(np.quantile(medias, a)),
            "ic_sup": float(np.quantile(medias, 1 - a)), "bloque_medio": bloque}


def spa_reality_check(perdidas: pd.DataFrame, referencia: str, alfa: float = 0.05,
                      reps: int = 1000, agrupar: int = 1,
                      semilla: int = config.SEMILLA) -> dict:
    """Prueba de habilidad predictiva superior frente a un modelo de referencia.

    Responde si **algun** modelo del conjunto supera de verdad a la referencia o si
    la aparente superioridad es producto de haber probado muchos (data snooping).
    Se calculan, con bootstrap estacionario sobre las diferencias de perdida:

    * el SPA de Hansen (2005), estudentizado, con sus tres valores p (inferior,
      consistente y superior, segun como se recentra la distribucion nula);
    * el Reality Check de White (2000): la misma prueba sin estudentizar y sin
      recentrar (valor p superior);
    * el StepM de Romano y Wolf (2005), que identifica *cuales* modelos superan a
      la referencia controlando el error familiar.

    Parameters
    ----------
    perdidas : DataFrame
        Observaciones (filas, en su orden natural) x modelos (columnas); menor es mejor.
    referencia : str
        Columna del modelo de referencia (benchmark).
    alfa : float
        Nivel del StepM.
    reps : int
        Replicas bootstrap.
    agrupar : int
        Si > 1, promedia las perdidas en bloques consecutivos de ese tamano (como en
        :func:`model_confidence_set`).

    Returns
    -------
    dict
        Valores p del SPA y del Reality Check, y la lista de modelos superiores.
    """
    from arch.bootstrap import SPA, StepM
    L = perdidas
    if agrupar > 1:
        g = np.arange(len(L)) // agrupar
        L = L.groupby(g).mean()
    b = L[referencia]
    M = L.drop(columns=[referencia])
    # un modelo identico a la referencia no aporta y anula la varianza de su diferencia
    M = M.loc[:, [(M[c] - b).std() > 1e-12 for c in M.columns]]
    spa = SPA(b, M, reps=reps, bootstrap="stationary", studentize=True, seed=semilla)
    spa.compute()
    rc = SPA(b, M, reps=reps, bootstrap="stationary", studentize=False, seed=semilla)
    rc.compute()
    stepm = StepM(b, M, size=alfa, reps=reps, bootstrap="stationary", studentize=True,
                  seed=semilla)
    stepm.compute()
    pv = spa.pvalues
    return {"referencia": referencia, "modelos": int(M.shape[1]),
            "p_spa_inferior": float(pv["lower"]), "p_spa_consistente": float(pv["consistent"]),
            "p_spa_superior": float(pv["upper"]), "p_reality_check": float(rc.pvalues["upper"]),
            "superiores_stepm": [str(c) for c in stepm.superior_models]}
