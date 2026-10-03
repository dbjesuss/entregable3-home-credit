"""Herramientas para la Seccion 4 (optimizacion computacional).

* :func:`medir` - tiempo de reloj promediado sobre repeticiones.
* :func:`medir_memoria` - pico de memoria asignada (tracemalloc, incluye numpy).
* :func:`exponente_empirico` - pendiente log-log de tiempo frente a n.
* :func:`perfilar` - cProfile de una funcion, resumido en un DataFrame.
* :data:`COMPLEJIDAD` - complejidad teorica de cada modelo (estandar y optimizado).
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np
import pandas as pd


def medir(fn, repeticiones: int = 3, calentamiento: bool = False) -> dict:
    """Ejecuta ``fn()`` varias veces y devuelve media y desviacion del tiempo (s)."""
    if calentamiento:
        fn()
    t = []
    salida = None
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        salida = fn()
        t.append(time.perf_counter() - t0)
    return {"media": float(np.mean(t)), "sd": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
            "repeticiones": repeticiones, "salida": salida}


def medir_memoria(fn) -> tuple:
    """Pico de memoria (MB) asignada por Python y numpy durante ``fn()``."""
    tracemalloc.start()
    try:
        salida = fn()
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pico / 2**20, salida


def medir_memoria_rss(fn, intervalo: float = 0.05) -> tuple:
    """Pico de memoria residente del proceso (MB por encima del nivel previo)
    durante ``fn()``. A diferencia de tracemalloc, incluye la memoria que reservan
    las bibliotecas en C/C++ (FAISS, XGBoost, BLAS)."""
    from memory_profiler import memory_usage
    base = max(memory_usage(-1, interval=0.01, timeout=0.05))
    pico, salida = memory_usage((fn, (), {}), max_usage=True, retval=True,
                                interval=intervalo, include_children=True)
    pico = float(pico if np.isscalar(pico) else max(pico))
    return max(pico - base, 0.0), salida


def experimento_en_disco(ruta, fn, verboso: bool = True):
    """Ejecuta ``fn() -> DataFrame`` y guarda el resultado en ``ruta`` (.csv).

    Si el archivo ya existe lo lee en lugar de recalcular; si ``fn`` falla,
    imprime la traza y devuelve None sin interrumpir el notebook.
    """
    import traceback
    from pathlib import Path
    ruta = Path(ruta)
    if ruta.is_file():
        return pd.read_csv(ruta)
    try:
        t0 = time.perf_counter()
        df = fn()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)
        if verboso:
            print(f"{ruta.stem}: {(time.perf_counter() - t0) / 60:.1f} min")
        return df
    except Exception:
        traceback.print_exc()
        return None


def exponente_empirico(n, tiempos) -> dict:
    """Ajuste ``log t = b log n + c``: ``b`` estima el exponente de la complejidad.

    Returns
    -------
    dict
        exponente, intercepto y R^2 del ajuste log-log.
    """
    x, y = np.log(np.asarray(n, float)), np.log(np.asarray(tiempos, float))
    b, c = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (b * x + c)) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"exponente": float(b), "intercepto": float(c), "r2": float(r2)}


def perfilar(fn, top: int = 15) -> tuple:
    """Perfila ``fn()`` con cProfile.

    Returns
    -------
    (DataFrame, object)
        Las ``top`` funciones con mayor tiempo acumulado y la salida de ``fn``.
    """
    pr = cProfile.Profile()
    pr.enable()
    salida = fn()
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    filas = []
    for (archivo, linea, nombre), (cc, nc, tt, ct, _) in st.stats.items():
        filas.append({"funcion": f"{nombre} ({archivo.split('/')[-1].split(chr(92))[-1]}:{linea})",
                      "llamadas": nc, "t_propio_s": tt, "t_acumulado_s": ct})
    df = pd.DataFrame(filas).sort_values("t_acumulado_s", ascending=False).head(top)
    return df.reset_index(drop=True), salida


COMPLEJIDAD = pd.DataFrame([
    # modelo, version, entrenamiento, inferencia (por consulta o por lote)
    ("KNN", "estandar (fuerza bruta, p dims)", "O(1) (memoriza)", "O(n p) por consulta"),
    ("KNN", "KD-Tree / Ball-Tree", "O(n log n)", "O(log n) en dims bajas; degenera a O(n p) en dims altas"),
    ("KNN", "PCA(d) + FAISS plano", "O(n p d) (PCA)", "O(n d) por consulta, vectorizado/multihilo"),
    ("KNN", "PCA(d) + fuerza bruta en GPU (PyTorch)", "O(n p d) (PCA)", "O(n d) por consulta, bloques de 1024 consultas en paralelo"),
    ("KNN", "PCA(d) + FAISS IVF", "O(n p d + n d nlist) (k-medias)", "O((n/nlist) d nprobe) por consulta (aproximado)"),
    ("Naive Bayes", "fit", "O(n p)", "O(p) por consulta"),
    ("Naive Bayes", "partial_fit (lotes)", "O(n p) con memoria O(b p)", "O(p) por consulta"),
    ("Logistica / Ridge / Lasso", "lbfgs / Cholesky", "O(n p^2 + p^3) (Ridge) | O(i n p) (lbfgs)", "O(p)"),
    ("Logistica / Ridge / Lasso", "SAGA", "O(n p) por epoca", "O(p)"),
    ("Lasso", "coordenadas + Gram", "O(n p^2) una vez + O(i p^2)", "O(p)"),
    ("Arbol", "CART", "O(p n log n) por nivel ~ O(p n log^2 n)", "O(profundidad)"),
    ("Random Forest", "scikit-learn: CART x T arboles", "O(T m n log^2 n), m = max_features", "O(T profundidad)"),
    ("Random Forest", "XGBoost XGBRF (hist, GPU)", "O(T m n) + O(p n log n) una vez (cuantiles), B bins", "O(T profundidad)"),
    ("XGBoost", "exact", "O(T K p n log n) (ordena cada columna)", "O(T profundidad)"),
    ("XGBoost", "hist", "O(T K p n) + O(p n log n) una vez (cuantiles), B bins", "O(T profundidad)"),
    ("SVM / SVR", "SVC con nucleo RBF", "O(n^2 p) a O(n^3)", "O(n_sv p) por consulta"),
    ("SVM / SVR", "LinearSVC / SGD", "O(i n p)", "O(p)"),
    ("SVM / SVR", "Nystroem(m) + lineal", "O(n m p + m^3) + O(i n m)", "O(m p) por consulta"),
], columns=["Modelo", "Version", "Entrenamiento", "Inferencia"])
"""Complejidad asintotica: n = observaciones, p = columnas, d = dimension del PCA,
T = arboles, K = hojas por arbol, i = iteraciones, m = componentes de Nystroem."""
