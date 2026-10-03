"""Modelos finales: reentrenamiento sobre todo desarrollo y evaluacion en prueba.

Para cada uno de los 14 modelos se elige su mejor combinacion (balanceo,
optimizador) segun el AUC-ROC medio del bucle externo, y sus hiperparametros son
los del pliegue externo con mayor AUC interno. Se reentrena sobre el desarrollo
completo (80 %) y se evalua **una sola vez** en la prueba (20 %), la misma de la
linea base logistica.

Cada modelo final se guarda en ``resultados/finales/`` apenas termina; volver a
ejecutar la celda salta los ya hechos y un fallo en uno no afecta a los demas.
"""
from __future__ import annotations

import json
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from . import config
from . import experimento as ex
from . import metricas as met
from . import modelos as mod
from . import preprocesamiento as prep


def seleccionar_finalistas(maestra: pd.DataFrame, tarea: str) -> pd.DataFrame:
    """Mejor combinacion (balanceo, optimizador) de cada modelo por AUC externo medio.

    Solo se consideran combinaciones con todos sus pliegues externos completos.
    """
    t = maestra[(maestra["tarea"] == tarea) & (maestra["pliegues"] == config.N_EXTERNOS)]
    idx = t.groupby("modelo")["m_auc_roc_media"].idxmax()
    return t.loc[idx].sort_values("m_auc_roc_media", ascending=False).reset_index(drop=True)


def params_finales(tarea, modelo, balanceo, optimizador) -> tuple:
    """Hiperparametros del pliegue externo con mayor AUC interno, y umbral final
    (mediana de los umbrales elegidos en cada pliegue externo).

    Returns
    -------
    (dict, float | None, pd.DataFrame)
        Parametros, umbral y la tabla de parametros por pliegue (para discutir su
        estabilidad).
    """
    regs = [r for r in ex.leer_corridas(tarea)
            if (r["modelo"], r["balanceo"], r["optimizador"]) == (modelo, balanceo, optimizador)]
    if not regs:
        raise ValueError(f"No hay corridas de {tarea}/{modelo}/{balanceo}/{optimizador}")
    mejor = max(regs, key=lambda r: r["auc_interno"])
    umbrales = [r["umbral"] for r in regs if r.get("umbral") is not None]
    tabla = pd.DataFrame([{"pliegue": r["pliegue"], "auc_interno": r["auc_interno"],
                           "auc_externo": r["metricas"]["auc_roc"], **r["mejor_params"]}
                          for r in sorted(regs, key=lambda r: r["pliegue"])])
    return mejor["mejor_params"], (float(np.median(umbrales)) if umbrales else None), tabla


def _ruta(tarea, modelo, sufijo=""):
    return config.DIR_FINALES / f"{tarea}__{modelo}{sufijo}"


def entrenar_final(particion, tarea, modelo, balanceo, params, semilla=config.SEMILLA,
                   dispositivo=None):
    """Ajusta sobre ``particion.X_tr`` y puntua ``particion.X_va``.

    Returns
    -------
    (estimador, ndarray, dict)
    """
    est, info = mod.entrenar(tarea, modelo, params, balanceo, particion.X_tr, particion.y_tr,
                             semilla=semilla, dispositivo=dispositivo)
    t0 = time.perf_counter()
    s = mod.puntuar(est, particion.X_va, tarea)
    info["t_inferencia"] = time.perf_counter() - t0
    return est, s, info


def evaluar_finalistas(conj, tarea: str, finalistas: pd.DataFrame, guardar_modelo: bool = True,
                       verboso: bool = True) -> pd.DataFrame:
    """Entrena y evalua en prueba cada finalista (reanudable, fallos aislados)."""
    config.DIR_FINALES.mkdir(parents=True, exist_ok=True)
    part = None
    filas = []
    for _, f in finalistas.iterrows():
        m, b, o = f["modelo"], f["balanceo"], f["optimizador"]
        ruta_json = _ruta(tarea, m, ".json")
        if ruta_json.is_file():
            filas.append(json.loads(ruta_json.read_text(encoding="utf-8")))
            if verboso:
                print(f"  {m:14s} ya evaluado")
            continue
        try:
            if part is None:
                part = prep.preparar_final(conj)
            params, umbral, _ = params_finales(tarea, m, b, o)
            t0 = time.perf_counter()
            est, s, info = entrenar_final(part, tarea, m, b, params)
            y = np.asarray(part.y_va).astype(int)
            if tarea == "clasificacion":
                metr = met.metricas_clasificacion(y, s, umbral)
                if mod.es_probabilidad(tarea, m):
                    metr.update(met.resumen_calibracion(y, s))
            else:
                metr = met.metricas_regresion(y, s)
            np.savez_compressed(_ruta(tarea, m, ".npz"), fila=part.fila_va, y=y.astype(np.int8),
                                s=s.astype(np.float64))
            if guardar_modelo:
                joblib.dump(est, _ruta(tarea, m, ".joblib"), compress=3)
            reg = {"tarea": tarea, "modelo": m, "balanceo": b, "optimizador": o,
                   "params": params, "umbral": umbral, "metricas": metr,
                   "t_ajuste": info["t_ajuste"], "t_balanceo": info["t_balanceo"],
                   "t_inferencia": info["t_inferencia"], "filas_ajuste": info["filas_ajuste"],
                   "t_total": time.perf_counter() - t0, "semilla": config.SEMILLA,
                   "auc_cv_media": float(f["m_auc_roc_media"]),
                   "auc_cv_sd": float(f["m_auc_roc_sd"])}
            ruta_json.write_text(json.dumps(ex._json_seguro(reg), indent=1), encoding="utf-8")
            filas.append(reg)
            if verboso:
                print(f"  {m:14s} AUC prueba {metr['auc_roc']:.4f} "
                      f"({reg['t_total'] / 60:.1f} min)", flush=True)
            del est
        except Exception as e:
            (config.DIR_FINALES / f"ERROR_{tarea}__{m}.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
            print(f"  {m:14s} ERROR {type(e).__name__}: {e}")
    return tabla_finales(filas)


def tabla_finales(registros) -> pd.DataFrame:
    """Tabla con una fila por modelo final: combinacion elegida, AUC de CV, metricas en prueba y tiempos."""
    filas = []
    for r in registros:
        fila = {"modelo": r["modelo"], "balanceo": r["balanceo"],
                "optimizador": r["optimizador"], "auc_cv": r["auc_cv_media"],
                "auc_cv_sd": r["auc_cv_sd"]}
        fila.update(r["metricas"])
        fila.update({"t_ajuste_s": r["t_ajuste"], "t_inferencia_s": r["t_inferencia"]})
        filas.append(fila)
    return pd.DataFrame(filas)


def cargar_finales(tarea: str) -> dict:
    """Predicciones de prueba y registro de cada modelo final ya evaluado."""
    out = {}
    for f in sorted(config.DIR_FINALES.glob(f"{tarea}__*.json")):
        if f.stem.count("__") != 1:          # solo los modelos finales
            continue
        reg = json.loads(f.read_text(encoding="utf-8"))
        z = np.load(f.with_suffix(".npz"))
        out[reg["modelo"]] = {"registro": reg, "fila": z["fila"], "y": z["y"].astype(int),
                              "s": z["s"]}
    return out


def cargar_modelo(tarea: str, modelo: str):
    """Estimador final guardado con joblib."""
    return joblib.load(_ruta(tarea, modelo, ".joblib"))


# ---------------------------------------------------------------------------
# Sensibilidad a la semilla (Seccion 5.4)
# ---------------------------------------------------------------------------
def sensibilidad_semillas(conj, tarea, modelo, semillas=(0, 1, 2, 3, 4),
                          verboso=True) -> pd.DataFrame:
    """Reentrena el modelo final con varias semillas y mide su AUC en prueba.

    Solo tiene sentido para algoritmos con aleatoriedad interna (Random Forest,
    XGBoost, PCA aleatorizado de KNN, Nystroem, SMOTE/ADASYN). Reanudable por semilla.
    """
    reg = json.loads(_ruta(tarea, modelo, ".json").read_text(encoding="utf-8"))
    part = None
    filas = []
    for sem in semillas:
        ruta = config.DIR_FINALES / "semillas" / f"{tarea}__{modelo}__s{sem}.json"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        if ruta.is_file():
            filas.append(json.loads(ruta.read_text(encoding="utf-8")))
            continue
        try:
            if part is None:
                part = prep.preparar_final(conj)
            _, s, info = entrenar_final(part, tarea, modelo, reg["balanceo"], reg["params"], semilla=sem)
            y = np.asarray(part.y_va).astype(int)
            fila = {"modelo": modelo, "semilla": sem,
                    "auc_roc": float(met.roc_auc_score(y, s)), "t_ajuste": info["t_ajuste"]}
            ruta.write_text(json.dumps(fila), encoding="utf-8")
            filas.append(fila)
            if verboso:
                print(f"  {modelo} semilla {sem}: AUC {fila['auc_roc']:.4f}", flush=True)
        except Exception as e:
            print(f"  {modelo} semilla {sem}: ERROR {type(e).__name__}: {e}")
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Ajuste entrenamiento / validacion / prueba (Seccion 5.2.1)
# ---------------------------------------------------------------------------
def ajuste_tres_conjuntos(conj, tarea, modelo, balanceo, params) -> pd.DataFrame:
    """Ajusta con la particion de la linea base (64 % entrenamiento) y predice
    entrenamiento, validacion (16 %) y prueba (20 %), para la grafica de ajuste.
    Se guarda en disco y se reutiliza."""
    ruta = _ruta(tarea, modelo, "__tres_conjuntos.parquet")
    if ruta.is_file():
        return pd.read_parquet(ruta)
    p, X_te, y_te, fila_te = prep.preparar_base(conj)
    est, info = mod.entrenar(tarea, modelo, params, balanceo, p.X_tr, p.y_tr)
    partes = []
    for nombre, X, y, fila in (("entrenamiento", p.X_tr, p.y_tr, p.fila_tr),
                               ("validacion", p.X_va, p.y_va, p.fila_va),
                               ("prueba", X_te, y_te, fila_te)):
        partes.append(pd.DataFrame({"conjunto": nombre, "fila": fila, "y": y.astype(int),
                                    "y_hat": mod.puntuar(est, X, tarea)}))
    df = pd.concat(partes, ignore_index=True)
    df.to_parquet(ruta, index=False)
    return df
