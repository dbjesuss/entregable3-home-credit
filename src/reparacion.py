"""Revision y reparacion de las evaluaciones que fallaron durante el estudio.

Dentro de una unidad, una evaluacion que falla puntua 0 y el optimizador sigue
(ver :class:`src.optimizadores._Evaluador`). Al terminar el estudio este modulo:

1. **Lista** todas las evaluaciones fallidas (``resultados/evaluaciones_fallidas.csv``)
   y las clasifica por causa:

   * ``memoria_gpu``: la configuracion no cabe en la memoria de la GPU (6 GB).
     Es una propiedad de la configuracion, no un accidente.
   * ``sistema``: fallos pasajeros del sistema operativo (``WinError 1450``,
     procesos de ``loky`` terminados). No dependen de la configuracion.
   * ``otro``: cualquier otra excepcion (se revisa a mano).

2. **Repara** sin fuga de informacion:

   * ``sistema`` -> la unidad completa se vuelve a ejecutar desde cero con la
     maquina libre (su resultado se descarta y se recalcula igual que las demas).
   * ``memoria_gpu`` en **rejilla** o **aleatoria** -> las configuraciones a
     evaluar se fijan antes de conocer ningun resultado, de modo que evaluarlas
     despues es equivalente a haberlas evaluado en su momento. Se evaluan en la
     CPU (mismo algoritmo de XGBoost, otro dispositivo), con los mismos pliegues
     internos. La ganadora se elige **solo por AUC interno**; si cambia, se
     reentrena en la CPU y se recalcula el AUC externo, suba o baje. Toda unidad
     reparada queda marcada (``reparacion`` en su JSON).
   * ``memoria_gpu`` en **bayesiana** o **genetica** -> no se repara: cada punto
     depende de los anteriores, asi que reevaluar cambiaria la trayectoria. Se
     documenta como configuracion no factible en el hardware.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config
from . import experimento as ex
from . import metricas as met
from . import modelos as mod

REPARABLES = ("rejilla", "aleatoria")


def causa(error: str) -> str:
    """Clasifica un mensaje de error en ``memoria_gpu``, ``sistema`` u ``otro``."""
    e = (error or "").lower()
    if "cudaerrormemoryallocation" in e or "out of memory" in e or "bad allocation" in e:
        return "memoria_gpu"
    if ("winerror" in e or "terminatedworker" in e or "brokenprocesspool" in e
            or "recursos insuficientes" in e):
        return "sistema"
    return "otro"


def listar_fallidas(guardar: bool = True) -> pd.DataFrame:
    """Una fila por evaluacion fallida en todo el estudio."""
    filas = []
    for ruta in sorted(config.DIR_CORRIDAS.glob("*.json")):
        r = json.loads(ruta.read_text(encoding="utf-8"))
        for h in r.get("historial", []):
            if h.get("error"):
                filas.append({"clave": r["clave"], "tarea": r["tarea"], "modelo": r["modelo"],
                              "balanceo": r["balanceo"], "optimizador": r["optimizador"],
                              "pliegue": r["pliegue"], "evaluacion": h["t"],
                              "causa": causa(h["error"]), "params": json.dumps(h["params"]),
                              "error": str(h["error"])[:200],
                              "reparada": bool(h.get("reparada_cpu"))})
    df = pd.DataFrame(filas, columns=["clave", "tarea", "modelo", "balanceo", "optimizador",
                                      "pliegue", "evaluacion", "causa", "params", "error",
                                      "reparada"])
    if guardar:
        df.to_csv(config.DIR_RESULTADOS / "evaluaciones_fallidas.csv", index=False)
    return df


def _unidad(r) -> ex.Unidad:
    return ex.Unidad(r["tarea"], r["modelo"], r["balanceo"], r["optimizador"], int(r["pliegue"]))


def repetir_unidad(clave: str) -> None:
    """Borra el resultado de una unidad y la vuelve a ejecutar completa."""
    r = json.loads((config.DIR_CORRIDAS / f"{clave}.json").read_text(encoding="utf-8"))
    u = _unidad(r)
    anterior = {"auc_externo": r["metricas"]["auc_roc"], "fecha": r["fecha"]}
    u.archivo.unlink()
    (config.DIR_PREDICCIONES / f"{clave}.npz").unlink(missing_ok=True)
    ex.ejecutar(u.tarea, modelos=[u.modelo], balanceos=[u.balanceo],
                optimizadores=[u.optimizador], pliegues=[u.pliegue])
    nuevo = json.loads(u.archivo.read_text(encoding="utf-8"))
    nuevo["reparacion"] = {"tipo": "unidad_repetida", "motivo": "fallo del sistema",
                           "anterior": anterior, "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
    ex._escribir_atomico(u.archivo, json.dumps(ex._json_seguro(nuevo), indent=1))


def reparar_memoria(clave: str, cache: ex.CacheParticiones) -> dict:
    """Evalua en la CPU las configuraciones que no cupieron en la GPU."""
    ruta = config.DIR_CORRIDAS / f"{clave}.json"
    r = json.loads(ruta.read_text(encoding="utf-8"))
    u = _unidad(r)
    if u.optimizador not in REPARABLES:
        return {"clave": clave, "resultado": "no reparable (optimizador secuencial)"}
    pendientes = [h for h in r["historial"]
                  if h.get("error") and causa(h["error"]) == "memoria_gpu"
                  and not h.get("reparada_cpu")]
    if not pendientes:
        return {"clave": clave, "resultado": "nada que reparar"}

    externa, internas = cache.pliegue(u.pliegue)
    entrenos = [cache.entrenamiento(p, u.tarea, u.modelo, u.balanceo, r["semilla"])
                for p in internas]
    nuevos = {}
    for h in pendientes:
        t0 = time.perf_counter()
        try:
            res = [ex._evaluar_interna(u.tarea, u.modelo, h["params"], u.balanceo, Xt, yt,
                                       p.X_va, p.y_va, r["semilla"], "cpu", Xb, es,
                                       config.NUCLEOS)
                   for p, (Xt, yt, Xb, es) in zip(internas, entrenos)]
        except Exception as e:                       # tampoco cabe en la CPU
            h["error_cpu"] = f"{type(e).__name__}: {str(e)[:300]}"
            continue
        h["error_original"], h["error"] = h["error"], None
        h["puntaje"] = float(np.mean([x[0] for x in res]))
        h["segundos_cpu"] = time.perf_counter() - t0
        h["reparada_cpu"] = True
        nuevos[h["t"]] = res

    mejor_previo = r["auc_interno"]
    ganador = max(r["historial"], key=lambda h: (h["error"] is None, h["puntaje"]))
    mejor_run = -np.inf
    for h in r["historial"]:                         # recalcula la curva "mejor hasta t"
        mejor_run = max(mejor_run, h["puntaje"])
        h["mejor"] = mejor_run
    cambio = ganador["t"] in nuevos and ganador["puntaje"] > mejor_previo
    info = {"clave": clave, "evaluaciones_reparadas": len(nuevos),
            "auc_interno_antes": mejor_previo, "auc_externo_antes": r["metricas"]["auc_roc"],
            "cambio_ganador": bool(cambio)}

    if cambio:
        res = nuevos[ganador["t"]]
        umbral = None
        if u.tarea == "clasificacion":
            y_oof = np.concatenate([np.asarray(p.y_va) for p in internas])
            umbral = met.umbral_coste(y_oof, np.concatenate([x[1] for x in res]))
        Xt, yt, Xb, es = cache.entrenamiento(externa, u.tarea, u.modelo, u.balanceo, r["semilla"])
        est, inf = mod.entrenar(u.tarea, u.modelo, ganador["params"], u.balanceo, Xt, yt,
                                semilla=r["semilla"], dispositivo="cpu", X_balanceado=Xb, es=es)
        s_ext = mod.puntuar(est, externa.X_va, u.tarea)
        del est
        y_ext = np.asarray(externa.y_va).astype(int)
        if u.tarea == "clasificacion":
            m = met.metricas_clasificacion(y_ext, s_ext, umbral)
            if mod.es_probabilidad(u.tarea, u.modelo):
                m.update(met.resumen_calibracion(y_ext, s_ext))
        else:
            m = met.metricas_regresion(y_ext, s_ext)
            m["brier"] = float(np.mean((np.clip(s_ext, 0, 1) - y_ext) ** 2))
        np.savez_compressed(config.DIR_PREDICCIONES / f"{clave}.npz",
                            fila=np.asarray(externa.fila_va), y=y_ext.astype(np.int8),
                            s=s_ext.astype(np.float32))
        r.update(mejor_params=ganador["params"], auc_interno=ganador["puntaje"],
                 auc_interno_pliegues=[x[0] for x in res], umbral=umbral, metricas=m,
                 dispositivo_xgb="cpu (reparacion)")
        info["auc_externo_despues"] = m["auc_roc"]
    info["auc_interno_despues"] = r["auc_interno"]
    r["reparacion"] = {"tipo": "memoria_gpu_evaluada_en_cpu",
                       "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    ex._escribir_atomico(ruta, json.dumps(ex._json_seguro(r), indent=1))
    return info


def reparar_todo(verboso: bool = True) -> pd.DataFrame:
    """Repite las unidades con fallos del sistema y repara las de memoria de GPU."""
    df = listar_fallidas()
    salida = []
    for clave in sorted(df.loc[df["causa"] == "sistema", "clave"].unique()):
        if verboso:
            print(f"Repitiendo {clave} (fallo del sistema)...", flush=True)
        repetir_unidad(clave)
        salida.append({"clave": clave, "resultado": "unidad repetida"})
    cache = ex.CacheParticiones()
    for clave in sorted(df.loc[df["causa"] == "memoria_gpu", "clave"].unique()):
        if verboso:
            print(f"Reparando {clave} (memoria GPU)...", flush=True)
        salida.append(reparar_memoria(clave, cache))
    cache.liberar()
    listar_fallidas()                                    # actualiza el CSV
    tabla = pd.DataFrame(salida)
    if len(tabla):
        tabla.to_csv(config.DIR_RESULTADOS / "reparaciones.csv", index=False)
    return tabla
