"""Motor del estudio combinatorio con validacion cruzada anidada.

Unidad de trabajo
-----------------
El estudio se descompone en unidades atomicas::

    (tarea, modelo, balanceo, optimizador, pliegue externo)

Clasificacion: 7 x 4 x 4 x 5 = 560 unidades. Regresion: 7 x 1 x 4 x 5 = 140.

Cada unidad:

1. corre el optimizador sobre los ``N_INTERNOS`` pliegues internos del externo
   (bucle interno); el objetivo es el AUC-ROC medio en validacion interna;
2. elige el umbral de decision (coste FN=10:1) con las predicciones internas
   fuera de muestra de la mejor configuracion;
3. reajusta la mejor configuracion sobre todo el entrenamiento del externo y la
   evalua en su validacion (bucle externo). **Solo esta metrica se reporta.**

Robustez
--------
* Cada unidad se guarda en disco (``resultados/corridas/<clave>.json``) apenas
  termina, con escritura atomica. Si el kernel se cae o se apaga el computador,
  al volver a ejecutar la celda se saltan las unidades hechas y se continua.
* Una unidad que falla se registra en ``resultados/errores.csv`` (con la traza
  completa en ``resultados/errores/``) y el estudio sigue con la siguiente.
* Una evaluacion que falla dentro de una optimizacion puntua 0 y no detiene la
  optimizacion (ver :class:`src.optimizadores._Evaluador`).
"""
from __future__ import annotations

import csv
import gc
import json
import os
import platform
import shutil
import time
import traceback
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import balanceo as bal
from . import config
from . import metricas as met
from . import modelos as mod
from . import preprocesamiento as prep
from .espacios import MODELOS, espacio
from .optimizadores import OPTIMIZADORES, optimizar

BALANCEOS_TAREA = {"clasificacion": bal.BALANCEOS, "regresion": ("ninguno",)}


# ---------------------------------------------------------------------------
# Unidades
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Unidad:
    """Unidad de trabajo del estudio: (tarea, modelo, balanceo, optimizador, pliegue externo)."""
    tarea: str
    modelo: str
    balanceo: str
    optimizador: str
    pliegue: int

    @property
    def clave(self) -> str:
        """Identificador unico de la unidad, usado como nombre de sus archivos."""
        return f"{self.tarea}__{self.modelo}__{self.balanceo}__{self.optimizador}__e{self.pliegue}"

    @property
    def archivo(self):
        """Ruta del JSON donde se guarda el resultado de la unidad."""
        return config.DIR_CORRIDAS / f"{self.clave}.json"

    @property
    def hecha(self) -> bool:
        """True si la unidad ya esta terminada y guardada en disco."""
        return self.archivo.is_file()


def unidades(tarea, modelos=None, balanceos=None, optimizadores=None, pliegues=None) -> list:
    """Lista ordenada de unidades: modelo -> balanceo -> pliegue -> optimizador.

    Con este orden, el balanceo de cada (pliegue, tecnica) se calcula una vez y lo
    reutilizan los cuatro optimizadores.
    """
    modelos = modelos or MODELOS[tarea]
    balanceos = balanceos or BALANCEOS_TAREA[tarea]
    optimizadores = optimizadores or OPTIMIZADORES
    pliegues = range(config.N_EXTERNOS) if pliegues is None else pliegues
    return [Unidad(tarea, m, b, o, k) for m in modelos for b in balanceos
            for k in pliegues for o in optimizadores]


# ---------------------------------------------------------------------------
# Cache de particiones y balanceos
# ---------------------------------------------------------------------------
def _memmap(ruta):
    return np.load(ruta, mmap_mode="r")


class CacheParticiones:
    """Mantiene en memoria mapeada las particiones del pliegue externo en curso,
    la separacion de parada temprana de XGBoost y los entrenamientos balanceados
    de la tecnica en curso.

    Los derivados (balanceos, separacion de parada temprana) se escriben como
    ``.npy`` temporales y se abren mapeados: asi los procesos paralelos de joblib
    los comparten sin copiarlos, y se calculan una sola vez por particion aunque
    los usen los cuatro optimizadores (SMOTE/ADASYN sobre ~130 mil filas tardan
    del orden de un minuto; recalcularlos en cada evaluacion multiplicaria ese
    coste por cientos).
    """

    def __init__(self):
        self.k = None
        self._externa = None
        self._internas = None
        self._bal_clave = None
        self._bal = {}
        self._es = {}
        # carpeta propia de este proceso: dos estudios en paralelo nunca comparten
        # (ni borran) los temporales del otro
        self.dir_tmp = config.DIR_CACHE / "temporal" / f"p{os.getpid()}"

    def _cargar(self, directorio, nombre):
        cols = json.loads((directorio / "columnas.json").read_text(encoding="utf-8"))
        arr = {a: _memmap(directorio / f"{a}.npy") for a in
               ("X_tr", "y_tr", "X_va", "y_va", "fila_tr", "fila_va")}
        return prep.Particion(nombre=nombre, columnas=cols, **arr)

    def _tmp(self, subcarpeta, nombre, arr):
        # nombre unico: en Windows un archivo mapeado que un proceso aun retiene no
        # se puede borrar ni sobrescribir; con nombres unicos nunca hay choque y la
        # limpieza es de mejor esfuerzo.
        d = self.dir_tmp / subcarpeta
        d.mkdir(parents=True, exist_ok=True)
        ruta = d / f"{nombre}_{uuid.uuid4().hex[:8]}.npy"
        np.save(ruta, arr)
        return _memmap(ruta)

    def pliegue(self, k):
        """Particion externa e internas del pliegue ``k``."""
        if self.k != k:
            self.liberar()
            self.k = k
            base = prep.dir_particiones() / f"e{k}"
            self._externa = self._cargar(base, f"e{k}")
            self._internas = [self._cargar(base / f"i{j}", f"e{k}/i{j}")
                              for j in range(config.N_INTERNOS)]
        return self._externa, self._internas

    def entrenamiento(self, part, tarea, modelo, tecnica, semilla=config.SEMILLA):
        """Entrenamiento listo para :func:`src.modelos.entrenar`.

        Returns
        -------
        (X, y, X_balanceado, es)
            ``X, y`` sin balancear (sin la parte de parada temprana si el modelo es
            XGBoost), el par balanceado (o None) y el conjunto de parada temprana
            (o None).
        """
        nombre = part.nombre.replace("/", "_")
        X, y, es = part.X_tr, part.y_tr, None
        if modelo == "xgboost":
            if part.nombre not in self._es:
                Xa, ya, Xe, ye = mod.separar_parada_temprana(part.X_tr, part.y_tr, tarea, semilla)
                self._es[part.nombre] = (self._tmp("es", f"{nombre}_Xa", Xa),
                                         self._tmp("es", f"{nombre}_ya", ya),
                                         self._tmp("es", f"{nombre}_Xe", Xe),
                                         self._tmp("es", f"{nombre}_ye", ye))
                del Xa, ya, Xe, ye
            X, y, Xe, ye = self._es[part.nombre]
            es = (Xe, ye)
        X_bal = None
        if tarea == "clasificacion" and tecnica in bal.REMUESTREO:
            clave = (self.k, tecnica)
            if self._bal_clave != clave:
                self.liberar_balanceo()
                self._bal_clave = clave
            k_bal = f"{nombre}_{modelo == 'xgboost'}"
            if k_bal not in self._bal:
                Xb, yb = bal.aplicar(tecnica, np.asarray(X), np.asarray(y), semilla)
                self._bal[k_bal] = (self._tmp("balanceo", f"{k_bal}_{tecnica}_X", Xb),
                                    self._tmp("balanceo", f"{k_bal}_{tecnica}_y", yb))
                del Xb, yb
            X_bal = self._bal[k_bal]
        return X, y, X_bal, es

    def liberar_balanceo(self):
        """Descarta de memoria y de disco los conjuntos balanceados de la particion actual."""
        self._bal = {}
        self._bal_clave = None
        gc.collect()
        shutil.rmtree(self.dir_tmp / "balanceo", ignore_errors=True)

    def liberar(self):
        """Libera toda la memoria y los temporales en disco de la cache."""
        self.liberar_balanceo()
        self._es = {}
        gc.collect()
        shutil.rmtree(self.dir_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Evaluacion de una configuracion en un pliegue interno
# ---------------------------------------------------------------------------
def _evaluar_interna(tarea, modelo, params, tecnica, X_tr, y_tr, X_va, y_va,
                     semilla, dispositivo, X_bal=None, es=None, hilos=None):
    """Ajusta en el entrenamiento interno y devuelve (AUC, puntajes de validacion, info).

    Es una funcion de modulo (no un cierre) para poder ejecutarse en procesos
    ``loky`` en paralelo.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=hilos):
        est, info = mod.entrenar(tarea, modelo, params, tecnica, X_tr, y_tr,
                                 semilla=semilla, dispositivo=dispositivo,
                                 X_balanceado=X_bal, es=es)
        t0 = time.perf_counter()
        s = mod.puntuar(est, X_va, tarea)
        info["t_prediccion"] = time.perf_counter() - t0
    del est
    return float(roc_auc_score(np.asarray(y_va).astype(int), s)), s, info


def _json_seguro(o):
    """Convierte tipos numpy para json."""
    if isinstance(o, dict):
        return {str(k): _json_seguro(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_seguro(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _escribir_atomico(ruta, texto):
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    tmp.write_text(texto, encoding="utf-8")
    os.replace(tmp, ruta)


def _versiones():
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__,
         "pandas": pd.__version__, "sklearn": sklearn.__version__}
    for lib in ("xgboost", "imblearn", "optuna", "deap", "faiss"):
        try:
            v[lib] = __import__(lib).__version__
        except Exception:
            v[lib] = None
    return v


def ejecutar_unidad(u: Unidad, cache: CacheParticiones, n_eval: int | None = None,
                    semilla: int = config.SEMILLA, dispositivo: str | None = None) -> dict:
    """Ejecuta una unidad completa (bucle interno + bucle externo) y la guarda.

    Returns
    -------
    dict
        El registro que se escribe en ``resultados/corridas/<clave>.json``.
    """
    n_eval = n_eval or config.N_EVALUACIONES
    dispositivo = dispositivo or config.DISPOSITIVO_XGB
    esp = espacio(u.tarea, u.modelo)
    externa, internas = cache.pliegue(u.pliegue)
    entrenos = [cache.entrenamiento(p, u.tarea, u.modelo, u.balanceo, semilla)
                for p in internas]

    paralelo = (config.PARALELIZAR_INTERNOS and u.modelo in mod.MODELOS_UN_HILO
                and len(internas) > 1)
    n_proc = min(len(internas), config.NUCLEOS) if paralelo else 1
    hilos = max(1, config.NUCLEOS // n_proc)
    mejor = {"puntaje": -np.inf, "s": None}

    def objetivo(params):
        """AUC medio de ``params`` en los pliegues internos; recuerda las predicciones del mejor."""
        args = [(u.tarea, u.modelo, params, u.balanceo, Xt, yt, p.X_va, p.y_va,
                 semilla, dispositivo, Xb, es, hilos)
                for p, (Xt, yt, Xb, es) in zip(internas, entrenos)]
        if paralelo:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=n_proc, backend="loky")(
                delayed(_evaluar_interna)(*a) for a in args)
        else:
            res = [_evaluar_interna(*a) for a in args]
        auc = float(np.mean([r[0] for r in res]))
        if auc > mejor["puntaje"]:
            mejor.update(puntaje=auc, s=[r[1] for r in res], aucs=[r[0] for r in res])
        return auc

    t0 = time.perf_counter()
    opt = optimizar(u.optimizador, esp, objetivo, n_eval, semilla=semilla)
    t_opt = time.perf_counter() - t0

    # --- umbral de decision con las predicciones internas fuera de muestra -----
    umbral = None
    if u.tarea == "clasificacion" and mejor["s"] is not None:
        y_oof = np.concatenate([np.asarray(p.y_va) for p in internas])
        s_oof = np.concatenate(mejor["s"])
        umbral = met.umbral_coste(y_oof, s_oof)

    # --- bucle externo: reajuste en todo el entrenamiento del externo ---------
    Xt, yt, Xb, es = cache.entrenamiento(externa, u.tarea, u.modelo, u.balanceo, semilla)
    est, info = mod.entrenar(u.tarea, u.modelo, opt.mejor_params, u.balanceo,
                             Xt, yt, semilla=semilla, dispositivo=dispositivo,
                             X_balanceado=Xb, es=es)
    t1 = time.perf_counter()
    s_ext = mod.puntuar(est, externa.X_va, u.tarea)
    t_inf = time.perf_counter() - t1
    del est
    y_ext = np.asarray(externa.y_va).astype(int)

    if u.tarea == "clasificacion":
        m = met.metricas_clasificacion(y_ext, s_ext, umbral)
        if mod.es_probabilidad(u.tarea, u.modelo):
            m.update(met.resumen_calibracion(y_ext, s_ext))
    else:
        m = met.metricas_regresion(y_ext, s_ext)
        m["brier"] = float(np.mean((np.clip(s_ext, 0, 1) - y_ext) ** 2))

    # --- guardar predicciones y registro -----------------------------------------
    config.DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.DIR_PREDICCIONES / f"{u.clave}.npz",
                        fila=np.asarray(externa.fila_va), y=y_ext.astype(np.int8),
                        s=s_ext.astype(np.float32))

    historial = [{k: h[k] for k in ("t", "puntaje", "mejor", "segundos", "generacion", "error")}
                 | {"params": h["params"]} for h in opt.historial]
    registro = {
        "clave": u.clave, "tarea": u.tarea, "modelo": u.modelo, "balanceo": u.balanceo,
        "optimizador": u.optimizador, "pliegue": u.pliegue, "semilla": semilla,
        "estado": "ok", "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dispositivo_xgb": dispositivo if u.modelo == "xgboost" else None,
        "motor_knn": config.KNN_MOTOR if u.modelo == "knn" else None,
        "mejor_params": opt.mejor_params, "auc_interno": opt.mejor_puntaje,
        "auc_interno_pliegues": mejor.get("aucs"),
        "n_evaluaciones": opt.n_evaluaciones, "presupuesto": n_eval,
        "t_optimizacion": t_opt,
        "t_por_evaluacion": float(np.mean([h["segundos"] for h in opt.historial])),
        "t_reentreno": info["t_ajuste"] + info["t_balanceo"],
        "t_inferencia": t_inf, "filas_ajuste": info["filas_ajuste"],
        "convergio": info["convergio"], "mejor_iteracion_xgb": info["mejor_iteracion"],
        "umbral": umbral, "metricas": m, "historial": historial,
        "extra": {k: v for k, v in opt.extra.items() if k != "estudio"},
        "paralelo_interno": paralelo, "versiones": _versiones(),
    }
    config.DIR_CORRIDAS.mkdir(parents=True, exist_ok=True)
    _escribir_atomico(u.archivo, json.dumps(_json_seguro(registro), indent=1))
    return registro


def _registrar_error(u: Unidad, e: Exception):
    d = config.DIR_RESULTADOS / "errores"
    d.mkdir(parents=True, exist_ok=True)
    traza = traceback.format_exc()
    (d / f"{u.clave}.txt").write_text(traza, encoding="utf-8")
    ruta = config.DIR_RESULTADOS / "errores.csv"
    nuevo = not ruta.is_file()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "clave", "tipo", "mensaje"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), u.clave, type(e).__name__,
                    str(e).replace("\n", " ")[:500]])


def limpiar_temporales_huerfanos() -> None:
    """Borra los temporales de procesos que ya no existen (p. ej. tras un apagado)."""
    base = config.DIR_CACHE / "temporal"
    if not base.is_dir():
        return
    try:
        import psutil
    except ImportError:
        return
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("p") and d.name[1:].isdigit():
            if not psutil.pid_exists(int(d.name[1:])):
                shutil.rmtree(d, ignore_errors=True)


def _dir_bloqueos():
    return config.DIR_RESULTADOS / "bloqueos"


def _proceso_vivo(pid: int) -> bool:
    """True si ``pid`` es un proceso de Python vivo (en Windows los PID se reciclan)."""
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        return "python" in psutil.Process(pid).name().lower()
    except Exception:
        return False


def _reclamar(u: Unidad) -> bool:
    """Reserva la unidad para este proceso con un archivo de bloqueo exclusivo.

    Permite que varias ventanas (``correr_estudio.py``) trabajen sobre la misma
    lista sin repetir unidades: la que llega primero la reserva y las demas la
    saltan. Un bloqueo de un proceso que ya no existe (ventana cerrada, apagado)
    se considera abandonado y se toma.
    """
    d = _dir_bloqueos()
    d.mkdir(parents=True, exist_ok=True)
    ruta = d / f"{u.clave}.lock"
    for _ in range(2):
        try:
            fd = os.open(ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            if u.hecha:                 # otra ventana la termino justo antes
                _liberar(u)
                return False
            return True
        except FileExistsError:
            try:
                pid = int(ruta.read_text().strip() or 0)
            except Exception:
                pid = 0
            if pid == os.getpid():
                return True
            if pid and _proceso_vivo(pid):
                return False
            try:                        # bloqueo abandonado
                ruta.unlink()
            except OSError:
                return False
    return False


def _liberar(u: Unidad) -> None:
    try:
        (_dir_bloqueos() / f"{u.clave}.lock").unlink()
    except OSError:
        pass


def ejecutar(tarea: str, modelos=None, balanceos=None, optimizadores=None, pliegues=None,
             n_eval: int | None = None, verboso: bool = True) -> pd.DataFrame:
    """Ejecuta (o reanuda) el estudio para las combinaciones indicadas.

    Las unidades ya guardadas se saltan; las que fallan se registran y el bucle
    continua. Puede interrumpirse en cualquier momento (boton de detener del
    kernel): lo terminado queda en disco.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    modelos, balanceos, optimizadores, pliegues : list, optional
        Subconjuntos a ejecutar (por defecto, todos).
    n_eval : int, optional
        Presupuesto de evaluaciones (por defecto ``config.N_EVALUACIONES``).

    Returns
    -------
    DataFrame
        Estado de cada unidad solicitada: ``hecha``, ``nueva``, ``error``.
    """
    lista = unidades(tarea, modelos, balanceos, optimizadores, pliegues)
    limpiar_temporales_huerfanos()
    cache = CacheParticiones()
    filas = []
    pendientes = [u for u in lista if not u.hecha]
    if verboso:
        print(f"{len(lista)} unidades | ya hechas: {len(lista) - len(pendientes)} | "
              f"pendientes: {len(pendientes)}")
    t_inicio = time.perf_counter()
    for i, u in enumerate(lista):
        if u.hecha:
            filas.append({"clave": u.clave, "estado": "hecha", "segundos": 0.0})
            continue
        if not _reclamar(u):              # la esta corriendo otra ventana
            filas.append({"clave": u.clave, "estado": "otra_ventana", "segundos": 0.0})
            continue
        t0 = time.perf_counter()
        try:
            r = ejecutar_unidad(u, cache, n_eval=n_eval)
            seg = time.perf_counter() - t0
            filas.append({"clave": u.clave, "estado": "nueva", "segundos": seg})
            if verboso:
                print(f"[{i + 1}/{len(lista)}] OK  {u.clave:62s} AUC ext "
                      f"{r['metricas']['auc_roc']:.4f} | {r['n_evaluaciones']} eval | "
                      f"{seg / 60:6.1f} min", flush=True)
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario. Lo terminado esta guardado; "
                  "vuelva a ejecutar la celda para continuar.")
            cache.liberar()
            _liberar(u)
            raise
        except Exception as e:            # aisla el fallo: el estudio sigue
            _registrar_error(u, e)
            filas.append({"clave": u.clave, "estado": "error", "segundos": time.perf_counter() - t0})
            if verboso:
                print(f"[{i + 1}/{len(lista)}] ERROR {u.clave}: {type(e).__name__}: {e}", flush=True)
        _liberar(u)
        gc.collect()
    cache.liberar()
    estado = pd.DataFrame(filas)
    if verboso:
        c = estado["estado"].value_counts().to_dict()
        print(f"\nResumen: {c} | tiempo de esta ejecucion: "
              f"{(time.perf_counter() - t_inicio) / 3600:.2f} h")
    return estado


def avance(tarea: str | None = None) -> pd.DataFrame:
    """Unidades hechas / totales / con error, por tarea y modelo."""
    errores = set()
    ruta = config.DIR_RESULTADOS / "errores.csv"
    if ruta.is_file():
        errores = set(pd.read_csv(ruta)["clave"])
    filas = []
    for t in ([tarea] if tarea else list(MODELOS)):
        for u in unidades(t):
            filas.append({"tarea": u.tarea, "modelo": u.modelo, "hecha": u.hecha,
                          "error_previo": (u.clave in errores) and not u.hecha})
    df = pd.DataFrame(filas)
    return (df.groupby(["tarea", "modelo"], sort=False)
              .agg(hechas=("hecha", "sum"), totales=("hecha", "size"),
                   con_error=("error_previo", "sum")).reset_index())


# ---------------------------------------------------------------------------
# Consolidacion: tabla maestra
# ---------------------------------------------------------------------------
def leer_corridas(tarea: str | None = None) -> list:
    """Lee todos los registros json (opcionalmente de una tarea)."""
    regs = []
    for f in sorted(config.DIR_CORRIDAS.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if tarea is None or r["tarea"] == tarea:
            regs.append(r)
    return regs


def consolidar(guardar: bool = True):
    """Construye el registro por pliegue y la tabla maestra (Seccion 7.3).

    Returns
    -------
    (DataFrame, DataFrame)
        ``pliegues``: una fila por unidad. ``maestra``: una fila por combinacion
        (tarea, modelo, balanceo, optimizador) con media y desviacion estandar de
        cada metrica a traves de los pliegues externos, tiempos, evaluaciones,
        hiperparametros de cada pliegue y semilla.
    """
    regs = leer_corridas()
    if not regs:
        raise RuntimeError("No hay corridas guardadas todavia.")
    filas = []
    for r in regs:
        f = {k: r[k] for k in ("tarea", "modelo", "balanceo", "optimizador", "pliegue",
                               "semilla", "auc_interno", "n_evaluaciones", "t_optimizacion",
                               "t_por_evaluacion", "t_reentreno", "t_inferencia",
                               "filas_ajuste", "convergio", "umbral")}
        f.update({f"m_{k}": v for k, v in r["metricas"].items()
                  if k not in ("tn", "fp", "fn", "tp")})
        f["params"] = json.dumps(r["mejor_params"])
        filas.append(f)
    pliegues = pd.DataFrame(filas)

    claves = ["tarea", "modelo", "balanceo", "optimizador"]
    metricas = [c for c in pliegues.columns if c.startswith("m_")]
    agg = pliegues.groupby(claves, sort=False)
    maestra = agg[metricas].mean().add_suffix("_media")
    maestra = maestra.join(agg[metricas].std(ddof=1).add_suffix("_sd"))
    maestra["pliegues"] = agg.size()
    maestra["auc_interno_media"] = agg["auc_interno"].mean()
    maestra["evaluaciones_total"] = agg["n_evaluaciones"].sum()
    maestra["t_total_h"] = (agg["t_optimizacion"].sum() + agg["t_reentreno"].sum()) / 3600
    maestra["t_por_evaluacion_s"] = agg["t_por_evaluacion"].mean()
    maestra["t_inferencia_s"] = agg["t_inferencia"].mean()
    maestra["semilla"] = agg["semilla"].first()
    maestra["params_por_pliegue"] = agg["params"].agg(lambda s: "[" + ",".join(s) + "]")
    maestra = maestra.reset_index()
    maestra = maestra.sort_values(["tarea", "m_auc_roc_media"], ascending=[True, False])

    if guardar:
        config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
        pliegues.to_csv(config.DIR_RESULTADOS / "registro_pliegues.csv", index=False)
        maestra.to_csv(config.DIR_RESULTADOS / "tabla_maestra.csv", index=False)
        try:
            pliegues.to_parquet(config.DIR_RESULTADOS / "registro_pliegues.parquet", index=False)
            maestra.to_parquet(config.DIR_RESULTADOS / "tabla_maestra.parquet", index=False)
        except Exception:
            pass
    return pliegues, maestra


def predicciones_oof(tarea: str, modelo: str, balanceo: str, optimizador: str) -> pd.DataFrame:
    """Predicciones fuera de muestra de una combinacion sobre todo desarrollo
    (union de las validaciones externas), ordenadas por fila original."""
    partes = []
    for k in range(config.N_EXTERNOS):
        u = Unidad(tarea, modelo, balanceo, optimizador, k)
        f = config.DIR_PREDICCIONES / f"{u.clave}.npz"
        if not f.is_file():
            raise FileNotFoundError(f"Falta {f.name}")
        z = np.load(f)
        partes.append(pd.DataFrame({"fila": z["fila"], "y": z["y"], "s": z["s"],
                                    "pliegue": k}))
    return pd.concat(partes).sort_values("fila").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estimacion de tiempos antes de lanzar el estudio
# ---------------------------------------------------------------------------
def estimar_tiempos(tarea: str, modelos=None, balanceos=None) -> pd.DataFrame:
    """Mide una evaluacion real (configuracion central del espacio, pliegue interno
    e0/i0) por (modelo, balanceo) y proyecta el tiempo total del estudio.

    La proyeccion por combinacion es::

        N_EXTERNOS x (N_OPTIMIZADORES x N_EVALUACIONES x N_INTERNOS x t_eval
                      + t_reentreno_externo)

    con ``t_reentreno_externo ~ t_eval x (n_externo / n_interno)``, suponiendo coste
    lineal en n (optimista para arboles y SVM con nucleo aproximado).
    """
    modelos = modelos or MODELOS[tarea]
    balanceos = balanceos or BALANCEOS_TAREA[tarea]
    cache = CacheParticiones()
    externa, internas = cache.pliegue(0)
    p = internas[0]
    razon_n = len(externa.y_tr) / len(p.y_tr)
    filas = []
    for m in modelos:
        for b in balanceos:
            try:
                t0 = time.perf_counter()
                Xt, yt, Xb, es = cache.entrenamiento(p, tarea, m, b)   # balanceo: 1 vez por particion
                t_bal = time.perf_counter() - t0
                t0 = time.perf_counter()
                auc, _, info = _evaluar_interna(tarea, m, espacio(tarea, m).centro(), b,
                                                Xt, yt, p.X_va, p.y_va,
                                                config.SEMILLA, config.DISPOSITIVO_XGB, Xb, es)
                t_eval = time.perf_counter() - t0
                err = None
            except Exception as e:
                t_eval, t_bal, auc, err = np.nan, np.nan, np.nan, f"{type(e).__name__}: {e}"
            # los pliegues internos de los modelos de un hilo se ajustan en paralelo
            paralelo = (config.PARALELIZAR_INTERNOS and m in mod.MODELOS_UN_HILO)
            factor_int = config.N_INTERNOS / min(config.N_INTERNOS, config.NUCLEOS) if paralelo else config.N_INTERNOS
            por_combo = config.N_EXTERNOS * (len(OPTIMIZADORES) * config.N_EVALUACIONES
                                             * factor_int * t_eval + t_eval * razon_n)
            por_combo += config.N_EXTERNOS * (config.N_INTERNOS + 1) * (t_bal or 0)
            filas.append({"modelo": m, "balanceo": b, "t_eval_s": t_eval, "t_balanceo_s": t_bal,
                          "auc_centro": auc, "horas_combinacion": por_combo / 3600, "error": err})
            print(f"  {m:14s} {b:13s} {t_eval:8.1f} s/eval  (balanceo {t_bal:6.1f} s)", flush=True)
        cache.liberar_balanceo()
    cache.liberar()
    df = pd.DataFrame(filas)
    total = df["horas_combinacion"].sum()
    print(f"\nProyeccion {tarea}: {total:,.1f} h ({total / 24:,.1f} dias) con "
          f"{config.N_EXTERNOS}x{config.N_INTERNOS} pliegues y {config.N_EVALUACIONES} evaluaciones")
    return df
