"""Los cuatro metodos de optimizacion de hiperparametros (Seccion 3.3) y
Successive Halving como tecnica multi-fidelidad.

Interfaz comun
--------------
Cada metodo recibe un :class:`~src.espacios.Espacio`, una funcion objetivo
``f(params) -> puntaje`` (mayor es mejor; aqui, AUC-ROC medio en los pliegues
internos) y un **presupuesto de evaluaciones**. Todos devuelven un
:class:`ResultadoOptimizacion` con el historial completo, que alimenta las curvas
de desempeno en cualquier momento (*anytime performance*) del capitulo 04.

Se cuentan solo evaluaciones **distintas**: si un metodo propone una
configuracion ya evaluada (frecuente en el genetico cuando la poblacion converge,
o en Grid con parametros inactivos) se reutiliza el resultado sin gastar
presupuesto. Asi el presupuesto mide lo mismo para los cuatro metodos: numero de
modelos realmente entrenados.

Metodos
-------
* ``rejilla``   : Grid Search sobre :meth:`Espacio.rejilla`.
* ``aleatoria`` : Random Search (Bergstra y Bengio, 2012).
* ``bayesiana`` : Optuna con Tree-structured Parzen Estimator (TPE). El modelo
  sustituto no es un proceso gaussiano: TPE modela dos densidades, ``l(x)`` para
  las configuraciones buenas (el mejor cuantil gamma) y ``g(x)`` para el resto, y
  propone el candidato que maximiza ``l(x)/g(x)``, que es equivalente a maximizar
  la Mejora Esperada (Expected Improvement) bajo ese modelo (Bergstra et al., 2011).
* ``genetica``  : algoritmo genetico con DEAP; genoma en ``[0, 1]^d``, seleccion
  por torneo, cruce blend, mutacion gaussiana y elitismo.
"""
from __future__ import annotations

import math
import random
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import config
from .espacios import Espacio

OPTIMIZADORES = ("rejilla", "aleatoria", "bayesiana", "genetica")

# Parametros del algoritmo genetico (justificados en el capitulo 04)
GA_TORNEO = 3          # presion de seleccion moderada: con poblaciones pequenas un
                       # torneo mayor converge prematuramente
GA_PROB_CRUCE = 0.7
GA_ALFA_BLEND = 0.5    # BLX-0.5: los hijos pueden salir del segmento entre padres
GA_PROB_MUTACION = 0.4
GA_SIGMA = 0.15        # desviacion de la mutacion en el espacio unitario
GA_ELITE = 1


@dataclass
class ResultadoOptimizacion:
    """Resultado de una optimizacion.

    Attributes
    ----------
    metodo : str
    mejor_params : dict
    mejor_puntaje : float
    historial : list of dict
        Una entrada por evaluacion distinta: ``t`` (1..n), ``params``, ``puntaje``,
        ``mejor`` (mejor puntaje hasta t), ``segundos``, ``generacion`` (genetico).
    t_total : float
        Tiempo de reloj de toda la optimizacion (s).
    extra : dict
        Informacion especifica del metodo (diversidad del genetico, configuracion
        del TPE, tamano de la rejilla, ...).
    """

    metodo: str
    mejor_params: dict
    mejor_puntaje: float
    historial: list
    t_total: float
    extra: dict = field(default_factory=dict)

    @property
    def n_evaluaciones(self) -> int:
        """Numero de evaluaciones realizadas."""
        return len(self.historial)

    def curva(self) -> np.ndarray:
        """Mejor puntaje encontrado hasta la evaluacion t (curva anytime)."""
        return np.array([h["mejor"] for h in self.historial])


class _Evaluador:
    """Envuelve la funcion objetivo: memoiza, mide tiempos y aisla errores."""

    def __init__(self, espacio: Espacio, objetivo, presupuesto: int):
        self.espacio = espacio
        self.objetivo = objetivo
        self.presupuesto = presupuesto
        self.memo = {}
        self.historial = []
        self.mejor = -math.inf
        self.mejor_params = None
        self.generacion = None
        self.errores = 0

    @property
    def agotado(self) -> bool:
        """True si ya se consumio el presupuesto de evaluaciones."""
        return len(self.historial) >= self.presupuesto

    def __call__(self, params: dict) -> float:
        params = self.espacio.canonico(params)
        k = self.espacio.clave(params)
        if k in self.memo:
            return self.memo[k]
        if self.agotado:
            return -math.inf
        t0 = time.perf_counter()
        error = None
        try:
            valor = float(self.objetivo(params))
            if not np.isfinite(valor):
                raise ValueError(f"puntaje no finito: {valor}")
        except Exception as e:                       # una evaluacion fallida no
            valor = 0.0                              # detiene la optimizacion
            error = f"{type(e).__name__}: {e}"
            self.errores += 1
            if self.errores == 1:
                traceback.print_exc()
        seg = time.perf_counter() - t0
        self.memo[k] = valor
        if valor > self.mejor:
            self.mejor, self.mejor_params = valor, dict(params)
        self.historial.append({"t": len(self.historial) + 1, "params": dict(params),
                               "puntaje": valor, "mejor": self.mejor, "segundos": seg,
                               "generacion": self.generacion, "error": error})
        return valor

    def resultado(self, metodo, t0, extra=None) -> ResultadoOptimizacion:
        """Empaqueta el resultado; falla si mas de la mitad de las evaluaciones dieron error."""
        if self.errores and self.errores >= max(1, len(self.historial) // 2 + 1):
            raise RuntimeError(f"{metodo}: fallaron {self.errores} de "
                               f"{len(self.historial)} evaluaciones")
        return ResultadoOptimizacion(metodo, self.mejor_params or {}, self.mejor,
                                     self.historial, time.perf_counter() - t0,
                                     extra or {})


# ---------------------------------------------------------------------------
# Metodos
# ---------------------------------------------------------------------------
def rejilla(espacio, objetivo, presupuesto, semilla=config.SEMILLA):
    """Grid Search: evalua la rejilla completa (a lo sumo ``presupuesto`` puntos)."""
    t0 = time.perf_counter()
    ev = _Evaluador(espacio, objetivo, presupuesto)
    puntos = espacio.rejilla(presupuesto)
    for p in puntos:
        ev(p)
    return ev.resultado("rejilla", t0, {"tamano_rejilla": len(puntos)})


def aleatoria(espacio, objetivo, presupuesto, semilla=config.SEMILLA):
    """Random Search: ``presupuesto`` configuraciones distintas muestreadas al azar."""
    t0 = time.perf_counter()
    ev = _Evaluador(espacio, objetivo, presupuesto)
    rng = np.random.default_rng(semilla)
    intentos = 0
    while not ev.agotado and intentos < 50 * presupuesto:
        ev(espacio.muestrear(rng))
        intentos += 1
    return ev.resultado("aleatoria", t0)


def bayesiana(espacio, objetivo, presupuesto, semilla=config.SEMILLA,
              n_arranque=None, devolver_estudio=False):
    """Optimizacion bayesiana con Optuna (TPE multivariado).

    Parameters
    ----------
    n_arranque : int, optional
        Evaluaciones aleatorias iniciales antes de que el TPE tome el control
        (exploracion pura). Por defecto ``max(3, presupuesto // 4)``.
    devolver_estudio : bool
        Si es True, ``extra['estudio']`` contiene el ``optuna.Study`` (para las
        visualizaciones del capitulo 04). No se serializa.
    """
    import warnings
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    t0 = time.perf_counter()
    ev = _Evaluador(espacio, objetivo, presupuesto)
    n_arranque = n_arranque or max(3, presupuesto // 4)
    muestreador = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=n_arranque,
                                             multivariate=True, n_ei_candidates=24)
    estudio = optuna.create_study(direction="maximize", sampler=muestreador)

    def f(trial):
        """Funcion objetivo de Optuna: evalua la configuracion sugerida por el trial."""
        v = ev(espacio.sugerir(trial))
        return v if np.isfinite(v) else 0.0

    def parar(study, _trial):
        """Callback de Optuna que detiene el estudio al agotar el presupuesto."""
        if ev.agotado:
            study.stop()

    estudio.optimize(f, n_trials=20 * presupuesto, callbacks=[parar],
                     gc_after_trial=True, show_progress_bar=False)
    extra = {"sustituto": "TPE (l(x)/g(x), multivariado)",
             "adquisicion": "Mejora Esperada (EI) via maximizacion de l(x)/g(x)",
             "n_arranque": n_arranque, "n_candidatos_ei": 24,
             "trials_optuna": len(estudio.trials)}
    if devolver_estudio:
        extra["estudio"] = estudio
    return ev.resultado("bayesiana", t0, extra)


def _diversidad(poblacion) -> float:
    """Distancia euclidea media entre pares de genomas, normalizada por sqrt(d)."""
    G = np.array([list(i) for i in poblacion], dtype=float)
    if len(G) < 2:
        return 0.0
    d = np.sqrt(((G[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(G), 1)
    return float(d[iu].mean() / math.sqrt(G.shape[1]))


def genetica(espacio, objetivo, presupuesto, semilla=config.SEMILLA, poblacion=None):
    """Algoritmo genetico con DEAP.

    * Representacion: genoma real en ``[0, 1]^d`` decodificado por
      :meth:`Espacio.decodificar` (escala log incluida).
    * Seleccion: torneo de tamano 3.
    * Cruce: blend (BLX-alfa, alfa=0.5) con probabilidad 0.7.
    * Mutacion: gaussiana (sigma=0.15, probabilidad por gen max(1/d, 0.2)) con
      probabilidad 0.4 por individuo; los genes se recortan a [0, 1].
    * Elitismo: el mejor individuo pasa intacto a la siguiente generacion.
    * Poblacion: ``max(4, presupuesto // 4)``; se generan generaciones hasta agotar
      el presupuesto de evaluaciones distintas.

    ``extra['generaciones']`` registra por generacion la diversidad genetica, el
    mejor y la media del puntaje, para detectar convergencia prematura.
    """
    from deap import base, creator, tools
    if not hasattr(creator, "AptitudE3"):
        creator.create("AptitudE3", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividuoE3"):
        creator.create("IndividuoE3", list, fitness=creator.AptitudE3)

    random.seed(semilla)                      # DEAP usa el generador de `random`
    rng = np.random.default_rng(semilla)
    t0 = time.perf_counter()
    ev = _Evaluador(espacio, objetivo, presupuesto)
    d = espacio.dimension
    P = poblacion or max(4, presupuesto // 4)
    indpb = max(1.0 / d, 0.2)

    def evaluar(ind):
        """Decodifica el genoma, evalua la configuracion y asigna la aptitud."""
        v = ev(espacio.decodificar(ind))
        ind.fitness.values = (v,)

    def recortar(ind):
        """Mantiene cada gen dentro de [0, 1] tras el cruce y la mutacion."""
        for i in range(len(ind)):
            ind[i] = min(1.0, max(0.0, ind[i]))

    pob = [creator.IndividuoE3(rng.random(d).tolist()) for _ in range(P)]
    ev.generacion = 0
    for ind in pob:
        evaluar(ind)
    generaciones = []

    def registrar(g):
        """Guarda diversidad, mejor y media de la poblacion en la generacion ``g``."""
        vals = [i.fitness.values[0] for i in pob if np.isfinite(i.fitness.values[0])]
        generaciones.append({"generacion": g, "diversidad": _diversidad(pob),
                             "mejor": max(vals) if vals else np.nan,
                             "media": float(np.mean(vals)) if vals else np.nan,
                             "evaluaciones": len(ev.historial)})

    registrar(0)
    g = 0
    while not ev.agotado and g < 200:
        g += 1
        ev.generacion = g
        elite = [creator.IndividuoE3(list(i)) for i in tools.selBest(pob, GA_ELITE)]
        for e, o in zip(elite, tools.selBest(pob, GA_ELITE)):
            e.fitness.values = o.fitness.values
        hijos = [creator.IndividuoE3(list(i)) for i in
                 tools.selTournament(pob, P - GA_ELITE, tournsize=GA_TORNEO)]
        for a, b in zip(hijos[::2], hijos[1::2]):
            if random.random() < GA_PROB_CRUCE:
                tools.cxBlend(a, b, alpha=GA_ALFA_BLEND)
        for h in hijos:
            if random.random() < GA_PROB_MUTACION:
                tools.mutGaussian(h, mu=0.0, sigma=GA_SIGMA, indpb=indpb)
            recortar(h)
            evaluar(h)
        pob = elite + hijos
        registrar(g)

    extra = {"poblacion": P, "torneo": GA_TORNEO, "prob_cruce": GA_PROB_CRUCE,
             "alfa_blend": GA_ALFA_BLEND, "prob_mutacion": GA_PROB_MUTACION,
             "sigma": GA_SIGMA, "indpb": indpb, "elite": GA_ELITE,
             "generaciones": generaciones}
    return ev.resultado("genetica", t0, extra)


METODOS = {"rejilla": rejilla, "aleatoria": aleatoria,
           "bayesiana": bayesiana, "genetica": genetica}


def optimizar(metodo: str, espacio: Espacio, objetivo, presupuesto: int,
              semilla: int = config.SEMILLA, **kw) -> ResultadoOptimizacion:
    """Punto de entrada comun a los cuatro metodos."""
    return METODOS[metodo](espacio, objetivo, presupuesto, semilla=semilla, **kw)


# ---------------------------------------------------------------------------
# Multi-fidelidad
# ---------------------------------------------------------------------------
def mitades_sucesivas(espacio: Espacio, objetivo_fidelidad, n_inicial: int = 27,
                      eta: int = 3, semilla: int = config.SEMILLA):
    """Successive Halving (Jamieson y Talwalkar, 2016).

    Se muestrean ``n_inicial`` configuraciones y se evaluan con la fidelidad mas
    baja; en cada ronda sobrevive la mejor fraccion ``1/eta`` y la fidelidad se
    multiplica por ``eta``, hasta evaluar la ultima con fidelidad completa.

    Parameters
    ----------
    objetivo_fidelidad : callable
        ``f(params, fidelidad) -> puntaje``, con ``fidelidad`` en (0, 1]: la
        fraccion del recurso usada (aqui, del entrenamiento).
    n_inicial, eta : int

    Returns
    -------
    (dict, list)
        Mejor configuracion y registro por ronda (configuraciones, fidelidad,
        coste en unidades de "entrenamiento completo", tiempo).
    """
    rng = np.random.default_rng(semilla)
    rondas = max(1, int(round(math.log(n_inicial, eta))))
    configs = [espacio.muestrear(rng) for _ in range(n_inicial)]
    registro = []
    for r in range(rondas + 1):
        fid = float(eta ** (r - rondas))
        t0 = time.perf_counter()
        puntajes = [float(objetivo_fidelidad(c, fid)) for c in configs]
        registro.append({"ronda": r, "configuraciones": len(configs), "fidelidad": fid,
                         "coste_equivalente": len(configs) * fid,
                         "mejor": max(puntajes), "segundos": time.perf_counter() - t0})
        orden = np.argsort(puntajes)[::-1]
        if r == rondas or len(configs) == 1:
            return configs[int(orden[0])], registro
        configs = [configs[i] for i in orden[:max(1, len(configs) // eta)]]
    return configs[0], registro
