"""Espacios de busqueda de hiperparametros.

Un mismo espacio se traduce a los cuatro optimizadores:

* Grid Search  -> :meth:`Espacio.rejilla` (niveles equiespaciados, en escala log
  cuando el parametro lo pide).
* Random Search -> :meth:`Espacio.muestrear`.
* Bayesiana (Optuna) -> :meth:`Espacio.sugerir`.
* Genetica (DEAP) -> :meth:`Espacio.decodificar`, que traduce un genoma en
  ``[0, 1]^d`` al espacio real.

Definirlo una sola vez garantiza que los cuatro metodos buscan exactamente en la
misma region, condicion necesaria para comparar su eficiencia.

La justificacion de cada rango esta en el capitulo 02; los comentarios de aqui
resumen el criterio.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Tipos de parametro
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Real:
    """Parametro continuo en ``[bajo, alto]``; ``log=True`` usa escala logaritmica."""

    bajo: float
    alto: float
    log: bool = False

    def desde_unitario(self, u: float) -> float:
        """Valor del hiperparametro que corresponde a la posicion ``u`` en [0, 1] (escala log si aplica)."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            return float(math.exp(math.log(self.bajo) + u * (math.log(self.alto) - math.log(self.bajo))))
        return float(self.bajo + u * (self.alto - self.bajo))

    def niveles(self, n: int) -> list:
        """``n`` niveles equiespaciados del rango (en escala log si aplica), para Grid Search."""
        if n == 1:
            return [self.desde_unitario(0.5)]
        return [self.desde_unitario(u) for u in np.linspace(0, 1, n)]

    def sugerir(self, trial, nombre):
        """Pide el valor a un ``trial`` de Optuna con el mismo rango y escala."""
        return trial.suggest_float(nombre, self.bajo, self.alto, log=self.log)


@dataclass(frozen=True)
class Entero:
    """Parametro entero en ``[bajo, alto]``; ``log=True`` usa escala logaritmica."""

    bajo: int
    alto: int
    log: bool = False

    def desde_unitario(self, u: float) -> int:
        """Valor del hiperparametro que corresponde a la posicion ``u`` en [0, 1] (escala log si aplica)."""
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            v = math.exp(math.log(self.bajo) + u * (math.log(self.alto) - math.log(self.bajo)))
        else:
            v = self.bajo + u * (self.alto - self.bajo)
        return int(min(self.alto, max(self.bajo, round(v))))

    def niveles(self, n: int) -> list:
        """Hasta ``n`` niveles enteros distintos y equiespaciados del rango, para Grid Search."""
        if n == 1:
            return [self.desde_unitario(0.5)]
        return sorted({self.desde_unitario(u) for u in np.linspace(0, 1, n)})

    def sugerir(self, trial, nombre):
        """Pide el valor a un ``trial`` de Optuna con el mismo rango y escala."""
        return trial.suggest_int(nombre, self.bajo, self.alto, log=self.log)


@dataclass(frozen=True)
class Categorico:
    """Parametro categorico."""

    opciones: tuple

    def desde_unitario(self, u: float):
        """Valor del hiperparametro que corresponde a la posicion ``u`` en [0, 1] (escala log si aplica)."""
        i = min(len(self.opciones) - 1, int(float(np.clip(u, 0, 1)) * len(self.opciones)))
        return self.opciones[i]

    def niveles(self, n: int) -> list:
        """Hasta ``n`` opciones repartidas uniformemente entre las categorias, para Grid Search."""
        if n >= len(self.opciones):
            return list(self.opciones)
        idx = np.round(np.linspace(0, len(self.opciones) - 1, n)).astype(int)
        return [self.opciones[i] for i in sorted(set(idx))]

    def sugerir(self, trial, nombre):
        """Pide el valor a un ``trial`` de Optuna con el mismo rango y escala."""
        return trial.suggest_categorical(nombre, list(self.opciones))


# ---------------------------------------------------------------------------
# Espacio
# ---------------------------------------------------------------------------
@dataclass
class Espacio:
    """Conjunto ordenado de hiperparametros de un modelo.

    Attributes
    ----------
    parametros : dict
        ``nombre -> Real | Entero | Categorico``.
    condiciones : dict
        ``nombre -> (param_padre, valores)``: el parametro solo tiene efecto
        cuando ``param_padre`` toma uno de ``valores`` (p. ej. ``gamma`` solo con
        el nucleo RBF). Se usa para no evaluar dos veces la misma configuracion.
    """

    parametros: dict
    condiciones: dict = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Numero de hiperparametros del espacio."""
        return len(self.parametros)

    # -- utilidades ----------------------------------------------------------
    def canonico(self, params: dict) -> dict:
        """Elimina los parametros inactivos (sin efecto dada la configuracion)."""
        out = {}
        for n, v in params.items():
            cond = self.condiciones.get(n)
            if cond is not None and params.get(cond[0]) not in cond[1]:
                continue
            out[n] = v
        return out

    def clave(self, params: dict) -> str:
        """Clave hashable de una configuracion (redondeada), para memoizar."""
        c = self.canonico(params)
        partes = []
        for n in sorted(c):
            v = c[n]
            if isinstance(v, float):
                v = f"{v:.6g}"
            partes.append(f"{n}={v}")
        return "|".join(partes)

    # -- traducciones ----------------------------------------------------------
    def muestrear(self, rng: np.random.Generator) -> dict:
        """Una configuracion al azar (Random Search)."""
        return {n: p.desde_unitario(rng.random()) for n, p in self.parametros.items()}

    def decodificar(self, genoma) -> dict:
        """Traduce un individuo ``[0, 1]^d`` del algoritmo genetico."""
        return {n: p.desde_unitario(g) for (n, p), g in zip(self.parametros.items(), genoma)}

    def sugerir(self, trial) -> dict:
        """Configuracion propuesta por Optuna."""
        return {n: p.sugerir(trial, n) for n, p in self.parametros.items()}

    def centro(self) -> dict:
        """Configuracion central del espacio (se usa para estimar tiempos)."""
        return {n: p.desde_unitario(0.5) for n, p in self.parametros.items()}

    def rejilla(self, presupuesto: int) -> list:
        """Rejilla completa con a lo sumo ``presupuesto`` configuraciones distintas.

        Los niveles se reparten de forma voraz: en cada paso se agrega un nivel al
        parametro que menos tiene, mientras el producto no exceda el presupuesto.
        Despues se eliminan duplicados por parametros inactivos. Es la limitacion
        estructural de Grid Search que el capitulo 04 discute: su tamano crece como
        producto y no puede usar un presupuesto arbitrario.
        """
        nombres = list(self.parametros)
        maximos = {n: (len(p.opciones) if isinstance(p, Categorico) else presupuesto)
                   for n, p in self.parametros.items()}
        n_niv = {n: 1 for n in nombres}
        while True:
            candidatos = [n for n in nombres if n_niv[n] < maximos[n]]
            candidatos.sort(key=lambda n: (n_niv[n], nombres.index(n)))
            avanzo = False
            for n in candidatos:
                prueba = dict(n_niv, **{n: n_niv[n] + 1})
                if math.prod(prueba.values()) <= presupuesto:
                    n_niv = prueba
                    avanzo = True
                    break
            if not avanzo:
                break
        niveles = [self.parametros[n].niveles(n_niv[n]) for n in nombres]
        vistos, rejilla = set(), []
        for combo in itertools.product(*niveles):
            cfg = dict(zip(nombres, combo))
            k = self.clave(cfg)
            if k not in vistos:
                vistos.add(k)
                rejilla.append(cfg)
        return rejilla


# ---------------------------------------------------------------------------
# Espacios por modelo
# ---------------------------------------------------------------------------
_KNN = Espacio({
    # k entre 5 y 300 en escala log: con 8 % de positivos, k < 5 da puntajes con
    # a lo sumo 5 valores distintos (AUC muy escalonado); k > 300 promedia regiones
    # enteras del espacio y el modelo tiende a la tasa base.
    "n_vecinos": Entero(5, 300, log=True),
    "ponderacion": Categorico(("uniform", "distance")),
    # dimension del PCA previo: la distancia euclidea pierde contraste en ~600
    # dimensiones (maldicion de la dimensionalidad) y FAISS escala con d.
    "n_componentes": Entero(10, 80),
})

_ARBOL = {
    # profundidad 3-20; min_samples_leaf 10-2000 (log). Hojas de menos de 10 casos
    # con 8 % de positivos estiman la probabilidad con 0 o 1 positivo.
    "max_depth": Entero(3, 20),
    "min_samples_leaf": Entero(10, 2000, log=True),
    "max_features": Categorico(("sqrt", 0.3, 1.0)),
}

_BOSQUE = {
    # el AUC de un bosque se estabiliza pasados ~200-300 arboles; mas arboles casi
    # no reducen el error y el coste crece linealmente (Probst y Boulesteix, 2018)
    "n_estimators": Entero(100, 300),
    # con hojas >= 20 casos y ~80 mil filas por arbol hay a lo sumo ~4 mil hojas,
    # que se alcanzan con 12-16 niveles: mas profundidad casi no cambia el modelo
    # y en la GPU los niveles profundos son los mas caros
    "max_depth": Entero(6, 16),
    # hojas >= 20: acota la varianza de la probabilidad en cada hoja (8 % positivos)
    "min_samples_leaf": Entero(20, 1000, log=True),
    "max_features": Real(0.02, 0.30),
    # ~0.63 es la fraccion de filas unicas del bootstrap clasico; por encima de 0.7
    # los arboles se parecen mas entre si (menos diversidad) y el coste sube
    "max_samples": Real(0.30, 0.70),
}

_XGB = {
    "learning_rate": Real(0.01, 0.30, log=True),
    "max_depth": Entero(3, 10),
    "min_child_weight": Real(1.0, 100.0, log=True),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.2, 1.0),
    "reg_lambda": Real(1e-3, 100.0, log=True),
    "reg_alpha": Real(1e-3, 10.0, log=True),
}

_SVM_COND = {"gamma": ("nucleo", ("rbf_nystroem",))}

ESPACIOS = {
    "clasificacion": {
        "knn": _KNN,
        "naive_bayes": Espacio({
            # suavizado de varianza: fraccion de la mayor varianza sumada a todas.
            "var_smoothing": Real(1e-12, 1e-1, log=True),
        }),
        "logistica": Espacio({
            # C en [1e-4, 10] log: la rejilla de la linea base encontro el optimo en
            # el borde (C=0.01 con L1) y avisaba que habia que ampliar el rango.
            "C": Real(1e-4, 10.0, log=True),
            "penalizacion": Categorico(("l1", "l2")),
        }),
        "arbol": Espacio(dict(_ARBOL)),
        "random_forest": Espacio(dict(_BOSQUE)),
        "xgboost": Espacio(dict(_XGB)),
        "svm": Espacio({
            "C": Real(1e-4, 10.0, log=True),
            "nucleo": Categorico(("lineal", "rbf_nystroem")),
            # gamma alrededor de 1/p ~ 1.6e-3 para ~600 columnas estandarizadas
            "gamma": Real(1e-4, 1e-2, log=True),
        }, condiciones=dict(_SVM_COND)),
    },
    "regresion": {
        "knn": _KNN,
        "ridge": Espacio({"alpha": Real(1e-2, 1e5, log=True)}),
        # con un objetivo 0/1 (varianza 0.074) alpha > 0.1 anula todos los coeficientes
        "lasso": Espacio({"alpha": Real(1e-6, 1e-1, log=True)}),
        "arbol": Espacio(dict(_ARBOL)),
        "random_forest": Espacio(dict(_BOSQUE)),
        "xgboost": Espacio(dict(_XGB)),
        "svr": Espacio({
            "C": Real(1e-4, 10.0, log=True),
            "epsilon": Real(0.0, 0.3),
            "nucleo": Categorico(("lineal", "rbf_nystroem")),
            "gamma": Real(1e-4, 1e-2, log=True),
        }, condiciones=dict(_SVM_COND)),
    },
}

MODELOS = {t: list(e) for t, e in ESPACIOS.items()}
"""Los siete modelos de cada tarea, en el orden de la guia."""


def espacio(tarea: str, modelo: str) -> Espacio:
    """Devuelve el espacio de busqueda de ``modelo`` en ``tarea``."""
    return ESPACIOS[tarea][modelo]


def tabla_espacios():
    """DataFrame legible con todos los espacios (para el libro)."""
    import pandas as pd
    filas = []
    for tarea, modelos in ESPACIOS.items():
        for m, esp in modelos.items():
            for n, p in esp.parametros.items():
                if isinstance(p, Categorico):
                    rango, escala = ", ".join(map(str, p.opciones)), "categorica"
                else:
                    rango = f"[{p.bajo:g}, {p.alto:g}]"
                    escala = "log" if p.log else "lineal"
                cond = esp.condiciones.get(n)
                filas.append({"Tarea": tarea, "Modelo": m, "Hiperparametro": n,
                              "Rango": rango, "Escala": escala,
                              "Activo si": f"{cond[0]} en {cond[1]}" if cond else "siempre"})
    return pd.DataFrame(filas)
