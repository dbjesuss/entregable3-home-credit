"""Definicion de los 14 modelos (7 de clasificacion y 7 de regresion).

Todos se entrenan con el mismo objetivo, ``TARGET`` (1 = dificultades de pago), y
todos devuelven un **puntaje de riesgo** con :func:`puntuar`, sobre el que se
calcula el AUC-ROC, la metrica comun del proyecto:

* clasificadores -> ``predict_proba[:, 1]`` o, si no hay probabilidades
  (LinearSVC), ``decision_function``;
* regresores     -> ``predict``: con error cuadratico sobre un objetivo 0/1, el
  regresor estima ``E[Y|X] = P(Y=1|X)``.

Versiones escalables (Seccion 4 de la guia)
-------------------------------------------
Con ~200 mil filas y ~600 columnas, varias implementaciones "de libro" no terminan
en tiempo razonable. El estudio principal usa directamente la version optimizada y
el capitulo 05 la compara con la estandar:

* KNN       -> PCA + busqueda exacta de vecinos en la GPU (PyTorch) o, sin GPU,
  con FAISS (``IndexFlatL2``, multihilo), en lugar de la busqueda exhaustiva de
  scikit-learn sobre 600 dimensiones.
* Random Forest -> modo bosque aleatorio de XGBoost (``XGBRFClassifier``) en la
  GPU, en lugar de ``RandomForestClassifier`` en CPU (ver :func:`_bosque_xgb`).
* SVM / SVR -> ``LinearSVC`` / ``LinearSVR`` (lineal) o aproximacion de Nystroem al
  nucleo RBF + modelo lineal, en lugar del ``SVC`` con nucleo completo
  (O(n^2)-O(n^3)).
* XGBoost   -> ``tree_method='hist'``, parada temprana y GPU cuando existe.
* Lasso     -> descenso por coordenadas con matriz de Gram precalculada
  (O(n p^2) una vez, luego O(p^2) por iteracion).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn import __version__ as _VERSION_SK
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from . import balanceo as bal
from . import config

_SK = tuple(int(p) for p in _VERSION_SK.split(".")[:2])
USAR_L1_RATIO = _SK >= (1, 8)   # en scikit-learn 1.8 `penalty` quedo deprecado

MODELOS_UN_HILO = {"naive_bayes", "logistica", "arbol", "svm", "ridge", "lasso", "svr"}
"""Modelos cuyo ajuste usa un solo hilo: sus pliegues internos se paralelizan."""


# ---------------------------------------------------------------------------
# KNN con FAISS
# ---------------------------------------------------------------------------
class KNNFaiss(BaseEstimator):
    """K vecinos mas cercanos sobre una proyeccion PCA, con indice FAISS.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    n_vecinos : int
    ponderacion : {'uniform', 'distance'}
    n_componentes : int
        Dimension del PCA previo (ajustado solo con el entrenamiento).
    pesos_clase : bool
        Equivalente de ``class_weight='balanced'``: el voto de cada vecino se
        multiplica por el peso de su clase. Nota: como el factor es constante por
        clase, el puntaje resultante es una funcion monotona del no ponderado, de
        modo que el AUC no cambia; solo se desplaza el umbral.
    indice : {'plano', 'ivf'}
        ``'plano'`` = busqueda exacta; ``'ivf'`` = aproximada (celdas de Voronoi,
        ``nprobe`` celdas visitadas por consulta).
    nlist, nprobe : int
        Parametros del indice IVF.
    semilla : int
    """

    def __init__(self, tarea="clasificacion", n_vecinos=50, ponderacion="uniform",
                 n_componentes=40, pesos_clase=False, indice="plano", nlist=1024,
                 nprobe=16, semilla=config.SEMILLA, motor=None):
        self.tarea = tarea
        self.n_vecinos = n_vecinos
        self.ponderacion = ponderacion
        self.n_componentes = n_componentes
        self.pesos_clase = pesos_clase
        self.indice = indice
        self.nlist = nlist
        self.nprobe = nprobe
        self.semilla = semilla
        self.motor = motor

    # -- serializacion (los indices FAISS no se serializan con pickle) --------
    def __getstate__(self):
        estado = self.__dict__.copy()
        idx = estado.pop("indice_", None)
        if idx is not None:
            try:
                import faiss
                estado["_indice_bytes"] = faiss.serialize_index(idx)
            except ImportError:
                estado["indice_"] = idx
        return estado

    def __setstate__(self, estado):
        datos = estado.pop("_indice_bytes", None)
        self.__dict__.update(estado)
        if datos is not None:
            import faiss
            self.indice_ = faiss.deserialize_index(datos)

    def fit(self, X, y, sample_weight=None):
        """Ajusta el PCA con el entrenamiento y guarda las proyecciones y etiquetas para la busqueda de vecinos."""
        from sklearn.decomposition import PCA
        X = np.asarray(X, dtype=np.float32)
        d = int(min(self.n_componentes, X.shape[1], X.shape[0] - 1))
        self.pca_ = PCA(n_components=d, svd_solver="randomized",
                        random_state=self.semilla).fit(X)
        Z = np.ascontiguousarray(self.pca_.transform(X), dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.float32)
        n1 = float(self.y_.sum())
        n0 = float(len(self.y_) - n1)
        self.w_clase_ = np.array([len(self.y_) / (2 * max(n0, 1)),
                                  len(self.y_) / (2 * max(n1, 1))], dtype=np.float32)
        self.k_ = int(min(self.n_vecinos, len(self.y_)))
        motor = self.motor or config.KNN_MOTOR
        if motor == "torch" and self.indice == "plano":
            # busqueda exacta por fuerza bruta en la GPU: se guarda la matriz
            # proyectada y las distancias se calculan por bloques en `vecinos`
            self.Z_ = Z
            self.indice_ = None
            self.motor_ = "torch"
            return self
        try:
            import faiss
            faiss.omp_set_num_threads(config.NUCLEOS)
            if self.indice == "ivf":
                nlist = int(max(1, min(self.nlist, len(Z) // 39)))
                cuant = faiss.IndexFlatL2(d)
                idx = faiss.IndexIVFFlat(cuant, d, nlist)
                idx.train(Z)
                idx.nprobe = int(min(self.nprobe, nlist))
            else:
                idx = faiss.IndexFlatL2(d)
            idx.add(Z)
            self.indice_ = idx
            self.motor_ = "faiss"
        except ImportError:                       # respaldo sin FAISS
            from sklearn.neighbors import NearestNeighbors
            self.indice_ = NearestNeighbors(n_neighbors=self.k_, n_jobs=config.NUCLEOS).fit(Z)
            self.motor_ = "sklearn"
        return self

    def _vecinos_torch(self, X, bloque=1024):
        """Busqueda exacta en la GPU: ||q - z||^2 = ||q||^2 - 2 q.z + ||z||^2 por
        bloques de consultas, y los k menores con ``topk``. Es el mismo calculo que
        hace ``faiss.IndexFlatL2``, de modo que devuelve los mismos vecinos."""
        import torch
        disp = "cuda" if torch.cuda.is_available() else "cpu"
        Zg = torch.from_numpy(np.ascontiguousarray(self.Z_)).to(disp)
        z2 = (Zg * Zg).sum(1)
        Dl, Il = [], []
        with torch.no_grad():
            for i in range(0, len(X), bloque):
                Q = np.ascontiguousarray(self.pca_.transform(np.asarray(X[i:i + bloque], np.float32)),
                                         dtype=np.float32)
                Qg = torch.from_numpy(Q).to(disp)
                d2 = (Qg * Qg).sum(1, keepdim=True) - 2.0 * (Qg @ Zg.T) + z2[None, :]
                D, I = torch.topk(d2, self.k_, dim=1, largest=False, sorted=True)
                Dl.append(D.clamp_min_(0).cpu().numpy())
                Il.append(I.cpu().numpy())
        del Zg, z2
        if disp == "cuda":            # devuelve la memoria: XGBoost comparte la GPU
            torch.cuda.empty_cache()
        return np.vstack(Dl), np.vstack(Il)

    def vecinos(self, X, bloque=config.BLOQUE_PREDICCION):
        """Distancias al cuadrado e indices de los k vecinos, por bloques."""
        if self.motor_ == "torch":
            return self._vecinos_torch(X)
        Dl, Il = [], []
        for i in range(0, len(X), bloque):
            Z = np.ascontiguousarray(self.pca_.transform(np.asarray(X[i:i + bloque], np.float32)),
                                     dtype=np.float32)
            if self.motor_ == "faiss":
                D, I = self.indice_.search(Z, self.k_)
            else:
                D, I = self.indice_.kneighbors(Z, n_neighbors=self.k_)
                D = D ** 2
            Dl.append(D)
            Il.append(I)
        return np.vstack(Dl), np.vstack(Il)

    def puntaje(self, X):
        """Proporcion (ponderada) de positivos entre los k vecinos: probabilidad o prediccion de regresion."""
        D, I = self.vecinos(X)
        I = np.where(I < 0, 0, I)               # IVF puede devolver -1 si faltan vecinos
        yk = self.y_[I]
        if self.ponderacion == "distance":
            w = 1.0 / (np.sqrt(np.maximum(D, 0)) + 1e-6)
        else:
            w = np.ones_like(yk)
        if self.pesos_clase and self.tarea == "clasificacion":
            w = w * self.w_clase_[yk.astype(int)]
        return (w * yk).sum(1) / w.sum(1)

    def predict_proba(self, X):
        """Probabilidades de las dos clases, en el formato de scikit-learn."""
        p = self.puntaje(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Clase con umbral 0.5 en clasificacion; el puntaje continuo en regresion."""
        p = self.puntaje(X)
        return (p >= 0.5).astype(int) if self.tarea == "clasificacion" else p


# ---------------------------------------------------------------------------
# Construccion
# ---------------------------------------------------------------------------
def _logistica(params, balanceo, semilla):
    from sklearn.linear_model import LogisticRegression
    l1 = params["penalizacion"] == "l1"
    kw = dict(C=float(params["C"]), solver="saga" if l1 else "lbfgs",
              max_iter=3000, random_state=semilla,
              class_weight="balanced" if balanceo == "class_weight" else None)
    if USAR_L1_RATIO:
        kw["l1_ratio"] = 1.0 if l1 else 0.0
    else:
        kw["penalty"] = "l1" if l1 else "l2"
    return LogisticRegression(**kw)


def _svm(tarea, params, balanceo, semilla):
    from sklearn.pipeline import Pipeline
    from sklearn.kernel_approximation import Nystroem
    if tarea == "clasificacion":
        from sklearn.svm import LinearSVC
        lineal = LinearSVC(C=float(params["C"]), dual="auto", max_iter=5000,
                           random_state=semilla,
                           class_weight="balanced" if balanceo == "class_weight" else None)
    else:
        from sklearn.svm import LinearSVR
        # perdida epsilon-insensible estandar de SVR (dual obligatorio en liblinear)
        lineal = LinearSVR(C=float(params["C"]), epsilon=float(params["epsilon"]),
                           loss="epsilon_insensitive", dual=True, max_iter=5000,
                           random_state=semilla)
    if params.get("nucleo") == "rbf_nystroem":
        return Pipeline([("nystroem", Nystroem(kernel="rbf", gamma=float(params["gamma"]),
                                               n_components=config.NYSTROEM_COMPONENTES,
                                               random_state=semilla)),
                         ("lineal", lineal)])
    return lineal


def _bosque_xgb(tarea, params, cw, semilla, dispositivo, pos_neg, h0):
    """Random Forest con el modo bosque aleatorio de XGBoost (una sola ronda de
    boosting con ``n_estimators`` arboles en paralelo, ``learning_rate=1``).

    Equivalencias con el espacio de busqueda (definido en terminos de scikit-learn):

    * ``n_estimators`` -> numero de arboles del bosque;
    * ``max_depth``    -> igual;
    * ``max_features`` -> ``colsample_bynode``: fraccion de columnas candidatas en
      cada nodo, que es exactamente lo que hace ``max_features`` en scikit-learn;
    * ``max_samples``  -> ``subsample``: fraccion de filas por arbol (XGBoost
      muestrea sin reemplazo; scikit-learn usa bootstrap con reemplazo);
    * ``min_samples_leaf`` -> ``min_child_weight``: XGBoost limita la suma de
      hessianos de cada hoja, no el numero de filas. Con perdida cuadratica el
      hessiano vale 1 por fila (equivalencia exacta); con perdida logistica vale
      p(1-p), de modo que se multiplica por p0(1-p0), con p0 la tasa de positivos
      del entrenamiento.

    Los cortes se buscan con histogramas (``tree_method='hist'``, 256 bins), igual
    que en XGBoost; es la principal diferencia con el CART exacto de scikit-learn.
    """
    import xgboost as xgb
    kw = dict(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
              colsample_bynode=float(params["max_features"]),
              subsample=float(params["max_samples"]),
              min_child_weight=float(params["min_samples_leaf"]) * (h0 if tarea == "clasificacion" else 1.0),
              tree_method="hist", device=dispositivo, max_bin=256,
              random_state=semilla, n_jobs=config.NUCLEOS)
    if tarea == "clasificacion":
        return xgb.XGBRFClassifier(scale_pos_weight=(pos_neg if cw and pos_neg else 1.0), **kw)
    return xgb.XGBRFRegressor(objective="reg:squarederror", **kw)


def construir(tarea: str, modelo: str, params: dict, balanceo: str = "ninguno",
              semilla: int = config.SEMILLA, dispositivo: str | None = None,
              pos_neg: float | None = None, h0: float = 0.0807 * (1 - 0.0807)):
    """Construye el estimador (sin ajustar) de ``modelo`` con ``params``.

    Parameters
    ----------
    tarea : {'clasificacion', 'regresion'}
    modelo : str
        Una clave de :data:`src.espacios.MODELOS`.
    params : dict
        Hiperparametros en la nomenclatura de :mod:`src.espacios`.
    balanceo : str
        Solo importa para ``class_weight``: activa la ponderacion propia de cada
        modelo (``class_weight='balanced'``, ``scale_pos_weight`` en XGBoost,
        votos ponderados en KNN; Naive Bayes la recibe como ``sample_weight`` en
        :func:`entrenar`).
    semilla : int
    dispositivo : {'cpu', 'cuda'}, optional
        Dispositivo de XGBoost. Por defecto ``config.DISPOSITIVO_XGB``.
    pos_neg : float, optional
        Razon negativos/positivos, para ``scale_pos_weight``.

    Returns
    -------
    estimador scikit-learn
    """
    cw = balanceo == "class_weight"
    clf = tarea == "clasificacion"
    dispositivo = dispositivo or config.DISPOSITIVO_XGB

    if modelo == "knn":
        return KNNFaiss(tarea=tarea, n_vecinos=int(params["n_vecinos"]),
                        ponderacion=params["ponderacion"],
                        n_componentes=int(params["n_componentes"]),
                        pesos_clase=cw, semilla=semilla)
    if modelo == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=float(params["var_smoothing"]))
    if modelo == "logistica":
        return _logistica(params, balanceo, semilla)
    if modelo == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(params["alpha"]), random_state=semilla)
    if modelo == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=float(params["alpha"]), max_iter=5000, precompute=True,
                     random_state=semilla)
    if modelo == "arbol":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        kw = dict(max_depth=int(params["max_depth"]),
                  min_samples_leaf=int(params["min_samples_leaf"]),
                  max_features=params["max_features"], random_state=semilla)
        return (DecisionTreeClassifier(class_weight="balanced" if cw else None, **kw)
                if clf else DecisionTreeRegressor(**kw))
    if modelo == "random_forest":
        if config.RF_MOTOR == "xgbrf":
            return _bosque_xgb(tarea, params, cw, semilla, dispositivo, pos_neg, h0)
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        kw = dict(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                  min_samples_leaf=int(params["min_samples_leaf"]),
                  max_features=float(params["max_features"]),
                  max_samples=float(params["max_samples"]),
                  n_jobs=config.NUCLEOS, random_state=semilla)
        return (RandomForestClassifier(class_weight="balanced" if cw else None, **kw)
                if clf else RandomForestRegressor(**kw))
    if modelo == "xgboost":
        import xgboost as xgb
        kw = dict(n_estimators=2000, early_stopping_rounds=50, tree_method="hist",
                  device=dispositivo, max_bin=256, random_state=semilla,
                  n_jobs=config.NUCLEOS,
                  learning_rate=float(params["learning_rate"]),
                  max_depth=int(params["max_depth"]),
                  min_child_weight=float(params["min_child_weight"]),
                  subsample=float(params["subsample"]),
                  colsample_bytree=float(params["colsample_bytree"]),
                  reg_lambda=float(params["reg_lambda"]),
                  reg_alpha=float(params["reg_alpha"]))
        if clf:
            return xgb.XGBClassifier(eval_metric="auc",
                                     scale_pos_weight=(pos_neg if cw and pos_neg else 1.0), **kw)
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **kw)
    if modelo in ("svm", "svr"):
        return _svm(tarea, params, balanceo, semilla)
    raise ValueError(f"Modelo desconocido: {tarea}/{modelo}")


# ---------------------------------------------------------------------------
# Entrenamiento y puntaje
# ---------------------------------------------------------------------------
def separar_parada_temprana(X, y, tarea, semilla=config.SEMILLA):
    """Separa del entrenamiento *real* el conjunto de parada temprana de XGBoost.

    Returns
    -------
    (X_ajuste, y_ajuste, X_es, y_es)
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    i_aj, i_es = train_test_split(idx, test_size=config.PROPORCION_PARADA_TEMPRANA,
                                  random_state=semilla,
                                  stratify=y if tarea == "clasificacion" else None)
    i_aj.sort()
    i_es.sort()
    return np.asarray(X[i_aj]), y[i_aj], np.asarray(X[i_es]), y[i_es]


def entrenar(tarea: str, modelo: str, params: dict, balanceo: str,
             X: np.ndarray, y: np.ndarray, semilla: int = config.SEMILLA,
             dispositivo: str | None = None, X_balanceado=None, es=None):
    """Ajusta un modelo aplicando el balanceo **solo** a su entrenamiento.

    Orden de operaciones (sin fuga):

    1. XGBoost separa primero ``PROPORCION_PARADA_TEMPRANA`` del entrenamiento
       *real* como conjunto de parada temprana (estratificado). Nunca se usa la
       validacion para parar.
    2. Se aplica el balanceo (SMOTE/ADASYN) al resto del entrenamiento.
    3. Se ajusta el modelo; ``class_weight`` se traduce a la forma que admite cada
       algoritmo.

    Parameters
    ----------
    tarea, modelo, params, balanceo, semilla, dispositivo
        Ver :func:`construir`.
    X, y : ndarray
        Entrenamiento preprocesado, sin balancear. Si ``es`` se entrega, ``X`` ya
        es la parte de ajuste (sin el conjunto de parada temprana).
    X_balanceado : tuple (ndarray, ndarray), optional
        ``(X, y)`` ya balanceados (cache del llamador).
    es : tuple (ndarray, ndarray), optional
        Conjunto de parada temprana ya separado (cache del llamador).

    Returns
    -------
    (estimador, dict)
        Estimador ajustado e informacion del ajuste (tiempos, filas, convergencia,
        iteracion optima de XGBoost).
    """
    info = {"filas_ajuste": 0, "t_balanceo": 0.0, "t_ajuste": 0.0,
            "convergio": True, "mejor_iteracion": None}
    y = np.asarray(y)
    X_es = y_es = None
    if modelo == "xgboost":
        if es is None:
            X, y, X_es, y_es = separar_parada_temprana(X, y, tarea, semilla)
            X_balanceado = None
        else:
            X_es, y_es = es

    t0 = time.perf_counter()
    if X_balanceado is not None and balanceo in bal.REMUESTREO:
        Xb, yb = X_balanceado
    elif tarea == "clasificacion":
        Xb, yb = bal.aplicar(balanceo, X, y, semilla)
    else:
        Xb, yb = X, y
    info["t_balanceo"] = time.perf_counter() - t0
    info["filas_ajuste"] = int(len(yb))

    n1 = float(np.sum(yb))
    p0 = n1 / max(len(yb), 1)
    if balanceo == "class_weight" and tarea == "clasificacion":
        p0 = 0.5                     # con pesos balanceados ambas clases pesan igual
    est = construir(tarea, modelo, params, balanceo, semilla, dispositivo,
                    pos_neg=(len(yb) - n1) / max(n1, 1.0), h0=p0 * (1 - p0))
    kw = {}
    if modelo == "naive_bayes" and balanceo == "class_weight":
        kw["sample_weight"] = bal.pesos_balanceados(yb)
    if modelo == "xgboost":
        kw["eval_set"] = [(X_es, y_es)]
        kw["verbose"] = False
    yb_fit = yb.astype(np.float32) if tarea == "regresion" else yb

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        est.fit(Xb, yb_fit, **kw)
    info["t_ajuste"] = time.perf_counter() - t0
    info["convergio"] = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
    if modelo == "xgboost":
        info["mejor_iteracion"] = int(getattr(est, "best_iteration", -1))
    return est, info


def puntuar(est, X: np.ndarray, tarea: str,
            bloque: int = config.BLOQUE_PREDICCION) -> np.ndarray:
    """Puntaje de riesgo por bloques (mayor = mas riesgo).

    * clasificacion: ``predict_proba[:, 1]`` si existe; si no, ``decision_function``;
    * regresion: ``predict``.
    """
    if isinstance(est, KNNFaiss):
        return est.puntaje(X).astype(np.float64)
    if tarea == "regresion":
        f = est.predict
    elif hasattr(est, "predict_proba"):
        def f(Z):
            """Probabilidad de la clase positiva (funcion que se pasa al explicador)."""
            return est.predict_proba(Z)[:, 1]
    else:
        f = est.decision_function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        partes = [np.asarray(f(X[i:i + bloque]), dtype=np.float64)
                  for i in range(0, len(X), bloque)]
    return np.concatenate(partes) if partes else np.empty(0)


def es_probabilidad(tarea: str, modelo: str) -> bool:
    """True si el puntaje del modelo es una probabilidad en [0, 1]."""
    return tarea == "clasificacion" and modelo != "svm"
