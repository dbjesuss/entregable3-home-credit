# Entregable 3: modelos de Machine Learning para el riesgo de crédito de Home Credit

**Integrantes:** Jesús David Barrios Valdes, Samuel David Chamorro Solorzano y Jacobo Londoño Baquero.  
**Curso:** Machine Learning, Pregrado en Ciencia de Datos.

Este libro reúne el tercer entregable del proyecto del curso. Sobre el mismo problema de las entregas
anteriores (predecir si un solicitante de Home Credit tendrá dificultades de pago, `TARGET` = 1) se
entrenan, optimizan y comparan **140 modelos** con un pipeline reproducible:

| Tarea | Modelos | Balanceo | Optimización | Total |
|---|---|---|---|---|
| Clasificación | KNN, Naive Bayes, Logística L1/L2, Árbol, Random Forest, XGBoost, SVM | sin balanceo, SMOTE, ADASYN, `class_weight` | Grid, Random, Bayesiana (Optuna), Genética (DEAP) | 7 × 4 × 4 = 112 |
| Regresión | KNN, Ridge, Lasso, Árbol, Random Forest, XGBoost, SVR | no aplica | las mismas 4 | 7 × 4 = 28 |

**Decisión de diseño.** Todo el proyecto usa la misma variable objetivo, `TARGET`, y la misma métrica
principal, el **AUC-ROC** (la métrica oficial del problema en Kaggle). Los regresores se entrenan sobre
`TARGET` con error cuadrático, de modo que estiman la probabilidad de incumplimiento y su predicción
funciona como puntaje de riesgo. Así los 14 algoritmos se evalúan en una sola escala; las métricas propias
de regresión (RMSE, MAE, análisis de residuos) se reportan además, como exige la guía. El capítulo 03
desarrolla las consecuencias teóricas de esta decisión.

## Estructura

| Capítulo | Contenido | Secciones de la guía |
|---|---|---|
| 01 Datos, EDA y línea base | EDA en tres fases, embudo de selección, logística de referencia y traspaso de los datos | 7.1 |
| 02 Clasificación | Estudio combinatorio de 112 modelos con validación cruzada anidada | 2, 3.2, 5.1.2, 7.2 |
| 03 Regresión | Estudio de 28 modelos sobre `TARGET` | 2, 3.2, 7.2 |
| 04 Optimización de hiperparámetros | Comparación de los 4 métodos, convergencia, TPE, genético, Successive Halving | 3 |
| 05 Optimización computacional | Complejidad teórica frente a tiempos medidos, perfilamiento, paralelismo, GPU | 4 |
| 06 Evaluación de clasificación | Métricas, matrices de confusión, ROC, balanceo, calibración | 5.1 |
| 07 Evaluación de regresión | RMSE, MAE, ajuste en tres conjuntos, residuos (White, BDS, Ljung-Box) | 5.2 |
| 08 Interpretabilidad | SHAP global y local, SHAP frente a LIME en XGBoost | 5.3 |
| 09 Robustez y comparación estadística | Semillas, Friedman, Nemenyi, CD, DeLong, BCa, MCS, GW, CW, DM, SPA y Reality Check, tamaños del efecto | 5.4, 6 |
| 10 Conclusiones | Síntesis | 9 |
| Anexo A Ejecución del estudio | Organización de la ejecución, estimación frente a tiempo real, fallos y reparaciones | 4, 7.3 |

## Reproducibilidad (Sección 7.3)

* **Entorno fijado:** `requirements.txt` con versiones exactas (Python 3.10 o 3.12; el estudio se ejecutó
  con Python 3.12 en Windows, y las versiones de cada librería quedan registradas en cada unidad).
* **Semilla única:** `SEMILLA = 42` en `src/config.py`, propagada a numpy, scikit-learn, imbalanced-learn,
  XGBoost, FAISS, Optuna y DEAP. Es la misma de la línea base, de modo que la partición de prueba
  coincide.
* **Registro de experimentos:** cada una de las 700 unidades de trabajo (140 combinaciones × 5 pliegues
  externos) se guarda en `resultados/corridas/` con hiperparámetros, métricas del bucle externo, tiempos,
  historial de la optimización, versiones de las librerías y semilla. `resultados/tabla_maestra.csv` y
  `.parquet` consolidan una fila por combinación.
* **Código modular:** el paquete `src/` separa configuración (`config`), datos (`datos`),
  preprocesamiento (`preprocesamiento`), espacios de búsqueda (`espacios`), modelos (`modelos`),
  balanceo (`balanceo`), optimizadores (`optimizadores`), motor del estudio (`experimento`), métricas
  (`metricas`), diagnóstico de residuos (`diagnostico`), modelos finales (`finales`), pruebas estadísticas
  (`estadistica`) y herramientas de complejidad (`complejidad`). Todas las funciones reutilizadas tienen
  docstrings estilo NumPy.
* **Sin rutas absolutas:** la raíz del libro se localiza sola; los datos de Kaggle se buscan
  automáticamente o con la variable de entorno `HOME_CREDIT_DIR`.

## Cómo se ejecuta

1. Instalar el entorno: `pip install -r requirements.txt` (Python 3.10 o 3.12) y, para el KNN en la
   GPU, PyTorch con CUDA (opcional: sin GPU se usa FAISS en la CPU, con los mismos vecinos).
2. Comprobar el entorno: `python verificar_entorno.py` (versiones, GPU, espacio en disco).
3. Ejecutar **completo** el capítulo 01 en un kernel limpio. Su última sección guarda el conjunto de
   modelado en `datos/`.
4. Ejecutar los capítulos 02 y 03. Son los que más tiempo toman. **Se pueden interrumpir y reanudar**:
   cada unidad terminada queda en disco y al volver a ejecutar la celda se continúa donde quedó. Un error
   en una unidad se registra en `resultados/errores.csv` y no detiene el estudio.
5. Ejecutar los capítulos 04 a 09 en orden. Leen los resultados desde disco y sus experimentos propios
   también son reanudables.
6. Construir el libro: `jupyter-book build .` (los notebooks no se re-ejecutan al construir).

En la práctica, el estudio de los capítulos 02 y 03 suma cientos de horas, y se ejecutó fuera de Jupyter
con el mismo código: `correr_estudio.py --grupo <gpu|cpu|...>`, lanzado en varias ventanas a la vez,
guarda cada unidad en disco; después, `correr_capitulos.py` repara las
evaluaciones fallidas, ejecuta los capítulos 01 a 09 de arriba abajo en un kernel nuevo y compila el libro.
Como las unidades ya están en disco, los capítulos 02 y 03 las encuentran hechas y muestran los resultados
que produce su propio código. El anexo A documenta esa ejecución.

Espacio en disco aproximado: ~10 GB para la caché de particiones (`cache/`), más los resultados.
