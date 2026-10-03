# Conclusiones

Este capítulo resume lo que los 140 modelos permiten afirmar sobre el problema de Home Credit. Cada
apartado indica el capítulo del que salen las cifras; todas se refieren a la variable `TARGET` y, salvo
que se diga otra cosa, al AUC-ROC sobre la partición de prueba (61,503 solicitudes que ningún modelo
vio durante el estudio) o al AUC externo medio de la validación cruzada anidada 5 × 3.

## 1. ¿Qué modelo ordena mejor el riesgo?

*Fuente: capítulos 06 y 09.* **XGBoost**, sin ambigüedad. Su mejor combinación (sin balanceo,
optimizador genético) alcanza un AUC de **0.7754** en la prueba, con un intervalo de confianza del
95 % BCa de [0.7685, 0.7816], y de 0.7721 ± 0.0038 en la validación cruzada. Supera al segundo y al
tercer modelo, SVM (0.7595) y la logística (0.7597), por unas 0.016 de AUC, con p de Holm < 0.0001
según DeLong y un tamaño de efecto grande (delta de Cliff ≈ 0.58 sobre los bloques), y es el primero
del diagrama de diferencia crítica con un rango medio de 1.04 entre 7 modelos.

Frente a la línea base logística del entregable anterior (AUC 0.7603 sobre la misma prueba), XGBoost
gana **+0.0151** con un intervalo [0.0119, 0.0179] que excluye el cero. La logística optimizada de
este entregable no mejora a la línea base (0.7597): la búsqueda de hiperparámetros encontró la misma
región (penalización L1 con C entre 0.013 y 0.018, frente al C = 0.01 de la línea base), lo que
confirma que la línea base ya estaba bien ajustada y que la ganancia viene del cambio de familia de
modelos, no de la optimización.

El orden completo en la prueba es XGBoost (0.775), logística y SVM (0.760), Random Forest (0.754),
árbol (0.702), KNN (0.698) y Naive Bayes (0.648). Entre SVM, logística y Random Forest, y entre árbol
y KNN, las diferencias no son significativas (capítulo 09, sección 3).

## 2. ¿Qué aportan el balanceo y el umbral?

*Fuente: capítulo 06, secciones 5 y 6.* **El balanceo no mejora el ordenamiento; con sobremuestreo lo
empeora.** SMOTE y ADASYN bajan el AUC en seis de las siete familias, entre 0.013 (logística y SVM) y
0.048 (Naive Bayes), y solo en XGBoost el efecto es despreciable (−0.001). `class_weight` deja el AUC
prácticamente igual en todas las familias. La predicción teórica de invariancia se cumple: en Naive
Bayes y KNN la diferencia de AUC con `class_weight` no supera 0.0011 y no es significativa.

Lo que sí importa para el negocio es el **umbral**. Elegido por coste en la validación interna, lleva a
todas las técnicas a puntos de operación parecidos: el XGBoost final detecta el 66.7 % de los
incumplidores en la prueba (recall) con una precisión del 18.7 %. Balancear no aporta recall
adicional que no se obtenga moviendo el umbral.

El balanceo sí tiene un **coste en calibración**: la logística y Random Forest con `class_weight`
llegan con un ECE de 0.34 y 0.27, porque entrenaron con las clases igualadas. Naive Bayes llega
descalibrado por su propio supuesto de independencia (ECE 0.34). La recalibración con predicciones
fuera de muestra lo corrige (ECE de 0.001 a 0.006); la regresión isotónica es la opción segura, porque
Platt empeora a los modelos que ya estaban calibrados (XGBoost pasa de 0.004 a 0.021).

## 3. ¿Qué método de optimización conviene y por qué?

*Fuente: capítulo 04.* Con 20 evaluaciones por optimizador, **el método no cambia el resultado**: las
medias de AUC externo por método difieren en unas 0.002, por debajo de la variabilidad entre pliegues,
y en el AUC externo el orden de los métodos incluso se invierte (la rejilla tiene el mejor rango medio).
Las diferencias están en la eficiencia de la búsqueda. Por área bajo la curva de brecha al mejor
conocido, la búsqueda aleatoria y la genética son las más eficientes (ABCB 0.0048), la bayesiana es la
que termina más cerca del óptimo (brecha final 0.0009) y la más estable entre semillas (desviación
0.0002 en el estudio de convergencia), y la rejilla es claramente la peor (ABCB 0.0136; efecto pequeño
a mediano frente a las otras). Las diferencias entre los tres métodos no exhaustivos son de tamaño
insignificante (delta de Cliff ≤ 0.04). Por minuto de cómputo, la búsqueda aleatoria es tan buena como
cualquiera.

El genético pierde diversidad pronto (por debajo de 0.1 en dos de tres semillas) sin que eso produzca
un estancamiento perjudicial. Successive Halving no compensó: con el coste de 4 entrenamientos
completos obtuvo 0.7662, peor que Random Search con el mismo coste (0.7677), porque con ~4,900 filas el
orden de las configuraciones de XGBoost no se conserva. Por último, el AUC interno del ganador resultó
ligeramente *menor* que el externo (−0.0015 a −0.003): los modelos internos aprenden con dos tercios de
los datos, y ese efecto supera al sesgo de selección.

**Recomendación:** búsqueda aleatoria cuando el presupuesto es pequeño y el coste por evaluación
variable; bayesiana cuando importa la estabilidad del resultado final; rejilla, nunca en espacios de
más de dos o tres dimensiones.

## 4. ¿Qué optimizaciones computacionales fueron decisivas?

*Fuente: capítulo 05 y anexo A.* Las decisivas fueron **cambios de algoritmo**, no de hardware:

* **Parada temprana en XGBoost**: 3.8 veces más rápida y +0.011 de AUC.
* **`hist` en lugar de `exact`**: 3.0 veces más rápido, mismo AUC.
* **Nystroem + modelo lineal en lugar de SVC con núcleo RBF**: 54 veces más rápido con 16,000 filas;
  el SVC exacto escala con exponente 2.2 y tardaría ~4 horas por ajuste con los datos completos.
* **PCA + búsqueda vectorizada en KNN**: entre 7 y 13 veces más rápida por consulta y, además, +0.048 de
  AUC, porque en ~600 dimensiones las distancias pierden significado.
* **La matriz de Gram en Lasso** (37 veces) y **`partial_fit` en Naive Bayes** (−58 % de memoria).

La GPU aportó 2.0 veces en XGBoost y 2.2 en el bosque aleatorio: útil, pero su memoria de 6 GB fue
también el origen de los fallos del estudio. El perfilamiento mostró que el balanceo es una fracción
modesta de cada evaluación (~9 % con SMOTE), pero que calcularlo una sola vez por partición y
compartirlo entre los cuatro optimizadores evita 79 de cada 80 repeticiones. Dos "optimizaciones" de la
guía no lo fueron en este problema: SAGA es 38 a 150 veces más lento que lbfgs y Cholesky con datos
densos y p moderado, y es la causa de que la logística L1 consumiera casi una quinta parte de todo el
cómputo del estudio.

## 5. Clasificadores y regresores sobre el mismo objetivo

*Fuente: capítulos 03, 07 y 09 (sección 5).* La equivalencia teórica se cumple. Entrenados sobre
`TARGET` con error cuadrático, el árbol y KNN regresores dan **exactamente** el mismo AUC que sus
contrapartes de clasificación (0.7017 y 0.6975 en la prueba), y Random Forest es indistinguible
(−0.0002, p = 0.85). En XGBoost el clasificador supera al regresor por 0.0017 (p = 0.058), en el
límite de la significación: la log-verosimilitud se ajusta algo mejor a un objetivo binario. Lasso
(0.7604) iguala en la prueba a la logística. La excepción es **SVR**: su pérdida ε-insensible produce
predicciones prácticamente constantes (alrededor de 0.30), R² negativo y un RMSE peor que predecir la
tasa base, aunque conserva algo de orden (AUC 0.666).

Las métricas propias de regresión confirman que, con un objetivo binario, **el AUC es el criterio que
importa**. El MAE premia predecir siempre 0 (ningún modelo lo supera); el R² del mejor modelo es 0.10,
lo esperable para un evento del 8 %, y no contradice un AUC de 0.77; y las pruebas de White y
Jarque-Bera rechazan por construcción, porque el residuo solo toma dos valores por observación. Las
pruebas informativas, BDS y Ljung-Box, no encuentran dependencia en el orden de `SK_ID_CURR`. En el
MCS sobreviven solo dos regresores, ambos XGBoost, y Diebold-Mariano confirma su ventaja sobre Lasso y
Ridge con un efecto real pero pequeño por observación. El SPA de Hansen y el Reality Check de White
descartan que esa ventaja sea un artefacto de haber probado 28 modelos (p < 0.001 frente al mejor
modelo lineal), y el StepM muestra que solo los cuatro XGBoost y, por muy poco, un Random Forest
superan de forma genuina al modelo lineal. El ajuste en los tres conjuntos confirma además que,
aunque XGBoost, Random Forest y KNN memorizan parte del entrenamiento, su AUC de validación y de
prueba coinciden, de modo que las estimaciones fuera de muestra son fiables.

## 6. ¿Qué explica el riesgo?

*Fuente: capítulo 08.* El puntaje externo `EXT_SOURCE_2` es, con diferencia, la variable más
influyente, con un efecto monótono: puntajes bajos elevan el riesgo. Le siguen variables de historial
construidas en la Fase 3 —deuda relativa y mora máxima en la central de riesgo, pagos tardíos de cuotas
anteriores, solicitudes rechazadas en el pasado— y variables de capacidad de pago y perfil (anualidad,
antigüedad laboral, edad, sexo, estado civil, educación). Es el mismo cuadro que trazó el EDA, salvo
por una ausencia: `EXT_SOURCE_1` y `EXT_SOURCE_3`, que el embudo de la Fase 3.5 excluyó por cobertura
insuficiente (43.6 % y 80.2 %, por debajo del umbral de 85 %).

Los casos locales muestran el límite del modelo. El incumplidor con el puntaje más alto reunía casi
todas las señales de riesgo a la vez; el incumplidor con el puntaje más bajo tenía el perfil de un
cliente seguro en todas sus variables, de modo que ningún modelo sobre estos datos podía anticiparlo.
LIME coincidió poco con SHAP (1-2 variables comunes entre las 10 principales) y su modelo local explica
solo el 20 % del comportamiento en el vecindario: para modelos de árboles, SHAP es la explicación a
usar.

## 7. Limitaciones

* **Optimismo residual del EDA.** El conjunto de variables se diseñó en el EDA mirando `TARGET` sobre
  todas las filas de entrenamiento. La validación anidada protege la selección de hiperparámetros, pero
  no esa etapa previa.
* **Desplazamiento entre particiones.** Tras el embudo de estabilidad, la validación adversarial entre
  `application_train` y `application_test` sigue en un AUC de 0.82 (capítulo 01). La evaluación sobre la
  prueba interna, extraída de `train`, puede ser optimista respecto del comportamiento sobre la
  población de `test`.
* **Variables excluidas.** La exclusión por cobertura de `EXT_SOURCE_1`, `EXT_SOURCE_3` y de toda la
  tabla `credit_card_balance` protege la comparabilidad entre particiones, pero probablemente cuesta
  capacidad predictiva: los modelos basados en árboles manejan los valores faltantes de forma nativa.
* **Bloques no independientes.** Los 50 bloques de Friedman y Nemenyi se forman dividiendo los 5
  pliegues externos, y los de un mismo pliegue comparten los modelos entrenados; la inferencia es algo
  liberal y por eso se complementa con DeLong sobre la prueba.
* **Sin orden temporal.** Home Credit no registra fechas, de modo que BDS, Ljung-Box, Giacomini-White y
  Diebold-Mariano se aplican sobre el orden de `SK_ID_CURR`, no sobre un orden temporal real.
* **Presupuesto de búsqueda.** 20 evaluaciones por optimizador bastan para comparar familias, pero no
  para agotar el potencial de XGBoost, que en el estudio de convergencia siguió ganando milésimas hasta
  las 30-40 evaluaciones.
* **Hardware.** Algunas configuraciones de Random Forest no cupieron en los 6 GB de la GPU y se
  evaluaron después en la CPU; ninguna resultó ganadora (anexo A).

## 8. Referencia externa y trabajo futuro

Como referencia, las soluciones ganadoras de la competencia de Home Credit en Kaggle alcanzaron un AUC
cercano a 0.80 en su tabla privada, con cientos de variables construidas a mano y conjuntos de modelos.
La comparación no es directa: esa cifra se mide sobre `application_test`, con otras variables y otro
protocolo. Aun así, sitúa el resultado de este proyecto (0.775 en una prueba interna) como competitivo
para un modelo único sin ingeniería de variables específica, y señala dónde está el margen de mejora:

1. **Recuperar información excluida**, en particular `EXT_SOURCE_1`, `EXT_SOURCE_3` y
   `credit_card_balance`, tratando el desplazamiento entre particiones de otra forma (por ejemplo,
   indicadores de disponibilidad en lugar de exclusión).
2. **Ingeniería de variables** sobre los puntajes externos y el historial: razones, interacciones como
   `EXT_SOURCE_2 × EXT_SOURCE_3` y ventanas temporales.
3. **Otros modelos de boosting y combinaciones**: LightGBM y CatBoost, y un *stacking* sobre las
   predicciones fuera de muestra que el estudio ya guarda para los 140 modelos.
4. **Más presupuesto de búsqueda para XGBoost**, con búsqueda aleatoria o bayesiana y parada temprana.
