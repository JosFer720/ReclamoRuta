# Verificación con modelos reales

Fecha: 13 de noviembre de 2026.

## Alcance y decisiones previas a test

Se descargaron los tres modelos por Git LFS y se comprobó su integridad. Se reprodujeron sus métricas sobre las 996 consultas de `validation_model.csv`: accuracy de 87.1486 % en el clásico, 87.6506 % en V1 y 92.0683 % en V2. Macro-F1: 0.864138, 0.876652 y 0.919783, respectivamente. Coinciden con `ENTRENAMIENTO.md`.

Se elige V2 para la demostración por su desempeño en esa validación, antes de consultar los resultados finales. La comparación de test no se usa para volver a elegir hiperparámetros, temperatura o umbral.

Se corrigió una incompatibilidad entre versiones de Transformers: el tokenizador de los artefactos reales se identifica como `BertTokenizer` y producía `token_type_ids`, que DistilBERT no acepta. Entrenamiento e inferencia desactivan ese campo explícitamente. Dos regresiones reprodujeron el error antes de corregirlo. Los pesos y manifiestos de los modelos permanecen intactos.

## Escenario económico

`configs/value_scenario.json` declara los parámetros estimados: clasificación manual de 30 segundos, aceptación de 5, corrección de 20 segundos adicionales y revisión completa de 35; Q40 por hora, detección humana de errores del 80 % y penalización estimada de Q25 por ruta equivocada no detectada.

Se excluyen costos de inferencia, operación, entrenamiento, energía y compra de equipo de este escenario local. Los resultados son estimaciones para un lote del tamaño indicado.

La calibración se ajustó exclusivamente en `validation_calibration` (498 filas). Los umbrales se eligieron por valor estimado en `validation_threshold` (539 filas), con la cuadrícula 0, 0.05, ..., 1 y desempate al umbral más alto. `evaluation_scenario_v1/freeze.json` registra hashes de modelos, datos, código fuente, rutas, escenario, calibraciones y decisiones antes de acceder al test. La plantilla `configs/decision.json` continúa pendiente; no representa este escenario.

| Modelo | Temperatura | Pérdida logarítmica antes → después, en calibración | Umbral |
|---|---:|---:|---:|
| Clásico | 0.502628 | 0.866756 → 0.472476 | 0.85 |
| V1 | 1.165337 | 0.510205 → 0.500804 | 0.75 |
| V2 | 1.223654 | 0.375193 → 0.353320 | 0.85 |

Mejorar la pérdida en el conjunto utilizado para ajustar la temperatura no es una evaluación independiente de calibración. La temperatura no cambia la categoría ganadora; modifica puntuaciones y, por tanto, qué casos superan el umbral. La curva de valor y cobertura en validación está en `CURVA_VALOR_ESCENARIO.svg`.

## Evaluación final con decisiones congeladas

Se evaluó el test oficial de 3,080 consultas y su vista única de 3,079, sin volver a ajustar los modelos o sus decisiones. Son dos vistas del mismo conjunto, no dos pruebas independientes.

| Modelo | Accuracy oficial | Macro-F1 oficial | Exactitud de ruta propuesta | Cobertura sobre el umbral | Valor estimado por lote |
|---|---:|---:|---:|---:|---:|
| Clásico | 87.86 % | 0.879113 | 95.84 % | 72.86 % | Q482.64 |
| V1 | 86.30 % | 0.862239 | 94.22 % | 76.62 % | Q458.84 |
| V2 | 91.98 % | 0.919965 | 97.34 % | 88.05 % | Q638.31 |

V1 quedó por debajo del clásico en este test; deep learning no mejora automáticamente el resultado. V2 superó a V1 en 5.68 puntos de accuracy y 0.057727 de macro-F1. Bootstrap pareado por grupo, 1,000 remuestreos y semilla 42: intervalo del 95 % de la diferencia de accuracy [4.58, 6.82] puntos y de macro-F1 [0.047433, 0.070282]. Cuantifica variación de la muestra, no variación entre semillas de entrenamiento ni desempeño en otras organizaciones.

En V2, 2,712 casos superaron el umbral y 368 fueron derivados a revisión obligatoria. Entre los que superaron el umbral hubo 82 errores de intención (3.02 %) y 16 errores de ruta (0.59 %). Superar el umbral no significa ausencia de error y todas las sugerencias requieren confirmación humana. La exactitud de ruta usa el mapa académico de siete rutas; BANKING77 no incluye departamentos bancarios reales.

Bajo los supuestos indicados, V2 representa 17.96 horas de ahorro estimado en el lote y Q638.31 netos después de la penalización por errores residuales, con los costos excluidos ya descritos. Su diferencia estimada frente a V1 es Q179.47 por estas 3,080 consultas, con intervalo de muestreo condicional [Q121.02, Q240.66]. No es un intervalo sobre ahorro comercial real y no incorpora incertidumbre en salario, tiempos, detección humana ni penalizaciones. La sensibilidad individual de tiempos, salario y volumen en ±20 % y ±50 % queda en el JSON de resultados.

En test único, las accuracies fueron 87.85 %, 86.29 % y 91.98 %; el valor estimado de V2 fue Q638.03 para 3,079 consultas. Los resultados completos, matrices, métricas por clase, calibración y sensibilidad están en `evaluation_scenario_v1/evaluation.json`.

## Interfaz y pruebas

La integración se ejecutó con `Streamlit AppTest`, el predictor real V2 y su calibración y decisión congeladas, en modo offline. Se probaron tres consultas sintéticas nuevas, ausentes de train, la confirmación sin cambios, una corrección que cambia de ruta y el rechazo de entrada vacía. No hubo excepciones y los hashes del artefacto permanecieron válidos. La evidencia agregada está en `INTEGRACION_UI_REAL.json`.

Una consulta nueva sobre entrega de tarjeta fue clasificada incorrectamente como débito no reconocido, con puntuación 0.605; quedó por debajo de 0.85 y se exigió revisión. Otra sobre una tarjeta robada identificó la intención de pérdida/robo, aunque también pidió revisión. Una entrada ambigua obtuvo 0.189 y pidió revisión. Estos casos ilustran funcionamiento y límites, no constituyen una medida de precisión en entradas nuevas.

Suite completa después de integrar los cambios: **454 pruebas aprobadas y una omitida**, porque Windows no permitió crear un enlace simbólico. La prueba contra los datos oficiales se habilitó con `BANKING77_RAW_DIR`. Se verificó también el servidor local de Streamlit mediante su endpoint de salud, con respuesta `ok`. Esto no equivale a despliegue público ni a ensayo humano.

## Latencia en CPU

Medición de 200 consultas de validación por modelo, batch 1, diez llamadas de calentamiento y dos hilos OMP/MKL. Los archivos `BENCHMARK_*_CPU.json` contienen equipo y entorno.

| Modelo | Mediana | Percentil 95 | Consultas por segundo |
|---|---:|---:|---:|
| Clásico | 7.64 ms | 11.01 ms | 124.58 |
| V1 | 22.21 ms | 30.61 ms | 43.51 |
| V2 | 20.54 ms | 29.96 ms | 46.38 |

Es una medición local sin concurrencia ni tráfico web. Los tres modelos se midieron secuencialmente en un mismo proceso: sus tiempos de arranque incluyen efectos de importaciones y caché diferentes y no deben compararse como arranques fríos independientes. Los tiempos de inferencia excluyen el calentamiento.

Entorno de verificación: Python 3.13.2, PyTorch 2.14.0+cu126 ejecutado en CPU, Transformers 4.56.2, scikit-learn 1.9.1 y Streamlit 1.49.0. Para cargar el pickle clásico se instaló en el entorno temporal la misma versión de scikit-learn del entrenamiento. Esta ejecución no repitió el entrenamiento del encoder.

## Seguridad y límites

Ship Safe 11.0.0 no encontró secretos en el código fuente revisado. Marcó tres patrones: dos corresponden a `model.eval()` de PyTorch, que no evalúa código recibido del usuario; el tercero es la carga de pickle del modelo clásico. Ese último es un riesgo real al cargar artefactos no confiables: solo usar el modelo propio obtenido del repositorio confiable. SHA-256 comprueba integridad, no autenticidad de un archivo y su manifiesto modificados juntos.

La puntuación automática fue 90/100, sin auditoría de dependencias; no equivale a certificar seguridad de producción. La aplicación no acepta archivos de modelo enviados por usuarios y no guarda el texto de sus consultas. Falta comprobar un despliegue público con HTTPS y límites de uso.

Para sustituir los supuestos por evidencia, consultar `GUIA_MEDICION_TIEMPOS.md`. Una calibración o un umbral alto tampoco garantiza reconocer consultas fuera del dominio bancario ni consultas en español.
