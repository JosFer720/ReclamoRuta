# Detección humana y costos de cómputo

La comparación conserva las decisiones de la evaluación final: umbral 0.85 para el clásico y V2, y 0.75 para V1. Se cambia únicamente la fracción de errores que una persona detecta y corrige: 50 %, 80 % y 100 %. Los modelos y umbrales permanecen fijos.

## Base de cálculo

El lote principal contiene las 3,079 consultas de `test_unique`. Se usan 30 segundos para clasificación manual, 5 para aceptar una sugerencia, 20 segundos adicionales para corregirla, 35 para revisión completa, Q40 por hora de trabajo y Q25 por ruta equivocada no detectada. Los parámetros están en `configs/value_scenario.json`.

Para incluir los equipos se calcula con un capital de **Q10,000 por equipo**, 36 meses de vida útil, 160 horas mensuales asignadas al equipo de entrenamiento y 10,000 consultas mensuales en el equipo de inferencia. Son las cantidades que definen esta comparación; el archivo `configs/compute_costs.json` permite cambiarlas. Los equipos de entrenamiento e inferencia se contabilizan por separado.

La tarifa aplicada es Q1.51/kWh, correspondiente a EEGSA, tarifa no social BTS, para noviembre de 2026 a enero de 2027. Fuente CNEE, tabla de tarifa no social:

https://cnee.gob.gt/wp-content/uploads/2026/10/Comunicado-CNEE-Ajustes-Tarifarios-1-noviembre-2026-al-31-enero-2027-1.pdf

La energía equivalente se calcula con referencias nominales de componentes: 65 W del Ryzen 5 7600 para inferencia CPU y 300 W de la RTX 5070 Ti para entrenamiento DL. TDP y TGP son parámetros de diseño; la lectura eléctrica del sistema completo requiere un medidor. La potencia adicional del resto del equipo puede incorporarse en los campos de potencia adicional del archivo de costos.

https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-5-7600.html

https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/

El tiempo de entrenamiento proviene de `fit_seconds` o `train_seconds` de cada artefacto. En el clásico se aplica el perfil CPU de 65 W como referencia de cálculo: su archivo de entrenamiento registra duración, pero no identifica el procesador usado. El servicio usa el tiempo medio batch 1 medido en el Ryzen 5 7600: 8.027 ms para el clásico, 22.982 ms para V1 y 21.559 ms para V2.

## Fórmulas

```text
tiempo_inferencia_h = consultas × media_batch1_ms / 3,600,000
energía_kWh = potencia_nominal_W / 1,000 × tiempo_h
costo_energía = energía_kWh × tarifa_Q_kWh

capital_inferencia_lote = capital_CPU / (meses_vida × consultas_mes) × consultas_lote
capital_entrenamiento = capital_equipo / (meses_vida × horas_uso_mes) × horas_entrenamiento

valor_recurrente = ahorro_horas × costo_hora
                  − costo_rutas_erróneas_no_detectadas
                  − energía_inferencia − capital_inferencia − otros_costos

valor_primer_lote = valor_recurrente − energía_entrenamiento − capital_entrenamiento
```

El entrenamiento se carga una vez al primer lote. Los lotes posteriores reutilizan el modelo. El arranque y calentamiento se mantienen separados de la media de inferencia. El parámetro de otros cargos mensuales permite agregar hosting y almacenamiento; esta ejecución local usa cero en ese campo. El costo de inferencia de este lote resulta de multiplicar su tamaño por la media medida en 200 solicitudes, no de cronometrar 3,079 solicitudes individuales.

## Costos incorporados

| Modelo | Entrenamiento, segundos | Energía de entrenamiento, Q | Capital de entrenamiento, Q | Energía de inferencia del lote, Q | Capital de inferencia del lote, Q |
|---|---:|---:|---:|---:|---:|
| Clásico | 1.058 | 0.000029 | 0.000510 | 0.000674 | 85.527778 |
| V1 | 295.938 | 0.037239 | 0.142717 | 0.001929 | 85.527778 |
| V2 | 227.169 | 0.028585 | 0.109553 | 0.001810 | 85.527778 |

El costo de la corrida seleccionada, sumando energía y capital, es Q0.000539 para el clásico, Q0.179956 para V1 y Q0.138139 para V2. V2 tarda menos que V1 en estas corridas porque V1 ejecutó 37 épocas y V2 ejecutó 10; no significa que el ajuste completo sea siempre más rápido.

El cuadro cobra una corrida seleccionada por modelo. El reporte de entrenamiento también registra nueve corridas DL, cuyos tiempos publicados suman 1,385 segundos. Aplicando los mismos parámetros a esos tiempos, el costo conjunto de búsqueda DL es Q0.842200: Q0.174279 de energía equivalente y Q0.667921 de capital. La suma usa los segundos redondeados de ese reporte y se mantiene separada del costo de servir un modelo ya entrenado.

## Comparación de detección humana

Valor neto del **primer lote de 3,079 consultas**, incluyendo energía, capital y una corrida de entrenamiento:

| Errores detectados y corregidos | Clásico | V1 | V2 | Diferencia V2 − V1 |
|---|---:|---:|---:|---:|
| 50 % | Q272.80 | Q169.12 | Q437.83 | Q268.71 |
| 80 % | Q396.84 | Q372.86 | Q552.37 | Q179.51 |
| 100 % | Q479.53 | Q508.68 | Q628.72 | Q120.04 |

V2 conserva el mayor valor en los tres niveles. La ventaja frente a V1 disminuye al aumentar la detección humana, porque la revisión corrige una mayor proporción de los errores de V1. Cuando la detección es del 100 %, la penalización por rutas equivocadas no detectadas llega a cero, pero corregir una sugerencia todavía consume tiempo.

V1 queda por debajo del clásico al detectar el 50 % y 80 % de los errores. Lo supera con detección del 100 %. Esa comparación muestra por qué conviene evaluar el flujo completo y no elegir un modelo únicamente por usar deep learning.

Con detección del 80 %, V2 produce Q552.50 por lote posterior, frente a Q373.04 de V1 y Q396.84 del clásico. La diferencia del primer lote frente al clásico es Q155.53. En V2, los componentes de ese primer lote son:

```text
Q718.033333 por horas liberadas
− Q80.000000 por riesgo residual de rutas incorrectas
− Q85.527778 por capital de inferencia
− Q0.001810 por energía de inferencia
− Q0.138139 por entrenamiento: energía y capital
= Q552.365607
```

Los importes expresan valor del tiempo y costo de errores dentro de la función; no son ventas. La curva está en `DETECCION_Y_VALOR.svg`. El JSON incluye también el test oficial de 3,080 consultas, separado de la vista única.

## Costo del equipo y volumen de consultas

La siguiente tabla mantiene V2 y detección del 80 %. Muestra valor neto por lote posterior de 3,079 consultas al distribuir el capital entre distintos volúmenes mensuales:

| Capital del equipo | 1,000 consultas/mes | 10,000 consultas/mes | 100,000 consultas/mes |
|---|---:|---:|---:|
| Q5,000 | Q210.39 | Q595.27 | Q633.76 |
| Q10,000 | −Q217.25 | Q552.50 | Q629.48 |
| Q15,000 | −Q644.89 | Q509.74 | Q625.20 |

Con bajo volumen, distribuir el capital entre pocas consultas puede consumir el beneficio e incluso volver negativo el valor. Con los demás parámetros fijos, el volumen y el costo de los errores influyen mucho más en el resultado que las pequeñas diferencias de energía de inferencia.

## Reproducción

El archivo `ANALISIS_ECONOMICO.json` contiene las tres tasas de detección, los componentes del costo, las diferencias entre modelos y nueve combinaciones de capital y volumen para cada modelo y tasa. Los hashes vinculan la ejecución con las decisiones, resultados, métricas de entrenamiento, benchmarks y parámetros utilizados.

```powershell
$env:PYTHONPATH = 'src;.'
python scripts/analyze_economics.py --evaluation-dir reports/evaluation_scenario_v1 --value-config configs/value_scenario.json --cost-config configs/compute_costs.json --artifacts artifacts/baseline artifacts/distilbert-v1 artifacts/distilbert-v2 --benchmarks reports/BENCHMARK_baseline_CPU.json reports/BENCHMARK_distilbert-v1_CPU.json reports/BENCHMARK_distilbert-v2_CPU.json --out data/cache/economia-nueva-ejecucion.json
```

El destino debe ser un archivo nuevo. Este análisis consume conteos fijos de la evaluación y no modifica modelos, temperaturas ni umbrales.
