# Entrenamiento — baseline, DistilBERT V1 y V2

Fecha: 13 de noviembre de 2026.

Todas las cifras de este reporte son de `validation_model.csv` (996 filas). Ningún entrenamiento leyó el test ni las partes de calibración o umbral. No son resultados finales: la evaluación en test se realiza después del congelamiento.

## Resumen

| Modelo | Versión | Accuracy | Macro-F1 | Tiempo de entrenamiento | Tamaño |
|---|---|---|---|---|---|
| Clásico (TF-IDF + regresión logística) | `baseline-tfidf` | 0.8715 | 0.8641 | 1.1 s en CPU | 5.5 MB |
| DL V1 (encoder congelado) | `distilbert-v1` | 0.8765 | 0.8767 | 296 s en GPU | 256 MB |
| DL V2 (fine-tuning completo) | `distilbert-v2` | 0.9207 | 0.9198 | 227 s en GPU | 256 MB |

- V2 supera al clásico por 4.9 puntos de accuracy y 5.6 de macro-F1.
- V1 queda prácticamente empatado con el clásico: 0.5 puntos más de accuracy y 1.3 de macro-F1, con unas 270 veces más tiempo de entrenamiento.
- Estas diferencias vienen de una sola semilla y 996 filas, sin intervalos de confianza. Una fila equivale a 0.1 puntos de accuracy. La incertidumbre debe calcularse con bootstrap.

## Entorno

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB, driver 616.56 |
| Sistema | Windows 11 (10.0.26200) |
| Python | 3.12.14, entorno aislado `.venv` |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| Transformers / safetensors | 5.23.0 / 0.8.0 |
| scikit-learn | 1.9.1 |
| Modelo base | `distilbert/distilbert-base-uncased`, revisión `12040accade4e8a0f71eabdb258fecc2e7e948be`, licencia Apache 2.0 |

## Verificación previa (FB3)

Diagnóstico de sobreajuste con 64 ejemplos de train de clases mezcladas, V2 con LR 5e-5:

- Pérdida inicial 4.36, cercana a ln(77) = 4.34.
- 100 % de accuracy sobre esos mismos 64 ejemplos desde la época 13.
- Pérdida finita en todos los pasos; 1.3 GB de memoria máxima.

Esto solo comprueba que el código aprende. No es una medida de calidad.

## Configuración final

Común a V1 y V2: cross-entropy, AdamW (weight decay por defecto 0.01), dropout original, semilla 42, máximo 128 tokens, microbatch 8 con acumulación 2 (batch efectivo 16), precisión fp32, calentamiento lineal en el primer 10 % de los pasos y decaimiento lineal hasta cero, y early stopping con paciencia 3.

| | V1 | V2 |
|---|---|---|
| Qué se entrena | Solo la cabeza (649,805 parámetros) | Todo (67,012,685 parámetros) |
| Tasa de aprendizaje | 1e-3 | 5e-5 |
| Máximo de épocas | 40 | 10 |
| Configuración | `configs/train_v1.json` | `configs/train_v2.json` |

- La época se elige por el mayor macro-F1 en `validation_model.csv`.
- V1 y V2 parten de la misma revisión base, cada una con cabeza y optimizador nuevos.
- Truncamiento: 0 textos. El texto más largo de train y validación tiene 98 tokens.

## Cambios de configuración

La primera ronda usó la configuración inicial. Como V1 no convergió y V2 terminó en su última época permitida, se hizo una segunda ronda con tres cambios:

| Parámetro | Inicial (ronda 1) | Final (ronda 2) |
|---|---|---|
| Tasa de aprendizaje | Constante | Calentamiento 10 % y decaimiento lineal |
| Máximo de épocas | V1: 10, V2: 5 | V1: 40, V2: 10 |
| Paciencia | 2 | 3 |

No cambiaron el modelo base, los datos, la semilla, el batch ni la regla de selección. La decisión de hacer la segunda ronda se tomó viendo solo `validation_model.csv`.

## Corridas

En total hubo 9 corridas de DL, todas evaluadas sobre las mismas 996 filas. Al elegir entre tantas, la cifra de la ganadora puede estar algo inflada; la medida sin ese sesgo es el test.

| Ronda | Versión | LR | Mejor época | Accuracy | Macro-F1 | Tiempo | Memoria GPU máxima |
|---|---|---|---|---|---|---|---|
| 1 | V1 | 1e-3 | 10 de 10 | 0.7821 | 0.7780 | 65 s | 430 MB |
| 1 | V1 | 3e-4 | 10 de 10 | 0.7610 | 0.7520 | 58 s | 430 MB |
| 1 | V2 | 2e-5 | 5 de 5 | 0.9207 | 0.9195 | 73 s | 1,310 MB |
| 1 | V2 | 5e-5 | 4 de 5 | 0.9086 | 0.9082 | 84 s | 1,310 MB |
| 2 | **V1** | **1e-3** | 34 de 37 | 0.8765 | 0.8767 | 296 s | 430 MB |
| 2 | V1 | 3e-3 | 21 de 24 | 0.8323 | 0.8353 | 185 s | 430 MB |
| 2 | V2 | 2e-5 | 5 de 8 | 0.9197 | 0.9176 | 169 s | 1,310 MB |
| 2 | V2 | 3e-5 | 9 de 10 | 0.9207 | 0.9190 | 228 s | 1,310 MB |
| 2 | **V2** | **5e-5** | 7 de 10 | 0.9207 | 0.9198 | 227 s | 1,310 MB |

En negrita, las corridas elegidas. Todas están completas en `artifacts/runs/`, con sus curvas en `metrics.json`. «34 de 37» significa que la mejor época fue la 34 y el early stopping cortó en la 37.

### Curvas de V1 final (LR 1e-3)

| Época | Pérdida train | Accuracy train | Pérdida val | Accuracy val | Macro-F1 val |
|---|---|---|---|---|---|
| 1 | 4.208 | 0.045 | 3.788 | 0.1476 | 0.0961 |
| 5 | 1.192 | 0.652 | 1.021 | 0.6827 | 0.6672 |
| 10 | 0.737 | 0.776 | 0.678 | 0.8102 | 0.7994 |
| 15 | 0.543 | 0.833 | 0.585 | 0.8353 | 0.8365 |
| 20 | 0.438 | 0.861 | 0.575 | 0.8303 | 0.8239 |
| 25 | 0.344 | 0.893 | 0.512 | 0.8574 | 0.8559 |
| 30 | 0.286 | 0.910 | 0.490 | 0.8645 | 0.8629 |
| 34 | 0.246 | 0.925 | 0.475 | 0.8765 | 0.8767 |
| 37 | 0.224 | 0.931 | 0.468 | 0.8715 | 0.8716 |

### Curvas de V2 final (LR 5e-5)

| Época | Pérdida train | Accuracy train | Pérdida val | Accuracy val | Macro-F1 val |
|---|---|---|---|---|---|
| 1 | 3.291 | 0.338 | 1.243 | 0.7500 | 0.6866 |
| 2 | 0.688 | 0.854 | 0.411 | 0.8886 | 0.8887 |
| 3 | 0.239 | 0.945 | 0.359 | 0.8916 | 0.8866 |
| 4 | 0.109 | 0.977 | 0.323 | 0.9197 | 0.9193 |
| 5 | 0.057 | 0.989 | 0.322 | 0.9137 | 0.9119 |
| 6 | 0.033 | 0.994 | 0.344 | 0.9137 | 0.9117 |
| 7 | 0.019 | 0.998 | 0.341 | 0.9207 | 0.9198 |
| 8 | 0.011 | 0.999 | 0.343 | 0.9157 | 0.9140 |
| 9 | 0.008 | 0.999 | 0.341 | 0.9207 | 0.9187 |
| 10 | 0.007 | 1.000 | 0.344 | 0.9177 | 0.9165 |

La pérdida y la accuracy de train se miden durante la época, con dropout activo. Por eso no son comparables directamente con las de validación.

## Observaciones

- **V1 mejoró 9.4 puntos de accuracy con más épocas** (0.7821 a 0.8765). En la ronda 1 estaba subajustado. Aun así seguía mejorando despacio cuando cortó el early stopping.
- **V2 no mejoró.** Las cinco corridas buenas de V2 quedan entre 0.9176 y 0.9198 de macro-F1, una diferencia de una o dos filas. El calendario, más épocas y otra tasa no movieron el resultado: parece el techo de DistilBERT en esta partición.
- **La elección entre las corridas de V2 es casi un empate.** Se eligió LR 5e-5 por la regla fijada (mayor macro-F1) y porque es reproducible con el código actual; la ganadora de la ronda 1 se entrenó con tasa constante, que el código ya no usa.
- **V2 sobreajusta después de la época 5.** La pérdida de validación sube de 0.322 a 0.344 mientras la de train llega casi a cero y la accuracy de train a 1.000. El macro-F1 ya no mejora de forma clara.
- **El clásico también sobreajusta**: 0.9868 de accuracy en train frente a 0.8715 en validación.
- **Costo.** V2 tardó unas 200 veces más que el clásico en entrenar (227 s frente a 1.1 s), necesita GPU y su artefacto pesa unas 47 veces más. Si esa diferencia de calidad compensa el costo es parte del análisis de valor.
- El diagnóstico exploratorio anterior del clásico (0.8721 de accuracy) usó la validación completa de 2,033 filas. La cifra de este reporte usa solo las 996 de selección de modelo, por eso no coinciden.

## Artefactos y carga en CPU

Cada artefacto se cargó con `Predictor.load(..., device="cpu")` y predijo las 996 filas de `validation_model.csv`. La accuracy coincide con la del entrenamiento en los tres casos.

| Artefacto | Carga | Tiempo por texto | SHA-256 de `manifest.json` |
|---|---|---|---|
| `artifacts/baseline` | 0.03 s | 0.29 ms | `8947628a1593c81776ca81236262d5edf32e4399324d1326c2a35c77f416606a` |
| `artifacts/distilbert-v1` | 3.68 s | 7.2 ms | `288180e594ffbfc01faa5faf75ff33012e950383c68e68e04948018a029c6319` |
| `artifacts/distilbert-v2` | 0.36 s | 7.2 ms | `56b7d3565d69a7f30d51a4b1a822bce50a81c3b3770be6061b0be68242c2c910` |

- El tiempo por texto es en lotes de 32 en la CPU de este equipo. No es la latencia de batch 1; esa se mide con `benchmark.py`.
- V1 se cargó primero en el mismo proceso, así que su tiempo probablemente incluye la importación inicial de PyTorch. No se midió por separado.
- El SHA-256 de `manifest.json` es el `artifact_sha256` que va en `configs/decision.json`. El manifiesto lista el hash de cada archivo y `Predictor.load` los verifica.
- Estas corridas se hicieron antes de que `baseline.py` y `train.py` verificaran los datos contra su manifiesto. Se comprobó después: el `data_manifest_sha256` de los tres artefactos coincide con el manifiesto de datos actual, y `train.csv`, `validation_model.csv` y `label_to_id.json` coinciden con su `output_sha256`. Los artefactos no se reentrenaron.
- El baseline se guarda con pickle. Cargar solo artefactos cuyo hash coincida con esta tabla.

## Fallos registrados

- El PyTorch global del equipo era solo CPU. Se creó `.venv` con la build cu128.
- El primer lanzamiento de los entrenamientos falló antes de entrenar, porque faltaba una carpeta temporal para las configuraciones alternativas. Se relanzó sin cambios en el código.

## Pendiente

- Compartir los tres artefactos con el equipo y confirmar el hash y la carga en otras computadoras.
- Comunicar los cambios de configuración (calendario, épocas y paciencia).
- Muestra de explicabilidad por oclusión de palabras.
- Redacción de metodología para el informe.
