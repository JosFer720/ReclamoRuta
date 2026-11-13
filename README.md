# ReclamoRuta

Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes, UVG.

Aplicación propuesta para clasificar consultas bancarias en inglés, sugerir una ruta de atención y permitir revisión humana. No opera cuentas ni toma decisiones financieras.

## Estado real

Al 23 de octubre de 2026 se descargaron y auditaron los CSV oficiales de BANKING77: 10,003 filas de entrenamiento y 3,080 de prueba, con 77 categorías. El proyecto ya genera limpieza, particiones, catálogo, EDA y manifiesto reproducibles. Los CSV procesados están en `data/processed/banking77/`, ignorados por Git; los reportes públicos pequeños están en `reports/`.

Al 13 de noviembre de 2026, los tres modelos entrenados están disponibles mediante Git LFS y sus métricas se reprodujeron en CPU: accuracy de 87.15 % para el clásico, 87.65 % para V1 y 92.07 % para V2 sobre las 996 filas de `validation_model.csv`. Son métricas de validación, no de prueba final. La interfaz usa pesos reales y el análisis económico presenta valores estimados. La demo funciona localmente.

## Instalación y ejecución

Desde esta carpeta, con Python 3.11 o superior y los cuatro archivos oficiales en `data/raw/banking77/`. La fuente y los archivos necesarios se indican en «Datos y límites»; los hashes están en `reports/BANKING77_MANIFEST.json`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python -m reclamoruta.data --raw data/raw/banking77 --out data/processed/banking77
$env:BANKING77_RAW_DIR = (Resolve-Path data/raw/banking77).Path
.\.venv\Scripts\python -m pytest -q
```

`requirements-dev.txt` instala el paquete en modo editable con el extra `dev`: pytest, Streamlit y las dependencias de modelos. Sin ellas, las pruebas de interfaz y de modelos fallan; no se omiten. Para las pruebas basta PyTorch para CPU. Las pruebas de modelos usan un DistilBERT diminuto creado en local: no descargan pesos ni repiten entrenamientos reales. Los módulos de datos, evaluación, valor, contratos, benchmark y estudio solo usan la biblioteca estándar.

| Extra | Contenido | Para qué |
|---|---|---|
| `app` | `streamlit>=1.49` | Interfaz |
| `model` | `torch>=2.5`, `transformers>=4.45,<6`, `safetensors>=0.4`, `scikit-learn>=1.5` | Entrenamiento y predictor |
| `dev` | `pytest==9.1.1` + `app` + `model` | Suite completa |

Suite verificada con Python 3.12.14 en tres entornos aislados: versiones mínimas declaradas (torch 2.5.1, Transformers 4.45.2, safetensors 0.4.5, scikit-learn 1.5.2, Streamlit 1.49.0), Transformers 4.56.2 y Transformers 5.23.0 con torch 2.11.0. Streamlit 1.40–1.48 no sirve: `app.py` usa `form_submit_button(key=...)`. No se ha ejecutado una matriz de versiones de Python.

La preparación no descarga datos ni sobrescribe archivos arbitrarios. Comprueba SHA-256 antes de procesar y permite repetir la misma preparación sin reescribir archivos; otro contenido/semilla en un destino existente se rechaza. Para una política diferente, elegir un directorio de salida nuevo, no borrar datos ajenos para forzar el comando.

Particiones generadas: train 7,959; validación total 2,033, dividida en selección de modelo 996, calibración 498 y umbral 539; test oficial 3,080 y vista única 3,079. El padre `validation.csv` contiene las tres subdivisiones: no tratarlo como un cuarto conjunto independiente ni usarlo completo para todas las decisiones.

Si este sandbox de Windows niega acceso a la carpeta temporal global de pytest, usar una carpeta temporal nueva dentro de `data/cache/`:

```powershell
New-Item -ItemType Directory -Path data/cache -Force | Out-Null
$testScratch = Join-Path (Get-Location).Path ('data\cache\pytest-local-' + [Guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $testScratch) { throw 'El temporal debe ser nuevo' }
.\.venv\Scripts\python -m pytest -q --basetemp "$testScratch"
```

## Módulos disponibles

- `data.py`: preparación y CLI; catálogo `label_to_id.json` y `manifest.json` junto a los CSV. La revisión de muestra está en `reports/REVISION_CUALITATIVA_DATOS.md` y no equivale a validación experta completa.
- `evaluate.py`: `evaluate_predictions(labels, predictions, categories=..., routes=...)`, con accuracy, macro-F1, métricas por clase, confusiones, ECE, latencia y exactitud de ruta. Pasar siempre el catálogo completo para comparar modelos. Acepta objetos con los campos `Prediction` acordados; no crea ni sustituye el módulo de contratos de la aplicación.
- `uncertainty.py`: `paired_bootstrap(...)` para diferencias aditivas y `paired_classification_bootstrap(...)` para accuracy/macro-F1, remuestreando grupos completos. Diferencia = V2 menos V1; 1,000 remuestreos por defecto.
- `value.py`: `compute_value(...)`, `threshold_curve(...)`, `select_threshold(...)` y `sensitivity_analysis(...)`. Los docstrings y las pruebas documentan parámetros, unidades y ejemplos; `configs/value.json` define los campos económicos requeridos.
- `configs/value.json` y `configs/decision.json` son plantillas **pending**: se rechazan hasta completar evidencia/supuestos y artefactos reales. No hay umbral final ni costos medidos. La selección requiere configuración económica validada, revisión `validation_threshold` y metadatos de modelo/calibración.

La aplicación debe usar `validate_value_config` y `validate_decision` antes de consumir configuraciones, además de verificar hashes contra los archivos de modelo reales. Estos validadores comprueban estructura: no pueden certificar que un costo es verdadero o que un llamador no etiquetó mal un split. No publicar los CSV de mensajes ni usar el test para buscar parámetros. Los ejemplos sintéticos de pruebas no son métricas del producto.

### Aplicación, contratos y medición

- `contracts.py`: `Prediction` (inmutable y validado: `label`, `score` en [0, 1], `top3` descendente, `model_version`, `elapsed_ms`), `validate_message` (1–2,000 caracteres sin contar los espacios exteriores), `load_routes`/`validate_routes`, `artifact_digest`, `bind_decision`, `route_message` y `confirm`. `score >= umbral` → la sugerencia se puede aceptar; `score < umbral` → revisión obligatoria. La ruta sale de `configs/routes.json`, no del clasificador.
- `configs/routes.json`: **propuesta** `rutas-v0.1` con 7 rutas que cubren las 77 intenciones. Es una decisión de diseño del grupo y queda pendiente de aprobación; no son departamentos reales.
- `app.py`: interfaz en Streamlit. Antes de cargar los pesos verifica catálogo, rutas, decisión y hash del artefacto. Si algo falla muestra un error y se detiene, sin dar respuestas de respaldo. No guarda el texto de los mensajes y las correcciones no reentrenan.
- `benchmark.py`: `run_benchmark(...)` y CLI. Separa el arranque de la inferencia, hace calentamiento, usa batch 1 y reporta p50/p95 y throughput con reloj monotónico, junto con la identificación del equipo y del modelo. No acepta splits de test, no sobrescribe resultados y no guarda textos.
- `study.py`: estudio humano de tiempos, condición manual frente a asistida. Usa IDs anónimos `P-xxxxxxxx`, no guarda el texto salvo con `store_text=True`, exporta a un CSV nuevo neutralizando fórmulas, y `summarize` produce `t_manual`, `t_aceptar`, `t_corregir` y `t_revision` para `value.py`.

En ese resumen, `t_corregir` es tiempo **adicional**, estimado como mediana de los casos corregidos menos mediana de los aceptados sin cambios. El tiempo total registrado de los casos corregidos se conserva por separado. Si falta alguno de esos grupos o la diferencia es negativa, la estimación queda como `None` con un diagnóstico; no se inventa un cero. Es una diferencia de medianas entre grupos, no una medición separada de la fase de corrección ni una reconstrucción exacta del tiempo total de cualquier población.

**Contrato para el predictor:** `reclamoruta.predictor.Predictor.load(artifact_dir, device="cpu")` devuelve un objeto con el atributo `model_version`, el método `predict(texts) -> list[contracts.Prediction]` (una predicción por texto, en el mismo orden) y, de forma opcional, `categories`. El artefacto debe incluir un `manifest.json` con los hashes de todos sus archivos, y `Predictor.load` debe verificarlos. El `artifact_sha256` de `configs/decision.json` es el SHA-256 de ese `manifest.json` (`contracts.artifact_digest`).

Instalación editable por partes (alternativa a `requirements-dev.txt`):

```powershell
python -m pip install -e ".[dev]"        # suite completa: pytest + app + modelos
python -m pip install -e ".[app,model]"  # solo app con predictor
```

Ejecutar la app, con un artefacto real y la decisión ya validada:

```powershell
$env:RECLAMORUTA_ARTIFACT_DIR = 'artifacts/distilbert-v2'
streamlit run src/reclamoruta/app.py
```

Otras variables: `RECLAMORUTA_DECISION`, `RECLAMORUTA_ROUTES`, `RECLAMORUTA_CATALOG`, `RECLAMORUTA_DEVICE` y `RECLAMORUTA_CALIBRATION`. Con el `decision.json` pendiente actual, la app se niega a iniciar a propósito.

**Calibración.** Si la decisión declara `calibration.fitted = true`, la app lee el archivo de `RECLAMORUTA_CALIBRATION` (por defecto `configs/calibration.json`) antes de cargar los pesos y se detiene si falta, si su SHA-256 no es el `calibration.artifact_sha256` de la decisión o si pertenece a otro modelo o artefacto. Solo se admite escalado por temperatura (`method: "temperature"`); cualquier otro método se rechaza en lugar de mostrarse como calibrado. El predictor aplica `softmax(log p / T)` a la distribución completa antes de comparar con el umbral. Formato:

```json
{"method": "temperature", "temperature": 1.3, "split_role": "validation_calibration",
 "model_version": "distilbert-v2", "artifact_sha256": "<SHA-256 del manifest.json del modelo>"}
```

La temperatura se ajusta solo con `validation_calibration.csv`. El módulo `calibration.py` verifica los hashes de datos y modelo, minimiza la pérdida logarítmica con una temperatura entre 0.05 y 20 y escribe un archivo nuevo sin modificar los pesos:

```powershell
python -m reclamoruta.calibration --artifact artifacts/distilbert-v2 --data data/processed/banking77 --out data/cache/calibration-v2.json
```

El tokenizador puede haber sido guardado con metadatos de BERT por otra versión de Transformers. En entrenamiento e inferencia se desactiva explícitamente `token_type_ids`, que DistilBERT no acepta; no se modifican los artefactos ni sus hashes.

### Comparación y demo con escenario estimado

`configs/value_scenario.json` contiene el escenario estimado usado para comparar las versiones. `status: validated` indica estructura y unidades revisadas; todos sus parámetros conservan `estimated: true`. La guía para medir los tiempos está en `reports/GUIA_MEDICION_TIEMPOS.md`.

`finalize.py` separa dos pasos. `prepare` ajusta la temperatura con calibración, elige el umbral por valor en la validación de umbral y congela hashes del código, datos, rutas, modelos y decisiones. `evaluate` comprueba ese congelamiento antes de leer test, calcula métricas y valor con decisiones fijas, sensibilidad e intervalos con 1,000 remuestreos pareados por grupo. No guarda los mensajes. Los intervalos económicos solo reflejan variación de la muestra bajo los costos supuestos.

```powershell
python -m reclamoruta.finalize --stage prepare --data data/processed/banking77 --artifacts artifacts/baseline artifacts/distilbert-v1 artifacts/distilbert-v2 --value-config configs/value_scenario.json --routes configs/routes.json --out reports/evaluation_scenario_v1 --device cpu
python -m reclamoruta.finalize --stage evaluate --data data/processed/banking77 --artifacts artifacts/baseline artifacts/distilbert-v1 artifacts/distilbert-v2 --value-config configs/value_scenario.json --routes configs/routes.json --out reports/evaluation_scenario_v1 --device cpu --bootstrap 1000
```

Los resultados de esa ejecución están incluidos en `reports/evaluation_scenario_v1/`. Los comandos rechazan sobrescribirlos: para reproducir, usar otro directorio de salida nuevo en ambos pasos. Si cambian archivos de código o supuestos, se necesita otro congelamiento; no editar los hashes para forzar una comparación. Después de consultar test, cualquier ajuste posterior debe declararse y no tratarse como una nueva evaluación independiente.

Para ejecutar la demo con los pesos reales y la decisión del escenario:

```powershell
$env:RECLAMORUTA_ARTIFACT_DIR = 'artifacts/distilbert-v2'
$env:RECLAMORUTA_DECISION = 'reports/evaluation_scenario_v1/distilbert-v2/decision.json'
$env:RECLAMORUTA_CALIBRATION = 'reports/evaluation_scenario_v1/distilbert-v2/calibration.json'
python -m streamlit run src/reclamoruta/app.py
```

La decisión del escenario exige revisión por debajo de 0.85. La plantilla principal `configs/decision.json` se mantiene pendiente para impedir confundir este ejemplo con un umbral respaldado por un estudio real. La demo requiere revisión humana y consultas en inglés; no reconoce de forma fiable todos los mensajes fuera de dominio.

Contenedor: `docker build -t reclamoruta .` y luego `docker run -p 8501:8501 -v ${PWD}/artifacts/distilbert-v2:/app/artifacts:ro -v ${PWD}/configs/decision.json:/app/configs/decision.json:ro reclamoruta`. Con una decisión calibrada, montar también `-v ${PWD}/configs/calibration.json:/app/configs/calibration.json:ro`. Los pesos se montan en modo de solo lectura y no se copian a la imagen. El contenedor corre sin root, en modo offline para Hugging Face y sin mostrar trazas al usuario (`.streamlit/config.toml`). En el despliegue público, HTTPS y los límites de uso los pone la plataforma o un proxy inverso; no se agrega autenticación casera.

### Entrenamiento y predictor

- `baseline.py`: TF-IDF (1–2-gramas, `min_df=2`, 30,000 atributos) + regresión logística One-vs-Rest.
- `train.py`: DistilBERT con `configs/train_v1.json` (encoder congelado, solo se entrena la cabeza) o `configs/train_v2.json` (fine-tuning completo). Usa AdamW, cross-entropy, semilla 42, máximo 128 tokens, microbatch 8 con acumulación 2, calentamiento y decaimiento lineal de la tasa de aprendizaje y early stopping. Resultados y cambios de configuración: `reports/ENTRENAMIENTO.md`. La época se elige por macro-F1 en `validation_model.csv`; ningún módulo de entrenamiento lee el test.
- `predictor.py`: `Predictor.load(artifact_dir, device="cpu")` sirve para los tres modelos. Rechaza el artefacto si falta, sobra o cambió algún archivo respecto a `manifest.json`. `with_calibration(temperature, sha256)` devuelve una copia que aplica el escalado por temperatura.

Antes de entrenar, `baseline.py` y `train.py` comprueban `train.csv`, `validation_model.csv` y `label_to_id.json` contra `output_sha256` del `manifest.json` de los datos, y se detienen si falta el manifiesto, es inválido o algún hash no coincide. Leen cada archivo una sola vez, así que lo que se entrena es exactamente lo verificado. El `data_manifest_sha256` del artefacto es el hash de ese manifiesto ya comprobado.

Cada artefacto es una carpeta plana con el modelo, `label_to_id.json`, `metrics.json` (curvas por época, tiempo, memoria y truncamiento) y `manifest.json` con el SHA-256 de cada archivo. El baseline se guarda como `model.pkl`: cargar solo artefactos propios, porque pickle ejecuta código. Un directorio de salida existente se rechaza; nunca se sobrescribe un artefacto.

PyTorch con CUDA se instala desde su índice oficial (verificado: `2.11.0+cu128`, Python 3.12, RTX 5070 Ti):

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
python -m reclamoruta.baseline --data data/processed/banking77 --out artifacts/baseline
python -m reclamoruta.train --config configs/train_v1.json --data data/processed/banking77 --out artifacts/distilbert-v1
python -m reclamoruta.train --config configs/train_v2.json --data data/processed/banking77 --out artifacts/distilbert-v2
```

Las pruebas usan un DistilBERT diminuto aleatorio y no descargan nada; no son resultados. Las cifras de validación de los entrenamientos reales están en `reports/ENTRENAMIENTO.md`. La evaluación final congelada está en `reports/VERIFICACION_REAL.md`: V2 obtuvo 91.98 % de accuracy en el test oficial de 3,080 consultas. Sus resultados económicos son escenarios estimados, con costos excluidos explícitos.

### Pesos entrenados (Git LFS)

Los tres modelos finales están en el repositorio con Git LFS: `artifacts/baseline` (5 MB), `artifacts/distilbert-v1` y `artifacts/distilbert-v2` (256 MB cada uno). Instalar Git LFS una vez antes de clonar:

```powershell
git lfs install
git clone https://github.com/JosFer720/ReclamoRuta.git
```

En un clon anterior, o si `model.safetensors` pesa unos pocos bytes (es solo el puntero de LFS), ejecutar `git lfs pull`. Para comprobar la descarga: `python -c "from reclamoruta.predictor import Predictor; print(Predictor.load('artifacts/distilbert-v2').model_version)"`. `Predictor.load` verifica cada archivo contra su `manifest.json`; los hashes de cada manifiesto están en `reports/ENTRENAMIENTO.md`.

`.gitattributes` guarda `artifacts/` byte a byte, sin convertir saltos de línea, porque los hashes dependen de los bytes exactos. Solo se versionan estas tres carpetas: las corridas intermedias y cualquier modelo nuevo quedan fuera salvo acuerdo del equipo. Cada descarga consume la cuota de LFS de la cuenta dueña del repositorio.

El pickle clásico se guardó con scikit-learn 1.9.1; para reproducir su carga usar esa misma versión (`python -m pip install scikit-learn==1.9.1`). No cargar pickles enviados por fuentes desconocidas: un hash verifica integridad, pero no vuelve confiable un archivo y su manifiesto de origen desconocido.

## Análisis complementario

`reports/ANALISIS_ERRORES.md` explica las categorías más difíciles, confusiones y diferencias por longitud y rasgos de redacción. Sus métricas se reproducen con `scripts/analyze_errors.py` y los datos agregados están en `reports/ANALISIS_ERRORES.json`. El análisis conserva los modelos y umbrales fijados.

`reports/ANALISIS_ECONOMICO.md` compara detección humana del 50 %, 80 % y 100 %, incorpora energía y capital del equipo, y separa el primer lote de los lotes posteriores. `scripts/analyze_economics.py` genera el JSON correspondiente a partir de conteos fijos, benchmarks y parámetros de `configs/compute_costs.json`. Incluye nueve cruces de capital y volumen por modelo y tasa de detección. Los dos reportes contienen sus comandos de reproducción y las fuentes de los parámetros.

## Colaboración

Todo se trabaja y se integra sobre `main`. Cada persona usa su propia copia del repositorio, sincroniza antes de empezar y antes de publicar, y coordina cambios en archivos compartidos. No sobrescribir cambios de otro integrante ni forzar un push.

## Datos y límites

Fuente oficial:
https://github.com/PolyAI-LDN/task-specific-datasets

Revisión fijada: `57ec275d8078af65b7731c2a98be812d844a6d6b`. Descargar `train.csv`, `test.csv` y `categories.json` desde:
https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data

Descargar también `LICENSE` desde:
https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE

Guardar los cuatro archivos en `data/raw/banking77/`, ignorados por Git. La revisión y los SHA-256 están en `reports/BANKING77_MANIFEST.json`; la preparación comprueba automáticamente su integridad. Conservar la atribución a PolyAI y la licencia CC BY 4.0.

Solo se plantea soporte en inglés. Las rutas serán un mapa diseñado por el grupo; no son departamentos reales etiquetados por el dataset. Los tiempos, costos y demanda comercial deben medirse o declararse como supuestos.

La comprobación de los pesos reales, la latencia y los límites de la evaluación se documentan en `reports/VERIFICACION_REAL.md`. Los tiempos del escenario son supuestos hasta realizar el estudio humano. Repositorio configurado como remoto `origin`:
https://github.com/JosFer720/ReclamoRuta.git
