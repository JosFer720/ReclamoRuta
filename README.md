# ReclamoRuta

Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes, UVG.

Aplicación propuesta para clasificar consultas bancarias en inglés, sugerir una ruta de atención y permitir revisión humana. No opera cuentas ni toma decisiones financieras.

## Estado real

Al 23 de octubre de 2026 se descargaron y auditaron los CSV oficiales de BANKING77: 10,003 filas de entrenamiento y 3,080 de prueba, con 77 categorías. El proyecto ya genera limpieza, particiones, catálogo, EDA y manifiesto reproducibles. Los CSV procesados están en `data/processed/banking77/`, ignorados por Git; los reportes públicos pequeños están en `reports/`.

Una prueba exploratoria con TF-IDF y regresión logística obtuvo 87.21 % de accuracy y 87.14 % de macro-F1 en validación. No son resultados de deep learning ni de prueba final. Todavía no hay entrenamiento DL, aplicación desplegada ni ahorro económico medido.

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

Los módulos actuales solo usan la biblioteca estándar; pytest es dependencia de pruebas. No requieren CUDA, Transformers ni un modelo entrenado. La configuración de empaquetado y las dependencias de aplicación/GPU están pendientes. Python 3.13.2 es el entorno verificado aquí; no se ha ejecutado una matriz de versiones de Python.

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

La aplicación todavía no está implementada; por ahora se ejecutan los módulos de datos y evaluación. Repositorio configurado como remoto `origin`:
https://github.com/JosFer720/ReclamoRuta.git
