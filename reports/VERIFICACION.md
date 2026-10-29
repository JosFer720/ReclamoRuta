# Verificación del código

Fecha: 23 de octubre de 2026. Alcance: preparación de datos, evaluación, incertidumbre y función de valor, sin entrenar modelos ni ejecutar predicciones sobre test.

## Implementación

- Fuentes oficiales comprobadas por SHA-256 antes de generar CSV.
- Particiones persistentes: train 7,959; validación 2,033 (modelo 996, calibración 498, umbral 539); test oficial 3,080 y vista única 3,079.
- Manifiesto, EDA y catálogo: copias sin textos de mensajes en `reports/BANKING77_MANIFEST.json`, `BANKING77_EDA.json` y `BANKING77_LABEL_TO_ID.json`. Los CSV completos están ignorados por Git.
- Métricas y bootstrap agrupado; función económica, umbrales y sensibilidad; configuraciones pendientes rechazadas por defecto.

Las pruebas se escribieron antes de implementación en tres worktrees: datos (17 fallos iniciales), evaluación/incertidumbre (58) y valor (97 iniciales más nuevas pruebas por función). Después se integraron los commits y se volvió a ejecutar el conjunto completo. Esos fallos iniciales son evidencia del ciclo RED/GREEN, no fallos actuales del producto.

## Verificación integrada

Entorno observado: Windows, Python 3.13.2 y pytest 9.1.1. Sin dependencias externas de ejecución para estos cuatro módulos.

La ejecución integrada anterior a la revisión final obtuvo **227 passed, 1 skipped** en aproximadamente 0.86 s. La prueba omitida requiere crear enlaces simbólicos, privilegio no disponible en el entorno. El código contiene rechazo de enlaces/reparse points; esta omisión no se presenta como una prueba de enlaces superada.

La revisión de especificación pidió métricas completas por ruta y evitar un umbral validado sin observaciones. Ambas correcciones tuvieron pruebas RED/GREEN: se añadieron `routing_metrics` y rechazo de selección vacía, preservando cálculos vacíos definidos. Después de integrar las correcciones, **230 passed, 1 skipped** en aproximadamente 0.88 s. La revisión independiente de especificación confirmó el cumplimiento e hizo una reconstrucción externa de los IDs de todas las particiones.

Una segunda revisión independiente de calidad, posterior al cumplimiento de especificación, no encontró problemas críticos, importantes ni menores accionables dentro de F1/F2. Comprobó por separado clasificación sintética y un bootstrap con grupos desiguales. El coordinador contrastó además accuracy, macro-F1 y matriz de confusión de 200 filas sintéticas contra scikit-learn; coincidieron. Ninguna de estas pruebas sintéticas equivale a desempeño del modelo sobre test.

La primera ejecución integrada tuvo errores de fixture porque pytest intentó acceder a `Temp/pytest-of-Ana`, bloqueado por el sandbox. Se reprodujo el problema y se repitió la suite con una ruta temporal nueva dentro de `data/cache/`: las pruebas pasaron sin modificar la lógica de datos. El README explica la alternativa, sin reutilizar ni vaciar directorios ajenos.

El comando real de preparación se ejecutó dos veces en el mismo destino: ambas devolvieron los mismos conteos; una repetición válida no reescribe archivos. Test solo se leyó para calidad/deduplicación y no se utilizó para entrenamiento ni selección de decisiones.

## Revisión de seguridad

- Ship Safe **11.0.0**, repetido después de las correcciones: `scan . --json --no-cache` → 19 archivos analizados, 0 hallazgos.
- `score . --no-deps` → 100/100; 0 secretos y 0 vulnerabilidades de código detectadas, 67 archivos considerados. **Dependencias omitidas** en esa puntuación; no es una certificación de seguridad ni de despliegue.
- Revisión manual de entradas numéricas no finitas, conteos, configuraciones pendientes, rutas de salida, colisiones y hashes. No se añadieron HTTP, auth, bases de datos, CORS, deserialización ejecutable ni credenciales.
- `entry-point-analyzer` se inspeccionó por exigencia de AGENTS.md, pero su flujo es exclusivo de smart contracts y no aplica aquí. La entrada que escribe estado es la CLI local de datos; no existe una API pública en este alcance.

## Límites pendientes

Los contratos de aplicación y los modelos/predictor todavía no están implementados. Faltan integración con pesos reales, entrenamiento, calibración real, umbral con artefacto verificado, ensayo humano, costos medidos y despliegue. Las plantillas pendientes y los tests sintéticos no sustituyen esas actividades. La revisión cualitativa de 97 consultas de train está documentada aparte y no prueba ausencia de sesgos o datos personales en todo el corpus.
