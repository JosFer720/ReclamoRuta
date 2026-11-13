# Cómo sustituir los tiempos estimados

`configs/value_scenario.json` permite probar el flujo con valores estimados. El modelo puede ahorrar tiempo y aun así producir valor negativo si los errores cuestan demasiado.

## Piloto con validación

1. Invitar al menos a tres personas y explicarles que es una prueba académica voluntaria. Usar identificadores anónimos, sin nombres ni datos bancarios.
2. Usar consultas de validación. Alternar el orden de las condiciones manual y asistida; una persona no debe ver el mismo mensaje en ambas condiciones.
3. En la condición manual, medir desde que se muestra el mensaje hasta que la persona elige la intención y ruta. En la asistida, medir hasta que confirma su elección final, incluyendo la lectura de la sugerencia y cualquier corrección.
4. Guardar identificador del mensaje, condición, segundos, sugerencia, decisión final y si requería revisión. No publicar los textos ni datos de participantes. `study.StudySession` permite registrar estos campos y `study.export_trials` los exporta a un CSV nuevo.
5. Revisar errores finales comparando las decisiones humanas con las etiquetas. La tasa de corrección por sí sola no mide cuántos errores se detectaron: se necesita identificar cuáles sugerencias estaban equivocadas.

## Parámetros que se necesitan

| Parámetro | Cómo estimarlo |
|---|---|
| `t_manual` | Mediana del tiempo de clasificación manual. |
| `t_aceptar` | Mediana del tiempo asistido de sugerencias aceptadas sin cambios y sin revisión obligatoria. |
| `t_corregir` | Tiempo adicional: mediana total de los casos corregidos menos mediana de los aceptados. Conservar también el tiempo total registrado. |
| `t_revision` | Mediana del tiempo completo de los casos con revisión obligatoria. |
| `q_detectado` | Fracción de sugerencias incorrectas que las personas detectan y corrigen; declarar el número de errores registrados. |

`study.summarize` calcula los tiempos. Si faltan casos aceptados o corregidos, o la diferencia de medianas es negativa, `t_corregir` queda sin estimar. No convertirlo en cero para completar la tabla. La diferencia de medianas es una aproximación entre grupos; no equivale a cronometrar una fase de corrección.

Documentar por separado el costo por hora, la penalización por ruta incorrecta, el costo de inferencia y el costo de operación. Identificar los valores estimados y registrar equipo, tamaño del lote y costos excluidos.

## Después del piloto

Crear una versión nueva de los supuestos y generar decisiones nuevas usando únicamente validación. No sobrescribir el escenario ni sus resultados anteriores. Congelar modelos, rutas, calibración, umbrales y protocolo antes de evaluar la nueva versión.

La prueba final con personas debe usar mensajes diferentes de los del piloto y conservar las decisiones congeladas. Si ya se consultó el test, declararlo en el informe; cambiar parámetros después impide tratar una nueva evaluación sobre ese mismo test como una comprobación independiente. Un ensayo con estudiantes tampoco demuestra el rendimiento de agentes profesionales de un banco.
