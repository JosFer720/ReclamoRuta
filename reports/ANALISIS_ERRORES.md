# Análisis de errores

Se analizaron las 3,079 consultas de `test_unique`, con los modelos, temperaturas y umbrales ya fijados. Las predicciones reproducen la accuracy, macro-F1 y matriz de confusión de la evaluación final. El análisis conserva únicamente resultados agregados.

## Comparación general

| Modelo | Aciertos | Errores de intención | Errores de ruta |
|---|---:|---:|---:|
| Clásico | 87.85 % | 374 | 128 |
| V1 | 86.29 % | 422 | 178 |
| V2 | 91.98 % | 247 | 82 |

V2 reduce 127 errores frente al clásico y 175 frente a V1. De sus 247 errores de intención, 165 conservan la ruta correcta y 82 cambian de ruta. Por eso conviene distinguir la categoría exacta de la consulta del equipo al que se dirige.

## Categorías que más cuestan a V2

Cada categoría de esta tabla tiene 40 consultas. Recall indica qué porcentaje de sus consultas identifica correctamente el modelo.

| Categoría | Recall | Errores |
|---|---:|---:|
| Transferencia rechazada (`declined_transfer`) | 75.0 % | 10 |
| Tarjeta que no llega (`card_arrival`) | 77.5 % | 9 |
| Recarga revertida (`top_up_reverted`) | 77.5 % | 9 |
| Saldo sin actualizar tras transferencia (`balance_not_updated_after_bank_transfer`) | 80.0 % | 8 |
| Recargar mediante tarjeta (`topping_up_by_card`) | 80.0 % | 8 |
| Pago con tarjeta pendiente (`pending_card_payment`) | 82.5 % | 7 |
| Transferencia pendiente (`pending_transfer`) | 82.5 % | 7 |

En el clásico, la categoría más débil es `virtual_card_not_working`, con 15 errores y recall de 62.5 %. En V1 es `topping_up_by_card`, con 16 errores y recall de 60 %. V2 mejora el resultado general, pero todavía tiene dificultades para distinguir estados cercanos de una operación: pendiente, rechazada, fallida o revertida.

## Confusiones más frecuentes de V2

La dirección importa: la primera categoría es la correcta y la segunda es la predicción equivocada.

| Categoría correcta → predicción | Casos | Porcentaje de la categoría correcta | ¿Conserva ruta? |
|---|---:|---:|---|
| `card_arrival` → `card_delivery_estimate` | 7 | 17.5 % | Sí |
| `why_verify_identity` → `verify_my_identity` | 5 | 12.5 % | Sí |
| `declined_transfer` → `declined_card_payment` | 4 | 10.0 % | Sí |
| `pin_blocked` → `change_pin` | 4 | 10.0 % | Sí |
| `top_up_reverted` → `top_up_failed` | 4 | 10.0 % | Sí |
| `Refund_not_showing_up` → `reverted_card_payment?` | 3 | 7.5 % | Sí |

Estos pares distinguen detalles cercanos: preguntar por una tarjeta que no llega frente a preguntar por su plazo; preguntar por qué se verifica la identidad frente a cómo verificarla; o cambiar el PIN frente a resolver un bloqueo. En otros casos cambia el tipo de operación, como transferencia frente a pago con tarjeta. El vocabulario compartido ayuda a explicar el patrón, aunque la matriz por sí sola no identifica la causa de cada error.

Las rutas corresponden al mapa académico de siete áreas. Conservar una ruta no vuelve correcta una intención equivocada: la sugerencia todavía puede necesitar corrección y consumir tiempo.

## Diferencias por longitud

Se cuentan palabras con una expresión regular que mantiene juntas las contracciones con apóstrofo. Los intervalos se fijaron antes de ejecutar el análisis.

| Palabras | Consultas | Clásico | V1 | V2 |
|---|---:|---:|---:|---:|
| Hasta 5 | 264 | 84.85 % | 85.98 % | 90.91 % |
| 6–10 | 1,640 | 87.38 % | 87.56 % | 91.77 % |
| 11–20 | 929 | 88.81 % | 85.15 % | 92.79 % |
| 21 o más | 246 | 90.65 % | 82.52 % | 91.46 % |

V2 mantiene un desempeño parecido entre longitudes, con una diferencia de 1.88 puntos entre su mejor y peor grupo. Su mejor resultado aparece entre 11 y 20 palabras; escribir más no mejora el resultado de manera continua. V1 baja a 82.52 % en los mensajes más largos y V2 lo supera allí por 8.94 puntos.

## Diferencias por redacción en V2

Los rasgos se detectan con reglas explícitas: presencia de `?`, dígitos del 0 al 9, negaciones como `not`, `never`, `cannot` o `don't`, y los pronombres `i`, `my` o `me`. La lista completa está en el JSON. Un mensaje puede pertenecer a varios grupos.

| Rasgo | Con el rasgo: consultas y aciertos | Sin el rasgo: consultas y aciertos |
|---|---:|---:|
| Signo `?` | 1,974 · 92.20 % | 1,105 · 91.58 % |
| Dígitos | 49 · 100.00 % | 3,030 · 91.85 % |
| Negación | 681 · 90.01 % | 2,398 · 92.54 % |
| Pronombres `i`, `my` o `me` | 2,545 · 92.34 % | 534 · 90.26 % |

Los mensajes con negación tienen 2.52 puntos menos de accuracy. Conviene revisar especialmente negaciones al confirmar una sugerencia. El grupo con dígitos solo contiene 49 consultas; acertarlas todas no permite asegurar el mismo resultado con otros números o mensajes.

Estos grupos describen el texto, no características personales. También contienen distintas mezclas de categorías. Las diferencias no prueban que agregar un pronombre, un signo o un número cambie por sí solo la predicción.

## Efecto del umbral y prioridades

Con umbral 0.85, V2 presenta 2,711 sugerencias sobre el umbral y deriva 368 consultas a revisión obligatoria. Entre las primeras quedan 82 errores de intención y 16 errores de ruta. La revisión recibe 165 de las 247 predicciones de intención equivocadas, junto con 203 predicciones correctas de puntuación baja.

Las prioridades para una siguiente versión son diferenciar estados de transferencias y recargas, distinguir consultas sobre llegada y plazo de entrega, y revisar negaciones. Cualquier cambio se desarrolla con entrenamiento y validación; este análisis no modifica el sistema ya evaluado.

Datos completos: `ANALISIS_ERRORES.json`. Gráfica: `ERRORES_POR_LONGITUD.svg`. Reproducción desde la raíz del repositorio:

```powershell
$env:PYTHONPATH = 'src;.'
python scripts/analyze_errors.py --data data/processed/banking77 --artifacts artifacts/baseline artifacts/distilbert-v1 artifacts/distilbert-v2 --evaluation-dir reports/evaluation_scenario_v1 --routes configs/routes.json --out data/cache/errores-nueva-ejecucion.json --device cpu
```

El destino debe ser un archivo nuevo. No se guardan textos, identificadores de mensajes ni predicciones individuales.
