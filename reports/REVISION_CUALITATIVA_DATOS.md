# BANKING77 — revisión cualitativa de entrenamiento

Fecha: 23 de octubre de 2026. Revisión asistida durante la implementación; no es una validación por especialistas bancarios ni una auditoría completa de privacidad.

## Muestra examinada

Fuente: `train.csv` oficial, revisión `57ec275d8078af65b7731c2a98be812d844a6d6b`, con SHA-256 documentado en el plan de código. Se leyeron las primeras consultas de cada una de las 77 categorías y las 20 consultas más largas; unión de 97 filas originales distintas. IDs: `train.csv:<fila>` con fila de datos base 0. No se usaron mensajes del test para esta revisión.

La muestra es intencional, no aleatoria. Sirve para identificar riesgos y no para estimar una tasa de errores de etiqueta. No se reproducen los mensajes completos en este informe.

## Hallazgos

- **Consultas sin contexto suficiente:** filas 1475 (`lost_or_stolen_card`), 1782 (`contactless_not_working`), 3994 (`get_physical_card`) y 9515 (`card_about_to_expire`). La etiqueta puede depender de contexto que no está en la frase. No se afirma que la etiqueta sea necesariamente incorrecta.
- **Textos con estilo de respuesta de soporte:** filas 3905, 3910, 3917, 3932 y 3934 (`unable_to_verify_identity`), así como 5997 y 6105 (`pending_transfer`). Incluyen explicaciones/procedimientos en lugar de una solicitud típica del cliente. Esto limita la correspondencia con la entrada propuesta para la aplicación.
- **Paráfrasis muy similares:** varios de esos textos largos repiten ideas y estructura. Eliminar duplicados normalizados no elimina todo parentesco semántico ni prueba independencia entre clientes o sesiones.
- **Intenciones próximas o múltiples:** algunos mensajes mezclan una incidencia con una petición de acción. Es importante inspeccionar confusiones entre reclamo de pago, tarjeta comprometida y corrección de transacciones, sin asignar al clasificador facultades para actuar sobre cuentas.
- **Privacidad:** no se identificaron números de cuenta/tarjeta completos ni credenciales en esta muestra. Sí aparecen contextos personales y cantidades; esta revisión pequeña no acredita que todo el corpus esté libre de información personal.

## Decisiones

1. No cambiar etiquetas ni excluir nuevas filas por intuición. Conservar la política acordada y su reproducibilidad.
2. Informar estos límites en EDA y conclusiones; examinar errores por clase y estilo/longitud después de entrenar.
3. No interpretar un buen resultado sobre BANKING77 como desempeño demostrado en conversaciones reales, español, mensajes de múltiples intenciones o temas desconocidos.
4. Mantener revisión humana y no usar datos personales reales en la demo. Validación experta y un conjunto externo representativo quedan fuera de la evidencia disponible.

La clasificación sigue siendo un experimento académico ejecutable, pero la representatividad de producción no está demostrada.
