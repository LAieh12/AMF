# FASE8_5_COMPLETADA

Fase 8.5 corrige la comparacion: el benchmark es generacion de respuestas y
Pythia-70M se evalua como modelo de completion, no como chat model.

Entregables:

- `run_phase8_5.py`
- `results/phase8_5_latest.json`
- `results/FASE8_5_RESULTADOS.md`

Correccion aplicada:

- Pythia-70M se mide con templates de completion.
- Se prueban formatos `Usuario/Respuesta`, `Human/Assistant`,
  `Pregunta/Respuesta` y few-shot local.
- La fila principal de Pythia usa el mejor resultado de la suite final como
  upper bound optimista para Pythia.
- Tambien se guarda el protocolo estricto seleccionado por calibracion.

Resultado medido:

- AMF8 talk_score: 0.9047
- Pythia-70M prompted talk_score: 0.1893
- AMF8 inferencia: 0.2455 s / 90 prompts
- Pythia-70M inferencia: 40.2022 s / 90 prompts
- AMF8 memoria: 0.8897 MB
- Pythia-70M memoria: 267.0288 MB

Claim valido: AMF8 no demuestra ser un reemplazo general de Pythia-70M; demuestra
generacion local de dominio pequeno con mucha menos memoria, CPU, entrenamiento
local de 90 ejemplos y mayor score en esta suite controlada.

Limitacion critica: `talk_score` es un scorer local definido en `run_phase8.py`.
Favorece respuestas breves, completas, en el vocabulario esperado y con keywords
del intent. No mide calidad linguistica general ni preferencia humana.
