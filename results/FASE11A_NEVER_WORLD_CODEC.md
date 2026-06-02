# Fase 11A - Never World Codec

Fecha: 2026-06-02

## Proposito

Responder a la duda central de la fase:

```text
Estamos usando Never completo con AMF World Model, o solo piezas?
```

La respuesta previa era: solo piezas. Este probe implementa una ruta mas completa:

```text
frames reales -> encoder de slots -> memoria de transiciones con celulas -> rollout por horizonte -> decoder copy-skip
```

## Archivo

```text
phase11a_never_world_codec_probe.py
```

## Que cambia frente a probes anteriores

- No optimiza solo el decoder.
- No optimiza solo features del encoder.
- Usa una memoria de transiciones tipo AMF/celulas.
- Aprende residuales de movimiento desde secuencias reales.
- Decodifica preservando detalle visual del frame observado.
- Reporta `transition_cells` para saber si la capacidad realmente crece.

## Comandos

Smoke:

```powershell
python phase11a_never_world_codec_probe.py --train-sequences 80 --test-sequences 20 --tune-sequences 20 --out results/phase11a_never_world_codec_probe_smoke.json
```

Principal:

```powershell
python phase11a_never_world_codec_probe.py --train-sequences 220 --test-sequences 40 --tune-sequences 60 --out results/phase11a_never_world_codec_probe_220_40.json
```

Escalado:

```powershell
python phase11a_never_world_codec_probe.py --train-sequences 650 --test-sequences 40 --tune-sequences 100 --out results/phase11a_never_world_codec_probe_650_40.json
```

## Criterio de exito

Para ser una mejora real debe:

- mostrar `mse_skill_vs_last > 0` en h5/h10/h17;
- usar una cantidad no trivial de `transition_cells`;
- acercarse al slot-hybrid/ranker en horizontes largos;
- idealmente reducir la brecha learned-vs-oracle existente.

## Estado de verificacion

Compilacion Python realizada con exito una vez:

```powershell
python -m py_compile phase11a_never_bottleneck_audit.py phase11a_never_world_codec_probe.py phase11a_frontier_slot_warp_probe.py
```

Despues de esa compilacion, el sandbox de Windows volvio a fallar antes de iniciar procesos:

```text
windows sandbox: setup refresh failed with status exit code: 1
```

Por tanto:

- codigo implementado;
- compilacion basica observada;
- resultados numericos pendientes;
- no hay claim de mejora hasta producir JSON.
