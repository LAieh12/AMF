# Fase 11A - Frontier slot warp probe

Fecha: 2026-06-02

## Objetivo

Probar una mejora de encoder/decoder inspirada por modelos frontier:

- tokens/slots espacio-temporales;
- estimacion local de movimiento por matching;
- decoder copy-skip que preserva detalle del frame real;
- seleccion de parametros con datos reales de MovingMNIST descargados;
- comparacion contra baselines `last_frame` y `linear_frame`.

## Script

Archivo:

```text
phase11a_frontier_slot_warp_probe.py
```

Comando smoke sugerido:

```powershell
python phase11a_frontier_slot_warp_probe.py --train-sequences 80 --test-sequences 20 --tune-sequences 20 --out results/phase11a_frontier_slot_warp_probe_smoke.json
```

Comando principal sugerido:

```powershell
python phase11a_frontier_slot_warp_probe.py --train-sequences 220 --test-sequences 40 --tune-sequences 80 --out results/phase11a_frontier_slot_warp_probe_220_40.json
```

## Criterio de exito

El probe debe reducir MSE frente a `last_frame` y `linear_frame`, especialmente en h10/h17. La metrica clave es:

```text
mse_skill_vs_last = (last_frame_mse - frontier_slot_warp_mse) / last_frame_mse
```

Valores positivos implican mejora real sobre copiar el ultimo frame. Para ser una mejora fuerte de Fase 11A debe mostrar skill positivo consistente en h5/h10/h17, no solo h1.

## Estado

Implementado, pendiente de ejecucion local.

El intento de smoke test quedo bloqueado antes de iniciar Python por fallo del sandbox de Windows:

```text
windows sandbox: setup refresh failed with status exit code: 1
```

No se reporta como resultado positivo hasta tener JSON medido.
