# Fase 13C completada - AMF World Model Training Readiness / Freeze

Fecha: 2026-06-05

## Estado

Phase 13C pasa:

```text
passed = true
```

La arquitectura queda congelada para Phase 14:

- AMF residual/local transition cells
- temporal-energy
- identity/orientation
- energy/constraint
- AMF-LTM residual
- Regime Expert Selector 13B
- Ridge safety fallback

Despues del preflight no se cambiaron features, thresholds, selector ni formula de residual. Solo quedan permitidos fixes de loader, checkpoint, logging o bug critico.

## Dataset

Se verifico el paquete remoto de physics completo:

```text
81 shards
91.59 GB
```

Por eso no se descargo todo el paquete completo en 13C. En su lugar se preparo:

- Tier 1 completo: `objects_falling`, `dominoes`, `wrecking_ball`
- Tier 2 ya disponible/cacheado: `billiards`, `bowling`

Manifest creado:

```text
results/phase13c_dataset_manifest.json
```

El manifest contiene 30 shards con tracks, secuencias, frames/transiciones, campos disponibles, size y checksum prefix.

## Splits

Splits fijos por secuencia:

```text
results/phase13c_splits.json
```

No hay mezcla por frame y no se usa test para calibrar.

## Config unica

Config lista:

```text
configs/phase14_world_model_train.yaml
```

La config tiene:

- escenas y shards;
- h1/h5/h15/h30/h60/h120;
- `use_rgb=false`;
- max cells, memory/LTM/selector settings;
- checkpoint dir;
- output paths;
- split path;
- seed.

## Modelo formal congelado

Archivos:

```text
phase14_formal_amf_world_model.py
run_phase14.py
```

Comando listo:

```powershell
python run_phase14.py --config configs/phase14_world_model_train.yaml
```

## Checkpoint / resume

Preflight probo interrupcion/resume:

```text
first pass completed shards: 1
resume completed shards: 3
resume_used: true
```

Archivos:

```text
checkpoints/phase14/latest.ckpt
checkpoints/phase14/epoch_or_shard_<id>.ckpt
```

El checkpoint guarda config, arquitectura congelada, progreso por shard, metricas parciales, conteos de celdas/memorias y pesos del selector en los resultados parciales.

## Logging

Generado:

```text
results/phase14_train_log.jsonl
results/phase14_metrics_live.json
results/phase14_latest.json
```

Cada entrada JSONL registra escena, shard, split, horizonte, experto seleccionado, MSE, gains, memoria MB, celdas, LTM memories, confidence, residual on/off y tiempo.

## Preflight

Preflight corto multi-escena:

```text
results/FASE13C_PREFLIGHT.md
results/phase13c_preflight_latest.json
```

Paso porque:

- no crashea;
- genera reportes;
- checkpoint/resume funciona;
- h30/h60/h120 se calculan;
- memoria no explota;
- selector sigue sin usar test;
- arquitectura queda congelada.
