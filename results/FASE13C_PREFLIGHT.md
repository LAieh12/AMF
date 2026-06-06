# Fase 13C - Preflight

Architecture is frozen from Phase 13B. No selector/features/residual formula changed after this preflight.

## Dataset

Manifest shards: 30
Remote full physics package checked: 91.59 GB
do not download full physics package in 13C; download/prepare complete Tier 1 and cached Tier 2 physics metadata only

## Checkpoint/resume

First pass completed shards: 1
Resume pass completed shards: 3
Resume used: `True`

## Metrics sanity

| scene | horizons logged | finite MSE | max memory MB |
|---|---:|---|---:|
| dominoes | 6 | True | 0.142 |
| objects_falling | 6 | True | 0.056 |
| wrecking_ball | 6 | True | 0.096 |

## Pass criteria

- no leakage: sequence splits and no test calibration recorded
- no crash: preflight completed after resume
- reports generated: true
- checkpoint/resume works: true
- h30/h60 calculated: true
- memory below limit: true
