# Fase 11A - Never diagnostic runner

Fecha: 2026-06-02

## Archivo

```text
phase11a_run_never_diagnostics.py
```

## Motivo

El sandbox local fallo varias veces antes de iniciar procesos. Para reducir la dependencia de multiples arranques separados, este runner ejecuta dentro de un solo proceso:

- compilacion de scripts clave;
- auditoria de bottleneck;
- smoke del Never World Codec;
- smoke del Never Definitive Codec;
- opcionalmente pruebas 220/40.

## Comandos

Smoke completo:

```powershell
python phase11a_run_never_diagnostics.py --retries 2 --delay 3 --out results/phase11a_never_diagnostics_run.json
```

Prueba principal:

```powershell
python phase11a_run_never_diagnostics.py --full --retries 2 --delay 3 --out results/phase11a_never_diagnostics_full_run.json
```

## Como interpretar

El JSON `all_passed: true` solo significa que los comandos corrieron. La conclusion numerica sale de:

- `results/FASE11A_NEVER_BOTTLENECK_AUDIT.md`;
- `results/phase11a_never_world_codec_probe_smoke.json`;
- `results/phase11a_never_definitive_codec_probe_smoke.json`;
- `results/phase11a_never_definitive_codec_probe_220_40.json`;
- `results/phase11a_never_world_codec_probe_220_40.json`.

## Estado

Ejecutado correctamente en modo smoke:

```powershell
python phase11a_run_never_diagnostics.py --retries 1 --delay 1 --out results/phase11a_never_diagnostics_run.json
```

El runner genero:

```text
results/phase11a_never_diagnostics_run.json
results/phase11a_never_world_codec_probe_smoke.json
results/phase11a_never_definitive_codec_probe_smoke.json
results/FASE11A_NEVER_BOTTLENECK_AUDIT.md
```

Luego se corrieron pruebas principales independientes 220/40 y 650/40 para el definitive codec.
