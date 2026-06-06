# Fase 13B completada - Regime Expert Selector

Fecha: 2026-06-05

## Objetivo

Phase 13B corrigio el fallo de Phase 13: no intenta reemplazar a los mejores AMF previos con un AMF-LTM general. En su lugar entrena un selector de expertos por regimen fisico, horizonte y contexto.

La prediccion final es:

```text
weighted expert prediction + optional gated LTM residual
```

El selector usa solo fit/validation para pesos, bins de contexto, alphas y residual beta. Test queda solo para la evaluacion final.

## Expertos

- `temporal_energy`
- `energy_constraint`
- `identity_orientation`
- `ensemble_12c`
- `amf_ltm_residual`
- `ridge_safety`
- `amf_residual_base`

## Contexto del selector

Los grupos de seleccion usan:

- regimen fisico detectado;
- horizonte;
- energia / cambio de energia;
- orientacion;
- contacto / impacto;
- memoria recuperada y confianza;
- proxy de error reciente desde residual recuperado.

`H_event` y `H_workspace` siguen como contexto/router/residual; no se vuelven features densas del predictor.

## Resultado

Escenas Tier 1 reales:

- `objects_falling`
- `dominoes`
- `wrecking_ball`

Horizontes:

- h1/h5/h15/h30/h60/h120

Resultado principal:

```text
Phase13B passed = true
Long h30/h60/h120 W/T vs best previous AMF = 8/9
```

| horizonte | W/T/L vs best previous | mean gain vs previous | mean gain vs temporal |
|---|---|---:|---:|
| h30 | 3/0/0 | 0.064743 | 0.168018 |
| h60 | 2/0/1 | 0.052371 | 0.154123 |
| h120 | 1/2/0 | 0.025853 | 0.044442 |

## Lectura

13B logra lo que 13 no pudo: superar al mejor AMF previo en la mayoria clara de los horizontes largos, sin usar test para elegir pesos.

El selector aprende que no hay un experto universal:

- `ensemble_12c` domina varios casos h30 en `objects_falling` y regimens de impacto.
- `amf_ltm_residual` domina gran parte de `dominoes` y `wrecking_ball`, especialmente h30/h60/h120.
- `ridge_safety` aparece como experto util en algunos regimenes de `objects_falling` h60.
- `identity_orientation` ayuda en `wrecking_ball` h15 y algunos regimens radial/mixtos.

## Fallos restantes

- `objects_falling` h1/h5/h15 todavia pierde contra el experto previo corto plazo.
- `dominoes` h60 pierde ligeramente contra temporal-energy.
- El residual LTM a veces empeora muchas muestras aunque la mezcla final gane; falta gating mas fino para apagar correcciones por subregimen.

## Archivos

- `phase13b_regime_expert_selector.py`
- `run_phase13b.py`
- `results/phase13b_latest.json`
- `results/FASE13B_REGIME_EXPERT_SELECTOR.md`
- `results/FASE13B_PREVIOUS_BEST_MATRIX.md`
