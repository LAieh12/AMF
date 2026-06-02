# Fase 11A - Never Definitive Codec Probe

Fecha: 2026-06-02

## Hipotesis

El mayor cuello de botella actual no es un encoder o decoder aislado. Es la cadena completa:

```text
estado -> dinamica -> seleccion -> decoder
```

Los resultados anteriores muestran varias brechas learned-vs-oracle. Eso sugiere que el sistema genera o contiene candidatos utiles, pero no siempre escoge el correcto.

## Archivo

```text
phase11a_never_definitive_codec_probe.py
```

## Arquitectura

El probe combina:

- encoder de slots con posicion, tamano, masa, velocidad, aceleracion y horizonte;
- AMF World Memory densa con control de crecimiento por radio y `max_cells`;
- candidatos multiples:
  - `last`;
  - `pixel_linear`;
  - `slot_velocity`;
  - `amf_world`;
  - `slot_amf_mean`;
  - `slot_amf_max`;
  - `slot75_amf25`;
- selector ridge para diagnostico y politica robusta por horizonte para la salida principal;
- decoder copy-skip que mueve detalle visual del frame observado;
- oracle de candidatos para medir la brecha real del selector.

## Por que es distinto

Este probe ataca el bottleneck identificado:

- si `candidate_oracle_mse` es mucho mejor que `never_definitive_mse`, el problema sigue siendo selector;
- si `amf_world_mse` mejora a `slot_velocity_mse`, la memoria AMF ayuda;
- si `transition_cells` no crece, escalar celulas no esta aumentando cobertura efectiva;
- si todos los candidatos son malos, el encoder/dinamica base sigue fallando.

## Comandos

Smoke:

```powershell
python phase11a_never_definitive_codec_probe.py --train-sequences 80 --test-sequences 20 --selector-sequences 20 --out results/phase11a_never_definitive_codec_probe_smoke.json
```

Principal:

```powershell
python phase11a_never_definitive_codec_probe.py --train-sequences 220 --test-sequences 40 --selector-sequences 60 --out results/phase11a_never_definitive_codec_probe_220_40.json
```

Escalado:

```powershell
python phase11a_never_definitive_codec_probe.py --train-sequences 650 --test-sequences 40 --selector-sequences 100 --out results/phase11a_never_definitive_codec_probe_650_40.json
```

## Criterio de exito

Debe superar `last` en MSE, elegir entre prediccion cinematica, AMF y mezclas hibridas, y cerrar parte de la brecha con `candidate_oracle_mse` en h10/h17.

Una mejora fuerte necesita:

```text
mse_skill_vs_last > 0
oracle_gap_mse bajo
selector converge a una rama robusta y medible
transition_cells > baseline trivial
```

## Estado

Compilacion limpia:

```powershell
python -m py_compile phase11a_never_definitive_codec_probe.py
```

Tambien compilaron juntos los scripts clave:

```powershell
python -m py_compile phase11a_never_definitive_codec_probe.py phase11a_never_world_codec_probe.py phase11a_never_bottleneck_audit.py
```

## Resultados medidos

Smoke 80/20:

| horizonte | MSE definitive | MSE last | skill vs last | oracle gap | rama |
|---|---:|---:|---:|---:|---|
| h1 | 0.027017 | 0.051743 | 0.477865 | 0.002097 | amf_world |
| h5 | 0.065859 | 0.077887 | 0.154425 | 0.005616 | amf_world |
| h10 | 0.071182 | 0.078085 | 0.088404 | 0.004713 | amf_world |
| h17 | 0.076041 | 0.080142 | 0.051168 | 0.006682 | amf_world |

220/40 con politica por horizonte:

| horizonte | MSE definitive | MSE last | skill vs last | oracle gap | rama |
|---|---:|---:|---:|---:|---|
| h1 | 0.024519 | 0.046394 | 0.471503 | 0.000942 | slot_amf_mean |
| h5 | 0.051779 | 0.071819 | 0.279034 | 0.000992 | slot_amf_mean |
| h10 | 0.054426 | 0.078210 | 0.304111 | 0.000322 | slot_amf_mean |
| h17 | 0.054294 | 0.075827 | 0.283973 | 0.000301 | slot_amf_mean |

650/40 con politica por horizonte:

| horizonte | MSE definitive | MSE last | skill vs last | oracle gap | rama |
|---|---:|---:|---:|---:|---|
| h1 | 0.022883 | 0.047569 | 0.518941 | 0.000555 | slot_amf_mean |
| h5 | 0.049480 | 0.070158 | 0.294730 | 0.001239 | slot_amf_mean |
| h10 | 0.051417 | 0.071599 | 0.281871 | 0.000933 | slot_amf_mean |
| h17 | 0.054018 | 0.073821 | 0.268249 | 0.000402 | slot_amf_mean |

Las celulas efectivas crecieron de 864 en 220/40 a 2596 en 650/40. Eso confirma que este codec si escala memoria real, no solo `max_cells` nominal.

## Selector por muestra

Tambien se probo un selector KNN por muestra (`knn_test_metrics` en los JSON). Con los candidatos hibridos ya casi coincide con la politica por horizonte, pero no se promovio como salida principal porque la politica global es mas simple y estable:

- 650/40 KNN h10/h17 elige esencialmente `slot_amf_mean`;
- el gap principal ya no es selector grueso, sino casos puntuales de colision/occlusion.

Esto confirma que el bottleneck restante existe, pero necesita mejores features de incertidumbre/colision, no solo un selector no lineal simple.

## Diagnostico actualizado

El mejor resultado nuevo no viene de un decoder monolitico. Viene de usar Never como ciclo completo y un decoder hibrido conservador:

```text
encoder de slots -> slot dynamics + AMF world memory -> mezcla copy-skip -> politica por horizonte
```

El bottleneck restante quedo pequeno: la brecha contra `candidate_oracle_mse` esta cerca de 0.0004-0.0012 en 650/40. Lo siguiente es modelar colisiones/oclusiones y validar en un segundo dataset.
