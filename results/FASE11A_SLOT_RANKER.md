# Fase 11A - Slot/Object Ranker Multi-Hypothesis Encoder

Este probe baja la seleccion del encoder/decoder multi-hipotesis desde frame
completo a slot/objeto. La motivacion viene de modelos objeto-centricos y de
tracking moderno: en cruces y oclusiones, una hipotesis global puede acertar un
digito y fallar el otro. Aqui cada digito escoge su propia reconstruccion y luego
el decoder compone el frame final.

## Arquitectura

- Encoder causal simple: velocidad `t-1 -> t`.
- Encoder motion-token: memoria compacta de trayectoria.
- Dos AMF compactos, uno por hipotesis.
- Decoder por capas/slots: renderiza dos capas por digito.
- Candidatos por slot:
  - `simple`
  - `token`
  - `max_beta_{0.00,0.25,0.50,0.75,1.00}`
  - `blend_beta_{0.00,0.25,0.50,0.75,1.00}`
  - `wide_max_beta_1.00`
  - `wide_blend_beta_0.75`
  - `very_wide_max_beta_1.00`
  - `very_wide_blend_beta_0.75`
  - `shift_simple_mid`, `shift_token_mid`
  - `shift_simple_anti`, `shift_token_anti`
  - `max_shift_mid`
- Ranker por slot:
  - clasificador de mejor candidato por objeto,
  - regresor multi-output de IoU por candidato/objeto.
- Ranker marginal por slot:
  - aprende, para cada candidato de un digito, el mejor IoU de frame posible al
    combinarlo con el otro digito.
  - Esta variante alinea el entrenamiento con la metrica final sin entrenar las
    144 combinaciones completas como una salida gigante.
- Diagnostico adicional:
  - `slot_pair_reg_ranker` predice IoU de 144 combinaciones de slots, pero no
    supera al regresor por slot.
  - `slot_frame_oracle` mide el techo combinatorio real de escoger una hipotesis
    distinta para cada digito.
- Entrenado solo con train real, evaluado en test real.
- AMF no recibe pixeles crudos.

## Resultado Mediano Verificado

Comando:

```powershell
python phase11a_slot_ranker_probe.py --train-sequences 220 --test-sequences 20 --out results/phase11a_slot_ranker_probe.json
```

| horizonte | simple | token | max beta 1.0 | slot reg | frame reg | frame extra | frame ensemble | slot frame oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.5976 | 0.5989 | 0.6201 | 0.6313 | 0.6290 | 0.6417 | 0.6324 | 0.6955 |
| h5 | 0.2779 | 0.2562 | 0.3053 | 0.3420 | 0.3428 | 0.3457 | 0.3460 | 0.4653 |
| h10 | 0.2168 | 0.1688 | 0.2223 | 0.2566 | 0.2482 | 0.2372 | 0.2448 | 0.3374 |
| h17 | 0.1981 | 0.1546 | 0.1909 | 0.2206 | 0.2090 | 0.2189 | 0.2219 | 0.3169 |

## Resultado Grande Verificado

Comando:

```powershell
python phase11a_slot_ranker_probe.py --train-sequences 650 --test-sequences 40 --out results/phase11a_slot_ranker_probe_hybrid_650_40.json
```

| horizonte | simple | token | max beta 1.0 | slot reg | frame reg | frame extra | frame ensemble | slot frame oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.6029 | 0.6000 | 0.6259 | 0.6378 | 0.6433 | 0.6420 | 0.6440 | 0.7125 |
| h5 | 0.2623 | 0.2411 | 0.2950 | 0.3634 | 0.3517 | 0.3507 | 0.3534 | 0.4606 |
| h10 | 0.1976 | 0.1902 | 0.2231 | 0.2634 | 0.2716 | 0.2647 | 0.2657 | 0.3444 |
| h17 | 0.1715 | 0.1543 | 0.1829 | 0.2114 | 0.2196 | 0.2115 | 0.2199 | 0.3012 |

La extension de score-routing hibrido mezcla scores locales de objeto con scores
globales de frame, normalizados por ejemplo. No cambia el resultado calibrado
h10/h17, pero sube el techo aprendido de test: h1 `0.6447` con
`slot_hybrid_extra_ranker`, h10 `0.2775` con `slot_hybrid_ranker`, y h17
`0.2218` con `slot_frame_ensemble_w75_ranker`.

## Comparacion Contra Ranker De Frame

En la corrida grande `650/40`, el ranker anterior por frame tenia:

- h1 class-ranker `0.6430`
- h10 reg-ranker `0.2326`
- h17 reg-ranker `0.1921`

El nuevo slot-ranker tiene:

- h1 hybrid-extra `0.6447`, por encima del class-ranker de frame `0.6430`.
- h10 hybrid `0.2775`, una mejora clara sobre `0.2326`.
- h17 weighted frame-ensemble `0.2218`, una mejora clara sobre `0.1921`.
- slot-frame-oracle h10/h17 `0.3444 / 0.3012`, lo que demuestra que hay margen
  real en la seleccion por slots.

## Seleccion Calibrada Interna

Para evitar escoger manualmente el mejor ranker mirando test, el probe ahora
separa train en `570` secuencias de ajuste y `80` de calibracion. Con esa
calibracion escoge una familia aprendida por horizonte y luego reporta su IoU en
test:

| horizonte | selector calibrado | calib IoU | test IoU |
|---|---|---:|---:|
| h1 | `slot_reg_ranker` | 0.6221 | 0.6378 |
| h5 | `slot_reg_ranker` | 0.3242 | 0.3634 |
| h10 | `slot_reg_ranker` | 0.2758 | 0.2634 |
| h17 | `slot_frame_ensemble_ranker` | 0.2137 | 0.2199 |

Resultado calibrado h10/h17: `0.2634 / 0.2199`. El test-best h10/h17 queda
arriba (`0.2775 / 0.2218`), pero la cifra calibrada es la comparacion mas
defendible porque no usa test para elegir la familia.

## Pairwise Ranker Opcional

Inspirado por tokenizers/routing de video, tambien se probo un ranker pairwise
que aprende `contexto causal + features del par candidato -> IoU`. Queda
habilitable con `--enable-pairwise-ranker`, pero no forma parte de la ruta
default porque no escala bien. En `650/40`, su test h10/h17 fue
`0.2402 / 0.1929`, por debajo del score-routing hibrido `0.2775 / 0.2218` y de
la seleccion calibrada `0.2634 / 0.2199`.

## Cell/Tile Router Opcional

Tambien se probo `--cell-router`, un decoder espacial que enruta candidatos por
tiles de 16. El tile-oracle h10/h17 `0.3743 / 0.3225` confirma que la
granularidad espacial tiene techo, pero el router aprendido no escala:
`0.2383 / 0.2075`. Por eso queda como evidencia negativa, documentada en
`results/FASE11A_CELL_ROUTER.md`.

## Conclusion

El mejor avance nuevo combina candidatos `shift_*`, `wide_*` / `very_wide_*` y
score-routing hibrido. Los `shift_*` elevan mucho el techo combinatorio y el
router hibrido sube el test-best de h10/h17. Esto mejora el encoder/decoder
causal y confirma que una decision por objeto alineada a frame-IoU es mas
informativa que una decision unica por frame para rollouts largos. El selector
todavia no captura todo el oracle, asi que el siguiente salto debe estar en un
selector por celda/slot mas fuerte.

Los targets extremos siguen abiertos. El siguiente salto no es meter pixeles
crudos en AMF, sino entrenar un selector por celda visual/slot con objetivo de
frame-IoU y un decoder completivo mas fuerte para oclusiones.
