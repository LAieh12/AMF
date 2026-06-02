# Fase 11A - Cell/Tile Router Decoder Probe

## Objetivo

Probar una traduccion ligera de tokenizers/decoders modernos de video: elegir
hipotesis visuales por celda espacial, no solo por frame, slot o par de slots.
El probe usa solo hipotesis causales generadas por AMF (`simple`, `token`,
`wide_*`, `shift_*`, etc.). El frame real descargado se usa solo como target de
entrenamiento/evaluacion.

## Implementacion

Archivo: `phase11a_cell_router_probe.py`

Dispatcher:

```bash
python phase11a_slot_ranker_probe.py --cell-router --train-sequences 650 --test-sequences 40 --tile-size 16 --out results/phase11a_cell_router_probe_weighted_650_40_tile16.json
```

El decoder divide el frame en tiles y entrena dos routers por horizonte:

- `tile_router_class`: clasificador del mejor candidato por tile.
- `tile_router_reg`: regresor multi-output del IoU esperado por candidato.

La version final pondera tiles activos/dificiles para no aprender solo fondo
facil.

## Resultado 220/20

| tile | h10 router reg | h17 router reg | h10 tile oracle | h17 tile oracle |
|---:|---:|---:|---:|---:|
| 8 | 0.2109 | 0.2164 | 0.4271 | 0.4042 |
| 16 | 0.2256 | 0.2242 | 0.3720 | 0.3448 |

Tile 16 fue mas estable y por eso se escalo.

## Resultado 650/40

| horizonte | simple | token | max beta 1.0 | frame oracle | tile oracle | tile router class | tile router reg |
|---|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.6029 | 0.6000 | 0.6259 | 0.6878 | 0.7198 | 0.5910 | 0.6433 |
| h5 | 0.2623 | 0.2411 | 0.2950 | 0.4100 | 0.4767 | 0.3129 | 0.3451 |
| h10 | 0.1976 | 0.1902 | 0.2231 | 0.3040 | 0.3743 | 0.2079 | 0.2383 |
| h17 | 0.1715 | 0.1543 | 0.1829 | 0.2610 | 0.3225 | 0.1775 | 0.2075 |

## Conclusion

El tile oracle confirma que el decoder por celda tiene techo: h10/h17
`0.3743 / 0.3225`. Pero el router aprendido no escala: h10/h17
`0.2383 / 0.2075`, por debajo del slot-ranker hibrido (`0.2775 / 0.2218`
test-best, `0.2634 / 0.2199` calibrado).

Queda como evidencia negativa. El siguiente salto no debe ser simplemente mas
granularidad espacial; necesita memoria temporal latente o un selector
espacio-temporal entrenado con mejor objetivo de secuencia.
