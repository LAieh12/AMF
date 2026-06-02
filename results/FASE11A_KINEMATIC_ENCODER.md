# Fase 11A - Kinematic Encoder Probe

Este probe prueba una tercera hipotesis causal de encoder:
`kinematic_token_sequence`. Usa solo informacion observada hasta `t`: velocidad
reciente, velocidad suavizada, aceleracion acotada y amortiguacion cerca de
paredes. La motivacion viene de encoders de tracking/video con memoria temporal
y trayectoria objeto-centrica.

AMF sigue recibiendo solo latentes compactos, nunca pixeles crudos.

## Resultado Mediano Verificado

Comando:

```powershell
python phase11a_slot_ranker_probe.py --kinematic-probe --train-sequences 220 --test-sequences 20 --out results/phase11a_kinematic_encoder_probe.json
```

| horizonte | simple | token | kinematic | max all | rf ranker | extra ranker | ensemble | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 0.5976 | 0.5989 | 0.5766 | 0.6147 | 0.6087 | 0.5889 | 0.6098 | 0.6804 |
| h5 | 0.2779 | 0.2562 | 0.2768 | 0.3049 | 0.3093 | 0.3132 | 0.3163 | 0.3702 |
| h10 | 0.2168 | 0.1688 | 0.2108 | 0.2217 | 0.2360 | 0.2324 | 0.2340 | 0.2720 |
| h17 | 0.1981 | 0.1546 | 0.1982 | 0.1945 | 0.2150 | 0.2047 | 0.2039 | 0.2624 |

## Conclusion

La hipotesis kinematica no supera al slot-ranker actual. Aporta algo en h17
frente a `simple` aislado y tiene oracle razonable, pero el ranker aprendido no
alcanza el resultado ya validado con candidatos `wide_*`/`very_wide_*`.

Se mantiene como evidencia negativa: suavizar velocidad/aceleracion causal no
basta. El siguiente encoder prometedor tendria que representar incertidumbre de
trayectoria por slot/celda, no una sola velocidad suavizada.
