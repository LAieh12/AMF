# Fase 11A - Patch Decoder Probe

Este probe prueba un decoder externo por celda/pixel inspirado en tokenizers y
VAEs de video modernos: recibe hipotesis visuales causales, coordenadas,
horizonte, distancia a slots y features latentes, y aprende una mascara externa
con `HistGradientBoostingClassifier`.

AMF sigue sin recibir pixeles crudos. El decoder usa pixeles solo como objetivo
supervisado externo, igual que los otros decoders visuales.

## Resultado Mediano Verificado

Comando:

```powershell
python phase11a_slot_ranker_probe.py --patch-decoder --train-sequences 220 --test-sequences 20 --per-frame 384 --max-pixels-per-horizon 180000 --calibration-sequences 40 --out results/phase11a_patch_decoder_probe.json
```

| horizonte | simple | token | max beta 1.0 | frame oracle | patch decoder |
|---|---:|---:|---:|---:|---:|
| h1 | 0.5976 | 0.5989 | 0.6201 | 0.6705 | 0.3636 |
| h5 | 0.2779 | 0.2562 | 0.3053 | 0.3357 | 0.2459 |
| h10 | 0.2168 | 0.1688 | 0.2223 | 0.2608 | 0.1864 |
| h17 | 0.1981 | 0.1546 | 0.1909 | 0.2428 | 0.1639 |

## Conclusion

El decoder por patches puro no gana. Aprende una mascara plausible, pero pierde
estructura geometrica y queda debajo de las hipotesis renderizadas. La leccion
arquitectonica es util: el decoder local debe actuar como refinador/gate sobre
las hipotesis geometricas, no como reemplazo completo.

La mejora que si funciono fue agregar candidatos locales `wide_*`/`very_wide_*`/`shift_*`
al slot-ranker, que subio la ruta causal grande calibrada a h10/h17
`0.2634 / 0.2199` calibrado (`0.2775 / 0.2218` como test-best aprendido).
