# Fase 11A - Confidence Selector Probe

Este probe implementa el siguiente paso del encoder multi-hipotesis: no solo
fusionar causal-simple + motion-token con un peso global, sino aprender un
selector causal por horizonte.

## Arquitectura

- AMF simple: encoder causal `t-1 -> t`.
- AMF token: encoder causal con memoria de motion tokens.
- Selector: `RandomForestClassifier` entrenado solo en train real.
- Features del selector:
  - horizonte normalizado,
  - diferencia entre renders de hipotesis,
  - areas activas,
  - distancia entre centros predichos,
  - diferencia de velocidades,
  - features de pared/interaccion al inicio y en la prediccion.
- AMF no recibe pixeles crudos.

## Resultado Mediano Verificado

Comando:

```powershell
python phase11a_confidence_selector_probe.py --train-sequences 220 --test-sequences 20 --out results/phase11a_confidence_selector_probe.json
```

| horizonte | simple | token | max fusion | selected | oracle |
|---|---:|---:|---:|---:|---:|
| h1 | 0.5976 | 0.5989 | 0.6201 | 0.6020 | 0.6645 |
| h5 | 0.2779 | 0.2562 | 0.3053 | 0.2513 | 0.3327 |
| h10 | 0.2168 | 0.1688 | 0.2223 | 0.2238 | 0.2551 |
| h17 | 0.1981 | 0.1546 | 0.1909 | 0.1956 | 0.2421 |

## Conclusion

El selector aprende algo util en h10 y hay margen oracle real, pero todavia no
domina la fusion max ni la ruta causal simple en todos los horizontes. No se
integra como resultado principal. La evidencia indica que el selector debe
optimizar IoU directamente o usar una perdida/ranking por hipotesis, no solo un
clasificador binario por `token mejor que simple`.
