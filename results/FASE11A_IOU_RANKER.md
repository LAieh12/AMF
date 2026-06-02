# Fase 11A - IoU Ranker Multi-Hypothesis Encoder

Este probe reemplaza el selector binario por un ranker directo de IoU.

## Arquitectura

- Encoder causal simple: velocidad `t-1 -> t`.
- Encoder motion-token: memoria compacta de trayectoria.
- Dos AMF compactos, uno por hipotesis.
- Candidatos visuales:
  - `simple`
  - `token`
  - `max_beta_{0.00,0.25,0.50,0.75,1.00}`
  - `blend_beta_{0.00,0.25,0.50,0.75,1.00}`
- Ranker:
  - clasificador de mejor candidato,
  - regresor multi-output de IoU por candidato.
- Entrenado solo con train real, evaluado en test real.
- AMF no recibe pixeles crudos.

## Resultado Mediano Verificado

Comando:

```powershell
python phase11a_iou_ranker_probe.py --train-sequences 220 --test-sequences 20 --out results/phase11a_iou_ranker_probe.json
```

| horizonte | simple | token | max beta 1.0 | class ranker | reg ranker | oracle |
|---|---:|---:|---:|---:|---:|---:|
| h1 | 0.5976 | 0.5989 | 0.6201 | 0.6300 | 0.6282 | 0.6735 |
| h5 | 0.2779 | 0.2562 | 0.3053 | 0.2826 | 0.3052 | 0.3364 |
| h10 | 0.2168 | 0.1688 | 0.2223 | 0.2257 | 0.2396 | 0.2620 |
| h17 | 0.1981 | 0.1546 | 0.1909 | 0.2033 | 0.2101 | 0.2424 |

## Resultado Grande Verificado

Comando:

```powershell
python phase11a_iou_ranker_probe.py --train-sequences 650 --test-sequences 40 --out results/phase11a_iou_ranker_probe_650_40.json
```

| horizonte | simple | token | max beta 1.0 | class ranker | reg ranker | oracle |
|---|---:|---:|---:|---:|---:|---:|
| h1 | 0.6029 | 0.6000 | 0.6259 | 0.6430 | 0.6361 | 0.6798 |
| h5 | 0.2623 | 0.2411 | 0.2950 | 0.2734 | 0.2950 | 0.3219 |
| h10 | 0.1976 | 0.1902 | 0.2231 | 0.2251 | 0.2326 | 0.2661 |
| h17 | 0.1715 | 0.1543 | 0.1829 | 0.1757 | 0.1921 | 0.2229 |

La escala grande confirma que el resultado no era solo ruido del split
`220/20`: el clasificador es mejor en h1, mientras que el regresor directo de
IoU es la mejor ruta no-oracle en h10/h17.

## Conclusion

El regressor-ranker es la mejor ruta causal actual en h10/h17:

- h10 sube de causal simple `0.2168` a `0.2396`.
- h17 sube de causal simple `0.1981` a `0.2101`.
- h1 sube sobre max-fusion: `0.6201` a `0.6282`.
- En la corrida grande `650/40`, h10 sube de causal simple `0.1976` a
  `0.2326`, y h17 de `0.1715` a `0.1921`.

Todavia queda margen oracle (`0.2620` h10 y `0.2424` h17). El siguiente paso es
llevar este ranker al runner principal y despues entrenar un selector por objeto
o por celda visual, no solo por frame/horizonte.
