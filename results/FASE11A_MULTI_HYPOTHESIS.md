# Fase 11A - Multi-Hypothesis Encoder Probe

Este probe ataca el fallo que quedo expuesto en la evaluacion causal: una sola
velocidad causal no basta para dos digitos con cruces y oclusiones.

## Arquitectura

- Encoder causal simple: velocidad `t-1 -> t`.
- Encoder causal motion-token: memoria de trayectoria compacta inspirada en
  TAPIR/CoTracker.
- AMF separado por hipotesis: cada encoder alimenta su propio AMF compacto.
- Decoder fusionado: combina la hipotesis causal simple con la motion-token
  usando pesos aprendidos en train por horizonte.
- AMF no recibe pixeles crudos.

## Resultado Mediano Verificado

Comando:

```powershell
python phase11a_multi_hypothesis_probe.py --train-sequences 220 --test-sequences 20 --out results/phase11a_multi_hypothesis_probe.json
```

Dataset real descargado: `data/MovingMNIST/mnist_test_seq.npy`, shape
`(20, 10000, 64, 64)`.

| modelo | h1 IoU | h5 IoU | h10 IoU | h17 IoU | center err h17 px |
|---|---:|---:|---:|---:|---:|
| dual_ridge | 0.5696 | 0.2242 | 0.1161 | 0.1046 | 19.211 |
| dual_AMF | 0.6201 | 0.3053 | 0.2223 | 0.1909 | 14.598 |

Pesos aprendidos para AMF por horizonte:

```json
{"1": 1.0, "5": 1.0, "10": 1.0, "17": 1.0}
```

## Comparacion Contra Rutas Previas

En la corrida grande `650/40` previa:

- AMF causal simple: one-step `0.5805`, h10 `0.1976`, h17 `0.1715`.
- AMF motion-token: one-step `0.6191`, h10 `0.1902`, h17 `0.1543`.

En este probe mediano:

- dual_AMF h1 `0.6201`
- dual_AMF h10 `0.2223`
- dual_AMF h17 `0.1909`

Conclusion: el encoder multi-hipotesis mejora el perfil causal frente a usar
una sola trayectoria, especialmente en h1/h10. Todavia no resuelve los targets
extremos; el siguiente salto necesita selector de confianza por objeto/frame y
decoder generativo completivo mas fuerte.
