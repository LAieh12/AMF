# Fase 10b - Encoder/Decoder visual para AMF world model

Objetivo: comprimir frames visuales grandes a un vector latente compacto `S(t)`,
usar el AMF calentado para predecir `S(t+1)` y decodificar un frame fisicamente
coherente sin hacer que AMF cargue pixeles.

Flujo:

```text
frame visual -> encoder -> S(t) -> AMF world model -> S(t+1) -> decoder -> frame futuro
```

Reglas: encoder/decoder sin backprop = True,
AMF ve latent y no pixeles = True,
latent dim constante = 8.

## Escalabilidad por resolucion

| res | frame KB | latent bytes | compression | enc ms | dec ms | AMF ms | recon MSE | recon IoU | pred latent MSE | pred frame MSE | pred IoU | raw pixel AMF est MB | actual AMF MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16.000000 | 32 | 512.000000x | 0.632275 | 0.159775 | 0.611246 | 0.000049 | 0.694697 | 0.044428 | 0.000166 | 0.612688 | 281.352997 | 0.686646 |
| 64 | 64.000000 | 32 | 2048.000000x | 0.754560 | 0.212386 | 0.521762 | 0.000072 | 0.642592 | 0.038229 | 0.000157 | 0.596431 | 1125.102997 | 0.686646 |
| 128 | 256.000000 | 32 | 8192.000000x | 1.827670 | 0.883561 | 0.604238 | 0.000076 | 0.646557 | 0.037913 | 0.000159 | 0.594923 | 4500.102997 | 0.686646 |
| 256 | 1024.000000 | 32 | 32768.000000x | 6.632363 | 5.452876 | 0.867607 | 0.000077 | 0.646847 | 0.036390 | 0.000158 | 0.593726 | 18000.102997 | 0.686646 |

## Propiedades del latent

- permanencia identity distance: 0.003235
- permanencia shift error: 0.000034
- separabilidad identity distance: 0.750003
- continuidad mean step: 0.106556
- continuidad max step: 0.227218

## Export

- AMF usado: `data/phase10a_warm_amf.npz`
- sample frames: `data\phase10b_sample_frames.npz`
- codec metadata por resolucion en `data/phase10b_visual_codec_<res>.json`

## Lectura

La escala visual ya no entra al AMF como pixeles. Un frame `256x256x4` pesa
1048576 bytes en float32, pero el estado que ve AMF pesa 32 bytes. El estimado
de memoria si AMF guardara celdas crudas de pixeles llega a miles de MB, contra
menos de 1 MB de arrays del AMF latent actual. El encoder conserva posicion,
velocidad, identidad visual, distancia a paredes y continuidad; el decoder no
busca fotorrealismo, sino coherencia fisica para el mundo simulado.
