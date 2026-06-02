# Fase 10c - N.E.V.E.R. AMF Orchestrator

Objetivo: conectar el pipeline completo dentro de `NEVER`.

```text
prompt usuario
  -> GPT/API-compatible action provider
  -> vector accion A(t)
  -> frame(t) -> encoder -> S(t)
  -> AMF world model -> S(t+1)
  -> frame real NEVER -> encoder -> S(t+1) real
  -> feedback online: compara prediccion vs real
  -> metaplasticidad
  -> decoder vectorial -> frame renderizado
```

## Integracion realizada

- `NEVER/src/main.cu` ya no instancia `SNNEngine`.
- `NEVER/src/inference_core.cu` ya no ejecuta `snn_lif_forward`.
- `NEVER/src/engine/snn_engine.cuh` quedo como marcador retirado.
- `NEVER/scripts/train_dynamic_snn.py` redirige al orquestador AMF.
- `NEVER/src/engine/amf_world_model.cuh` agrega runtime AMF, latent state,
  action vector, identidad congelada y training online nativo.
- `NEVER/src/engine/amf_vector_decoder.cuh` agrega decoder vectorial CUDA.
- `NEVER/src/engine/action_orchestrator.hpp` agrega fallback local C++.
- `NEVER/scripts/phase10c_never_amf_orchestrator.py` ejecuta el loop real con
  el AMF y codec exportados en fases 10a/10b, incluyendo feedback de cada
  frame real hacia `learn_transition`.

## Demo verificada

Comando:

```powershell
python run_phase10c.py --offline --prompt "Personaje saltando, camara rotando 45 grados" --steps 36 --resolution 128
```

Resultado:

| metrica | valor |
|---|---:|
| latent bytes | 32 |
| frame bytes | 262144 |
| compression | 8192x |
| AMF memory MB | 0.686646 |
| cells start | 9000 |
| cells end | 9000 |
| mean frame MSE | 0.000473 |
| ms per step | 14.704744 |
| feedback events | 36 |
| mean prediction error antes online | 0.194045 |
| mean prediction error despues online | 0.193772 |
| online probe score | 0.25 -> 1.00 |
| online probe MSE | 0.094373 -> 0.000000 |
| session MSE | 0.003052 -> 0.002328 |

Metaplasticidad en runtime:

| status | count |
|---|---:|
| metaplasticity_adapted_cell | 14 |
| buffered_possible_noise | 19 |
| explained_by_existing_cell | 5 |
| created_confirmed_novelty | 1 |

El probe online repitio una transicion especifica del mundo NEVER y produjo:

```text
buffered_possible_noise
buffered_possible_noise
created_confirmed_novelty
explained_by_existing_cell
```

La puntuacion del probe subio de `0.25` a `1.00`, igual que el criterio de
aprendizaje online usado en Fase 9. El MSE del probe bajo de `0.094373` a
`0.000000`.

La curva de sesiones online tambien baja con el tiempo:

| medicion | valor |
|---|---:|
| session first MSE | 0.003052 |
| session last MSE | 0.002328 |
| session first score | 0.75 |
| session last score | 1.00 |

## Artefactos

- `NEVER/outputs/phase10c_never_loop.json`
- `NEVER/outputs/phase10c_never_loop_frames.npz`
- `NEVER/outputs/phase10c_identity_geometry.json`
- `results/phase10c_latest.json`

## Estado CUDA

Build CUDA verificado en Windows/MSVC + CUDA:

```powershell
cmake -S . -B build_phase10c
cmake --build build_phase10c --config Release
```

Resultado de configuracion:

```text
CUDA compiler identification is NVIDIA 13.3.33
Build files have been written to: J:/aa/NEVER/build_phase10c
```

Resultado de compilacion:

```text
never.vcxproj -> J:\aa\NEVER\build_phase10c\Release\never.exe
```

Ejecucion CUDA verificada:

```powershell
.\build_phase10c\Release\never.exe "Personaje saltando, camara rotando 45 grados"
```

Salida clave:

```text
[AMF Runtime] Warm AMF metadata loaded: ../data/phase10a_warm_amf.json
[AMF Runtime] cells=9000, arrays=0.686646 MB, delta-learning=true
[AMF Identity] Frozen geometry vertices=2048, style_code=1
[AMF Loop] frame=0 ... online_status=created_first_online_cell mse=0.00248913->0 online_cells=1
[AMF Loop] frame=7 ... online_status=explained_by_existing_cell mse=0.00199606->0.00161308 online_cells=1
[AMF Online Probe] score=0.25->1 mse=0.102484->0 final_status=explained_by_existing_cell online_cells=2
N.E.V.E.R. Phase 10c AMF core ready.
```

La validacion Python/NumPy sigue siendo el loop completo autoritativo para
exports reales de codec/AMF y metricas. La validacion CUDA confirma que el
entrypoint nativo de NEVER compila, carga metadata AMF caliente, mantiene
identidad congelada y ejecuta el camino action -> latent -> feedback online ->
AMF runtime.
