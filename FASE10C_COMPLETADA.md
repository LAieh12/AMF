# FASE10C_COMPLETADA

Fase 10c conecta N.E.V.E.R. al AMF world model.

Entregables:

- `run_phase10c.py`
- `NEVER/scripts/phase10c_never_amf_orchestrator.py`
- `NEVER/src/engine/amf_world_model.cuh`
- `NEVER/src/engine/amf_vector_decoder.cuh`
- `NEVER/src/engine/action_orchestrator.hpp`
- `NEVER/src/main.cu`
- `NEVER/src/inference_core.cu`
- `NEVER/src/engine/snn_engine.cuh`
- `NEVER/scripts/train_dynamic_snn.py`
- `NEVER/outputs/phase10c_never_loop.json`
- `NEVER/outputs/phase10c_never_loop_frames.npz`
- `NEVER/outputs/phase10c_identity_geometry.json`
- `results/phase10c_latest.json`
- `results/FASE10C_RESULTADOS.md`

Resultado local verificado:

- prompt: `Personaje saltando, camara rotando 45 grados`
- steps: 36
- resolution: 128
- latent bytes: 32
- frame bytes: 262144
- compression: 8192x
- AMF memory MB: 0.686646
- cells: 9000 -> 9000
- mean frame MSE: 0.000473
- ms per step: 14.704744
- identity frozen: true
- feedback events: 36
- online probe score: 0.25 -> 1.00
- online probe MSE: 0.094373 -> 0.000000
- online session MSE: 0.003052 -> 0.002328

Metaplasticidad:

- metaplasticity_adapted_cell: 14
- buffered_possible_noise: 19
- explained_by_existing_cell: 5
- created_confirmed_novelty: 1

La rama SNN activa fue retirada de `main.cu` e `inference_core.cu`. El archivo
`snn_engine.cuh` queda solo como marcador de migracion hacia
`AMFWorldModelRuntime`.

Build CUDA verificado:

- configure: `cmake -S . -B build_phase10c`
- CUDA compiler: NVIDIA 13.3.33
- build: `cmake --build build_phase10c --config Release`
- executable: `J:\aa\NEVER\build_phase10c\Release\never.exe`
- run: carga `../data/phase10a_warm_amf.json`, reporta `cells=9000`,
  `arrays=0.686646 MB`, identidad congelada, 8 frames de accion/latente con
  feedback online y probe nativo `score=0.25->1`.

La ruta Python/NumPy sigue siendo la validacion completa con exports reales de
Fase 10a y Fase 10b; la ruta CUDA ya compila y ejecuta el entrypoint nativo de
NEVER con training online en runtime.
