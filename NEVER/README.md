# N.E.V.E.R. AMF Core

**Status:** Phase 10c integrated prototype  
**Active engine:** AMF latent world model, not SNN  
**Target:** prompt-conditioned vectorial animation with frozen identity

N.E.V.E.R. now runs the AMF world-model path:

```text
user prompt
  -> external action orchestrator (GPT-5.5/API-compatible, local fallback)
  -> action vector A(t)
  -> frame(t) -> visual encoder -> latent S(t)
  -> AMF world model predicts S(t+1)
  -> real NEVER frame -> encoder -> real S(t+1)
  -> online feedback compares prediction vs real latent
  -> metaplasticity reinforces/buffers/creates cells
  -> vector decoder renders frame(t+1)
```

The previous dynamic SNN branch has been retired. The active code no longer
instantiates `SNNEngine`; `src/engine/snn_engine.cuh` is a migration marker that
points reviewers to `AMFWorldModelRuntime`.

## Core Features

- **External Orchestrator:** `scripts/phase10c_never_amf_orchestrator.py`
  maps text prompts to bounded action vectors. It can call the OpenAI Responses
  API when `OPENAI_API_KEY` is present, and has a deterministic offline fallback
  for reproducible local tests.
- **Frozen Identity:** one-shot geometry metadata is exported to
  `outputs/phase10c_identity_geometry.json`; identity/style remain locked.
- **AMF World Model:** loads the warmed Phase 10a export
  `../data/phase10a_warm_amf.npz`; the CUDA side exposes
  `AMFWorldModelRuntime` and an AMF delta kernel integration point.
- **Visual Latent Codec:** reuses Phase 10b encoder/decoder so large frames
  compress to a constant 8D latent (`32` bytes).
- **Metaplasticity:** low-error transitions reinforce existing cells, possible
  noise is buffered, and repeated high-error novelty creates confirmed cells
  while respecting the cell cap.
- **Online Training:** while NEVER runs, every real frame is encoded back to a
  latent and sent into `AMFDynamicsWorldModel.learn_transition`. No full
  retrain or backprop is used; the warmed AMF adapts to the current scene.
- **Vectorial Presenter:** `amf_decode_latent_to_rgba_kernel` renders a
  coherent RGBA frame from `S(t+1)`.

## Run The Local Loop

```powershell
python scripts\phase10c_never_amf_orchestrator.py --offline --prompt "Personaje saltando, camara rotando 45 grados" --steps 36 --resolution 128
```

Outputs:

- `outputs/phase10c_never_loop.json`
- `outputs/phase10c_never_loop_frames.npz`
- `outputs/phase10c_identity_geometry.json`

Verified local result:

- latent bytes: `32`
- frame bytes at `128x128x4`: `262144`
- compression: `8192x`
- AMF memory: `0.686645 MB`
- cell count: `9000 -> 9000`
- feedback events: `36`
- mean prediction error: `0.194045 -> 0.193772` after online feedback
- online probe: `0.25 -> 1.00`, MSE `0.094373 -> 0.000000`
- online session MSE: `0.003052 -> 0.002328`
- statuses: `explained_by_existing_cell`, `buffered_possible_noise`,
  `metaplasticity_adapted_cell`, `created_confirmed_novelty`

## CUDA Entrypoint

```powershell
cmake -S . -B build_phase10c
cmake --build build_phase10c --config Release
```

Then run:

```powershell
build_phase10c\Release\never.exe "Personaje saltando, camara rotando 45 grados"
```

Verified CUDA build:

- compiler: NVIDIA CUDA `13.3.33`
- executable: `build_phase10c\Release\never.exe`
- runtime: loads `../data/phase10a_warm_amf.json`
- AMF metadata: `cells=9000`, `arrays=0.686646 MB`
- identity: frozen, `2048` vertices
- online runtime: per-frame `learn_from_real_latent`, native probe
  `score=0.25->1`

The CUDA executable uses the same AMF latent state structs, frozen identity
metadata, action vector, online feedback loop, and vector decoder kernel. The
Python orchestrator remains the authoritative metric loop because it loads the
full NumPy AMF export and visual codec from Phases 10a/10b.
