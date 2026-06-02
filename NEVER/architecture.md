# N.E.V.E.R. Phase 10c Architecture

## 1. Orchestrator

Input: a user script such as `Personaje saltando, camara rotando 45 grados`.

Processing:

```text
prompt -> action provider -> A(t) = [force_x, force_y, camera_yaw, style_hold]
```

The production path can call an external model through the OpenAI-compatible
Responses API. Local tests use a deterministic fallback so the engine is
reproducible without secrets or network access.

## 2. One-Shot Identity Initialization

N.E.V.E.R. does not burn VRAM on a giant visual encoder for identity. Geometry
is ingested once, then frozen:

```text
Blender vertices / UV / style -> FrozenIdentityMemory -> identity_frozen=True
```

The local prototype writes this to
`outputs/phase10c_identity_geometry.json`.  The style code, radius and geometry
metadata stay locked while dynamics evolve.

## 3. AMF Physical Core

The SNN dynamic branch is retired. The active core is AMF:

```text
frame(t) -> encoder -> S(t)
S(t), A(t) -> AMFWorldModelRuntime -> Delta S
S(t+1) = S(t) + Delta S
real frame(t+1) -> encoder -> S(t+1) real
AMF compares prediction vs real latent -> online metaplastic update
```

Important properties:

- AMF stores deltas, not full next states.
- Existing cells reinforce if they explain the transition.
- Possible noise is buffered.
- Repeated novelty becomes a confirmed cell.
- Medium error triggers metaplastic adaptation of nearby cells.
- The runtime learns online from each real NEVER frame without full retraining.
- Low-use and similar cells are regulated by the warmed model.

## 4. Vectorial Decoder

`S(t+1)` is decoded into a coherent RGBA frame:

```text
S(t+1) -> amf_decode_latent_to_rgba_kernel -> presenter buffer / frame export
```

The decoder is not photorealistic. Its job is consistency: position, velocity,
identity/style and physically plausible motion.

## 5. Active Files

- `scripts/phase10c_never_amf_orchestrator.py`
- `src/engine/amf_world_model.cuh`
- `src/engine/amf_vector_decoder.cuh`
- `src/engine/action_orchestrator.hpp`
- `src/main.cu`
- `src/inference_core.cu`

Retired:

- `src/engine/snn_engine.cuh` is now a migration marker.
- `scripts/train_dynamic_snn.py` forwards to the AMF orchestrator.
