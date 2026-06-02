# N.E.V.E.R. Roadmap

## Completed: Phase 10c AMF Integration

- Retired the SNN dynamic branch from the active path.
- Added an external action orchestrator with local fallback.
- Loaded the Phase 10a warmed AMF world model.
- Reused the Phase 10b visual latent codec.
- Added frozen identity geometry metadata.
- Added metaplasticity statuses in the local loop:
  `explained_by_existing_cell`, `buffered_possible_noise`,
  `metaplasticity_adapted_cell`, `created_confirmed_novelty`.
- Added CUDA-facing AMF structs and vector decoder kernel.
- Added online training inside NEVER: frame feedback -> real latent -> AMF
  comparison -> metaplastic update. Verified probe: `0.25 -> 1.00`.

## Next: Phase 10d

- Load NPZ AMF cell arrays into resident CUDA memory.
- Replace the host fallback predictor with the full top-k AMF CUDA kernel.
- Persist online AMF cells across application sessions.
- Feed Blender-exported geometry instead of procedural identity metadata.
- Present frames through OpenGL/Vulkan interop or direct FFmpeg pipe.
- Add API provider adapters for OpenAI, Claude and DeepSeek while keeping the
  same bounded action-vector contract.
