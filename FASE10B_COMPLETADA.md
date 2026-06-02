# FASE10B_COMPLETADA

Fase 10b implementa el encoder/decoder visual para el AMF world model.

Entregables:

- `phase10b_visual_codec.py`
- `run_phase10b.py`
- `results/phase10b_latest.json`
- `results/FASE10B_RESULTADOS.md`
- `data/phase10b_sample_frames.npz`
- `data/phase10b_visual_codec_*.json`

Resultado en resolucion maxima (256):

- latent_dim: 8
- frame_bytes: 1048576
- latent_bytes: 32
- compression_ratio: 32768.0x
- reconstruction_iou: 0.6468
- predicted_frame_iou: 0.5937
- predicted_latent_mse: 0.036390
- AMF predict ms: 0.867607
- raw_pixel_amf_memory_mb_est: 18000.10
- actual_amf_memory_mb: 0.686646

Invariantes:

- permanence_identity_distance: 0.003235
- permanence_shift_error: 0.000034
- separability_identity_distance: 0.750003
- continuity_mean_step: 0.106556
- continuity_max_step: 0.227218

El AMF ya no escala con pixeles: escala con `S(t)` compacto.
