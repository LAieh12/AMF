# FASE10A_COMPLETADA

Fase 10a implementa pretraining sintetico de un world model AMF.

Entregables:

- `phase10a_toy_simulator.py`
- `phase10a_amf_world_model.py`
- `run_phase10a.py`
- `results/phase10a_latest.json`
- `results/FASE10A_RESULTADOS.md`
- `data\phase10a_warm_amf.npz`
- `data\phase10a_warm_amf.json`

Resultado AMF:

- train transitions: 53550
- test transitions: 9450
- cells: 9000
- one_step_mse: 0.01743034
- rollout_mse: 0.28755888
- bounce_mse: 0.08504881
- memory_mb_arrays: 0.686646
- export reload max abs diff: 0.0000000000
- metaplasticity_probe_passed: True
- raw_cells: 26375
- pruned_low_usage: 16090
- fused_similar: 37
- final_cells: 9000

El modelo fue calentado con miles de transiciones sinteticas y exportado para
Fase 10b. Guarda deltas, regula crecimiento por metaplasticidad y no usa LLM,
decoder denso ni backprop.
