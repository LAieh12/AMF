# Fase 12D - AMF-LTM router/retriever

Tar: `data\physicalai_hf_cache\datasets--nvidia--PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes\snapshots\ff9b2b0a93243d84925c6c7474ee8a5bc02886cb\physics\wrecking_ball\physics-wrecking_ball-00000.tar`
Tracks: 63712
Sequences: 1000 (750 train / 250 test)
Fit/validation/test: 600 / 150 / 250 (seed 123)
Stride: 30; window: 20 frames; router top-k: 32

## Discipline

- Temporal-energy is frozen as the strong baseline.
- H_event and H_workspace are used for routing/retrieval/confidence, not as dense predictor features.
- Memories, selector weights, and residual alphas are calibrated only on train/validation sequences.
- `oracle_selector_test_only_invalid` is a diagnostic ceiling only and is not a valid model.

## Metrics

| horizon | temporal-energy MSE | router MSE | residual MSE | router+residual MSE | router blend | best valid | gain vs temporal | oracle invalid MSE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| h1 | 0.000029 | 0.000029 | 0.000024 | 0.000024 | 0.00 | ltm_residual_no_router | 0.175261 | 0.000018 |
| h5 | 0.003250 | 0.003250 | 0.002925 | 0.002925 | 0.00 | ltm_residual_no_router | 0.099936 | 0.002616 |
| h15 | 0.073433 | 0.073433 | 0.066685 | 0.066685 | 0.00 | ltm_residual_no_router | 0.091894 | 0.060559 |
| h30 | 0.392941 | 0.392941 | 0.347153 | 0.347153 | 0.00 | ltm_residual_no_router | 0.116527 | 0.318632 |
| h60 | 1.068345 | 1.068345 | 0.967422 | 0.967422 | 0.00 | ltm_residual_no_router | 0.094467 | 0.725167 |

## LTM interpretation

AMF-LTM 12D changes the role of long-term memory: it no longer feeds a wider dense vector into the predictor. It writes episodic validation memories with physical regime, energy trend, radial/tangential state, orientation change, impact/change proxies, object identity, slot/color workspace identity, residual surprise, and the predictor that won locally.

At test time, each state retrieves nearby episodes and uses them either as a router over temporal-energy/energy/orientation/identity/static-ensemble or as a small residual on top of temporal-energy.

## Memory summaries

- h1: temporal_energy=38668, energy=1811, orientation=1946, identity=2637, static_ensemble=3998
- h5: temporal_energy=14011, energy=14871, orientation=6249, identity=11569, static_ensemble=2360
- h15: temporal_energy=12615, energy=9647, orientation=9289, identity=16216, static_ensemble=1293
- h30: temporal_energy=12498, energy=8172, orientation=9950, identity=7742, static_ensemble=886
- h60: temporal_energy=8622, energy=5107, orientation=7412, identity=6832, static_ensemble=1463
