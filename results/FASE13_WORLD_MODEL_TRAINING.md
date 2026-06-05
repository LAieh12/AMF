# Fase 13 - Formal AMF World Model Training

Scenes: objects_falling, dominoes, wrecking_ball
Horizons: h1, h5, h15, h30, h60, h120
Stride: 60; memory window: 20; max cells: 5000

## Architecture

AMF-LTM residual/router over temporal-energy; LTM levels are retrieval/residual selectors, not dense appended features

Temporal-energy remains the base predictor. AMF-LTM stores episodic validation memories and applies calibrated residual corrections only when confidence passes validation thresholds.

## Success audit

Passed: `False`
Long-horizon W/T vs temporal-energy: 9/9
Long-horizon W/T vs best previous AMF: 3/9

## Scene metrics

### objects_falling

Tracks: 19036; sequences: 142 (106 train / 36 test); elapsed: 173.86s

| horizon | temporal | best previous | AMF-LTM selected | selected branch | gain selected vs temporal | gain selected vs previous |
|---|---:|---:|---:|---|---:|---:|
| h1 | 0.000046 | 0.000038 (amf_residual) | 0.000046 | amf_ltm_full | 0.000152 | -0.201430 |
| h5 | 0.000966 | 0.000852 (energy) | 0.000962 | amf_ltm_no_h_workspace | 0.003808 | -0.128956 |
| h15 | 0.005431 | 0.005431 (temporal_energy) | 0.005430 | amf_ltm_no_h_workspace | 0.000184 | 0.000184 |
| h30 | 0.091119 | 0.077342 (amf_ensemble_12c) | 0.089465 | amf_ltm_full | 0.018153 | -0.156734 |
| h60 | 0.536858 | 0.377944 (amf_residual) | 0.497598 | amf_ltm_no_h_event | 0.073130 | -0.316590 |
| h120 | 0.778726 | 0.769398 (amf_ensemble_12c) | 0.772198 | amf_ltm_no_h_event | 0.008383 | -0.003639 |

### dominoes

Tracks: 138720; sequences: 1000 (750 train / 250 test); elapsed: 1457.19s

| horizon | temporal | best previous | AMF-LTM selected | selected branch | gain selected vs temporal | gain selected vs previous |
|---|---:|---:|---:|---|---:|---:|
| h1 | 0.000000 | 0.000000 (temporal_energy) | 0.000000 | amf_ltm_no_h_event | 0.049645 | 0.049645 |
| h5 | 0.000006 | 0.000006 (temporal_energy) | 0.000006 | amf_ltm_no_h_regime | 0.059974 | 0.059974 |
| h15 | 0.000151 | 0.000151 (temporal_energy) | 0.000136 | amf_ltm_no_h_event | 0.099154 | 0.099154 |
| h30 | 0.002109 | 0.001956 (amf_ensemble_12c) | 0.002068 | amf_ltm_no_h_event | 0.019699 | -0.057178 |
| h60 | 0.051092 | 0.051092 (temporal_energy) | 0.050291 | amf_ltm_no_h_event | 0.015679 | 0.015679 |
| h120 | 0.188043 | 0.182413 (amf_ensemble_12c) | 0.186433 | amf_ltm_no_h_regime | 0.008562 | -0.022033 |

### wrecking_ball

Tracks: 63712; sequences: 1000 (750 train / 250 test); elapsed: 616.18s

| horizon | temporal | best previous | AMF-LTM selected | selected branch | gain selected vs temporal | gain selected vs previous |
|---|---:|---:|---:|---|---:|---:|
| h1 | 0.000027 | 0.000027 (temporal_energy) | 0.000026 | amf_ltm_no_h_event | 0.029125 | 0.029125 |
| h5 | 0.002430 | 0.002430 (temporal_energy) | 0.002373 | amf_ltm_no_h_event | 0.023508 | 0.023508 |
| h15 | 0.032916 | 0.031564 (amf_ensemble_12c) | 0.031554 | amf_ltm_no_h_workspace | 0.041385 | 0.000300 |
| h30 | 0.133334 | 0.119021 (amf_ensemble_12c) | 0.119047 | amf_ltm_no_h_workspace | 0.107154 | -0.000219 |
| h60 | 0.705985 | 0.681515 (amf_ensemble_12c) | 0.641479 | amf_ltm_no_h_event | 0.091370 | 0.058745 |
| h120 | 1.701225 | 1.675586 (amf_ensemble_12c) | 1.595454 | amf_ltm_no_h_regime | 0.062174 | 0.047823 |

## Notes

- `oracle_no_valid` is present in JSON only as a test-only diagnostic ceiling.
- If AMF-LTM full loses, the report keeps the loss and the diagnostics identify whether confidence, regime retrieval, or residual aggression caused it.
