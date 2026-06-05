# Fase 13 - Cross-scene evaluation

Scenes: objects_falling, dominoes, wrecking_ball
Mean h30/h60 gain vs temporal-energy: 0.054197

| horizon | mean temporal | mean best previous | mean AMF-LTM selected | W/T/L vs temporal | W/T/L vs previous |
|---|---:|---:|---:|---|---|
| h1 | 0.000024 | 0.000022 | 0.000024 | 2/1/0 | 2/0/1 |
| h5 | 0.001134 | 0.001096 | 0.001114 | 2/1/0 | 2/0/1 |
| h15 | 0.012833 | 0.012382 | 0.012373 | 2/1/0 | 1/2/0 |
| h30 | 0.075521 | 0.066106 | 0.070193 | 3/0/0 | 0/1/2 |
| h60 | 0.431312 | 0.370184 | 0.396456 | 3/0/0 | 2/0/1 |
| h120 | 0.889331 | 0.875799 | 0.851361 | 3/0/0 | 1/1/1 |

## Per scene

### h1

| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |
|---|---:|---:|---:|---|---:|---:|---|
| objects_falling | 0.000046 | 0.000038 | 0.000046 | amf_ltm_full | 0.000152 | -0.201430 | amf_residual |
| dominoes | 0.000000 | 0.000000 | 0.000000 | amf_ltm_no_h_event | 0.049645 | 0.049645 | amf_ltm_no_h_regime |
| wrecking_ball | 0.000027 | 0.000027 | 0.000026 | amf_ltm_no_h_event | 0.029125 | 0.029125 | amf_ltm_no_h_event |

### h5

| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |
|---|---:|---:|---:|---|---:|---:|---|
| objects_falling | 0.000966 | 0.000852 | 0.000962 | amf_ltm_no_h_workspace | 0.003808 | -0.128956 | energy |
| dominoes | 0.000006 | 0.000006 | 0.000006 | amf_ltm_no_h_regime | 0.059974 | 0.059974 | amf_ltm_no_h_event |
| wrecking_ball | 0.002430 | 0.002430 | 0.002373 | amf_ltm_no_h_event | 0.023508 | 0.023508 | amf_ltm_no_h_event |

### h15

| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |
|---|---:|---:|---:|---|---:|---:|---|
| objects_falling | 0.005431 | 0.005431 | 0.005430 | amf_ltm_no_h_workspace | 0.000184 | 0.000184 | amf_ltm_no_h_regime |
| dominoes | 0.000151 | 0.000151 | 0.000136 | amf_ltm_no_h_event | 0.099154 | 0.099154 | amf_ltm_no_h_event |
| wrecking_ball | 0.032916 | 0.031564 | 0.031554 | amf_ltm_no_h_workspace | 0.041385 | 0.000300 | amf_ltm_no_h_workspace |

### h30

| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |
|---|---:|---:|---:|---|---:|---:|---|
| objects_falling | 0.091119 | 0.077342 | 0.089465 | amf_ltm_full | 0.018153 | -0.156734 | amf_ensemble_12c |
| dominoes | 0.002109 | 0.001956 | 0.002068 | amf_ltm_no_h_event | 0.019699 | -0.057178 | amf_ensemble_12c |
| wrecking_ball | 0.133334 | 0.119021 | 0.119047 | amf_ltm_no_h_workspace | 0.107154 | -0.000219 | amf_ensemble_12c |

### h60

| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |
|---|---:|---:|---:|---|---:|---:|---|
| objects_falling | 0.536858 | 0.377944 | 0.497598 | amf_ltm_no_h_event | 0.073130 | -0.316590 | ridge |
| dominoes | 0.051092 | 0.051092 | 0.050291 | amf_ltm_no_h_event | 0.015679 | 0.015679 | amf_ltm_no_h_event |
| wrecking_ball | 0.705985 | 0.681515 | 0.641479 | amf_ltm_no_h_event | 0.091370 | 0.058745 | amf_ltm_no_h_event |

### h120

| scene | temporal | best previous | AMF-LTM selected | selected branch | gain vs temporal | gain vs previous | best valid |
|---|---:|---:|---:|---|---:|---:|---|
| objects_falling | 0.778726 | 0.769398 | 0.772198 | amf_ltm_no_h_event | 0.008383 | -0.003639 | amf_ensemble_12c |
| dominoes | 0.188043 | 0.182413 | 0.186433 | amf_ltm_no_h_regime | 0.008562 | -0.022033 | amf_ensemble_12c |
| wrecking_ball | 1.701225 | 1.675586 | 1.595454 | amf_ltm_no_h_regime | 0.062174 | 0.047823 | amf_ltm_no_h_regime |

