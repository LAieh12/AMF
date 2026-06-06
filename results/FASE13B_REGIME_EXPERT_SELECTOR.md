# Fase 13B - Regime Expert Selector

Scenes: objects_falling, dominoes, wrecking_ball
Experts: temporal_energy, energy_constraint, identity_orientation, ensemble_12c, amf_ltm_residual, ridge_safety, amf_residual_base
Stride: 60; memory window: 20; selector step: 0.5

## Success

Passed: `True`
Long h30/h60/h120 W/T vs best previous AMF: 8/9

## Cross-scene

| horizon | W/T/L vs best previous | mean gain vs previous | mean gain vs temporal |
|---|---|---:|---:|
| h1 | 2/0/1 | -0.026994 | 0.040208 |
| h5 | 2/0/1 | 0.014986 | 0.058327 |
| h15 | 2/0/1 | 0.105607 | 0.119204 |
| h30 | 3/0/0 | 0.064743 | 0.168018 |
| h60 | 2/0/1 | 0.052371 | 0.154123 |
| h120 | 1/2/0 | 0.025853 | 0.044442 |

## Scene x horizon

### objects_falling

| horizon | selector MSE | previous best | previous MSE | gain vs previous | gain vs temporal | top expert weight | LTM beta |
|---|---:|---|---:|---:|---:|---|---:|
| h1 | 0.000046 | amf_residual_base | 0.000038 | -0.201573 | 0.000033 | amf_ltm_residual:0.97 | 0.50 |
| h5 | 0.000942 | energy_constraint | 0.000852 | -0.105647 | 0.024376 | energy_constraint:0.49 | 0.50 |
| h15 | 0.005709 | temporal_energy | 0.005431 | -0.051166 | -0.051166 | amf_ltm_residual:0.82 | 0.50 |
| h30 | 0.071281 | ensemble_12c | 0.077342 | 0.078372 | 0.217713 | ensemble_12c:0.39 | 0.50 |
| h60 | 0.349487 | amf_residual_base | 0.377944 | 0.075295 | 0.349014 | amf_ltm_residual:0.26 | 0.25 |
| h120 | 0.769218 | ensemble_12c | 0.769398 | 0.000234 | 0.012209 | ensemble_12c:0.46 | 0.25 |

### dominoes

| horizon | selector MSE | previous best | previous MSE | gain vs previous | gain vs temporal | top expert weight | LTM beta |
|---|---:|---|---:|---:|---:|---|---:|
| h1 | 0.000000 | temporal_energy | 0.000000 | 0.056702 | 0.056702 | amf_ltm_residual:0.93 | 0.25 |
| h5 | 0.000005 | temporal_energy | 0.000006 | 0.124868 | 0.124868 | amf_ltm_residual:0.90 | 0.25 |
| h15 | 0.000097 | temporal_energy | 0.000151 | 0.360571 | 0.360571 | amf_ltm_residual:0.84 | 0.50 |
| h30 | 0.001795 | ensemble_12c | 0.001956 | 0.082381 | 0.149109 | amf_ltm_residual:0.77 | 0.25 |
| h60 | 0.051517 | temporal_energy | 0.051092 | -0.008314 | -0.008314 | amf_ltm_residual:0.76 | 0.25 |
| h120 | 0.181813 | ensemble_12c | 0.182413 | 0.003291 | 0.033128 | amf_residual_base:0.97 | 0.25 |

### wrecking_ball

| horizon | selector MSE | previous best | previous MSE | gain vs previous | gain vs temporal | top expert weight | LTM beta |
|---|---:|---|---:|---:|---:|---|---:|
| h1 | 0.000025 | temporal_energy | 0.000027 | 0.063889 | 0.063889 | amf_ltm_residual:0.77 | 0.25 |
| h5 | 0.002368 | temporal_energy | 0.002430 | 0.025736 | 0.025736 | amf_ltm_residual:0.67 | 0.25 |
| h15 | 0.031330 | ensemble_12c | 0.031564 | 0.007415 | 0.048208 | identity_orientation:0.38 | 0.50 |
| h30 | 0.115036 | ensemble_12c | 0.119021 | 0.033478 | 0.137233 | amf_ltm_residual:0.62 | 0.25 |
| h60 | 0.620088 | ensemble_12c | 0.681515 | 0.090133 | 0.121670 | amf_ltm_residual:0.77 | 0.25 |
| h120 | 1.551538 | ensemble_12c | 1.675586 | 0.074033 | 0.087988 | amf_ltm_residual:0.69 | 0.25 |

## Dominant experts by regime

### objects_falling

- h1: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual, radial_transfer:amf_ltm_residual, freefall_like:amf_ltm_residual
- h5: rest_or_constraint_hold:energy_constraint, pendulum_radial_constraint:energy_constraint, impact_or_regime_change:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual, radial_transfer:amf_ltm_residual, freefall_like:amf_ltm_residual
- h15: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:ensemble_12c, mixed_workspace_motion:amf_ltm_residual, radial_transfer:amf_ltm_residual, freefall_like:amf_ltm_residual
- h30: rest_or_constraint_hold:identity_orientation, pendulum_radial_constraint:ensemble_12c, impact_or_regime_change:ensemble_12c, mixed_workspace_motion:identity_orientation, radial_transfer:identity_orientation, freefall_like:identity_orientation
- h60: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:ridge_safety, impact_or_regime_change:ensemble_12c, mixed_workspace_motion:amf_ltm_residual, radial_transfer:amf_ltm_residual, freefall_like:amf_ltm_residual
- h120: rest_or_constraint_hold:ensemble_12c, pendulum_radial_constraint:ensemble_12c

### dominoes

- h1: rest_or_constraint_hold:amf_ltm_residual, radial_transfer:identity_orientation, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual
- h5: rest_or_constraint_hold:amf_ltm_residual, radial_transfer:identity_orientation, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual
- h15: rest_or_constraint_hold:amf_ltm_residual, radial_transfer:energy_constraint, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual
- h30: rest_or_constraint_hold:amf_ltm_residual, radial_transfer:identity_orientation, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:ensemble_12c, mixed_workspace_motion:ensemble_12c
- h60: rest_or_constraint_hold:amf_ltm_residual, radial_transfer:ensemble_12c, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:ensemble_12c, mixed_workspace_motion:ensemble_12c
- h120: rest_or_constraint_hold:amf_residual_base, pendulum_radial_constraint:identity_orientation

### wrecking_ball

- h1: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:energy_constraint, radial_transfer:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual, freefall_like:amf_ltm_residual
- h5: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:ensemble_12c, radial_transfer:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual, freefall_like:amf_ltm_residual
- h15: rest_or_constraint_hold:identity_orientation, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:ensemble_12c, radial_transfer:amf_ltm_residual, mixed_workspace_motion:ensemble_12c, freefall_like:ensemble_12c
- h30: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:ensemble_12c, impact_or_regime_change:ensemble_12c, radial_transfer:ensemble_12c, mixed_workspace_motion:ensemble_12c, freefall_like:ensemble_12c
- h60: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:amf_ltm_residual, impact_or_regime_change:identity_orientation, radial_transfer:amf_ltm_residual, mixed_workspace_motion:amf_ltm_residual, freefall_like:amf_ltm_residual
- h120: rest_or_constraint_hold:amf_ltm_residual, pendulum_radial_constraint:amf_ltm_residual

## Average expert weights

### objects_falling

| horizon | weights | confidence | specific/regime/global sources |
|---|---|---:|---|
| h1 | amf_ltm_residual:0.97, amf_residual_base:0.03 | 0.299351 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:713, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=hi:832, impact_or_regime_change/fallback:1502, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=mid:916, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:2212 |
| h5 | energy_constraint:0.49, identity_orientation:0.25, ensemble_12c:0.04, amf_ltm_residual:0.22 | 0.299351 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:697, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=hi:426, impact_or_regime_change/fallback:1485, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:1228, pendulum_radial_constraint/fallback:1409 |
| h15 | identity_orientation:0.07, ensemble_12c:0.05, amf_ltm_residual:0.82, amf_residual_base:0.05 | 0.299351 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:601, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=hi:771, impact_or_regime_change/e=hi/o=hi/i=hi/c=lo/err=lo:1476, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=mid:875, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:2045 |
| h30 | energy_constraint:0.03, identity_orientation:0.27, ensemble_12c:0.39, amf_ltm_residual:0.31 | 0.305231 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:812, pendulum_radial_constraint/fallback:1209, impact_or_regime_change/fallback:1309, rest_or_constraint_hold/fallback:1840, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=mid:673 |
| h60 | ensemble_12c:0.24, amf_ltm_residual:0.26, ridge_safety:0.25, amf_residual_base:0.24 | 0.305231 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:748, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=mid:465, impact_or_regime_change/e=hi/o=hi/i=hi/c=lo/err=lo:1366, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=hi:855, rest_or_constraint_hold/fallback:1139 |
| h120 | ensemble_12c:0.46, amf_ltm_residual:0.33, ridge_safety:0.14, amf_residual_base:0.07 | 0.508608 | global:144, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=lo/err=lo:398, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=lo/err=mid:500, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=lo/err=hi:397, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=mid/err=mid:429 |

### dominoes

| horizon | weights | confidence | specific/regime/global sources |
|---|---|---:|---|
| h1 | identity_orientation:0.05, amf_ltm_residual:0.93, ridge_safety:0.01 | 0.357936 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:20025, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:14014, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=lo:937, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:8086, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=mid:1222 |
| h5 | identity_orientation:0.09, amf_ltm_residual:0.90 | 0.357936 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:19577, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:11149, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=mid:1563, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=mid:12005, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=lo:696 |
| h15 | identity_orientation:0.16, amf_ltm_residual:0.84 | 0.357936 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=hi:20185, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=hi:567, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=hi:7886, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=lo:1525, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=lo:10014 |
| h30 | identity_orientation:0.20, ensemble_12c:0.02, amf_ltm_residual:0.77 | 0.367646 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:10134, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=hi:8701, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=hi:673, rest_or_constraint_hold/fallback:455, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=hi:6108 |
| h60 | identity_orientation:0.13, ensemble_12c:0.11, amf_ltm_residual:0.76 | 0.367646 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:9354, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=hi:5647, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=mid/err=hi:888, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=hi:5248, rest_or_constraint_hold/e=hi/o=hi/i=hi/c=lo/err=hi:6889 |
| h120 | identity_orientation:0.02, amf_ltm_residual:0.02, amf_residual_base:0.97 | 0.547181 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=hi:5893, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:5395, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=hi:2871, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=mid:3515, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:4010 |

### wrecking_ball

| horizon | weights | confidence | specific/regime/global sources |
|---|---|---:|---|
| h1 | energy_constraint:0.08, identity_orientation:0.13, amf_ltm_residual:0.77, amf_residual_base:0.02 | 0.362068 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:6798, rest_or_constraint_hold/e=mid/o=lo/i=mid/c=mid/err=mid:338, rest_or_constraint_hold/e=mid/o=lo/i=mid/c=mid/err=hi:387, rest_or_constraint_hold/fallback:3886, rest_or_constraint_hold/e=mid/o=mid/i=mid/c=mid/err=mid:484 |
| h5 | identity_orientation:0.10, ensemble_12c:0.24, amf_ltm_residual:0.67 | 0.362068 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:6105, rest_or_constraint_hold/e=mid/o=lo/i=mid/c=mid/err=mid:481, rest_or_constraint_hold/fallback:4191, rest_or_constraint_hold/e=mid/o=mid/i=mid/c=mid/err=mid:759, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:7684 |
| h15 | identity_orientation:0.38, ensemble_12c:0.27, amf_ltm_residual:0.35 | 0.362068 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:5313, rest_or_constraint_hold/e=mid/o=lo/i=mid/c=mid/err=mid:463, rest_or_constraint_hold/fallback:4797, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:8104, rest_or_constraint_hold/e=mid/o=mid/i=mid/c=mid/err=mid:709 |
| h30 | energy_constraint:0.08, identity_orientation:0.03, ensemble_12c:0.27, amf_ltm_residual:0.62 | 0.377644 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:4320, rest_or_constraint_hold/e=mid/o=lo/i=mid/c=mid/err=mid:518, rest_or_constraint_hold/fallback:3467, rest_or_constraint_hold/e=mid/o=mid/i=mid/c=mid/err=mid:921, pendulum_radial_constraint/fallback:1112 |
| h60 | identity_orientation:0.12, amf_ltm_residual:0.77, amf_residual_base:0.11 | 0.377644 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:4202, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:2935, rest_or_constraint_hold/fallback:3372, rest_or_constraint_hold/e=mid/o=lo/i=mid/c=mid/err=mid:589, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=hi:3219 |
| h120 | energy_constraint:0.04, ensemble_12c:0.04, amf_ltm_residual:0.69, ridge_safety:0.11, amf_residual_base:0.13 | 0.519342 | rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=mid:2584, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=hi/err=lo:2283, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=lo/err=hi:459, rest_or_constraint_hold/e=lo/o=lo/i=lo/c=mid/err=hi:1779, pendulum_radial_constraint/e=lo/o=lo/i=lo/c=lo/err=lo:1247 |

## LTM residual effect

| scene | horizon | beta | corrected | off | improved | worsened | retrieved | memory MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| objects_falling | h1 | 0.50 | 13920 | 0 | 4707 | 5929 | 24 | 1.802 |
| objects_falling | h5 | 0.50 | 13920 | 0 | 2565 | 8198 | 24 | 1.802 |
| objects_falling | h15 | 0.50 | 13920 | 0 | 4872 | 5805 | 24 | 1.802 |
| objects_falling | h30 | 0.50 | 9280 | 0 | 3432 | 3713 | 24 | 1.201 |
| objects_falling | h60 | 0.25 | 9280 | 0 | 2887 | 4185 | 24 | 1.201 |
| objects_falling | h120 | 0.25 | 4640 | 0 | 2758 | 1882 | 24 | 0.601 |
| dominoes | h1 | 0.25 | 107556 | 0 | 88930 | 14252 | 24 | 13.930 |
| dominoes | h5 | 0.25 | 107556 | 0 | 79689 | 23705 | 24 | 13.930 |
| dominoes | h15 | 0.50 | 50785 | 56771 | 22409 | 28376 | 24 | 13.930 |
| dominoes | h30 | 0.25 | 71704 | 0 | 38114 | 29876 | 24 | 9.287 |
| dominoes | h60 | 0.25 | 71704 | 0 | 38818 | 29212 | 24 | 9.287 |
| dominoes | h120 | 0.25 | 21181 | 14671 | 9981 | 11200 | 24 | 4.643 |
| wrecking_ball | h1 | 0.25 | 47544 | 0 | 23876 | 15080 | 24 | 6.288 |
| wrecking_ball | h5 | 0.25 | 47544 | 0 | 24182 | 14832 | 24 | 6.288 |
| wrecking_ball | h15 | 0.50 | 47544 | 0 | 23270 | 15813 | 24 | 6.288 |
| wrecking_ball | h30 | 0.25 | 31696 | 0 | 15614 | 11883 | 24 | 4.192 |
| wrecking_ball | h60 | 0.25 | 31696 | 0 | 18912 | 8614 | 24 | 4.192 |
| wrecking_ball | h120 | 0.25 | 15848 | 0 | 10856 | 4992 | 24 | 2.096 |
