# Fase 13 - LTM diagnostic

## Global regime counts

| regime | count |
|---|---:|
| freefall_like | 375 |
| impact_or_regime_change | 49751 |
| mixed_workspace_motion | 2471 |
| pendulum_radial_constraint | 72274 |
| radial_transfer | 20392 |
| rest_or_constraint_hold | 643497 |

## Residual safety

| horizon | corrected | off | improved | worsened |
|---|---:|---:|---:|---:|
| h1 | 169020 | 0 | 132780 | 22452 |
| h5 | 169020 | 0 | 117382 | 36813 |
| h15 | 169020 | 0 | 114293 | 43596 |
| h30 | 112680 | 0 | 66890 | 39035 |
| h60 | 112680 | 0 | 72559 | 32328 |
| h120 | 56340 | 0 | 45543 | 10322 |

## By scene

### objects_falling

| horizon | selected branch | gain vs temporal | corrected | off | improved | worsened | alpha | confidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h1 | amf_ltm_full | 0.000152 | 13920 | 0 | 6607 | 4026 | 0.50 | 0.299351 |
| h5 | amf_ltm_no_h_workspace | 0.003808 | 13920 | 0 | 5774 | 5373 | 0.50 | 0.591239 |
| h15 | amf_ltm_no_h_workspace | 0.000184 | 13920 | 0 | 6030 | 5101 | 0.50 | 0.591239 |
| h30 | amf_ltm_full | 0.018153 | 9280 | 0 | 4190 | 2953 | 0.50 | 0.305231 |
| h60 | amf_ltm_no_h_event | 0.073130 | 9280 | 0 | 4212 | 2985 | 0.50 | 0.312970 |
| h120 | amf_ltm_no_h_event | 0.008383 | 4640 | 0 | 2950 | 1690 | 0.20 | 0.508608 |

### dominoes

| horizon | selected branch | gain vs temporal | corrected | off | improved | worsened | alpha | confidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h1 | amf_ltm_no_h_event | 0.049645 | 107556 | 0 | 97291 | 8217 | 0.50 | 0.366325 |
| h5 | amf_ltm_no_h_regime | 0.059974 | 107556 | 0 | 83194 | 20701 | 0.50 | 0.422925 |
| h15 | amf_ltm_no_h_event | 0.099154 | 107556 | 0 | 81373 | 24296 | 0.50 | 0.366325 |
| h30 | amf_ltm_no_h_event | 0.019699 | 71704 | 0 | 45907 | 23992 | 0.50 | 0.373904 |
| h60 | amf_ltm_no_h_event | 0.015679 | 71704 | 0 | 46821 | 23085 | 0.50 | 0.373904 |
| h120 | amf_ltm_no_h_regime | 0.008562 | 35852 | 0 | 31030 | 4347 | 0.20 | 0.551137 |

### wrecking_ball

| horizon | selected branch | gain vs temporal | corrected | off | improved | worsened | alpha | confidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| h1 | amf_ltm_no_h_event | 0.029125 | 47544 | 0 | 28882 | 10209 | 0.50 | 0.379652 |
| h5 | amf_ltm_no_h_event | 0.023508 | 47544 | 0 | 28414 | 10739 | 0.50 | 0.379652 |
| h15 | amf_ltm_no_h_workspace | 0.041385 | 47544 | 0 | 26890 | 14199 | 0.50 | 0.574702 |
| h30 | amf_ltm_no_h_workspace | 0.107154 | 31696 | 0 | 16793 | 12090 | 0.50 | 0.570313 |
| h60 | amf_ltm_no_h_event | 0.091370 | 31696 | 0 | 21526 | 6258 | 0.50 | 0.391379 |
| h120 | amf_ltm_no_h_regime | 0.062174 | 15848 | 0 | 11563 | 4285 | 0.50 | 0.519940 |

