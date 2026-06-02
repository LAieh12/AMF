# Fase 6 - Tabla tipo paper

| Dataset | n | d | C | AMF5 acc | Best sklearn | Gap | AMF5 fit s | AMF5 pred s | AMF5 MB | Cells | Worst AMF5 attack |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| iris | 150 | 4 | 3 | 0.913 +- 0.062 | sk_logistic 0.967 | -0.053 | 0.019 | 0.0001 | 0.001 | 7.8 | n/a |
| wine | 178 | 13 | 3 | 0.960 +- 0.050 | sk_extra_trees 0.997 | -0.037 | 0.037 | 0.0002 | 0.004 | 13.3 | n/a |
| wdbc | 569 | 30 | 2 | 0.937 +- 0.021 | sk_rbf_svc 0.978 | -0.042 | 0.175 | 0.0004 | 0.012 | 24.9 | n/a |
| madelon | 1800 | 500 | 2 | 0.636 +- 0.017 | sk_hist_gradient_boosting 0.801 | -0.164 | 8.419 | 0.0264 | 0.634 | 145.4 | top_feature_zero_12pct (0.478) |
| spambase | 4601 | 57 | 2 | 0.844 +- 0.016 | sk_extra_trees 0.954 | -0.110 | 9.195 | 0.0457 | 0.138 | 194.4 | top_feature_shuffle_12pct (0.695) |
| ionosphere | 351 | 34 | 2 | 0.846 +- 0.042 | sk_rbf_svc 0.934 | -0.089 | 0.154 | 0.0003 | 0.010 | 18.3 | n/a |
| sonar | 208 | 60 | 2 | 0.783 +- 0.042 | sk_extra_trees 0.874 | -0.090 | 0.090 | 0.0003 | 0.023 | 25.8 | n/a |
