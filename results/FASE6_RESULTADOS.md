# Fase 6 - Reproducibilidad externa y paperizacion

Pregunta de esta fase: ya no solo si AMF5 puede funcionar, sino si empieza a
compararse honestamente contra modelos clasicos oficiales de scikit-learn.

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Attack/stress seeds: [0]
Split: train 60%, validation 20%, test 20%, estratificado por seed.
Validacion: cada familia elige variante por macro-F1 en validation y se reentrena
en train+validation antes de medir test.
scikit-learn: 1.8.0
Tiempo total: 939.4 s

## Datasets

- iris: n=150, d=4, C=3, source=UCI Iris
- wine: n=178, d=13, C=3, source=UCI Wine
- wdbc: n=569, d=30, C=2, source=UCI Breast Cancer Wisconsin Diagnostic
- madelon: n=1800, d=500, C=2, source=UCI Madelon
- spambase: n=4601, d=57, C=2, source=UCI Spambase
- ionosphere: n=351, d=34, C=2, source=UCI Ionosphere
- sonar: n=208, d=60, C=2, source=UCI Sonar

## Tabla principal

| Dataset | n | d | C | AMF5 acc | Best sklearn | Gap | AMF5 fit s | AMF5 pred s | AMF5 MB | Cells | Worst AMF5 attack |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| iris | 150 | 4 | 3 | 0.913 +- 0.062 | sk_logistic 0.967 | -0.053 | 0.019 | 0.0001 | 0.001 | 7.8 | n/a |
| wine | 178 | 13 | 3 | 0.960 +- 0.050 | sk_extra_trees 0.997 | -0.037 | 0.037 | 0.0002 | 0.004 | 13.3 | n/a |
| wdbc | 569 | 30 | 2 | 0.937 +- 0.021 | sk_rbf_svc 0.978 | -0.042 | 0.175 | 0.0004 | 0.012 | 24.9 | n/a |
| madelon | 1800 | 500 | 2 | 0.636 +- 0.017 | sk_hist_gradient_boosting 0.801 | -0.164 | 8.419 | 0.0264 | 0.634 | 145.4 | top_feature_zero_12pct (0.478) |
| spambase | 4601 | 57 | 2 | 0.844 +- 0.016 | sk_extra_trees 0.954 | -0.110 | 9.195 | 0.0457 | 0.138 | 194.4 | top_feature_shuffle_12pct (0.695) |
| ionosphere | 351 | 34 | 2 | 0.846 +- 0.042 | sk_rbf_svc 0.934 | -0.089 | 0.154 | 0.0003 | 0.010 | 18.3 | n/a |
| sonar | 208 | 60 | 2 | 0.783 +- 0.042 | sk_extra_trees 0.874 | -0.090 | 0.090 | 0.0003 | 0.023 | 25.8 | n/a |

## Lectura por dataset

- iris: AMF5 0.913 +- 0.062; best sklearn sk_logistic 0.967; gap -0.053; cells 7.8; fit 0.019s, predict 0.0001s
- wine: AMF5 0.960 +- 0.050; best sklearn sk_extra_trees 0.997; gap -0.037; cells 13.3; fit 0.037s, predict 0.0002s
- wdbc: AMF5 0.937 +- 0.021; best sklearn sk_rbf_svc 0.978; gap -0.042; cells 24.9; fit 0.175s, predict 0.0004s
- madelon: AMF5 0.636 +- 0.017; best sklearn sk_hist_gradient_boosting 0.801; gap -0.164; cells 145.4; fit 8.419s, predict 0.0264s
- spambase: AMF5 0.844 +- 0.016; best sklearn sk_extra_trees 0.954; gap -0.110; cells 194.4; fit 9.195s, predict 0.0457s
- ionosphere: AMF5 0.846 +- 0.042; best sklearn sk_rbf_svc 0.934; gap -0.089; cells 18.3; fit 0.154s, predict 0.0003s
- sonar: AMF5 0.783 +- 0.042; best sklearn sk_extra_trees 0.874; gap -0.090; cells 25.8; fit 0.090s, predict 0.0003s

## Busqueda de fallos con mala intencion

- madelon: clean 0.628 +- 0.000; peor ataque top_feature_zero_12pct acc 0.478
- spambase: clean 0.845 +- 0.000; peor ataque top_feature_shuffle_12pct acc 0.695

## Estres de alta dimension por features basura

- madelon: +1.0x noise acc 0.578 +- 0.000, MB 1.036 +- 0.000
- spambase: +1.0x noise acc 0.834 +- 0.000, MB 0.250 +- 0.000

## Lectura honesta

AMF5 queda como candidato local/compacto e incremental, no como ganador universal
de accuracy. Esta fase mide el costo real de competir contra `sklearn`: cuando
los modelos globales clasicos explotan bien la frontera, AMF5 queda atras; cuando
hay muchas features distractoras o ataques a subconjuntos informativos, la
anatomia Fisher + voto local muestra donde puede valer la pena seguir.
