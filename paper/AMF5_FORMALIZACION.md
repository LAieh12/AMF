# AMF5 formalizacion matematica

AMF5 es un clasificador local por celulas morfogenicas con metrica adaptativa,
control de crecimiento y campo atencional de voto suave. Dado un ejemplo
`x in R^d`, el modelo mantiene un conjunto dinamico de celulas
`C = {c_i}_{i=1}^M`. Cada celula contiene:

- centro `w_i in R^d`
- histograma de clase `h_i in R^K`
- etiqueta principal `y_i`
- uso/confianza `u_i`
- radio local `r_i`
- importancia `I_i`

## Metrica Fisher

Para cada feature `j`, AMF5 estima una razon tipo Fisher:

```text
a_j = between_class_variance_j / (within_class_variance_j + eps)
```

Las features atencionales son el conjunto:

```text
A_m = top_m({a_j})
```

Si la metrica Fisher se desactiva, `a_j = 1` y `A_m` se elige por varianza
empirica.

## Campo local de prediccion

La distancia atencional entre una entrada `x` y una celula `c_i` es:

```text
d_i^2(x) = (1 / |A_m|) * sum_{j in A_m} a_j * (x_j - w_{i,j})^2
```

Sea `N_k(x)` el conjunto de las `k` celulas con menor `d_i^2(x)`. El peso de
voto de una celula es:

```text
eta_i(x) = exp(-d_i^2(x) / (2 * (s * r_i + rho)^2)) * I_i^gamma
```

donde `s` es `radius_scale`, `rho` es `min_radius` y `gamma` es
`importance_power`. La distribucion predictiva es:

```text
p(y = q | x) =
  sum_{i in N_k(x)} eta_i(x) * h_i[q]
  / (sum_{q'=1}^K sum_{i in N_k(x)} eta_i(x) * h_i[q'] + eps)
```

La prediccion final es:

```text
y_hat(x) = argmax_q p(y = q | x)
```

## Aprendizaje local y crecimiento

Para una muestra `(x_t, y_t)`, el modelo busca celulas candidatas con indexacion
LSH y evalua la mejor celula local. Si la celula correcta existe y la muestra
esta dentro del umbral morfogenico, se adapta:

```text
w_i <- w_i + alpha * (x_t - w_i)
h_i[y_t] <- h_i[y_t] + 1
r_i <- EMA(r_i, distance(x_t, w_i))
u_i <- u_i + 1
```

Si la distancia supera `theta`, la prediccion local falla, o la region requiere
nueva memoria, se crea una celula:

```text
c_new = (w_new = x_t, h_new = one_hot(y_t), y_new = y_t, u_new = 1)
```

Periodicamente se aplican fusion y poda para controlar `M`, el numero de
celulas.

## Complejidad

Con `M` celulas, dimension original `d`, dimension atencional `m`, `L` tablas
LSH y `k` votos:

| Operacion | Complejidad conceptual | Medicion en Fase 6 |
|---|---:|---|
| Actualizacion Fisher | `O(d)` por muestra amortizada | `fit_seconds`, `peak_fit_ram_mb` |
| Busqueda candidata LSH | `O(L * hash + B * m)` | `avg_candidates` |
| Voto local | `O(k * K)` despues de distancias | `avg_votes`, `predict_seconds` |
| Inferencia sin indice | `O(M * m)` | peor caso, visible en `avg_candidates` |
| Memoria AMF5 | `O(M * (d + K) + Kd)` | `model_mb`, `cells` |

La hipotesis de Fase 6 no es que AMF5 gane todo en accuracy puro. La hipotesis
es que puede ocupar una region util cuando importan aprendizaje local, memoria
compacta, adaptacion incremental y robustez ante features distractoras.
