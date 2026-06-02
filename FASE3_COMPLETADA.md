# Fase 3 completada

La fase 3 no agrega funciones al azar: intenta romper el prototipo de fase 2
con stress tests, baselines y ablaciones.

## Comando

```powershell
python run_phase3.py
```

Salidas:

- `results/phase3_latest.json`: datos completos de la corrida.
- `results/FASE3_RESULTADOS.md`: reporte legible.

## Cobertura de los ataques pedidos

- Datasets mas grandes: 4200 train / 1800 test, 320 dimensiones, 8 clases.
- Ruido adversarial: ataque dirigido hacia fronteras de prototipos, con sweep
  epsilon 0.03 a 0.20.
- Clases nuevas despues del entrenamiento: entrenamiento inicial en clases 0-3
  y actualizacion incremental con clases 4-7.
- Drift temporal: stream de 9 chunks donde los conceptos migran hacia regiones
  de otras clases.
- Olvido catastrofico: medicion de accuracy en clases viejas antes/despues de
  aprender clases nuevas.
- Comparacion contra kNN, SVM, random forest, MLP pequeno y prototipos.
- Medicion real: tiempo de fit/predict, RAM estimada/peak por tracemalloc y
  candidatos por consulta donde aplica.
- Ablaciones: sin Fisher, sin LSH, sin poda, sin memoria y sin ridge.

## Resultados mas fuertes

- En el dataset grande, la red morfogenica logra 0.994 clean y 0.802 con ataque
  adversarial epsilon 0.12, usando 48 candidatos medios frente a 4200 de kNN.
- Con clases nuevas, mantiene 0.991 en clases viejas y aprende nuevas con 0.936,
  con olvido de solo 0.008.
- En drift temporal, mantiene 1.000 en el ultimo chunk; SVM/prototipos estaticos
  caen a 0.000 y el MLP estatico a 0.002.
- Las ablaciones muestran que Fisher es critico (0.959 -> 0.181), LSH da un
  tradeoff fuerte de velocidad (2.526 s -> 0.232 s), la memoria temporal sube
  0.476 -> 1.000, ridge sube 0.759 -> 0.908, y la poda reduce 437 -> 117
  celulas sin perder accuracy en el dataset de crecimiento.

## Lectura honesta

No es una victoria total: random forest gana el benchmark limpio/adversarial
principal, y quitar LSH conserva algo mas de accuracy en una ablation a cambio
de mucha mas busqueda. Lo prometedor es la combinacion: precision alta, memoria
compacta, aprendizaje incremental, bajo olvido y adaptacion online.
