# Resultados morfogenicos de segunda generacion

Este reporte se genero con `python run_experiments.py` usando solo NumPy y los
desafios planteados en `Investigacion.md`.

## 1. Escalabilidad a alta dimension

- Dataset: 1024 dimensiones, solo 18 informativas.
- Base euclidiana: accuracy 0.249, 897 celulas.
- Metrica Fisher adaptativa + indice LSH: accuracy 0.963, 99 celulas.
- Recuperacion de dimensiones utiles en los pesos top: 1.000.
- Candidatos medios por consulta con indice: 7.9.

## 2. Control del crecimiento

- Red sin consolidacion: accuracy 0.999, 476 celulas.
- Fusion/poda informativa: accuracy 0.999, 122 celulas.
- Reduccion de celulas: 3.90x.

## 3. Secuencias y razonamiento

- Tarea: mismo saco de simbolos, clasificar si A aparece antes que B.
- Clasificador de vectores independientes: accuracy 0.476.
- Memoria temporal + reglas de composicion: accuracy 1.000.
- Dimension de rasgos temporales: 132.

## 4. Aprendizaje global hibrido

- Celulas locales: accuracy 0.797.
- Celulas locales + cabeza ridge global ocasional: accuracy 0.902.
- Ganancia absoluta: 0.105.

## Lectura corta

Los resultados son prometedores porque cada desafio deja de ser una nota
conceptual y pasa a tener una prueba operativa: la metrica adaptativa rescata
senal en alta dimension, la consolidacion reduce crecimiento sin colapsar la
precision, la composicion temporal resuelve una relacion invisible para bolsas
de vectores, y una cabeza global ligera mejora la frontera aprendida por celulas
locales sin volver al entrenamiento profundo end-to-end.
