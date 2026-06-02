# Desafios completados desde la investigación de gpt 5.5

Este folder partia solo de la investigación de gpt 5.5, que terminaba con cuatro
desafios abiertos. Se implemento un prototipo reproducible en NumPy que convierte
esos desafios en mecanismos ejecutables y pruebas comparativas.

## Como reproducir

```powershell
python run_experiments.py
```

La corrida genera:

- `results/latest_results.json`: metricas completas en formato estructurado.
- `results/RESULTADOS.md`: resumen legible de la ultima corrida.

## 1. Escalabilidad a alta dimension

Mecanismo agregado:

- Metrica adaptativa diagonal tipo Fisher, aprendida con momentos por clase.
- Indice LSH sobre prototipos ponderados para evitar comparar contra todas las
  celulas en inferencia.

Prueba:

- Dataset sintetico de 1024 dimensiones con solo 18 dimensiones informativas.
- La base euclidiana ve casi todo como ruido de alta dimension.

Resultado de la ultima corrida:

- Euclidiano base: accuracy 0.249.
- Metrica adaptativa + indice: accuracy 0.963.
- Recall de dimensiones utiles entre los pesos top: 1.000.
- Candidatos por consulta: 897.0 -> 7.9.

## 2. Control del crecimiento

Mecanismo agregado:

- Consolidacion periodica con fusion de celulas compatibles de la misma clase.
- Poda por importancia informativa: soporte, pureza, confianza, margen y radio.

Prueba:

- Flujo multiclase con modos, ruido y ejemplos atipicos.
- Comparacion contra una red que no consolida sus celulas.

Resultado de la ultima corrida:

- Sin consolidacion: 476 celulas, accuracy 0.999.
- Con fusion/poda informativa: 122 celulas, accuracy 0.999.
- Reduccion: 3.90x menos celulas sin perder precision medible.

## 3. Secuencias y razonamiento

Mecanismo agregado:

- Codificador temporal con memoria decaida.
- Reglas de composicion para transiciones, pares ordenados y primera aparicion.

Prueba:

- Todas las secuencias tienen exactamente el mismo saco de simbolos.
- La etiqueta depende solo de una relacion: si A aparece antes que B.

Resultado de la ultima corrida:

- Bolsa de vectores independientes: accuracy 0.476.
- Memoria temporal + composicion: accuracy 1.000.

## 4. Aprendizaje global hibrido

Mecanismo agregado:

- Cabeza global ridge entrenada ocasionalmente sobre activaciones RBF de las
  celulas locales.
- Las celulas siguen siendo el sustrato local; el modulo global solo calibra una
  frontera de alto nivel.

Prueba:

- Dataset espiral no lineal donde la frontera local por vecino mas cercano deja
  huecos de generalizacion.

Resultado de la ultima corrida:

- Solo celulas locales: accuracy 0.797.
- Celulas + cabeza global: accuracy 0.902.
- Ganancia absoluta: +0.105.

## Lectura final

Los resultados son muy prometedores: los cuatro puntos abiertos ya tienen una
extension tecnica concreta, una prueba adversarial minima y una mejora
cuantificada. No prueba aun rendimiento en lenguaje real o datos industriales,
pero si muestra que la arquitectura morfogenica puede avanzar mas alla del
clasificador de prototipos 2D original.
