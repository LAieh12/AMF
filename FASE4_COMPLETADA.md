# Fase 4 completada

La mejora fuerte no vino de agregar mas piezas, sino de cambiar la decision
central de la arquitectura.

## Hipotesis que funciono

Fase 3 aprendia buenas celulas, pero decidia con una sola celula ganadora. Eso
la hacia fragil cerca de fronteras adversariales. Fase 4 conserva las celulas
morfogenicas y cambia la inferencia:

- selecciona un subespacio de 32 dimensiones con mayor peso Fisher;
- compara las celulas en ese subespacio informativo;
- deja votar a 8 celulas cercanas;
- pondera cada voto por distancia, radio, importancia y pureza historica.

El resultado es `AttentionalMorphogenicClassifier` en
`phase4_architecture.py`.

## Comando

```powershell
python run_phase4.py
```

Salidas:

- `results/phase4_latest.json`
- `results/FASE4_RESULTADOS.md`

## Mejora principal

En el stress test grande de fase 3:

- fase 3 nearest-cell: clean 0.994, adversarial 0.802;
- fase 4 attentional field: clean 1.000, adversarial 0.988;
- random forest: clean 0.998, adversarial 0.945.

La mejora adversarial absoluta de fase 4 sobre fase 3 es +0.186, y supera al
random forest en epsilon 0.12 y tambien en el sweep hasta epsilon 0.16.

## Clases nuevas

Despues de entrenar con clases 0-3 y actualizar con 440 ejemplos few-shot de
clases 4-7:

- fase 3: old 0.991, nuevas 0.936, olvido 0.008;
- fase 4: old 1.000, nuevas 1.000, olvido 0.000.

## Drift temporal

En el stream con drift:

- fase 3: mean 1.000, ultimo chunk 1.000;
- fase 4: mean 1.000, ultimo chunk 1.000.

Aqui fase 4 no necesitaba superar mucho porque fase 3 ya estaba saturada, pero
mantiene la propiedad online sin romperla.

## Lectura honesta

La arquitectura fase 4 es claramente mejor en el punto mas debil encontrado en
fase 3: robustez de frontera. La mejora es impresionante porque no depende de
un modelo externo ni de guardar todo el dataset; sigue siendo una red de celulas
locales, solo que ahora decide como un campo morfogenico atencional en lugar de
un nearest-neighbor duro.
