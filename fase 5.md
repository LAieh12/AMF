Objetivo de Fase 5

Probar si AMF sigue funcionando cuando:

1\. el benchmark no fue diseñado para él,

2\. el ataque no está basado en prototipos,

3\. hay datasets reales,

4\. se repite con muchas seeds,

5\. cada pieza del campo atencional se abla cuidadosamente.



Yo la llamaría:



Fase 5 — Generalización y anatomía del campo morfogénico atencional



1\. Datasets reales, no solo sintéticos



La prioridad número uno es salir de datos sintéticos.



Empieza con datasets tabulares reales porque AMF está más cerca de ese tipo de estructura que de lenguaje bruto o visión cruda.



Tipo	Datasets sugeridos	Qué prueba

Tabular clásico	Iris, Wine, Breast Cancer, Digits	sanity check

Tabular más serio	OpenML/UCI, Adult, Covertype, Higgs pequeño	generalización real

Ruido/alta dimensión	Madelon, Arcene, Gisette	Fisher/subespacio

Drift	datos sintéticos + streams reales si consigues	adaptación online

Visión simple	MNIST/Fashion-MNIST con PCA o embeddings	escala visual inicial



No empieces con CIFAR-10 raw de golpe. Eso puede matar la arquitectura por una razón injusta: AMF no tiene todavía un extractor visual profundo.



Mejor prueba dos versiones:



A) AMF sobre pixeles reducidos con PCA.

B) AMF sobre embeddings congelados de un encoder pequeño.

2\. Seeds múltiples sí o sí



Una sola corrida ya no basta.



Para Fase 5, cada resultado importante debería tener:



mean accuracy ± std

mean fit time ± std

mean predict time ± std

mean cells ± std

mean model MB ± std



Ejemplo:



AMF-5: 0.941 ± 0.012 acc, 180 ± 15 cells, 0.42 ± 0.05 MB



Mínimo usa:



seeds = \[0, 1, 2, 3, 4]



Ideal:



10 seeds



Esto vuelve los resultados mucho más serios.



3\. Ataques nuevos, no hechos para AMF



El ataque de Fase 4 fue útil, pero estaba conectado a frontera de prototipos. Como AMF fue diseñado para arreglar esa frontera, ahora toca atacarlo con métodos más generales.



Prueba:



Ataque	Idea

Gaussian noise	ruido aleatorio normal

Feature dropout	apagar dimensiones al azar

Feature swap	intercambiar dimensiones informativas

Worst-feature perturbation	perturbar las features Fisher top

Random direction attack	mover ejemplos en direcciones aleatorias normalizadas

Boundary attack genérico	buscar cambios que alteren predicción sin usar estructura interna

Label noise	entrenar con etiquetas parcialmente corruptas



Especialmente importante:



Ataque a top Fisher features



Porque Fase 4 depende mucho de seleccionar 32 features. Según los datos completos, AMF usa top\_features=32, vote\_k=8 y células votantes sobre el subespacio Fisher.

Entonces hay que probar qué pasa cuando esas features se dañan.



4\. Anatomía del campo atencional



Esta es la parte más importante científicamente.



No basta decir “votar con 8 células funciona”. Hay que saber por qué funciona.



Haz ablaciones como:



Variante	Qué responde

vote\_k=1	equivale casi a Fase 3

vote\_k=3,5,8,16,32	curva de robustez vs costo

sin peso por distancia	si la distancia realmente importa

sin radio	si el tamaño de célula aporta

sin importancia	si el historial aporta

sin pureza	si la confiabilidad histórica aporta

sin Fisher	si el subespacio sigue siendo crítico

top\_features 8/16/32/64/128	tradeoff señal vs ruido

voto uniforme	si el campo gana por atención o solo por ensemble

voto por clase normalizado	si clases con más células dominan



La tabla clave sería algo así:



Modelo	Clean	Adv	Cells	Pred s	Interpretación

AMF full	?	?	?	?	modelo principal

k=1	?	?	?	?	frontera dura

sin pureza	?	?	?	?	rol de confiabilidad

sin radio	?	?	?	?	rol de geometría local

voto uniforme	?	?	?	?	atención vs mayoría simple



Si el full gana de forma consistente, ya tienes argumento fuerte.



5\. Comparación contra baselines más peligrosos



En Fase 4 ya superaste al random forest en el benchmark adversarial principal: AMF logró 1.000 clean y 0.988 adversarial, mientras random forest quedó en 0.998 clean y 0.945 adversarial.

Pero para Fase 5 necesitas rivales más duros.



Agrega:



ExtraTrees

GradientBoosting

HistGradientBoosting

RBF-SVM

weighted kNN

radius neighbors

nearest centroid

Gaussian Naive Bayes

online passive-aggressive

SGDClassifier



Especialmente RBF-SVM y gradient boosting, porque esos sí pueden pelear fronteras no lineales.



La pregunta no es “¿AMF gana todo?”. La pregunta correcta es:



¿En qué zona del mapa AMF domina?



Tal vez AMF no sea mejor que todos en accuracy puro, pero puede ganar en:



incrementalidad

bajo olvido

drift

memoria compacta

pocos candidatos

robustez con pocos ejemplos

explicabilidad por células

6\. Métricas nuevas



Accuracy ya no basta.



Fase 5 debe medir:



accuracy

balanced accuracy

macro F1

confusion matrix

fit time

predict time

update time

model MB

peak RAM

número de células

candidatos promedio

votos promedio

olvido catastrófico

adaptación a clases nuevas

drift recovery time

robustez por epsilon



Una métrica buena para tu proyecto sería:



Morphogenic Efficiency Score



Algo simple:



MES = accuracy / (model\_MB \* avg\_candidates)



O una variante:



MES = (accuracy \* robustness \* incremental\_score) / (model\_MB \* predict\_time)



No como métrica oficial absoluta, sino como métrica interna para comparar versiones.



7\. Tests de drift más duros



Fase 4 mantiene el drift temporal prácticamente saturado: mean 0.9998, último chunk 1.000, con solo 7 células y 0.014 MB.

Eso es buenísimo, pero puede significar que el drift todavía es demasiado fácil para AMF.



Haz tres tipos:



gradual drift:

las clases se mueven poco a poco



sudden drift:

las distribuciones cambian de golpe



recurring drift:

conceptos viejos vuelven después de desaparecer



Y mide:



accuracy antes de update

accuracy después de update

cuántas células nuevas crea

cuántas células poda

si recuerda conceptos viejos cuando vuelven



El recurring drift es clave. Ahí se ve si AMF “recuerda” o solo se adapta al presente.



8\. Clases nuevas más difíciles



Fase 4 logró 1.000 viejas, 1.000 nuevas y 0.000 olvido en el setup de clases nuevas.

Excelente, pero ahora hazlo más cruel:



1-shot por clase nueva

5-shot por clase nueva

10-shot por clase nueva

clases nuevas parecidas a clases viejas

clases nuevas con ruido

clases nuevas mezcladas con datos viejos

clases nuevas que aparecen desbalanceadas



Tabla ideal:



Shots	Old after	New acc	Forgetting	Cells added

1	?	?	?	?

5	?	?	?	?

10	?	?	?	?

50	?	?	?	?



Esto probaría si realmente aprende few-shot.



9\. Entregables de Fase 5



Yo organizaría los archivos así:



phase5\_architecture.py

phase5\_datasets.py

phase5\_attacks.py

phase5\_ablations.py

phase5\_baselines.py

phase5\_metrics.py

run\_phase5.py



Resultados:



results/phase5\_latest.json

results/FASE5\_RESULTADOS.md

FASE5\_COMPLETADA.md

FASE5\_NOTAS\_FALLOS.md



Ese último es importante. Documenta lo que salió mal. Eso hace que el research se vea real.



10\. La hipótesis principal de Fase 5



La hipótesis debe ser clara

