Investigación sobre una arquitectura morfogénica innata
Contexto y motivación

En 2026 los modelos generativos de lenguaje y otros modelos de deep learning ya alcanzan resultados sorprendentes, pero su entrenamiento sigue siendo extremadamente costoso. Para alcanzar rendimiento «SOTA» hay que entrenar redes con miles de millones de parámetros durante meses en clusters de GPUs, lo que consume energía y hardware en cantidades desproporcionadas. Esto contrasta con el cerebro humano, que aprende de pocos ejemplos, de forma incremental y con apenas 20 W de potencia. La desigualdad se debe a que la mayoría de los modelos artificiales:

Se inicializan casi sin estructura. Son grandes matrices de pesos aleatorios. Todo el conocimiento del mundo tiene que ser descubierto mediante millones de pasos de backpropagation sobre cantidades masivas de datos.
Actualizan globalmente todos sus parámetros. Cada nueva observación propaga gradientes por toda la red, obligando a recalcular millones de multiplicaciones.
Separan el aprendizaje de la utilización. El modelo se entrena en un pre‑training enorme y después se congela. Aprender algo nuevo exige volver a entrenar desde el principio o hacer fine‑tuning costoso.

Estas limitaciones han generado un interés creciente en aprendizaje local y arquitecturas inspiradas en el cerebro. Por ejemplo:

Las redes de espines superconductores y otros circuitos neuromórficos utilizan reglas de aprendizaje locales, con refuerzo, que se implementan físicamente en hardware. Sus simulaciones muestran que pueden ajustar pesos sin necesidad de programar manualmente cada sinapsis y que los tiempos de actualización son del orden de nanosegundos.
La técnica Forward‑Forward (FF) evita la retropropagación; entrena cada capa utilizando dos pasadas hacia delante y una función de energía local. Un trabajo reciente presentó Self‑Contrastive Forward–Forward (SCFF), que genera pares positivos/negativos automáticamente y alcanza una precisión competitiva en datasets como CIFAR‑10. Otra adaptación aplicó FF a redes de espigas y demostró que se puede entrenar un modelo completo sin backprop sobre redes neuromórficas.
Investigadores de la Universidad de Tsinghua propusieron un modelo híbrido que combina aprendizaje global y local. Este enfoque integra reglas de plasticidad locales con un mecanismo de error global y mejora el aprendizaje de pocas muestras y la tolerancia a fallos. Concluyen que ni el aprendizaje puramente global ni el puramente local son suficientes y que es necesaria una arquitectura que combine ambos.
Los modelos hopfield cuantizados y dispersos utilizan códigos discretos y reglas de aprendizaje locales inspiradas en la memoria asociativa, superando a las redes neuronales densas en tareas de memoria continua. El estudio destaca la importancia de aprender de forma online y continua mediante reglas locales.
En el ámbito de visión por computadora se han incorporado operadores morfológicos en redes profundas. Un trabajo de 2022 mostró que operadores de erosión y dilatación, integrados mediante meta‑aprendizaje, aportan descriptores topológicos que mejoran tareas como clasificación y detección de bordes.

Esta literatura indica que las ideas de plasticidad local, memoria estructural y morfología no solo son biológicamente plausibles, sino que pueden ofrecer eficiencia y adaptabilidad. Sin embargo, muchas propuestas dependen todavía de operaciones tipo gradient descent, redes densas o hardware especializado.

Propuesta: arquitectura morfogénica innata

Como ejercicio de investigación conceptual, se propuso diseñar una arquitectura de aprendizaje que rompa el paradigma de «matriz de pesos + retropropagación». La idea central es que el modelo nazca con mecanismos innatos de aprendizaje, no con un conjunto de pesos aleatorios. Estos mecanismos permiten detectar patrones, asociar causas y efectos, recordar experiencias y reorganizar su propia estructura. El modelo se compone de células morfogénicas que pueden crearse, fusionarse, dividirse o eliminarse según la experiencia, análogas a neuronas que crecen y se adaptan.

Unidades básicas: células morfogénicas

Cada célula morfogénica mantiene:

Un vector de patrón w_i ∈ R^d que describe la característica (o «prototipo») que la célula detecta.
Una etiqueta o identidad y_i (por ejemplo, la clase asociada en un problema de clasificación).
Un nivel de confianza c_i y contadores de uso, que reflejan la fiabilidad de la célula.

La red inicial se construye con un conjunto muy reducido de células (o incluso ninguna) y va creciendo a medida que recibe datos. Las células se activan solamente cuando la entrada es similar a su vector de patrón, de modo que no es necesario evaluar toda la red para cada ejemplo; sólo las células relevantes se actualizan.

Predicción y aprendizaje local

Ante una entrada x, la red calcula la distancia entre x y cada vector w_i de las células activas. La célula que minimiza la distancia da la predicción del modelo:

ŷ(x) = y_j donde j = argmin_i ||x - w_i||

El aprendizaje se realiza localmente según reglas simples:

Adaptación de la célula correcta: si la predicción coincide con la etiqueta verdadera (ŷ = y), el vector de patrón de la célula más cercana se mueve ligeramente hacia la entrada, refinando el prototipo:

w_i ← w_i + α (x - w_i)

donde α ∈ (0,1) es un ritmo de aprendizaje local. Sólo se modifica esa célula.

Creación de nuevas células: si la predicción es incorrecta, se comprueba la distancia d = ||x - w_i||. Si d es grande (mayor que un umbral θ) o si se quiere capturar una variación que la célula actual no puede representar, se crea una nueva célula con w_new = x y y_new = y. De esta manera la red crece cuando encuentra patrones novedosos.
Fusión y poda: periódicamente se pueden fusionar células cuyos vectores sean muy cercanos o eliminar aquellas que rara vez se activan. Estas operaciones morfogénicas permiten que la red mantenga un tamaño razonable y previenen la proliferación incontrolada.

Estas reglas no requieren retropropagación ni actualizar miles de parámetros a la vez; la decisión de crear o adaptar células depende exclusivamente de la entrada y de la célula activada. Además, se pueden ejecutar en paralelo para diferentes subconjuntos de células.

Ventajas conceptuales
Aprendizaje incremental: cada ejemplo modifica solo una pequeña parte de la red o genera una célula nueva. No hay necesidad de una fase de pre‑entrenamiento costosa.
Eficiencia computacional: las operaciones son simples (distancias Euclidianas, actualizaciones vectoriales) y pueden ejecutarse en hardware básico o incluso en microcontroladores. Una sola GPU puede gestionar miles de células sin saturarse.
Memoria y plasticidad: la red incorpora memoria explícita en forma de células; puede recordar ejemplos raros mediante células especializadas y generalizar mediante células ampliamente usadas.
Adaptación dinámica: cuando el entorno cambia, la red crea nuevas células o ajusta las existentes sin necesidad de re‑entrenar todo el sistema.
Experimentos de prueba de concepto

Para evaluar si este esquema podría competir con modelos tradicionales en pequeñas tareas de clasificación, se realizaron experimentos en Python comparando una red de células morfogénicas con modelos de referencia. Se generaron datasets sintéticos y se midió el tiempo de entrenamiento, la precisión y el número de células creadas.

Dataset lineal (dos clases)

Se generaron 400 muestras de dos clases en 2D. Se comparó la red morfogénica con un modelo de regresión logística entrenado mediante gradiente. La regresión logística necesitó ~0.012 s de entrenamiento y alcanzó precisión perfecta en train/test. La red morfogénica necesitó ~0.005 s, creó únicamente dos células, y también alcanzó precisión perfecta en train y test. Ambos modelos resolvieron la tarea, pero la red morfogénica lo hizo con la mitad de tiempo y sin actualizar toda una matriz de pesos.

Dataset multiclase (cuatro grupos separables)

Para un dataset de cuatro clases con 600 muestras, el clasificador logístico tardó ~0.039 s en entrenar y alcanzó precisión ≈1.0. La red morfogénica tardó ~0.031 s, consiguió precisión ≈1.0 y creó 14 células. Esto demuestra que las células pueden representar múltiples clases con pocos prototipos.

Dataset no lineal (círculos concéntricos)

Se generó un dataset con dos círculos concéntricos (problema no lineal). La regresión logística lineal fracasó (≈ 0.61 de precisión en test). Un perceptrón multicapa (MLP) alcanzó precisión ≈ 1.0, pero tardó ≈0.68 s en entrenar. La red morfogénica, en cambio, tardó sólo ≈0.02 s, creó 16 células y logró 99.3 % de precisión en el conjunto de prueba. Estos resultados muestran que la arquitectura propuesta compite con una red neuronal no lineal en un problema complejo, pero con entrenamiento local y rápido.

Discusión

Los experimentos evidencian que una red de células morfogénicas puede aprender representaciones útiles con operaciones locales, sin necesidad de retropropagación. Para datos sencillos y de dimensión baja, su rendimiento se acerca o incluso supera a modelos globales tradicionales, con tiempos de entrenamiento inferiores y con estructuras de memoria interpretables (células/prototipos). La red se comporta como un clasificador por prototipos, pero incorpora la capacidad de actualizarse dinámicamente y crecer según los errores.

No obstante, quedan desafíos antes de que una arquitectura morfogénica innata pueda escalar a problemas del mundo real:

Escalabilidad a alta dimensión: para datos de muy alta dimensión, la distancia Euclidiana puede volverse poco informativa. Habrá que estudiar métricas adaptativas y métodos de indexación eficientes.
Control del crecimiento: en tareas complejas la red podría crear demasiadas células. Se necesitarían mecanismos de fusión más sofisticados y criterios de importancia basados en información.
Extensión a secuencias y razonamiento: el esquema actual clasifica vectores independientes. Para tareas de lenguaje o tiempos largos habría que añadir memoria temporal y reglas de composición que permitan abstraer relaciones.
Combinación con aprendizaje global: las referencias citadas sugieren que integrar un mecanismo global (p. ej. ajuste fino ocasional o supervisión de alto nivel) puede mejorar la generalización. Un híbrido de células locales y módulos globales podría aportar lo mejor de ambos mundos.
Conclusiones

El análisis bibliográfico muestra que el campo está explorando activamente aprendizaje local, reglas de plasticidad y operadores morfológicos como alternativas al costoso backpropagation. Inspirándose en estas ideas y en la eficiencia del cerebro, se propuso una arquitectura morfogénica innata donde el modelo no nace con pesos aleatorios, sino con mecanismos de aprendizaje y transformación que le permiten construir y reorganizar su estructura a partir de la experiencia.

Los experimentos de juguete demuestran que una red basada en células morfogénicas puede aprender tareas de clasificación de forma rápida y local, incluso en problemas no lineales, utilizando una sola CPU/GPU y sin pre‑training. Esto sugiere que hay caminos prometedores para diseñar modelos eficientes que aprendan como sistemas vivos, aunque aún resta mucho para competir con arquitecturas profundas en dominios complejos. El siguiente paso consistiría en extender estas ideas a datos secuenciales, incorporar memoria de trabajo y explorar mecanismos de fusión/priorización que imiten procesos biológicos.