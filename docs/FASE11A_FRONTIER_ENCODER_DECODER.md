# Fase 11A - encoder/decoder frontier notes

Fecha: 2026-06-02

## Fuentes revisadas

- Seedance 2.0 model card / arXiv 2604.14148: arquitectura unificada para generacion audio-video multimodal.
- OpenAI Sora technical report y Sora 2 notes: compresion a latentes, patches espacio-temporales y transformer/diffusion sobre esos patches.
- SimVP, CVPR 2022: encoder, translator y decoder CNN simples; la clave no es un decoder magico, sino preservar informacion espacial y aprender evolucion temporal.
- PredFormer / Video Prediction Transformers without Recurrence or Convolution: transformer temporal puro con bloques gated y analisis de atencion 3D.
- MAGVIT-v2 / VideoPoet / Latte: tokenizer visual fuerte + modelado global sobre tokens/latentes.
- ProMAG / VFRTok / ViTok / RAE / DDT: el cuello de botella moderno esta en la calidad/capacidad del tokenizer, la compresion temporal y la separacion semantica-detalle.

## Lecciones para AMF

1. AMF no debe comprimir demasiado pronto.
   La evidencia de SimVP y tokenizers de video modernos apunta a conservar detalle local mediante latentes/patches y reconstruccion con rutas tipo skip/copy. Para MovingMNIST, esto favorece decoders que copian o desplazan patches/slots de frames anteriores antes de intentar reconstruir desde un unico vector.

2. El encoder necesita tokens espacio-temporales, no solo vectores globales.
   Sora/Latte/PredFormer usan patches o tokens 3D. En AMF, el equivalente practico son slots/tiles con estado temporal: posicion, masa/intensidad, velocidad, historia corta y confianza por patch.

3. El decoder debe estar desacoplado del encoder.
   DDT y RAE sugieren separar representacion semantica de reconstruccion de detalle. Para AMF, el encoder deberia decidir "que objeto/slot se mueve y hacia donde"; el decoder deberia hacer copy/warp/render conservador desde detalle local.

4. La compresion temporal debe crecer progresivamente.
   ProMAG muestra que forzar alta compresion temporal desde cero degrada reconstruccion. Para AMF, conviene entrenar/probar primero horizontes cortos y usar esos estados como guia para horizontes largos, en vez de pedir h17 directo desde un embedding comprimido.

5. Escalar celulas solo ayuda si aumenta cobertura efectiva.
   La prueba wide ya mostro que subir `max_cells` no crea mas capacidad por si solo: con 220 secuencias se quedaron 3740 celulas. Para escalar de verdad hay que bajar radios/umbrales, diversificar features o cambiar la regla de creacion; no basta con subir el limite.

## Direccion prometedora

La arquitectura mas prometedora para AMF no es un decoder pesado clasico, sino:

- encoder slot/tile con tokens espacio-temporales;
- ranker global ligero que elige candidatos/ramas por horizonte;
- AMF local denso para memoria de transiciones;
- decoder copy-skip que conserva pixels/patches del pasado;
- crecimiento progresivo por horizonte;
- control de celulas basado en importancia/cobertura, no solo en conteo.

## Estado empirico local

Hasta las pruebas actuales, el mejor camino sigue siendo el slot-hybrid/ranker:

- test-best h10/h17 aproximado en 650/40: 0.2775 / 0.2218;
- oracle h10/h17: 0.3444 / 0.3012;
- AMF wide en 220/20 no mejoro y creo las mismas 3740 celulas efectivas;
- patch-attention simple, multi-encoder y Mini-SimVP rapido fueron negativos.

Conclusion: la mejora debe venir de tokens/slots temporales con decoder de copia y seleccion global, no de un decoder monolitico ni de aumentar limites de celulas sin modificar la densidad real.
