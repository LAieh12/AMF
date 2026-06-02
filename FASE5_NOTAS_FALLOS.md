# FASE5_NOTAS_FALLOS

- No se uso scikit-learn porque no esta instalado en el entorno. Por eso los
  baselines se implementaron en NumPy como aproximaciones reproducibles:
  ExtraTrees, gradient boosting por stumps, RBF-SVM-like por landmarks, kNN
  ponderado, radius neighbors, Naive Bayes, Passive-Aggressive y SGD lineal.
- Los datasets reales descargados localmente fueron UCI Iris, Wine, WDBC,
  Optical Digits y Madelon. No se incluyo Adult/Covertype/Higgs por tiempo y
  tamano de descarga.
- Madelon valid no trajo labels desde el mirror usado, asi que se hizo split
  estratificado sobre train.
- MNIST/Fashion-MNIST con PCA/embeddings queda pendiente porque no habia un
  loader local confiable sin agregar dependencias grandes.
- El few-shot cruel ahora usa Optical Digits multiclass con clases viejas 0-4 y
  clases nuevas 5-9. Aun asi no cubre clases nuevas tabulares no visuales.
- El boundary attack generico es black-box por direcciones aleatorias; no es un
  ataque adversarial optimizado tipo gradient-based.
