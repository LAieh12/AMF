# FASE7_NOTAS_FALLOS

- AMF7 ya no es una arquitectura local pura. Es un supercampo hibrido: memorias
  AMF locales, expertos globales fuertes, MLP de mas de un millon de parametros
  y reglas de compuerta aprendidas en validation.
- La mejora fuerte aparece en datasets no saturados, especialmente Madelon. En
  Wine, WDBC e Ionosphere el techo de accuracy de los clasicos ya es muy alto,
  asi que AMF7 empata al mejor clasico en vez de separarse.
- La corrida final usa 3 seeds por costo. Es suficiente para validar la mejora
  arquitectonica inicial de Fase 7, pero una paperizacion futura deberia repetir
  con 10 o 20 seeds.
- Algunas reglas de routing son heuristicas dependientes de forma del problema:
  binario pequeno, multiclass pequeno, alta dimension y tabular grande. No usan
  labels de test, pero si codifican sesgos aprendidos durante esta fase.
- Los modelos de millones de parametros son `MLPClassifier(1024, 1024)` dentro
  de AMF7 y como baseline. No se usaron redes profundas externas ni GPU.
