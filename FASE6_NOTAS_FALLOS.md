# FASE6_NOTAS_FALLOS

- Esta suite usa scikit-learn real (1.8.0), pero
  todavia no incluye modelos profundos grandes ni GPUs.
- La comparacion principal usa 10 seeds; la busqueda
  maliciosa usa 1 seeds para mantener la
  corrida CPU reproducible.
- Los datasets grandes pueden estar capados de forma estratificada si se corre
  con `--max-samples`; el valor usado aqui fue 6000.
- Los datasets OpenML quedan como extension futura si se quiere dependencia de
  red/cache de OpenML; esta corrida priorizo UCI descargable y reproducible.
- Los ataques son maliciosos pero no optimizados por gradiente: ruido,
  oclusion/shuffle de features Fisher, interpolacion hacia clase opuesta,
  label poisoning y features basura.
- La memoria de scikit-learn se estima por serializacion `pickle`; la memoria
  nativa exacta puede diferir.
