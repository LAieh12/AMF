# FASE7_COMPLETADA

Se completo la Fase 7 con una arquitectura nueva: `AMF7SuperField`.

Entregables:

- `phase7_architecture.py`
- `run_phase7.py`
- `results/phase7_latest.json`
- `results/FASE7_RESULTADOS.md`
- `results/FASE7_TABLA_PAPER.md`
- `FASE7_NOTAS_FALLOS.md`

Resultado verificado en la corrida final:

```powershell
python run_phase7.py --seeds 3 --datasets iris wine wdbc ionosphere sonar spambase madelon
```

AMF7 gano 4 de 7 datasets y empato los otros 3 contra el mejor modelo clasico
por dataset. No tuvo perdidas en la corrida final. El gap promedio fue positivo
y la mejora mas grande aparecio en Madelon, el caso de alta dimension con muchas
features distractoras.
