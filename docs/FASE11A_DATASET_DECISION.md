# Fase 11A - decision de datasets para Never

Fecha: 2026-06-02

## Decision corta

Si, vale seguir usando MovingMNIST, pero no como unico benchmark.

MovingMNIST debe quedar como:

- smoke test;
- prueba rapida de dinamica visual;
- laboratorio para slots, rebotes, memoria AMF y decoder copy-skip;
- comparacion reproducible con pocos ejemplos.

No debe usarse para afirmar que Never tiene un encoder/decoder definitivo.

## Por que todavia sirve

MovingMNIST sigue siendo util porque separa un problema claro:

```text
puede el world model inferir estado y transicion visual con pocos datos?
```

Si Never no mejora aqui, fallara en video mas complejo. Es barato y permite iterar.

## Por que ya no basta

El dataset es demasiado simple:

- objetos casi siempre aislables por componentes;
- poca semantica;
- dinamica fisica simple;
- fondo negro;
- no prueba memoria larga ni razonamiento;
- puede favorecer heuristicas manuales de movimiento.

## Puerta de promocion

No pasar a claims fuertes hasta que el codec logre en MovingMNIST:

- skill positivo en h5/h10/h17 contra `last`;
- `amf_world_mse <= slot_velocity_mse`;
- `never_definitive_mse` cerca de `candidate_oracle_mse`;
- crecimiento real de `transition_cells`;
- mejora sobre el mejor slot-hybrid/ranker o explicacion clara de por que no.

## Siguiente dataset recomendado

Despues de pasar MovingMNIST, agregar uno de estos:

1. KTH Actions:
   video real simple, movimiento humano, fondos relativamente controlados.

2. BAIR Robot Pushing:
   dinamica fisica y objetos reales, mas dificil para slots.

3. TaxiBJ:
   dinamica espaciotemporal no visual-natural, util para probar world model sin depender de rendering.

4. Tarea simbolica/secuencial propia:
   necesaria si Never apunta a razonamiento y no solo prediccion visual.

## Regla practica

No cambiar dataset para esconder fallos.

Primero:

```text
arreglar selector/dinamica en MovingMNIST
```

Luego:

```text
validar que el mismo codec no colapse en un dataset mas complejo
```

Asi se evita optimizar un decoder vistoso que no resuelve el world model.
