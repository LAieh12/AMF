# AMF5 / AMF6 research suite

This folder contains the morphogenic attentional field experiments derived from
`Investigacion.md`. Fases 1-5 tested whether the idea can work. Fase 6 turns the
question into external reproducibility: compare AMF5 against official
scikit-learn baselines, with clear train/validation/test splits, multiple seeds,
paper-style tables, measured cost, and malicious failure search.

## Quick start

```powershell
python download_phase6_data.py
python run_phase6.py --seeds 10 --datasets iris wine wdbc ionosphere sonar spambase madelon --attack-seeds 1
```

The default run uses a stratified cap of 6000 samples per dataset. Use
`--max-samples 0` to disable the cap, or pass no `--datasets` list to run every
available dataset, including `optdigits`, `pendigits`, and `satimage`.

## Outputs

- `results/phase6_latest.json`: raw records, validation choices, environment,
  timing, memory, and attack/stress records.
- `results/FASE6_RESULTADOS.md`: readable phase report.
- `results/FASE6_TABLA_PAPER.md`: compact table for paper-style comparison.
- `FASE6_NOTAS_FALLOS.md`: honest limitations and failure notes.
- `paper/AMF5_FORMALIZACION.md`: mathematical definition of AMF5.
- `docs/AMF5_ARCHITECTURE.mmd`: Mermaid architecture diagram.
- `results/phase7_latest.json`: AMF7 hybrid superfield records against strong
  classical baselines and million-parameter MLP tests.
- `results/FASE7_RESULTADOS.md`: Fase 7 win/tie report.
- `results/phase8_latest.json`: morphogenic decoder benchmark with generated
  outputs from prompts not seen during training.
- `results/FASE8_RESULTADOS.md`: Fase 8 decoder report and example speech.
- `results/phase8_5_latest.json`: AMF8 vs Pythia-70M response-generation
  benchmark using the same prompt suite and scorer.
- `results/FASE8_5_RESULTADOS.md`: paradigm comparison: local morphogenic
  output vs dense pretrained LLM output.
- `results/phase9_latest.json`: AMF8 Domain Expansion scaling records for
  90, 300, 1000, 3000, and 10000 examples.
- `results/FASE9_RESULTADOS.md`: multi-domain assistant report with separated
  memories, structured outputs, baselines, and online learning probe.
- `results/phase10a_latest.json`: synthetic AMF world-model pretraining
  records with one-step, rollout, bounce, metaplasticity, and export checks.
- `results/FASE10A_RESULTADOS.md`: Phase 10a report for the toy simulator,
  AMF dynamics cells, baselines, metaplasticity, and warm export.
- `results/phase10b_latest.json`: visual latent codec benchmark across
  32/64/128/256 px frames.
- `results/FASE10B_RESULTADOS.md`: encoder/decoder report with latent
  invariants, visual prediction, compression, and raw-pixel AMF memory estimate.
- `results/phase10c_latest.json`: latest N.E.V.E.R. Phase 10c orchestrator run.
- `results/FASE10C_RESULTADOS.md`: N.E.V.E.R. Phase 10c integration report:
  prompt/action orchestrator, AMF latent loop, metaplasticity, decoder and SNN
  retirement.
- `results/phase11a_latest.json`: Moving MNIST real visual-rollout records.
- `results/FASE11A_RESULTADOS.md`: Fase 11A report using downloaded Moving
  MNIST, compact two-object latents, baselines, rollout metrics and honest
  target status.
- `results/FASE11A_SLOT_RANKER.md`: object/slot-level multi-hypothesis ranker
  report for causal Moving MNIST rollout.
- `results/FASE11A_PATCH_DECODER.md`: patch-level decoder probe and negative
  result for pure learned masks.
- `results/FASE11A_KINEMATIC_ENCODER.md`: kinematic temporal encoder probe and
  negative result for single smoothed-velocity hypotheses.
- `results/FASE11A_CELL_ROUTER.md`: cell/tile router decoder probe and negative
  scaled result for spatial-only routing.
- `results/FASE11A_NEVER_DEFINITIVE_CODEC.md`: current best N.E.V.E.R./AMF
  world-codec result. The 650/40 real MovingMNIST run uses slot encoding, AMF
  transition memory, hybrid copy-skip decoding, and `slot_amf_mean`, reaching
  MSE/skill-vs-last of h1 `0.022883/0.518941`, h5 `0.049480/0.294730`,
  h10 `0.051417/0.281871`, and h17 `0.054018/0.268249`.
- `results/FASE11A_NEVER_BOTTLENECK_AUDIT.md`: audit of best metrics and
  whether experiments use partial AMF pieces or a fuller Never world model.
- `results/FASE12A_PHYSICALAI_DATASET.md`: NVIDIA PhysicalAI manifest, scene
  ladder, shard counts, and streaming smoke for the next world-model stage.
- `results/FASE12A_PHYSICALAI_PHYSICS_SMOKE.md`: real `objects_falling`
  physics-shard inspection, confirming usable `com`, `velocity`, `spin`, and
  `rot` arrays without downloading RGB/depth shards.
- `results/FASE12A_PHYSICALAI_WORLD_PROBE.md`: PhysicalAI physics world
  probe. On `objects_falling` physics shard `00007`, the validation-selected
  hybrid AMF/Ridge world model reaches h1/h5/h15/h30 MSE
  `0.000194/0.006390/0.054260/0.225712`, beating Ridge at every horizon with
  h30 gain-vs-Ridge `0.149995` under a deterministic shuffled sequence split.
- `results/FASE12A_BILLIARDS_WORLD_PROBE.md`: second real PhysicalAI 12A
  scene. On `billiards` physics shard `00000`, the same validation-selected
  world probe reaches h1/h5/h15/h30 MSE
  `0.000003/0.000347/0.006750/0.051179`; h30 beats Ridge by `0.169266`.
- `results/FASE12B_BOWLING_WORLD_PROBE.md`: first real PhysicalAI 12B
  impact/collision probe. On `bowling` physics shard `00000`, the hybrid
  AMF/Ridge world model reaches h1/h5/h15/h30 MSE
  `0.000063/0.002284/0.024849/0.097678`, beating Ridge at every horizon with
  h30 gain-vs-Ridge `0.279631`.
- `results/FASE12B_DOMINOES_WORLD_PROBE.md`: real PhysicalAI 12B causal-chain
  probe. On `dominoes` physics shard `00000`, AMF/Ridge reaches h1/h5/h15/h30
  MSE `0.000000/0.000011/0.000342/0.004851`; h15/h30 beat Ridge by
  `0.124300/0.141610`.
- `results/FASE12B_BOWLING_CONTACT_WORLD_PROBE.md`: contact-context diagnostic
  with nearest-neighbor, relative velocity, closing speed, and local density
  features. At matched `stride=30`, it improves `bowling` h30 AMF MSE
  `0.096619 -> 0.092665`, while hurting shorter horizons; this identifies
  mask/object identity as the next encoder bottleneck. See
  `results/FASE12B_CONTACT_DIAGNOSTIC.md` for the matched table.
- `results/FASE12B_DOMINOES_IDENTITY_WORLD_PROBE.md`: slot-identity probe
  using `segmentation_colors`, object code, and slot index from real physics
  NPZ metadata. On `dominoes`, it improves h15/h30 AMF MSE
  `0.000342 -> 0.000339` and `0.004851 -> 0.004809`; see
  `results/FASE12B_IDENTITY_DIAGNOSTIC.md`.
- `results/FASE12B_DOMINOES_ORIENTATION_WORLD_PROBE.md`: orientation probe
  using PhysicalAI `rot` quaternions and recent rotation delta. On `dominoes`,
  it improves h15 MSE to `0.000335`; h30 is `0.004839`.
- `results/FASE12B_DOMINOES_ENCODER_SELECTOR_PROBE.md`: validation-selected
  encoder probe across base, identity, and orientation. It selects orientation
  for h15/h30 on validation; see `results/FASE12B_ENCODER_DIAGNOSTIC.md`.
- `results/FASE12B_DOMINOES_ENCODER_ENSEMBLE_PROBE.md`: validation-selected
  convex ensemble over base, identity, and orientation encoders. On `dominoes`,
  it improves h15/h30 MSE to `0.000330/0.004697`, beating the best individual
  encoder and best Ridge by h30 gain `0.159212`.

## What Fase 6 measures

- Official scikit-learn baselines: logistic regression, LinearSVC, RBF SVC,
  kNN, GaussianNB, RandomForest, ExtraTrees, HistGradientBoosting, Dummy, and a
  small MLP when enabled.
- AMF5 validation variants: compact/default/wide on small datasets, adaptive
  compact grid on large or high-dimensional datasets.
- Split discipline: train 60%, validation 20%, test 20%, stratified by seed.
- Complexity: fit seconds, predict seconds, samples/second, peak fit RAM,
  serialized model MB, AMF cells, AMF selected features, and candidate/vote
  counts.
- Malicious tests: Gaussian noise, top-Fisher zeroing, top-Fisher shuffling,
  nearest-opposite interpolation, label poisoning, and appended random features.

## Core files

- `phase5_architecture.py`: AMF5 implementation.
- `phase6_datasets.py`: reproducible UCI dataset loaders and splits.
- `phase6_sklearn_baselines.py`: official sklearn model grids.
- `phase6_metrics.py`: timing, memory, metrics, and summaries.
- `phase6_malicious.py`: failure-search attacks and high-dimensional stress.
- `run_phase6.py`: orchestration and report writing.
- `phase7_architecture.py`: AMF7 SuperField, a validated hybrid of local AMF
  memories, global experts, top-k ensemble routing, and shape-aware anchors.
- `run_phase7.py`: Fase 7 comparison against the best classical baselines.
- `phase8_corpus.py`: small controlled conversational corpus and prompt suite.
- `phase8_morphogenic_decoder.py`: non-dense morphogenic input field and
  multiple output decoders.
- `run_phase8.py`: decoder comparison and report generation.
- `run_phase8_5.py`: response-generation benchmark for AMF8 and optional
  Pythia-70M execution.
- `phase9_corpus_builder.py`: scalable multi-domain corpus generator.
- `phase9_domain_memory.py`: domain router and separated domain memories.
- `phase9_decoder_scaling.py`: Phase 9 morphogenic assistant and composition.
- `phase9_eval_prompts.py`: difficult multi-domain and structured prompt suite.
- `phase9_baselines.py`: global nearest and domain-template baselines.
- `run_phase9.py`: Phase 9 scaling benchmark and report writer.
- `phase10a_toy_simulator.py`: toy gravity/bounce simulator for synthetic
  `(S_t, action, S_t+1)` pretraining pairs.
- `phase10a_amf_world_model.py`: AMF dynamics cells that store deltas
  `S_t+1 - S_t` with metaplasticity.
- `run_phase10a.py`: Phase 10a world-model pretraining, evaluation, and export.
- `phase10b_visual_codec.py`: visual frame encoder/decoder for compact
  `S(t)` latents.
- `run_phase10b.py`: Phase 10b visual codec and AMF latent prediction benchmark.
- `run_phase10c.py`: wrapper for the N.E.V.E.R. AMF orchestrator.
- `phase11a_moving_mnist.py`: real Moving MNIST loader, object tracker,
  compact visual codec, AMF residual world model and baselines.
- `run_phase11a.py`: Fase 11A Moving MNIST benchmark runner.
- `phase11a_slot_ranker_probe.py`: object/slot-level causal multi-hypothesis
  ranker probe for Moving MNIST; includes optional pairwise ranker probe behind
  `--enable-pairwise-ranker`.
- `phase11a_patch_decoder_probe.py`: experimental patch-level decoder probe
  conditioned on causal visual hypotheses; tested as a negative result for pure
  learned masks.
- `phase11a_cell_router_probe.py`: experimental spatial tile router decoder;
  accessed via `phase11a_slot_ranker_probe.py --cell-router`.
- `phase11a_kinematic_encoder_probe.py`: experimental three-hypothesis encoder
  pool probe using simple, motion-token, and kinematic-token latents.
- `NEVER/scripts/phase10c_never_amf_orchestrator.py`: local/API-compatible
  Phase 10c loop.
- `NEVER/src/engine/amf_world_model.cuh`: CUDA-facing AMF latent runtime.
- `NEVER/src/engine/amf_vector_decoder.cuh`: latent-to-frame vector decoder.
- `NEVER/src/engine/action_orchestrator.hpp`: prompt-to-action fallback.

## Fase 7

```powershell
python run_phase7.py --seeds 3 --datasets iris wine wdbc ionosphere sonar spambase madelon
```

The final Fase 7 run includes a `MLPClassifier(1024, 1024)` expert/baseline,
which is above one million parameters on these datasets. In the verified run,
AMF7 wins 4/7 datasets, ties 3/7, and has no losses against the best classical
family selected per dataset.

## Fase 8

```powershell
python run_phase8.py
```

Fase 8 adds output capacity without an LLM and without a dense decoder. The
verified architecture is:

```text
Input -> morphogenic input field -> active latent state -> morphogenic decoder -> output
```

The best decoder is `resonant_morphogenic_decoder`, which chooses among local
response cells, transition cells, and frame-slot cells using a non-dense
resonance score.

## Fase 8.5

```powershell
python run_phase8_5.py --run-pythia
```

Fase 8.5 compares response generation directly: same prompts, same scorer,
speed, memory, and GPU requirement. The corrected run treats Pythia-70M fairly
as a completion model: it tries `Usuario/Respuesta`, `Human/Assistant`,
`Pregunta/Respuesta`, and few-shot local templates, then reports an optimistic
best-template upper bound for Pythia. In the verified local run, AMF8 reached
`talk_score=0.9047` in `0.2455s / 90 prompts`, while prompted Pythia-70M reached
`talk_score=0.1893` in `40.2022s / 90 prompts` on CPU. `talk_score` is a local
scorer for this controlled suite; it favors short, complete, keyword-aligned
responses and is not a universal language-quality metric.

## Fase 9

```powershell
python run_phase9.py
```

Fase 9 scales AMF8 as a morphogenic assistant rather than a dense LLM. It grows
the corpus from 90 to 10000 examples, separates memories into conversation,
architecture, research, code, structured, and safety domains, and evaluates
normal answers, steps, tables, JSON, pseudocode, experiment plans, diagnostics,
multi-intent prompts, baselines, latency, memory, and online learning. In the
verified run, the 10000-example assistant reached `service_score=0.9132`,
`talk_score=0.8720`, `domain_accuracy=1.0000`, `format_success=1.0000`,
`composition_success=1.0000`, `0.0000` repetition, `5.4907ms` per prompt, and
`28.4242 MB`.

## Fase 10a

```powershell
python run_phase10a.py
```

Fase 10a trains a synthetic AMF world model in pure Python/NumPy. A toy
gravity/bounce simulator generates `(S_t, action, S_t+1)` pairs, and AMF stores
local dynamics deltas rather than full next states. The final verified run used
`53,550` train transitions and exported `data/phase10a_warm_amf.npz`. AMF
reached `one_step_mse=0.017430` and `rollout_mse=0.287559`, beating the best
baseline (`ridge_linear_dynamics`, `one_step_mse=0.047347`,
`rollout_mse=0.348361`). Metaplasticity is active: existing cells explain
without growth, possible noise is buffered, repeated novelty is confirmed,
low-use cells are pruned, similar cells are fused, and identity memory is
frozen.

## Fase 10b

```powershell
python run_phase10b.py
```

Fase 10b adds the visual encoder/decoder around the warmed Phase 10a AMF. Frames
are rendered as visual/vectorial pixel tensors, encoded to an 8D latent
`S(t)`, predicted by AMF as `S(t+1)`, then decoded back to a physically
consistent frame. In the verified 256 px run, a frame is `1,048,576` float32
bytes while the latent is `32` bytes, a `32768x` compression. Estimated raw
pixel AMF memory would be about `18,000 MB`; the actual latent AMF arrays remain
`0.686646 MB`. The codec passed permanence, separability, and continuity checks,
with `predicted_frame_mse=0.000158` at 256 px.

## Fase 10c

```powershell
python run_phase10c.py --offline --prompt "Personaje saltando, camara rotando 45 grados" --steps 36 --resolution 128
```

Fase 10c connects the warmed AMF world model into `NEVER`. The old active SNN
branch is retired from `NEVER/src/main.cu` and `NEVER/src/inference_core.cu`.
The new loop is: prompt -> action vector -> visual encoder -> latent `S(t)` ->
AMF world model -> real frame feedback -> metaplasticity -> decoder -> rendered
frame. The verified local run used `32` latent bytes for a `262144` byte frame
(`8192x` compression), kept AMF at `0.686646 MB`, and executed `36` online
feedback events. Runtime metaplasticity produced
`metaplasticity_adapted_cell=14`, `buffered_possible_noise=19`,
`explained_by_existing_cell=5`, and `created_confirmed_novelty=1`. The online
probe rose from `0.25` to `1.00`, and session MSE fell from `0.003052` to
`0.002328`. The native CUDA entrypoint was also compiled with CUDA `13.3.33`
and verified at `NEVER/build_phase10c/Release/never.exe`, including per-frame
online updates and a native probe `score=0.25->1`.

## Fase 11A

```powershell
python run_phase11a.py --train-sequences 650 --test-sequences 40
```

Fase 11A uses the downloaded real Moving MNIST file at
`data/MovingMNIST/mnist_test_seq.npy` with verified shape
`(20, 10000, 64, 64)`. AMF never sees raw pixels: the encoder tracks two digit
objects into a compact `26`-float latent (`104` bytes), while identity layers
stay outside the AMF. The real frame is `16384` bytes, a `157.5x` compression.
The current verified run uses a visual-shift encoder, appearance-aware tracker,
metaplastic AMF cells, and a conservative global collision module
(`collision_box=0.317`, predictive residual scale `0.0`). The decoder now also
uses a causal identity bank of `12000` real train crops, inspired by modern
video tokenizers that keep compact latents while conditioning the decoder on
reference/context frames. `AMF_full` reached
one-step IoU `0.8867`, 480-step stability `1.0000`, identity drift `0.000000`,
and passed metaplasticity novelty/reinforcement probing. In ground-truth
rollout it beat the classical baselines at h5/h10/h17: h10 IoU `0.2161` vs
ridge `0.1293`, and h17 IoU `0.1750` vs ridge `0.1342`. The extreme visual
targets rollout-10 `>0.75` and rollout-17 `>0.60` remain open; the standard
downloaded dataset has only 20 frames, so 30/60/120/240/480 are reported as
stability rollouts rather than ground-truth IoU. The report also includes a
harder causal path where `S(t)` uses velocity from `t-1 -> t`, not `t -> t+1`;
there AMF_full reaches causal h10/h17 `0.1976 / 0.1715`, still above causal
ridge `0.1111 / 0.0900`, but causal one-step remains open at `0.5805`. A
motion-token encoder inspired by TAPIR/CoTracker raises AMF_full causal one-step
to `0.6191`, although its h10/h17 trade off down to `0.1902 / 0.1543`, so the
next encoder needs multi-hypothesis confidence rather than one collapsed
velocity. A separate multi-hypothesis probe in
`phase11a_multi_hypothesis_probe.py` combines causal-simple and motion-token
encoders; in `220/20` it reached dual_AMF h1/h10/h17
`0.6201 / 0.2223 / 0.1909` versus dual_ridge h10/h17
`0.1161 / 0.1046`. The research translation is recorded in
`docs/FASE11A_ENCODER_DECODER_RESEARCH.md`. A confidence-selector probe in
`phase11a_confidence_selector_probe.py` improved h10 slightly (`0.2223` max
fusion to `0.2238` selected) but did not dominate all horizons, so it is kept as
research evidence in `results/FASE11A_CONFIDENCE_SELECTOR.md`. The stronger
`phase11a_iou_ranker_probe.py` trains a direct IoU ranker over simple/token and
blend/max candidates; in `220/20` its reg-ranker reached h1/h10/h17
`0.6282 / 0.2396 / 0.2101`, with oracle `0.6735 / 0.2620 / 0.2424`, documented
in `results/FASE11A_IOU_RANKER.md`. In the larger `650/40` run, the same ranker
reached class-ranker h1 `0.6430` and reg-ranker h10/h17 `0.2326 / 0.1921`,
improving over causal simple `0.1976 / 0.1715` and max-fusion
`0.2231 / 0.1829`; oracle h10/h17 is still `0.2661 / 0.2229`. The newer
`phase11a_slot_ranker_probe.py` moves the decision from whole-frame to
object/slot level; in `650/40`, the frame-marginal slot-reg reached h10/h17
`0.2634 / 0.2199` with train-calibrated selector choice after adding `wide_*`,
`very_wide_*`, `shift_*`, and hybrid score-routing decoder candidates.
Test-best learned h10/h17 is `0.2775 / 0.2218`, and slot-frame-oracle is
`0.3444 / 0.3012`, documented in
`results/FASE11A_SLOT_RANKER.md`. A pairwise token-routing ranker is available
behind `--enable-pairwise-ranker`, but the scaled `650/40` run was negative
(`h10/h17 0.2402 / 0.1929`), so it is not the default path. The patch-level
decoder probe was implemented and tested in `220/20`, but its pure patch output
underperformed the slot ranker (`h10/h17 0.1864 / 0.1639`), so it is kept as
negative evidence rather than integrated. An optional real-data neural
completion probe is available in
`phase11a_neural_decoder_probe.py`; it improved AMF h10/h17 in a 220/20 probe
but is not integrated into the main benchmark because it still hurts h1 and the
actual-dynamics decoder ceiling.
The cell/tile router decoder probe found a high tile oracle in `650/40`
(`h10/h17 0.3743 / 0.3225`), but the learned spatial router underperformed the
hybrid slot ranker (`0.2383 / 0.2075`), so it is also kept as negative evidence.

## Reproducibility notes

The main paper comparison should use 10 or 20 seeds. Attack/stress seeds can be
lower because those suites retrain selected models under deliberately expensive
perturbations. Reports explicitly record both values; raise `--attack-seeds`
when you want a stronger failure-search run.
