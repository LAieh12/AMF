# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup for N.E.V.E.R. Architecture.
- Created base directory structure (`src/`, `include/`, `tests/`).
- Authored initial `README.md` and `CHANGELOG.md`.
- Basic `CMakeLists.txt` for CUDA and C++17 configuration.
- Phase 10c AMF integration: `AMFWorldModelRuntime`, vector decoder kernel,
  action orchestrator, local Python loop, frozen identity metadata and AMF
  metaplasticity statuses.
- Online AMF world-model training inside NEVER: every runtime frame feeds back
  through encoder -> real latent -> AMF comparison -> metaplastic update, with
  probe score `0.25 -> 1.00`.

### Changed
- Replaced the active dynamic SNN path with AMF latent world-model orchestration.
- `train_dynamic_snn.py` now forwards to the Phase 10c AMF orchestrator.

### Removed
- Active `SNNEngine` allocation from `main.cu` and `inference_core.cu`.
