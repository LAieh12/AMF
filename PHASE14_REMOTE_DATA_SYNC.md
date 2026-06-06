# Phase 14 remote data sync

Use this on the remote GPU machine before the long Phase 14 run.

## 1. Clone and install

```bash
git clone https://github.com/LAieh12/AMF.git
cd AMF
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Download only allowed data

This downloads:

- full `physics/**/*.tar`
- full `captions/**/*.tar`

It does not download RGB, depth, segmentation PNG, USDA scene files, or camera metadata.

```bash
python phase14_data_sync.py --download --local-dir data/physicalai_physics_captions_full
```

Expected remote size checked by the script:

```text
physics: ~91.59 GB
captions: ~1.4 GB class metadata/context payload
```

## 3. If files were uploaded manually

Place them under:

```text
data/physicalai_physics_captions_full/physics/<scene>/*.tar
data/physicalai_physics_captions_full/captions/<scene>/*.tar
```

Then regenerate manifest/splits/config without downloading:

```bash
python phase14_data_sync.py --local-dir data/physicalai_physics_captions_full
```

## 4. Verify outputs

The sync creates or updates:

```text
results/phase13c_dataset_manifest.json
results/phase13c_splits.json
results/phase14_data_sync_report.json
configs/phase14_world_model_train.yaml
```

Check that `forbidden_asset_check.passed` is `true` in:

```text
results/phase14_data_sync_report.json
```

Optional quick smoke, useful before the full manifest scan:

```bash
python phase14_data_sync.py --local-dir data/physicalai_physics_captions_full --scenes billiards --limit-shards 1 --skip-remote-summary --results-dir results/smoke_data_sync --config results/smoke_data_sync/phase14_world_model_train.yaml
```

## 5. Run Phase 14

```bash
python run_phase14.py --config configs/phase14_world_model_train.yaml
```

Resume after interruption:

```bash
python run_phase14.py --config configs/phase14_world_model_train.yaml --resume
```

The architecture/protocol remains frozen. This sync only changes data paths, manifest, splits, and config shard lists.
