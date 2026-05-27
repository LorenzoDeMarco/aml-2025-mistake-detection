# AML/DAAI 2025 — Mistake Detection

**Project setup**

**1. Reproduce the the V1 and V2 baselines：Propose a new baseline:LSTM**

**Evaluation: Reproduce the the V1 baselines**

```bash
python -m core.evaluate  --variant MLP --backbone omnivore  --modality video --ckpt  data/checkpoints/omnivore/MLP/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6
```

**Generate checkpoint on LSTM:**

use this command to generate step_data_split_combined.json

```bash
python core/generate_recordings_combined_splits.py 
```

if you want have error type analysis, use CaptainCookStepDataset.py to train the checkpoint

```bash
python train_er.py --backbone omnivore --variant LSTM  --split step
```

**Evaluation:**

```bash
python -m core.evaluate  --variant LSTM --backbone omnivore  --modality video --ckpt data/checkpoints/omnivore/LSTM/error_recognition_step_omnivore_LSTM_video_epoch_1.pt --split step --threshold 0.6
```


| Backbones | Baseline | Split | threshold |
| --------- | -------- | ----- | --------- |
| Omnivore  | V3: LSTM | step  | 0.6       |


test Sub Step Level Metrics:


| Metric    | Value               |
| --------- | ------------------- |
| Accuracy  | 0.6806857628639573  |
| Precision | 0.45546774882528834 |
| Recall    | 0.7195883246161633  |
| F1        | 0.5578444836832124  |
| AUC       | 0.7649622844068281  |
| PR-AUC    | 0.4062              |


test Step Level Metrics:


| Metric    | Value              |
| --------- | ------------------ |
| Accuracy  | 0.7894736842105263 |
| Precision | 0.6816143497757847 |
| Recall    | 0.6104417670682731 |
| F1        | 0.6440677966101694 |
| AUC       | 0.8272433998288234 |
| PR-AUC    | 0.5376             |


2.Four-stage extension pipeline: **step localization → step embeddings → task-graph matching → GNN verification**, using **PerceptionEncoder (PE)** 1s segment features.

```
CaptainCook pipeline
  → substep1_1 (ActionFormer JSON)
  → substep1_2 (train ActionFormer)
  → substep1_3 (infer boundaries → eval_results.pkl)
  → substep1_4 (step-level PE embeddings → .npz)
  → substep2 (Transformer baseline)
  → substep3 (Hungarian graph matching → .pt)
  → substep4 (GNN classifier)
```

**Video labels** (Substep 2 & 4): `1` = correct execution, `0` = incorrect (any step with `has_error` in `recordings.json`).

---

## Setup

```bash
python core/extractor/PerceptionEncoder_feature_extractor.py 
```

Place PE 1s features under `data/features/perception_encoder/` (extract with `core/extractor/PerceptionEncoder_feature_extractor.py`).

CaptainCook data: `captaincook/` (annotations, task graphs, splits).

---

## Pipeline Scripts

### `substep1_1_convert_to_action_former_json.py`

Convert CaptainCook step annotations to ActionFormer JSON.


| Item       | Value                                                               |
| ---------- | ------------------------------------------------------------------- |
| **Input**  | `captaincook/annotation_json/`, `captaincook/data_splits/`          |
| **Output** | `data/substep1_1_actionformer_annotations/combined/recordings.json` |
| **Args**   | None (runs all splits/categories)                                   |


```bash
python -m substep1_1_convert_to_action_former_json
```

---

### `substep1_2_train_checkpoint.py`

Train ActionFormer for temporal step localization.


| Param    | Required | Description                                       |
| -------- | -------- | ------------------------------------------------- |
| `config` | yes      | YAML path, e.g. `configs/perception_encoder.yaml` |


```bash
python substep1_2_train_checkpoint.py ./configs/perception_encoder.yaml
```

---

### `substep1_3_step_level_boundaries.py`

Run ActionFormer inference; save predicted step boundaries.


| Param          | Required | Description                                  |
| -------------- | -------- | -------------------------------------------- |
| `config`       | yes      | Same YAML as training                        |
| `ckpt`         | yes      | Checkpoint file                              |
| `--saveonly`   | no       | Skip mAP eval; write `eval_results.pkl` only |
| `-epoch`       | no       | Epoch to load (-1 = latest)                  |
| `-t`, `--topk` | no       | Max segments per video                       |
| `--score_thr`  | no       | Min detection score                          |


For all videos: set `val_split: ['training', 'validation', 'test']` in the config.

```bash
python substep1_3_step_level_boundaries.py \
  ./configs/perception_encoder.yaml \
  ./data/substep1_2_train_checkpoint/<run>/best_model.pth.tar \
  --saveonly
```

**Output**: `eval_results.pkl` (`video-id`, `t-start`, `t-end`, `label`, `score`).

---

### `substep1_4_step_localization.py`

Pool PE features inside predicted step intervals → per-video step embeddings.


| Param            | Required | Default | Description                        |
| ---------------- | -------- | ------- | ---------------------------------- |
| `--eval_pkl`     | yes      | —       | `eval_results.pkl` from substep1_3 |
| `--features_dir` | yes      | —       | PE `.npz` directory (1s segments)  |
| `--out_dir`      | yes      | —       | Output directory                   |
| `--segment_sec`  | no       | 1.0     | Seconds per feature segment        |
| `--score_thr`    | no       | 0.01    | Min prediction score               |


Applies temporal NMS + per-label cap (max 4) + recipe-level step budget.

```bash
python substep1_4_step_localization.py \
  --eval_pkl ./data/substep1_2_train_checkpoint/<run>/eval_results.pkl \
  --features_dir ./data/features/perception_encoder \
  --out_dir ./data/substep1_4_step_localization \
  --score_thr 0.05
```

**Output**: `*.npz` per video ('video_id', 'segments', 'labels', 'scores', 'embeddings', 'feature_counts', 'descriptions', 'feature_file', 'step_features', 'step_starts', 'step_ends', 'step_labels').

---

### `substep2_transformer_baseline.py`

**Substep 2 baseline**: Transformer on step-embedding sequences; binary task verification.


| Param               | Required | Default                                                                 | Description           |
| ------------------- | -------- | ----------------------------------------------------------------------- | --------------------- |
| `--substep1_dir`    | no       | `./data/substep1_4_step_localization`                                   | Step `.npz` directory |
| `--recordings_json` | no       | `./data/`substep1_1_actionformer_annotations/combined`/recordings.json` | Video labels          |
| `--output_dir`      | no       | `./data/substep2_transformer_baseline`                                  | Results directory     |
| `--epochs`          | no       | 15                                                                      | Epochs per LOO fold   |
| `--batch_size`      | no       | 64                                                                      | Batch size            |
| `--lr`              | no       | 1e-4                                                                    | Learning rate         |


**Eval**: leave-one-**recipe**-out (train on k−1 recipes, test on k-th).

```bash
python substep2_transformer_baseline.py \
  --substep1_dir ./data/substep1_4_step_localization \
  --recordings_json ./data/substep1_1_actionformer_annotations/combined/recordings.json \
  --output_dir ./data/substep2_transformer_baseline
```

**Output**: `substep2_leave_one_out_results.json`.

### `substep2_transformer_official_split.py`

Same Transformer model as above, but on **official train/val/test** (same protocol as Substep 4 — comparable test set).

```bash
python substep2_transformer_official_split.py  --substep1_dir ./data/substep1_4_step_localization  --recordings_json ./data/substep1_1_actionformer_annotations/combined/recordings.json  --output_dir ./data/substep2_transformer_official
```

**Output**: `best.pt`, `results.json` (`test_metrics` for comparison with Substep 4).

---

### `substep3_taskgraph_match.py`

**Substep 3**: Encode task-graph node texts (PE-Core), Hungarian-match visual steps to graph nodes.


| Param              | Required | Default  | Description                             |
| ------------------ | -------- | -------- | --------------------------------------- |
| `--substep1_dir`   | yes      | —        | Step `.npz` from substep1_4             |
| `--task_graph_dir` | yes      | —        | CaptainCook task graph JSONs            |
| `--avg_csv`        | yes      | —        | Recipe id → name CSV                    |
| `--out_dir`        | yes      | —        | Output `.pt` directory                  |
| `--device`         | no       | cuda/cpu | Device                                  |
| `--normalize`      | no       | off      | L2-normalize embeddings before matching |


```bash
python substep3_taskgraph_match.py \
  --substep1_dir ./data/substep1_4_step_localization \
  --task_graph_dir ./captaincook/task_graphs \
  --avg_csv ./captaincook/metadata/average_segment_length.csv \
  --out_dir ./data/substep3_taskgraph_match \
  --device cuda --normalize
```

**Output**: `{video_id}.pt` (`x_text`, `step_x`, `edge_index`, `node_to_step`, …).

---

### `substep4_gnn.py`

**Substep 4**: DAGNN  graph classifier — correct vs incorrect execution.


| Param               | Required | Default    | Description                            |
| ------------------- | -------- | ---------- | -------------------------------------- |
| `--graph_pt_dir`    | yes      | —          | Substep3 `.pt` files                   |
| `--recordings_json` | yes      | —          | Labels + official splits               |
| `--output_dir`      | yes      | —          | Checkpoints & results                  |
| `--split_mode`      | no       | `official` | `official` (train/val/test) or `kfold` |
| `--gnn_layer`       | no       | `dagnn`    | `dagnn` or `digraph`                   |


```bash
python substep4_gnn.py \
  --graph_pt_dir ./data/substep3_taskgraph_match \
  --recordings_json ./data/substep1_1_actionformer_annotations/combined/recordings.json \
  --output_dir ./data/substep4_gnn \
  --device cuda --gnn_layer dagnn --split_mode official
```

**Output**: `best.pt`, `results.json` (report `test_metrics` as final score).

---

## Results

### Substep 2 — Transformer baseline (leave-one-recipe-out, 382 videos)


| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.5942 |
| Precision | 0.5631 |
| Recall    | 0.3452 |
| F1        | 0.4280 |
| AUC       | 0.5957 |
| PR-AUC    | 0.5239 |


Confusion (pooled over all LOO folds): tp=58, tn=169, fp=45, fn=110.

### Substep 2 — Transformer baseline (official **test** split, 108 videos)

Same model as above; trained/evaluated with CaptainCook **train/val/test** (`substep2_transformer_official_split.py`). **Use this row to compare with Substep 4.**


| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.6481 |
| Precision | 0.5849 |
| Recall    | 0.6596 |
| F1        | 0.6200 |
| AUC       | 0.6418 |
| PR-AUC    | 0.5514 |


Confusion: tp=31, tn=39, fp=22, fn=16. Label: 1=correct, 0=incorrect.

Validation (early stopping ): accuracy=0.6452, F1=0.5769, AUC=0.7126.

### Substep 4 — DAGNN (official CaptainCook **test** split, 108 videos)


| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.5185 |
| Precision | 0.4658 |
| Recall    | 0.7234 |
| F1        | 0.5667 |
| AUC       | 0.5795 |
| PR-AUC    | 0.5078 |


Confusion: tp=34, tn=22, fp=39, fn=13. Label: 1=correct, 0=incorrect.

Validation (early stopping): accuracy=0.6452, F1=0.6071, AUC=0.6880.

### Official test split comparison (Substep 2 vs Substep 4)


| Method                  | Accuracy | F1     | AUC    |
| ----------------------- | -------- | ------ | ------ |
| Transformer (Substep 2) | 0.6481   | 0.6200 | 0.6418 |
| DAGNN (Substep 4)       | 0.5185   | 0.5667 | 0.5795 |


Both use the same 108-video CaptainCook test split.

---

## Key Paths


| Path                                        | Content                          |
| ------------------------------------------- | -------------------------------- |
| `captaincook/`                              | Annotations, task graphs, splits |
| `data/features/perception_encoder/`         | PE 1s segment features           |
| `data/substep1_1_actionformer_annotations/` | ActionFormer JSON                |
| `data/substep1_4_step_localization/`        | Step embeddings (`.npz`)         |
| `data/substep3_taskgraph_match/`            | Matched graphs (`.pt`)           |
| `configs/perception_encoder.yaml`           | ActionFormer + PE config         |


---

## References

- [CaptainCook4D — error_recognition](https://github.com/CaptainCook4D/error_recognition)
- [CaptainCook4D — feature_extractors](https://github.com/CaptainCook4D/feature_extractors)
- [CaptainCook4D — multi_step_localization](https://github.com/CaptainCook4D/multi_step_localization)
- Course repo: [aml-2025-mistake-detection](https://github.com/sapeirone/aml-2025-mistake-detection)

