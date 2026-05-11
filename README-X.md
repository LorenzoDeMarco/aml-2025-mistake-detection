# AML/DAAI 2025 - Mistake Detection Project

## Environment Setup

First of all, create a python environment with 

```
python -m venv .venv
pip install -r requirements.txt
```

Then, download the pre-extracted features for 1s segments and put them in the `data/features` directory.

## Step 1: Baselines reproduction

Download the official best checkpoints from [here](https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3) (`error_recognition_best` directory) and place them in the `checkpoints`. Then run the evaluation for the error recognition task.

**Example command**:

```
python -m core.evaluate --variant MLP --backbone omnivore --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6
```

You should be able to reproduce results close to those reported in the paper (Table 2):


| Split      | Model              | F1    | AUC   |
| ---------- | ------------------ | ----- | ----- |
| Step       | MLP (Omnivore)     | 24.26 | 75.74 |
| Recordings | MLP (Omnivore)     | 55.42 | 63.03 |
| Step       | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |


**NOTE**: Use the thresholds indicated in the official README.md of project (0.6 for step and 0.4 for recordings steps).

## Acknowledgements

This project builds on many repositories from the CaptainCook4D release. Please refer to the original codebases for more details.

**Error Recognition**: [https://github.com/CaptainCook4D/error_recognition](https://github.com/CaptainCook4D/error_recognition)

**Features Extraction**: [https://github.com/CaptainCook4D/feature_extractors](https://github.com/CaptainCook4D/feature_extractors)

# Project setup

## 2. Reproduce the the V1 and V2 baselines

### a. analyze the performance of the model on different error types

### b. Propose a new baseline:LSTM

### Evaluation: Reproduce the the V1 baselines

```bash
python -m core.evaluate  --variant MLP --backbone omnivore  --modality video --ckpt  data/checkpoints/omnivore/MLP/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6
```

### Generate checkpoint on LSTM:

if you want have error type analysis, use CaptainCookStepDataset.py to train the checkpoint

```bash
python train_er.py --backbone omnivore --variant LSTM  --split step 
```

### Evaluation:

```bash
python -m core.evaluate  --variant LSTM --backbone omnivore  --modality video --ckpt data/checkpoints/omnivore/LSTM/error_recognition_recordings_omnivore_LSTM_video_epoch_1.pt --split step --threshold 0.6
```

## 3. Extend the baselines to a new features extraction backbone

### Extract features on PerceptionEncoder backbone：

```bash
python core/extractor/PerceptionEncoder_feature_extractor.py 
```

### Generate checkpoint:

```bash
python train_er.py --backbone perception_encoder --variant MLP  --split recordings 
```

### Evaluation:

```bash
python -m core.evaluate --variant MLP --backbone perception_encoder --modality video --ckpt data/checkpoints/perception_encoder/MLP/error_recognition_recordings_perception_encoder_MLP_video_epoch_5.pt  --split recordings --threshold 0.4
```

## Outputs

\results\error_recognition\combined_results\step_True_substep_True_threshold_xxx.csv

# Extension: video-mistake-detection

## Overview

4-stage: step localization, step embedding, task-graph matching, and GNN-based verification.

## Repository Structure

- `substep1_1_convert_to_action_former_json.py`: Convert CaptainCook annotations to ActionFormer JSON
- `substep1_2_train_checkpoint.py`: Train localization model (feature folder set in YAML; use `configs/error_perception_encoder.yaml` for PE `.npz`)
- `substep1_3_step_level_boundaries.py`: Produce serialization file (outputs `eval_results.pkl`),which is model-predicted boundaries of step segments (start/end)
- `substep1_4_step_localization.py`: Generate step-level embedding features with the (start, end) boundaries of the step on PerceptionEncoder feature (outputs`.npz`)
- `substep2_task_verification.py`: Use step embeddings features to classify a recipe execution is correct or incorrect
- `substep3_taskgraph_match.py`: encodes task-graph node texts with PE, matches video step embeddings to graph nodes using one-to-one Hungarian matching, and saves a per-video graph package or Substep4 to learn (outputs `.pt`)
- `substep4_gnn.py`: GNN training and evaluation
- `configs/`: ActionFormer configs
- `libs/`: Modeling, datasets, utilities
- `captaincook/`: the data from github project: aml-2025-mistake-detection 
- `data/substep1_1_actionformer_annotations/`: the output of substep1_1
- `data/perception_encoder/actionformer_checkpoints/`: default output of substep1_2 / substep1_3 when using `error_perception_encoder.yaml`
- `data/features/perception_encoder/`: 1 Hz segment `.npz` from `PerceptionEncoder_feature_extractor.py` (step localization input)
- `data/pe_core/`:  the output of substep1_4 to substep4

## Data and code resource

- `captaincook/`: is annotations/ from project: aml-2025-mistake-detection 
[https://github.com/sapeirone/aml-2025-mistake-detection](https://github.com/sapeirone/aml-2025-mistake-detection)
- `configs/error_omnivore.yaml`: Omnivore features (`./data/omnivore/features`), from multi_step_localization / ActionFormer lineage
- `configs/error_perception_encoder.yaml`: PerceptionEncoder 1s `.npz` under `./data/features/perception_encoder` (`input_dim: 1024` for default PE-Core-B-16)
- `lib/`: is multi_step_localization/actionformer/libs from project:multi_step_localization
- `lib/error.py`: Copied dataloader script error_dataset.py from project:multi_step_localization then modified
- `data/omnivore/annotations/recordings-combined.json`: Copied the result from ./data/substep1_1_actionformer_annotations/combined\recordings-combined.json because this includes the currect and incurrect labels
- `substep1_1_convert_to_action_former_json.py`: Copied convert_to_action_former_json.py from project:multi_step_localization then modified
- `substep1_2_train_checkpoint.py`: Copied train.py from project:multi_step_localization then modified
- `substep1_3_step_level_boundaries.py`: Copied eval.py from project:multi_step_localization then modified
- `substep1_4_step_localization.py`: based on online resource to write
- `substep2_task_verification.py`: based on online resource to write
- `substep3_taskgraph_match.py`:based on online resource to write
- `substep4_gnn.py`:based on online resource to write

## Usage

### Substep1: Step Localization + Step Embeddings

1. Convert annotations:

```bash
python -m substep1_1_convert_to_action_former_json
```

1. Extract segment features (once per video corpus):

```bash
python core/extractor/PerceptionEncoder_feature_extractor.py --video_dir ./data/video --output_dir ./data/features
```

Outputs `./data/features/perception_encoder/*.npz` (1 Hz segments; key `features`).

1. Train localization model on those features:

```bash
python substep1_2_train_checkpoint.py ./configs/error_perception_encoder.yaml
```

1. Inference and save `eval_results.pkl` (pass the **run folder** that contains `epoch_*.pth.tar`, e.g. `error_perception_encoder_<timestamp>/` under `actionformer_checkpoints/`):

```bash
python substep1_3_step_level_boundaries.py ./configs/error_perception_encoder.yaml ./data/perception_encoder/actionformer_checkpoints/error_perception_encoder_<timestamp>/ --saveonly
```

1. Build step-level embeddings (same `.npz` directory as localization):

```bash
python substep1_4_step_localization.py --eval_pkl "./data/perception_encoder/actionformer_checkpoints/error_perception_encoder_<timestamp>/eval_results.pkl"
  --features_dir "./data/features/perception_encoder" --out_dir "./data/pe_core/substep1_out" --segment_sec 1.0 --score_thr 0.01 --topk 30
```

### Substep2: Video-Level Verification (Transformer)

```bash
python substep2_task_verification.py --substep1_dir  "./data/pe_core/substep1_out"
  --step_annotations_csv "./captaincook/annotation_csv/step_annotations.csv" 
  --out_dir "./data/pe_core/substep2_out" --epochs 5  --batch_size 8  --max_steps 256
```

### Substep3: Task-Graph Matching

```bash
python substep3_taskgraph_match.py  --substep1_dir ".\data\pe_core\substep1_out"
  --task_graph_dir ".\captaincook\task_graphs"   --avg_csv ".\captaincook\metadata\average_segment_length.csv"
  --out_dir ".\data\pe_core\substep3_out" --device cuda --normalize --sim_threshold 0.0
```

### Substep4: GNN Training/Evaluation

```bash
python substep4_gnn.py  --graph_pt_dir ".\data\pe_core\substep3_out" 
  --recordings_json ".\data\omnivore\annotations\recordings-combined.json"
  --output_dir ".\data\pe_core\substep4_out" --eval_only --resume ".\data\pe_core\substep4_out\best.pt" --device cuda
```

## Outputs

- Substep1:
  - `data/substep1_1_actionformer_annotations/<category>/<split>.json`
  - `epoch_XXX.pth.tar`
  - `eval_results.pkl`
  - `*.npz` per video (segments + embeddings)
- Substep2:
  - `substep2_leave_one_out_results.json`
- Substep3:
  - `*.pt` per video (graph samples)
- Substep4:
  - `best.pt`, `last.pt`

## Results

Add metrics: Accuracy, F1, AUC. PR-AUC .
 {
    'accuracy': 0.5370370370370371, 
    'precision': 0.4827586206896552, 
    'recall': 0.5833333333333334, 
    'f1': 0.5283018867924529, 
    'auc': 0.5888888888888889, 
    'pr_auc': 0.5811502899224296
  }