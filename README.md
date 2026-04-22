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

| Split | Model | F1 | AUC |
|-------|-------|----|-----|
| Step | MLP (Omnivore) | 24.26 | 75.74 |
| Recordings | MLP (Omnivore) | 55.42 | 63.03 |
| Step | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |

**NOTE**: Use the thresholds indicated in the official README.md of project (0.6 for step and 0.4 for recordings steps).

## Acknowledgements

This project builds on many repositories from the CaptainCook4D release. Please refer to the original codebases for more details.

**Error Recognition**: https://github.com/CaptainCook4D/error_recognition

**Features Extraction**: https://github.com/CaptainCook4D/feature_extractors


# Multi-Step Procedural Action Localization and Error Detection

A two-stage pipeline for analyzing complex multi-step procedures using egocentric video data. The system combines **Temporal Action Localization (TAL)** with **Sequential Reasoning** to detect both the occurrence of actions and procedural errors.

---

## Pipeline Overview

The pipeline is structured into two main sequential phases:

1. **Step 1 — Action Localization:** ActionFormer detects individual action segments, boundaries, and confidence scores within the video stream.
2. **Step 2 — Error Detection:** Detected steps are aggregated into a chronological sequence and classified as *Correct* or *Error* using a Transformer-based architecture.

The pipeline leverages **EgoVLP** features for robust visual representation and semantic alignment.

---

## Execution Workflow

### 1. Project Initialization

Prepare the directory structure for model checkpoints and evaluation results:

```bash
mkdir -p ./ckpt/ego4d/
```

### 2. Training — Substep 1

Train the ActionFormer model using the 5-fold cross-validation configuration:

```bash
bash run_actionformer.sh
```

The script iterates over each fold, trains a separate model, and saves checkpoints under `./ckpt/ego4d/egovlp_recordings_egovlp_fold{N}/`.

### 3. Inference & Evaluation

Generate raw prediction files (`.pkl`) for each fold to assess localization performance:

```bash
bash eval_actionformer.sh
```

### 4. Data Consolidation

Consolidate distributed fold results into a single comprehensive CSV file for sequential analysis:

```bash
python extract_predictions.py
```

Output: `dataset_substep2_predictions.csv`

### 5. Step-Level Embedding Generation

Transform localized segments into unique step-level embeddings:

```bash
python compute_step_embeddings.py
```

Output: `step_embeddings_dataset.npz`

---

## Technical Strategy

### Late Filtering & Overlap Retention

Unlike standard localization pipelines that apply aggressive Non-Maximum Suppression (NMS) at the first stage, this project adopts a **Late Filtering** strategy.

Instead of discarding overlapping segments, they are preserved to provide the Substep 2 Transformer with a dense temporal context. This allows the model to act as a soft arbitrator, resolving temporal ambiguities through its self-attention mechanism rather than relying on hard-coded geometric thresholds.

### Refinement & Filtering Criteria

During embedding generation, the following filters are applied to ensure a high signal-to-noise ratio:

- **Confidence Thresholding** (τ ≥ 0.05): Predictions below this score are discarded to eliminate low-probability background noise.
- **Top-K Constraining** (K = 100): Only the top 100 highest-scoring predictions per video are retained to optimize memory and sequence length.
- **Chronological Sorting**: Retained segments are re-ordered by `start_time` to reconstruct the logical flow of the procedure.

### Feature Aggregation — Temporal Average Pooling

For each detected step, a unique step-level embedding is computed by extracting all EgoVLP feature vectors (sampled at 1.876 Hz) within the predicted boundaries and applying **Temporal Average Pooling**. Each step is represented by a fixed-size **768-dimensional vector** summarizing the visual semantics of the action.

---

## Configuration

Key parameters in `configs/captaincook_egovlp.yaml`:

| Parameter | Value | Description |
|---|---|---|
| `num_classes` | 353 | Total action categories (1-indexed) |
| `input_dim` | 768 | EgoVLP feature dimensionality |
| `feat_stride` | 16 | Temporal stride (frames) |
| `num_frames` | 16 | Frames per feature vector |
| `default_fps` | 1.876 | Effective feature sampling rate |
| `max_seq_len` | 4096 | Maximum sequence length |
| `backbone_arch` | [2, 2, 5] | ConvTransformer architecture |

Regression ranges are calibrated on the CaptainCook4D training set distribution (action durations from ~13s to ~1068s, converted to feature grid units at stride 16).

---

## Data Structure

```
.
├── configs/
│   └── captaincook_egovlp.yaml
├── data/
│   └── egovlp_features/          # .npz feature files
├── captaincook_actionformer_annotations/
│   └── combined/
│       ├── recordings.json
│       └── recordings_fold{1..5}.json
├── ckpt/
│   └── ego4d/
│       └── egovlp_recordings_egovlp_fold{N}/
├── dataset_substep2_predictions.csv
└── step_embeddings_dataset.npz
```

---

## Dataset Statistics — CaptainCook4D

| Metric | Value |
|---|---|
| Total videos | 384 |
| Total action steps | 5,700 |
| Max video length | 2,470s |
| Avg video length | 886s |
| Avg action duration | 50.5s |
| Action duration P20/P40/P60/P80 | 13s / 26s / 42s / 74s |

---

## Target Task — Error Detection

The `step_embeddings_dataset.npz` serves as training data for a **Sequence-to-Label Transformer** that analyzes the logical consistency of detected steps and outputs a final classification:

- **Class 0** — Correct Procedure
- **Class 1** — Procedural Error
