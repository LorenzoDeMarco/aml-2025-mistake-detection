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

# Task Verification Baseline: Sequence Modeling via Temporal Transformers

This module implements the **Task Verification** baseline (Substep 2) for fine-grained procedural error detection. Given a sequence of frozen video embeddings extracted from a vision-language foundation model (e.g., EgoVLP), the architecture captures long-range temporal relationships to predict whether a procedural cooking execution contains a mistake ($y=1$) or is executed correctly ($y=0$).

---

## 1. Architectural Evolution and Rationale

The finalized architecture is the result of systematic engineering iterations designed to mitigate severe overfitting shortcuts and optimization stalles dictated by the data-scarce nature of the dataset (384 video samples evaluated under a rigorous Leave-One-Out cross-validation regime).

### 1.1 The "Model Laziness" and Majority Class Collapse Trap
Initial implementations of a standard unregularized Transformer baseline suffered from a structural collapse into the majority class. Because the dataset is imbalanced (with a higher concentration of videos containing errors), deep architectures quickly discovered an inductive shortcut: by bias-shifting their predictions heavily toward the positive class, they achieved an artificially inflated Recall. However, this came at the cost of catastrophic precision degradation and a high false-positive rate. The model became visually "lazy," choosing to systematically guess "Error" rather than learning the subtle temporal kinematics of the task.

### 1.2 The Negative Impact of Data Augmentation on Frozen Features
To enforce regularization, a data augmentation pipeline consisting of online Gaussian noise injection, step dropout, and feature scaling/jittering was introduced inside the dataset module. However, rigorous empirical testing proved that this approach actively harmed performance on frozen feature spaces. 

Foundation visual encoders map human actions into highly structured, hyperspherical latent spaces where fine-grained anomalies (e.g., pouring the wrong ingredient) are separated by subtle angular or distance thresholds. Adding artificial noise directly onto these frozen vectors corrupted their pre-existing geometric semantic relationships. Furthermore, the input features already underwent a **late filtering** stage from the preceding *multi-step localization* module, yielding clean and dense action proposals. Injecting feature-level perturbations disrupted this precise temporal alignment, drowning out the transient cinematic signatures of mistakes within a sparse data regime. Consequently, augmentation functions were explicitly disabled (`#Avoid to use because performs worse`).

### 1.3 Structural Regularization via Capacity Control
Hyperparameter sweeps originally gravitated toward deeper and high-capacity models (e.g., 4 Transformer layers). While deep configurations minimized training loss rapidly, they suffered from overconfidence and memorization during testing. Given the limited sample size, deep networks exploit non-generalizable shortcuts (such as actor-specific cues or kitchen background details). Restricting the network to a lightweight **2-layer Transformer Encoder** acted as a powerful structural regularizer, capping the model's capacity and forcing it to learn universal macro-temporal actions rather than localized noise.

### 1.4 Overcoming Temporal Dilution: Conv1D + [CLS] Token Paradigms
Early iterations utilized global pooling operations to compress the time dimension before classification. However, pooling completely flattens temporal dynamics; since a procedural mistake is an extremely transient anomaly localized within a brief window, averaging features across the entire video dilutes the error signature into the overwhelming mass of correct frames, making the model prone to class collapse.

To solve this, a dual-stage temporal aggregator was designed:
1. **Finer Temporal Downsampling via Conv1D:** Instead of broad pooling windows or loose strides (e.g., stride=4) that aggressively discard visual data, we implemented a 1D Convolutional layer with a fine-grained **`stride=2` and `kernel_size=4`**. This compresses the temporal sequence by exactly half, preserving fine-grained micro-temporal actions while smoothing local frame-to-frame noise.
2. **Dynamic Attention via the `[CLS]` Token:** A learnable classification token (`[CLS]`) is prepended directly to the downsampled sequence. Through self-attention query-key-value interactions, the `[CLS]` token assigns non-uniform attention weights across frames. If a procedural anomaly occurs, the attention matrices isolate that exact temporal frame window, routing its signature directly to the classification head without geometric dilution.

### 1.5 Loss Re-balancing: Shifting from Asymmetric Weights to Label Smoothing & Post-Hoc Calibration
To handle class imbalance, initial models relied on a heavily skewed `pos_weight` factor inside the cross-entropy function. This heavily compressed the model's logit outputs downward, clustering probabilities in a narrow, lower bound zone and rendering the standard $0.5$ decision threshold entirely ineffective.

The optimization was refactored as follows:
* **Symmetric Binary Cross-Entropy (BCE) Loss** was reinstated to preserve a natural probability center around the $0.5$ threshold.
* **Label Smoothing (0.1)** was integrated, softening hard targets from $[0, 1]$ to $[0.1, 0.9]$. This establishes a mathematical *entropy floor* (preventing the loss from dropping below $\approx 0.32$), which bounds the gradient flow, curbs extreme network overconfidence, and prevents logit divergence.
* This smooth, well-behaved probability space provides the ideal foundation for **post-hoc threshold calibration**, enabling post-training calibration on the final aggregated CSV to maximize global accuracy.

---

## 2. Current Model Architecture

The finalized model architecture is structured inside `task_verification/transformer.py` as a pipelined sequence model:

1. **Linear Feature Projection:** Maps the raw 768-dimensional foundation video embeddings to an optimized hidden space (`embed_dim=256`) via a linear layer (`nn.Linear(input_dim, embed_dim)`).
2. **Temporal Downsampling:** A 1D Convolutional layer (`kernel_size=4, stride=2, padding=0`) reduces the temporal dimension by half, cleaning the signal and densifying the action segments.
3. **`[CLS]` Token Prepending:** A learnable parameters tensor (`torch.zeros(1, 1, embed_dim)`) initialized with a standard deviation of $0.02$ is prepended to the feature sequence, expanding the tensor shape to `[B, T_new + 1, embed_dim]`.
4. **Parametric Positional Encoding:** Sinusoidal positional encodings are injected across the sequence. The maximum sequence length is dynamically bounded to `(max_seq_len // 2) + 50` to account for the stride-2 compression safely.
5. **Transformer Encoder Layer with Pre-LN:** A 2-layer Transformer Encoder utilizes Pre-Layer Normalization (`norm_first=True`). Applying normalization *before* the multi-head self-attention and feed-forward sub-layers guarantees deep gradient stability and smooth loss propagation.
6. **MLP Classification Head:** The output corresponding to index 0 (the `[CLS]` token) is extracted and passed through a robust non-linear MLP head: `Linear(256 -> 128) -> ReLU -> Dropout(0.2) -> Linear(128 -> 1)` to generate the final decision logit.

### Mask Compression and Alignment Logic
To ensure that padding regions are not processed during multi-head self-attention, a corresponding boolean mask reduction is performed. The mask is downsampled using a 1D Max Pooling operation (`F.max_pool1d(mask, kernel_size=4, stride=2)`) to mirror the shape changes caused by the Conv1D layer. Any minor shape mismatches stemming from integer division rounding are resolved via dynamic alignment (slicing or zero-padding). Finally, a valid token prefix (`1.0`) is added to ensure that the `[CLS]` token position is never masked during self-attention computation.

---

## 3. Hyperparameter Tuning & Training Strategy

The training routine employs a highly structured, accelerated optimization schedule executed via a single NVIDIA A100 GPU:

* **Optimizer:** `AdamW` with a constant weight decay of `1e-2` to regularize weights from expanding.
* **Learning Rate (LR):** Configured to `2e-4`, representing the optimal empirical sweet spot for fine-tuning frozen representations safely without inducing gradient explosions.
* **Batch Size:** Expanded to `64` to maximize the parallel tensor-core throughput of the hardware while generating stable, highly directional gradient updates.
* **Epochs & Scheduling:** Extended to `35` epochs to fully compensate for the large batch configuration. With 383 training samples per fold, the loader performs 6 optimization steps per epoch, totaling **210 gradient steps** over the full run. The learning rate is controlled by a `CosineAnnealingLR` scheduler, tapering down from `2e-4` to a minimum of `1e-6` at `T_max=35` to smoothly freeze the attention matrices inside the local minimum.

---

## 4. Computational Optimizations

To manage the massive overhead of executing a 384-fold Leave-One-Out loop sequentially, the execution script embeds several hardware-acceleration features:

* **Automatic Mixed Precision (AMP):** All forward steps and loss computations are executed within `torch.cuda.amp.autocast()` using float16 primitives. Backward propagation is handled via `torch.cuda.amp.GradScaler()` to scale losses and eliminate underflow errors.
* **Gradient Clipping:** A strict norm threshold (`max_norm=1.0`) is enforced via `nn.utils.clip_grad_norm_` right before the optimization step, safeguarding the Transformer layers against sudden gradient spikes.
* **High-Throughput Collating:** Batches are constructed dynamically via a custom `dynamic_collate_fn` that automatically pads irregular sequence dimensions, generating a precise matching attention mask on the fly.

---

## 5. Dataset Setup & Execution Guide

### 5.1 Data Placement
Before executing the pipeline, the pre-extracted foundation embeddings file (`step_embeddings_dataset.npz`) and the corresponding JSON ground-truth annotation file must be imported into the repository. Structure the root workspace as follows:

```text
repository_root/
├── step_embeddings_dataset.npz             # Target frozen video embeddings archive
├── annotations/
│   └── annotation_json/
│       └── complete_step_annotations.json    # Procedural step annotations and error labels
└── task_verification/
    ├── __init__.py
    ├── dataset.py
    ├── transformer.py
    ├── train_transformer.py
    └── transformer_sweep.py
```

### 5.2 Execution Commands
To guarantee correct absolute pathway indexing and prevent nested module lookup failures, always invoke the scripts as Python modules from the root directory of the repository.

* **Run the Hyperparameter Sweep (Weights & Biases integration)**
To launch or join an automated hyperparameter optimization sweep across alternative learning rates, layers, or dimensions, run:

```bash
python -m task_verification.transformer_sweep
```

* **Run the Leave-One-Out Main Training Pipeline**
To run the standard Leave-One-Out evaluation over all 384 videos using the optimal hyperparameter layout, run:

```bash
python -m task_verification.train_transformer
```
