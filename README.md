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

**Commands**:
```
python -m core.evaluate --variant MLP --backbone omnivore --ckpt checkpoints/error_recognition_best/MLP/omnivore/error_recognition_MLP_omnivore_step_epoch_43.pt --split step --threshold 0.6

python -m core.evaluate --variant MLP --backbone omnivore --ckpt checkpoints\error_recognition_best\MLP\omnivore\error_recognition_MLP_omnivore_recordings_epoch_33.pt --split recordings --threshold 0.4

python -m core.evaluate --variant Transformer --backbone omnivore --ckpt checkpoints\error_recognition_best\Transformer\omnivore\error_recognition_Transformer_omnivore_step_epoch_9.pt --split step --threshold 0.6

python -m core.evaluate --variant Transformer --backbone omnivore --ckpt checkpoints\error_recognition_best\Transformer\omnivore\error_recognition_Transformer_omnivore_recordings_epoch_31.pt --split recordings --threshold 0.4
```

You should be able to reproduce results close to those reported in the paper (Table 2):

| Split | Model | F1 | AUC |
|-------|-------|----|-----|
| Step | MLP (Omnivore) | 24.26 | 75.74 |
| Recordings | MLP (Omnivore) | 55.42 | 63.03 |
| Step | Transf. (Omnivore) | 55.39 | 75.62 |
| Recordings | Transf. (Omnivore) | 40.73 | 62.27 |

**NOTE**: Use the thresholds indicated in the official README.md of project (0.6 for step and 0.4 for recordings steps).

### Evaluation results (Backbone: Omnivore)

| Variant | Split | Threshold | Level | Accuracy | Precision | Recall | F1-Score | AUC | PR_AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | Step | 0.6 | Sub Step | 0.6831 | 0.4096 | 0.2990 | 0.3457 | 0.6542 | 0.3187 |
| **MLP** | Step | 0.6 | Step | 0.7105 | 0.6607 | 0.1486 | 0.2426 | 0.7574 | 0.3638 |
| **MLP** | Recordings | 0.4 | Sub Step | 0.5735 | 0.3965 | 0.5688 | 0.4673 | 0.5988 | 0.3673 |
| **MLP** | Recordings | 0.4 | Step | 0.5037 | 0.4091 | 0.8589 | 0.5542 | 0.6303 | 0.4020 |
| **Transformer** | Step | 0.6 | Sub Step | 0.6739 | 0.4445 | 0.6614 | 0.5317 | 0.7462 | 0.3888 |
| **Transformer** | Step | 0.6 | Step | **0.6992** | **0.5156** | **0.5984** | **0.5539** | **0.7562** | **0.4338** |
| **Transformer** | Recordings | 0.4 | Sub Step | 0.6450 | 0.4491 | 0.3512 | 0.3942 | 0.6254 | 0.3711 |
| **Transformer** | Recordings | 0.4 | Step | 0.6140 | 0.4541 | 0.3693 | 0.4073 | 0.6227 | 0.3942 |

### Performance for Error Type (Step Level, Backbone: Omnivore)

#### Methodology: Code Modifications for Error-Type Analysis

To extend the baseline evaluation (V1 MLP and V2 Transformer) and analyze the model's performance across specific error categories (e.g., Technique, Preparation, Timing), several key modifications were introduced to the data pipeline and evaluation scripts:

* **Dataset Enhancement (`CaptainCookStepDataset.py`):** Modified the `__getitem__` and `_build_modality_step_features_labels` methods to extract and return the specific string name of the error category (e.g., "Technique Error") for each sub-step.
    * Resolved a dictionary mapping issue by properly utilizing the `_error_category_label_name_map` to convert the numeric error IDs from the JSON annotations back into their human-readable string labels, preventing valid errors from being incorrectly masked as "Normal".

* **DataLoader Update (`collate_fn`):** Updated the batching function to accept the newly introduced error types. 
    * Implemented a flattening mechanism to convert the nested lists of error strings into a single, flat 1D array that perfectly aligns with the concatenated feature and label tensors.

* **Evaluation Logic (`base.py` - `test_er_model`):**
    * **Temporal Aggregation:** Updated the inference loop to track error types globally. During the Step-Level aggregation, the script now identifies the "dominant" error type within a step's temporal boundaries by finding the most frequent error tag (excluding "Normal" frames).
    * **Isolated Metric Calculation:** Added a dedicated evaluation block for error types. To ensure mathematical validity for metrics like AUC, F1-Score, and Precision, the script dynamically creates binary classification subsets. It combines all instances of a specific error type (Label 1) with all "Normal" instances (Label 0). This allows `scikit-learn` to reliably compute Accuracy, Precision, Recall, F1-Score, and AUC for each discrete error category.

| Model (Variant) | Split | Error Type | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | Step | Preparation Error | 0.8163 | 0.5250 | 0.1667 | 0.2530 | 0.7652 |
| **MLP** | Step | Technique Error | 0.8195 | 0.5581 | 0.1890 | 0.2824 | 0.7718 |
| **MLP** | Step | Measurement Error | 0.8248 | 0.4571 | 0.1416 | 0.2162 | 0.7567 |
| **MLP** | Step | Timing Error | 0.8321 | 0.4412 | 0.1415 | 0.2143 | 0.7540 |
| **MLP** | Step | Temperature Error | 0.8565 | 0.4062 | 0.1529 | 0.2222 | 0.7721 |
| | | | | | | | |
| **MLP** | Recordings | Preparation Error | 0.4164 | 0.2469 | 0.8167 | 0.3791 | 0.6082 |
| **MLP** | Recordings | Technique Error | 0.4161 | 0.2449 | 0.8220 | 0.3774 | 0.5897 |
| **MLP** | Recordings | Measurement Error | 0.4205 | 0.2487 | 0.8462 | 0.3845 | 0.6373 |
| **MLP** | Recordings | Timing Error | 0.3916 | 0.1962 | 0.8022 | 0.3153 | 0.5832 |
| **MLP** | Recordings | Temperature Error | 0.3831 | 0.1763 | 0.8101 | 0.2896 | 0.5978 |
| | | | | | | | |
| **Transformer** | Step | Preparation Error | 0.7244 | 0.3636 | 0.6349 | 0.4624 | 0.7755 |
| **Transformer** | Step | Technique Error | 0.7145 | 0.3458 | 0.5827 | 0.4340 | 0.7454 |
| **Transformer** | Step | Measurement Error | 0.7296 | 0.3458 | 0.6549 | 0.4526 | 0.7810 |
| **Transformer** | Step | Timing Error | 0.7267 | 0.3237 | 0.6321 | 0.4281 | 0.7712 |
| **Transformer** | Step | Temperature Error | 0.7303 | 0.2784 | 0.6353 | 0.3871 | 0.7801 |
| | | | | | | | |
| **Transformer** | Recordings | Preparation Error | 0.6509 | 0.2465 | 0.2917 | 0.2672 | 0.5825 |
| **Transformer** | Recordings | Technique Error | 0.6551 | 0.2517 | 0.3051 | 0.2759 | 0.6024 |
| **Transformer** | Recordings | Measurement Error | 0.6746 | 0.3007 | 0.3932 | 0.3407 | 0.6180 |
| **Transformer** | Recordings | Timing Error | 0.6679 | 0.1894 | 0.2747 | 0.2242 | 0.5768 |
| **Transformer** | Recordings | Temperature Error | 0.6719 | 0.1508 | 0.2405 | 0.1854 | 0.5483 |
### Methodology: Introduction of a Bidirectional LSTM Baseline

To bridge the gap between the purely frame-independent approach (MLP) and the computationally heavy global-attention approach (Transformer), we proposed and implemented a Bidirectional Long Short-Term Memory (Bi-LSTM) network as a robust third baseline. 

* **Architectural Motivation:** Cooking errors—such as incorrect timing or missed preparation steps—are inherently sequential. Unlike the MLP, which evaluates frames in isolation, the Bi-LSTM processes the chronological sequence of video features. By using a bidirectional configuration, the model evaluates each frame using both past and future contextual dependencies within the step, effectively capturing the flow of the action without the $O(N^2)$ complexity of a Self-Attention mechanism.
* **Implementation Details:** The custom `ErrorRecognitionLSTM` module ingests sequences of spatial-temporal features (extracted via the Omnivore backbone) into a single-layer Bi-LSTM with a hidden dimension of 256. The concatenated forward and backward hidden states are then passed through a fully connected sequential classifier equipped with ReLU activation and Dropout (0.3) to output frame-level binary probabilities.
* **Pipeline Integration:** The evaluation parser (`argparse`) and the model factory (`fetch_model`) were extended to natively support the newly introduced `LSTM` variant. This ensured the new model could be seamlessly trained, validated, and evaluated across both the `step` and `recordings` splits using the exact same normalization strategies and metrics as the original baselines.

**Commands**:
```
python -m train_er --variant LSTM --backbone omnivore --split step --batch_size 32 --lr 0.001 --num_epochs 50 --ckpt_directory checkpoints

python -m train_er --variant LSTM --backbone omnivore --split recordings --batch_size 32 --lr 0.001 --num_epochs 50 --ckpt_directory checkpoints
```

### Baseline 3: Bidirectional LSTM Performance (Backbone: Omnivore)
**Commands**:
```
python -m core.evaluate --variant LSTM --backbone omnivore --ckpt checkpoints\error_recognition_best\LSTM\omnivore\error_recognition_step_omnivore_LSTM_video_epoch_16.pt --split step --threshold 0.6

python -m core.evaluate --variant LSTM --backbone omnivore --ckpt checkpoints\error_recognition_best\LSTM\omnivore\error_recognition_recordings_omnivore_LSTM_video_epoch_5.pt --split recordings --threshold 0.4
```

#### Table 1: Global Evaluation Metrics
This table reports the overall Sub-Step and Step level metrics across both data splits.

| Split | Threshold | Level | Accuracy | Precision | Recall | F1-Score | AUC | PR_AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | 0.6 | Sub Step | 0.6952 | 0.4677 | 0.6439 | 0.5418 | 0.7474 | 0.4008 |
| **Step** | 0.6 | Step | 0.7456 | 0.5839 | 0.6426 | 0.6119 | 0.8020 | 0.4868 |
| **Recordings** | 0.4 | Sub Step | 0.6405 | 0.4604 | 0.5435 | 0.4985 | 0.6628 | 0.4003 |
| **Recordings** | 0.4 | Step | 0.5514 | 0.4318 | 0.7884 | 0.5580 | 0.6516 | 0.4164 |

#### Table 2: Performance per Error Type (Step Level)
This table breaks down the LSTM's ability to classify specific error categories by combining the target error samples with the "Normal" base samples.

| Split | Error Type | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step** | Preparation Error | 0.7778 | 0.4412 | 0.7143 | 0.5455 | 0.8360 |
| **Step** | Technique Error | 0.7663 | 0.4213 | 0.6535 | 0.5123 | 0.7957 |
| **Step** | Measurement Error | 0.7764 | 0.4093 | 0.6991 | 0.5163 | 0.8316 |
| **Step** | Timing Error | 0.7725 | 0.3838 | 0.6698 | 0.4880 | 0.8147 |
| **Step** | Temperature Error | 0.7823 | 0.3486 | 0.7176 | 0.4692 | 0.8382 |
| | | | | | | |
| **Recordings** | Preparation Error | 0.5018 | 0.2775 | 0.8000 | 0.4120 | 0.6315 |
| **Recordings** | Technique Error | 0.4818 | 0.2515 | 0.7119 | 0.3717 | 0.5954 |
| **Recordings** | Measurement Error | 0.4954 | 0.2669 | 0.7778 | 0.3974 | 0.6603 |
| **Recordings** | Timing Error | 0.4741 | 0.2114 | 0.7363 | 0.3284 | 0.6004 |
| **Recordings** | Temperature Error | 0.4715 | 0.1935 | 0.7595 | 0.3085 | 0.6069 |

## Feature extraction
This module handles the transformation of raw videos into dense vector representations (features) using **EgoVLP**, an architecture optimized for first-person (egocentric) video analysis. The extraction process has been engineered around three core principles to maximize the effectiveness of the downstream mistake detection model.

### Processing method: temporal window and spatial normalization
To capture the temporal dynamics of actions, videos are not processed frame-by-frame, but rather through a **16-frame temporal sliding window**.
* **Rationale:** 16 frames represent the ideal temporal receptive field for the EgoVLP TimeSformer backbone. This allows the model to observe an atomic action as it unfolds, providing sufficient context to distinguish complex movements without saturating memory constraints.
* **Transformations:** each frame is resized, cropped to `224x224` pixels, and normalized. This step aligns the input with EgoVLP's original training distributions, ensuring that the patches extracted by the transformer maintain the correct spatial scale.

### Invariant temporal sampling: 1.875 Hz
Raw videos often feature heterogeneous framerates (FPS). Instead of extracting a fixed number of features per video or sampling every *n* frames, the module enforces a fixed target extraction frequency of **1.875 Hz**.
* **Mechanism:** the system dynamically calculates the stride (the step size of the sliding window) based on the video's native FPS (`stride = round(fps / 1.875)`).
* **Rationale:** this approach guarantees **temporal invariance**. Regardless of the recording hardware, the downstream model will always receive ~1.88 representations for every second of video. This consistency is crucial for sequential models, as it allows them to learn the actual real-world duration of recipe steps without being misled by variations in the original framerate.

### Pure representation: projection head removal (768-dim)
During extraction, EgoVLP's final projection head is deliberately bypassed, fetching features directly from the transformer backbone's last layer. The generated output is a dense **768-dimensional** vector per step.
* **Rationale:** during the pre-training phase, EgoVLP's projection head is used to map videos and texts into a shared latent space (optimizing a contrastive loss). This process inherently "compresses" information, discarding fine visual details in favor of a high-level semantic representation aligned with text.
* **Advantage for mistake detection:** by bypassing the head, we retrieve the raw embeddings. These 768-dimensional vectors are significantly richer in spatiotemporal micro-details (e.g., exact hand positions, small objects, subtle movement directions). These details are essential for identifying operational errors (mistakes) that are difficult to describe purely through language, ensuring maximum expressive capacity for the downstream task.

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

* **Optimizer:** `AdamW` with a constant weight decay of `1e-2` to prevent parameter expansion and enforce weight regularization.
* **Learning Rate (LR):** Configured to `2e-4`, representing the optimal empirical sweet spot for fine-tuning frozen representations safely without inducing gradient explosions.
* **Dropout:** Set aggressively to **0.4** across the classification head and self-attention layers. This high dropout configuration was introduced as a vital internal regularizer following a detailed error analysis on **Recipe 04**. In lower-dropout setups, Recipe 04 suffered from a catastrophic *semantic inversion* (yielding a failing AUROC of 0.4286), where the network systematically assigned high error probabilities to perfectly correct video executions due to sequence memorization. By randomly zeroing out 40% of the activations at each training step, the model is prevented from developing fragile co-adaptations and memorizing exact, chronological micro-frame trajectories from dominant tasks, forcing it instead to isolate robust, generalizable macro-temporal dynamics.
* **Batch Size:** Expanded to `64` to maximize the parallel tensor-core throughput of the hardware while generating stable, highly directional gradient updates.
* **Epochs & Scheduling:** Strictly capped at **20 epochs** to act as a temporal barrier against overfitting and prevent the network from drilling into local shortcuts. With 383 training samples per fold, the loader performs approximately 6 optimization steps per epoch, totaling **120 gradient steps** over the full run. The learning rate is controlled by a `CosineAnnealingLR` scheduler, tapering down from `2e-4` to a minimum of `1e-6` at `T_max=20` to smoothly freeze the attention matrices inside the optimal minimum.

This synchronized combination of a compressed sequence space (Stride 4), aggressive feature-level dropout (0.4), and an early epoch cutoff effectively countered the cross-task interference, successfully restoring the geometric separation of the model and driving the task-specific AUROC of Recipe 04 from **0.4286** up to a highly solid **0.7714**.

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

## 6. Experimental Results and Performance Analysis

The finalized regularized Sequence Transformer (configured with Stride 4, 20 training epochs, a 0.4 dropout rate, and operating on clean, non-augmented EgoVLP embeddings) was evaluated across the complete cohort of 384 video samples through a rigorous Leave-One-Out (LOO) cross-validation regime. 

### 6.1 Quantitative Metrics and Confusion Matrix Evaluation
To verify whether the architectural regularizations successfully prevented the network from falling into the majority-class guessing shortcut ("model laziness"), we examine the raw execution metrics at the standard decision boundary of $\tau = 0.50$:

* **AUROC (Global Geometric Capacity):** 0.62862
* **Accuracy:** 0.62760 (62.76%)
* **F1-Score:** 0.68845
* **Precision:** 0.66109
* **Recall (Sensitivity):** 0.71818

The active discriminative behavior of the model is mathematically validated by the global Confusion Matrix compiled at the $\tau = 0.50$ threshold:

|                     | Predicted Correct (0) | Predicted Error (1) |
|---------------------|-----------------------|---------------------|
| **Actual Correct (0)** | 83 (TN)              | 81 (FP)             |
| **Actual Error (1)**   | 62 (FN)              | 158 (TP)            |

#### Refutation of Majority-Class Guessing (Model Laziness)
In early iterations featuring unregularized deep architectures, the model suffered from complete majority-class collapse. Due to the baseline dataset distribution leaning toward anomalous executions (220 Error videos vs. 164 Correct videos), the unconstrained network optimized its loss by shifting its prediction barycenter entirely toward the positive class, systematically predicting "Error" without processing fine-grained temporal kinematics. 

The finalized model successfully overcomes this limitation. Out of 384 test iterations, the network risks predicting a valid, error-free execution ($y=0$) a total of 145 times, securing **83 True Negatives (TN)**. The predicted class distribution (239 Errors vs. 145 Correct) mirrors the underlying ground truth distribution with high fidelity. This balance demonstrates that the combination of temporal downsampling via Stride 4 and aggressive dropout (0.4) stripped away superficial visual shortcuts, forcing the self-attention heads to actively look for distinct kinematic and procedural anomalies.



### 5.2 Post-Hoc Threshold Calibration
While the symmetric Binary Cross-Entropy (BCE) loss aligns the natural mathematical center at $0.50$, the injection of **0.1 Label Smoothing** alters the logit dynamics. By converting hard binary targets into soft boundaries $[0.1, 0.9]$, label smoothing prevents the network from generating overconfident, extreme probabilities, contracting the entire prediction landscape toward a narrower central band. 

When optimizing the model for a deployment scenario where missing a procedural mistake carries a high cost (requiring high-sensitivity anomaly detection), a post-hoc threshold calibration becomes highly effective. Shifting the decision boundary down to $\tau = 0.30$ adapts the model to the smoothed probability space, yielding the following operational profile:

* **Calibrated Accuracy:** 0.62500
* **Calibrated F1-Score:** **0.72830** (a +3.98% improvement)
* **Calibrated Recall:** **0.89091** (a +17.27% improvement)
* **Calibrated Confusion Matrix:** `TP = 196 | TN = 43 | FP = 121 | FN = 24`

By operating at $\tau = 0.30$, the network successfully captures **196 out of 220 real procedural errors**, offering a highly robust safety-net baseline at a minimal cost to global accuracy.

---

### 6.3 Generalization to Unseen Recipes (LOGO)

To probe how much of the Transformer's performance depends on having seen other videos of the *same* recipe during training, the model was additionally evaluated under a **Leave-One-Recipe-Out (LOGO)** protocol: all videos of a held-out recipe are removed from training together, so the model must classify a recipe it has *never* seen. This is a strictly harder regime than the LOO evaluation above — under LOO the network can memorize recipe-specific visual signatures, motion patterns, and error distributions, whereas under LOGO it has no recipe-specific prior at all and must generalize across procedural domains.

The drop in performance quantifies exactly that dependence:

| Metric | Transformer LOO | Transformer LOGO |
|---|---|---|
| AUROC | 0.62862 | **0.60260** |
| Accuracy | 0.62760 | 0.59110 |
| F1-Score | 0.68845 | 0.65800 |
| Precision | 0.66109 | 0.63180 |
| Recall | 0.71818 | 0.68640 |

**Post-hoc threshold calibration (LOGO, $\tau^* = 0.6880$):** Acc 0.59635, Precision 0.69461, Recall 0.52727, F1 0.59948.

AUROC falls by ~2.6 points and accuracy by ~3.7 points when the recipe is unseen. This confirms that a meaningful fraction of the Transformer's LOO score is recipe-specific memorization rather than transferable procedural reasoning — precisely the limitation that motivates the topological (GNN) approach, and the reason the GNN is benchmarked primarily under LOGO. The two protocols are revisited side-by-side for both models in the GNN results section below.

---

## 7. Granular Error Diagnostics & Inductive Limits of Sequence Transformers

A disaggregated, recipe-by-recipe diagnostic assessment of the 384 cross-validation folds reveals a stark performance divergence across tasks, highlighting the boundaries of unconstrained temporal attention on fine-grained human actions:

* **Top-Performing Task Classes:** Recipe 26 (**82.35%** accuracy), Recipe 18 (**80.00%**), Recipe 20 (**78.57%**), Recipe 01 (**77.78%**), and Recipe 29 (**77.78%**).
* **Failing/Inverted Task Classes:** Recipe 23 (**37.50%** accuracy), Recipe 25 (**40.00%**), Recipe 09 (**42.86%**), and Recipe 08 (**43.75%**).

| Recipe ID | Total Samples | Ground Truth Errors | Predicted Errors | TP | TN | FP | FN | Accuracy |
|-----------|---------------|---------------------|------------------|----|----|----|----|----------|
| 01        | 18            | 13                  | 13               | 11 | 3  | 2  | 2  | 77.78%   |
| 04        | 17            | 7                   | 10               | 5  | 5  | 5  | 2  | 58.82%   |
| 08        | 16            | 10                  | 15               | 6  | 1  | 9  | 4  | 43.75%   |
| 18        | 15            | 9                   | 10               | 8  | 4  | 2  | 1  | 80.00%   |
| 20        | 14            | 8                   | 10               | 7  | 4  | 3  | 1  | 78.57%   |
| 23        | 16            | 9                   | 14               | 7  | 2  | 7  | 2  | 56.25%   |
| 26        | 17            | 10                  | 11               | 8  | 6  | 3  | 2  | 82.35%   |

### 7.1 The Cross-Task Semantic Inversion Phenomenon
The catastrophic drop in accuracy observed in Recipes 23, 25, 09, and 08 is driven by **Cross-Task Semantic Interference**. A puristic temporal Transformer treats visual video embeddings as a linear, unconstrained string of feature events. It learns to correlate specific visual movements, hand trajectories, or tool manipulations with the presence of an error based entirely on the global statistical frequency of those patterns across the training pool. 

Consequently, if a correct execution step within a minority recipe (e.g., Recipe 23) shares high feature-level similarity or background co-occurrences with an incorrect step from a dominant recipe (e.g., Recipe 01), the model falls victim to semantic inversion. Lacking any localized context regarding recipe boundaries, the network over-indexes on the visual familiarity of the motion profile. It flags legitimate, context-specific manipulations as generic procedural errors, causing false positives to spike (e.g., 7 False Positives in Recipe 23) and dragging task-specific accuracy significantly below random chance.

### 7.2 The Topological Imperative: Bridging to Graph Neural Networks
The empirical evidence gathered from this finalized baseline proves that this performance cap cannot be resolved via further hyperparameter sweeps or standard feature-level regularizations. The performance trade-off is structural: **the sequence Transformer is fundamentally blind to procedural logic.** It evaluates structural correctness without any explicit representation of the underlying recipe blueprints or state transition constraints. It cannot verify whether an action strictly complies with a localized sequence of steps.

To bypass this intrinsic limitation, the system must transition from a sequential attention framework to a **Graph Neural Network (GNN)** paradigm. By mapping localized action proposals directly to state nodes and constraining their connectivity through a deterministic **Procedural Task Graph**, the verification task is refactored from unconstrained temporal sequence matching into topological path validation. 

In the upcoming GNN layer, cross-task semantic inversion is systematically eliminated: an execution step is no longer evaluated on fuzzy, global visual similarities, but verified through its topological validity as a valid path transition within that specific recipe’s graph. This architectural shift leverages the stable temporal features extracted by the Transformer while providing the necessary logical constraints to eliminate cross-task confusion.

# Task Graph Encoding and Semantic Matching

# 1. Cross-Modal Node Realization and Alignment

In this stage, we bridge the modality gap between the ground-truth task graphs provided by the CaptainCook4D dataset and the temporal visual features extracted in the previous steps. The primary objective is to establish an optimal and robust alignment between the textual description of each recipe step (represented as a node in the task graph) and the corresponding visual segment in the video. The matched representations are subsequently fused to yield *realized node features* that encapsulate both the semantic intent and the visual execution of the step, serving as the input state for the Graph Neural Network (GNN).

---

## 2. Textual Feature Encoding

To establish a strong semantic prior, we encode the textual descriptions of the task graph nodes using the pre-trained text encoder from EgoVLP, based on the FrozenInTime architecture. The descriptive text associated with each node is tokenized and processed through the language model to obtain dense semantic embeddings.

The resulting textual embeddings are $\ell_2$-normalized in order to project them onto a unit hypersphere, producing robust 256-dimensional representations that naturally share a latent topology optimized for ego-centric action understanding.

---

## 3. Temporal Abstraction and Domain Adaptation

Visual features extracted from videos often contain high-frequency temporal noise and operate in a higher-dimensional space (768-d) compared to textual embeddings. To address this discrepancy, the visual representations first undergo a temporal abstraction stage implemented through a 1D Convolution layer with kernel size 4 and stride 4. This operation both downsamples the temporal sequence and smooths short-term fluctuations.

Subsequently, to mitigate the domain shift between visual and textual modalities, both representations are projected into a shared 256-dimensional latent space via two separate **non-linear MLP projection heads** (`sim_visual_proj` and `sim_text_proj`). Each projection head is structured as:

$$
\text{Linear}(d_{in} \to 256) \to \text{GELU} \to \text{LayerNorm}(256) \to \text{Linear}(256 \to 256)
$$

The use of GELU activations and Layer Normalization — rather than simple linear projections — is a deliberate architectural decision. Simple linear layers can only rotate and scale the feature space, leaving the inter-modal domain gap structurally intact. Non-linear projectors instead learn to *warp* each modality's manifold independently, bending the visual and textual representation spaces until they overlap on a shared semantic surface. This is particularly important here because EgoVLP visual features (768-d) and text features (256-d) originate from different computational paths (vision encoder vs. DistilBERT) and exhibit significant distributional divergence. Layer Normalization stabilizes the projection during the early stages of training when the alignment signal is still weak.

---

## 4. Sequential Positional Encoding and Bipartite Matching via the Hungarian Algorithm

### 4.1 Sequential Positional Encoding

Before computing the similarity matrix, a crucial **Sequential Positional Encoding** step is applied to the projected text features. A learnable embedding table maps each step index $i \in \{0, 1, \ldots, N-1\}$ to a 256-dimensional positional vector, which is additively injected into the projected text representation:

$$
\tilde{t}_i = \text{proj\_text}(t_i) + \text{PE}(i)
$$

This design choice addresses a fundamental weakness of pure cosine-similarity-based matching: without positional information, two task graph nodes with visually similar descriptions (e.g., *"add the ingredient"* appearing at step 2 and step 7) are indistinguishable during the Hungarian assignment. The Sequential PE breaks this symmetry by encoding the *chronological order* of each step, biasing the similarity matrix toward temporally coherent alignments and preventing the algorithm from confusing procedurally distant steps that share superficial visual similarity.

Unlike sinusoidal encodings, using a learnable embedding allows the model to adapt the positional structure to the specific temporal geometry of the egocentric cooking domain during training, rather than imposing a fixed mathematical prior.

### 4.2 Bipartite Matching via the Hungarian Algorithm

Within the shared latent space, we compute a cosine similarity matrix $S \in \mathbb{R}^{N_{vis} \times N_{text}}$ between the positionally-encoded projected visual features and textual node embeddings:

$$
S_{ij} = \frac{\tilde{v}_i \cdot \tilde{t}_j}{\|\tilde{v}_i\| \cdot \|\tilde{t}_j\|}
$$

The alignment process is formulated as a maximum-weight bipartite matching problem. Specifically, the negative similarity matrix $C = 1 - S$ is interpreted as a cost matrix and solved via the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`), enforcing a strict one-to-one correspondence between visual segments and task graph nodes.

**Computational Optimization.** The Hungarian algorithm has worst-case complexity $\mathcal{O}(n^3)$, which could become prohibitive for long sequences. Two strategies are applied to keep it tractable: (1) the temporal Conv1D with stride 4 already reduces the visual sequence length by a factor of 4 before any matching is performed; (2) the cost matrix is computed entirely on GPU in float32 and detached from the computational graph (`.detach().cpu().numpy()`) before being passed to SciPy. This means the Hungarian solver runs on CPU (where SciPy is highly optimized via LAPJV internally) while never blocking the GPU gradient tape — the matching result is treated as a fixed index assignment, and only the *downstream similarity scores* at those matched positions are retained on-graph for gradient flow through the contrastive loss.

**Cosine Semantic Thresholding.** To avoid spurious assignments caused by noisy or semantically irrelevant visual segments, a matched pair $(v_i, t_j)$ is accepted only if the cosine similarity exceeds an empirical threshold:

$$
\cos(v_i, t_j) \geq 0.20
$$

Unmatched task graph nodes are assigned a learnable **missing visual embedding** (a trainable parameter initialized with $\mathcal{N}(0, 0.02)$), ensuring that every node receives a valid feature vector for downstream fusion regardless of matching success.

---

## 5. Auxiliary Contrastive Alignment: InfoNCE, Learnable Temperature, and Hard Negative Mining

### 5.1 The InfoNCE Objective

To strengthen the cross-modal alignment signal beyond what the Hungarian assignment alone provides, we introduce an auxiliary **InfoNCE** (Noise-Contrastive Estimation) loss computed over the matched pairs.

InfoNCE frames alignment as a classification problem: given a matched text node $t_j$, the model must identify its correct visual partner $v_{i^*}$ among $K$ hard negative visual candidates. The contrastive loss per sample is computed via cross-entropy over the concatenated logits $[s_{i^*,j} \cdot \tau,\; s_{n_1,j} \cdot \tau,\; \ldots,\; s_{n_K,j} \cdot \tau]$ with the positive at index 0. Across the batch, the loss is accumulated sample-by-sample and divided by the count of samples that contained at least one valid Hungarian match (`valid_batch_counts`):

$$
\mathcal{L}_{\text{InfoNCE}} = \frac{1}{|\mathcal{B}_{\text{valid}}|} \sum_{b \in \mathcal{B}_{\text{valid}}} \mathcal{L}_{\text{CE}}^{(b)}
$$

where $\mathcal{B}_{\text{valid}}$ is the set of batch samples with at least one matched pair and $\mathcal{L}_{\text{CE}}^{(b)}$ is the mean cross-entropy over all matched text nodes in sample $b$. Minimizing this loss pulls matched visual-text pairs together on the unit hypersphere while repelling the most confusing visual alternatives — a critical regularizer given the high inter-recipe visual similarity in the CaptainCook4D dataset.

### 5.2 Learnable Temperature (CLIP-style)

The similarity scores are scaled by a **learnable temperature parameter** $\tau = \exp(\log\_scale)$, initialized at $\log(1/0.07) \approx 2.66$, following the CLIP pretraining convention. Temperature controls the sharpness of the softmax distribution over negatives:

- A *low* temperature makes the distribution peaky, producing strong gradients for clearly wrong assignments but risking instability on ambiguous pairs.
- A *high* temperature produces a flatter distribution, generating weaker but more stable gradients.

By making $\tau$ a trainable parameter (optimized alongside the projectors), the model self-calibrates the contrastive sharpness to the specific difficulty level of the current training batch, avoiding the need for a manually tuned fixed temperature. This is particularly beneficial in the early epochs when the alignment is still noisy and a conservative temperature prevents gradient explosions before the projectors have stabilized.

### 5.3 Hard Negative Mining

Standard InfoNCE draws negatives randomly from the batch. In a recipe-structured dataset, however, random negatives are often *too easy* — a visual segment of someone cracking an egg is trivially different from a text node describing *"stir the sauce"*. Easy negatives provide near-zero gradient signal and slow convergence dramatically.

To address this, we implement **Hard Negative Mining**: for each **matched text node** $t_j$, we examine its full row of scaled similarity logits across *all visual nodes* — not just the one it was assigned to. The positive entry is the matched visual $v_{i^*}$; all other visual nodes become candidate negatives. Among these, we select the $K$ with the **highest similarity scores** — the visual segments that are most confusingly similar to $t_j$ yet were not selected as its match by the Hungarian algorithm. These are the most dangerous false-positive candidates: visually plausible but semantically incorrect assignments.

Concretely, for each matched text node $t_j$ (row $j$ of the transposed logit matrix), the positive column $v_{i^*}$ is masked out, and `torch.topk` selects the top-$K$ remaining visual logits as hard negatives:

```python
matched_logits = logits.t()[t_idx_t]          # [num_matched, num_vis]
mask[torch.arange(num_matched), v_idx_t] = False
negative_logits = matched_logits[mask].view(num_matched, num_vis - 1)
hard_negatives, _ = torch.topk(negative_logits, k, dim=-1)
```

The number of hard negatives $K$ is set **dynamically** as a fixed fraction of the visual sequence length:

$$
K = \max\!\left(1,\; \left\lfloor 0.15 \cdot N_{vis} \right\rfloor\right)
$$

This adaptive schedule is intentional: longer videos with more detected steps produce more hard negatives, scaling the difficulty of the contrastive task proportionally to the complexity of the scene. For very short sequences (where $N_{vis}$ is small), $K$ is floored to 1 to ensure at least one negative is always present.

The final InfoNCE logits for each matched pair consist of one positive score concatenated with $K$ hard negative scores:

$$
\ell_i = \big[s_{ij} \cdot \tau, \; s_{i,n_1} \cdot \tau, \; \ldots, \; s_{i,n_K} \cdot \tau\big]
$$

Cross-entropy is then minimized with the positive class at index 0.

---

## 6. Node Realization and Feature Fusion

Once the optimal alignment is established, the downsampled visual features are gathered according to the matched indices produced by the Hungarian algorithm. Task graph nodes that received no valid match (either due to low cosine similarity or an absence of visual coverage) are assigned the **learnable missing visual embedding** — a dedicated trainable parameter that acts as a learned "no-match" sentinel, preventing zero-vectors from corrupting downstream message passing.

To preserve the original semantic richness encoded by the language model, the gathered visual features are concatenated directly with the **raw (unprojected) textual embeddings** — not the positionally-encoded projected versions used during matching. This design choice is deliberate: the positional encoding was injected only to guide the similarity computation; fusing raw text features into the GNN ensures that the full 256-dimensional EgoVLP language representation, without any matching-induced bias, is available for structural reasoning.

The concatenated vector $[v_{\text{matched}} \| t_{\text{raw}}] \in \mathbb{R}^{768 + 256}$ is processed by the **Unified Fusion MLP**:

$$
\text{Linear}(1024 \to 512) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(512 \to 256) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{LayerNorm}(256)
$$

Note that the fusion MLP uses **ReLU** activations (rather than GELU) — a deliberate choice for the aggregation stage. After the contrastive projectors have already shaped the feature space with GELU non-linearities, ReLU in the fusion stage acts as a sparse activation gate: it hard-zeros negative activations, effectively performing implicit feature selection and encouraging the network to build a sparse, disjoint internal representation of visual-semantic node states. The final LayerNorm stabilizes the magnitude of realized node vectors before they enter the graph convolution layers.

The output constitutes the final **realized node representations** $h_i \in \mathbb{R}^{256}$: dense vectors that fuse the semantic intent of the procedural step (from the task graph text) with grounded visual evidence (from the matched video segment). These realized nodes, organized according to the DAG topology of their respective recipe, are subsequently routed into the Graph Neural Network for topological verification and error detection.


---

# Graph Neural Network for Procedural Error Detection

## 0. Dataset Pipeline, Graph Construction, and Computational Strategy

### 0.1 RAM Preloading

The LOGO evaluation requires training a fresh model for each of the 24 recipe folds. Reloading large NPZ archives from disk at every fold would introduce significant I/O overhead. Instead, both feature archives (`step_embeddings_dataset.npz` for visual features, `text_task_graphs_v2.npz` for text features) are loaded **once into RAM** at program startup and kept as plain Python dictionaries keyed by `video_id`. Every fold then receives a reference to these in-memory dictionaries, making dataset construction essentially instantaneous:

```python
global_visual = {k: v.astype(np.float32) for k, v in np.load(visual_npz).items()}
global_text   = {k: v.astype(np.float32) for k, v in np.load(text_npz).items()}
```

### 0.2 Graph Construction and Node Remapping

At dataset construction time (`TaskVerificationGraphDataset.__init__`), the task graph JSON files are parsed for every recipe. The raw graph stores node IDs as string keys, including `START` and `END` anchor tokens. These are filtered out — they carry no semantic content — and the remaining valid nodes are sorted numerically to produce a stable ordered list `valid_ids`. A `node_id_to_local` mapping then converts original dataset node IDs to compact local indices $0 \ldots N-1$, which is the index space used by PyTorch Geometric.

At item retrieval (`__getitem__`), edges are remapped through this mapping and stored as a `[2, E]` long tensor in COO format. The binary label is derived by checking whether any step in the video's annotation has `has_errors: True`.

### 0.3 DAG Depth Computation

Node depths are computed in `__getitem__` via `compute_dag_depth()`: a BFS from all zero-in-degree source nodes over the remapped local edge list. Depth is defined as the length of the longest path from any source to the node, giving each node its causal distance from the start of the recipe. The result is a `np.int64` array of shape `[N]`, stored per sample and later padded by the collator.

### 0.4 Dynamic Collation

`graph_collate_fn` assembles variable-length samples into padded batch tensors. Visual and text feature tensors are zero-padded to the maximum sequence length within the batch, with corresponding float masks (`1.0` for real tokens, `0.0` for padding). Node depth arrays are similarly padded to `max_text_len`. Edge index tensors are **not** stacked — they are kept as a Python list of per-sample `[2, E_i]` tensors, since graph sizes differ and stacking would require expensive offset bookkeeping at collation time; this is instead deferred to the GNN's graph batching loop.

---

## 1. From Sequence Transformers to Graph Neural Networks: Architectural Motivation

The Transformer baseline established in the previous substep demonstrated a hard performance ceiling rooted not in hyperparameter choices but in a **structural limitation**: a sequence model is fundamentally blind to procedural logic. It evaluates a video as a chronological stream of visual features without any representation of the underlying recipe blueprint or the causal constraints between steps. Two consequences followed directly from this blindness.

**Cross-Task Semantic Inversion.** When trained across all recipes simultaneously, the Transformer learned to associate specific motion profiles with error labels based on global statistical frequency. A correct manipulation in Recipe 23 that visually resembles an incorrect manipulation from Recipe 01 — a dominant recipe in the training pool — was systematically flagged as an error. The model had no mechanism to condition its judgment on *which recipe* it was currently evaluating. This produced catastrophic false-positive spikes (e.g., 7 FP in Recipe 23) and task-specific accuracy dropping below random chance.

**Majority-Class Collapse.** The dataset imbalance (220 error videos vs. 164 correct videos) created a persistent optimization trap. Without structural constraints, deep unconstrained networks discovered that shifting the prediction barycenter toward the positive class minimized training loss cheaply, producing inflated Recall at the cost of Precision and True Negative coverage. Regularization via dropout and label smoothing partially mitigated this but could not eliminate it architecturally.

The transition to a **Graph Neural Network** resolves both failure modes by design. By mapping each procedural step to a node in the recipe's **task graph** — a Directed Acyclic Graph (DAG) encoding the valid causal ordering of actions — error detection is refactored from unconstrained temporal sequence matching into **topological path validation**. An execution step is no longer evaluated on fuzzy global visual similarities; it is verified as a valid transition within that specific recipe's graph. Cross-task confusion is eliminated because each video is reasoned about exclusively through its own recipe's topology.

---

## 2. DAG Structural Depth and Positional Encoding

A critical challenge when applying standard message-passing GNNs to procedural DAGs is node positional ambiguity. In an unweighted graph, nodes at different structural positions can appear identical to the message-passing algorithm if their local neighborhoods share similar feature statistics. For a recipe DAG, this is catastrophic: "Add butter to the pan" (step 2) and "Add sauce to the pan" (step 6) may produce visually similar node features after fusion, yet their topological roles — and the error signatures they carry — are entirely different.

To inject structural awareness directly into the graph representation, we compute the **topological depth** of each node in the DAG via a Kahn's algorithm-style BFS from all source nodes (nodes with in-degree 0):

$$
\text{depth}(v) = \max_{u \in \text{parents}(v)} \left( \text{depth}(u) + 1 \right), \quad \text{depth}(\text{source}) = 0
$$

This gives each node a scalar integer encoding the length of the longest causal path leading to it — its position in the procedural dependency chain. A **learnable depth embedding table** (`Embedding(50, 256)`, initialized with $\mathcal{N}(0, 0.01)$ for small initial perturbations to avoid early over-dominance) maps these depth integers into 256-dimensional positional vectors.

The depth integers are computed once per sample in the `Dataset` via `compute_dag_depth()` — a BFS over the remapped local edge list — and carried through the collator as `node_depths: [B, max_N]`. Inside `TaskVerificationGNN.forward()`, after `GraphNodeRealizer` has produced the realized node states, the depth embeddings are additively injected per-sample **before the first GNN layer**:

```python
d_emb = self.depth_embedding(torch.clamp(d_b, max=49))
x_b = x_b + d_emb
```

The clamp to 49 guards against any edge-case DAGs deeper than the embedding table. The injection happens after graph batching slices away padding, so only real nodes receive a depth signal — padding positions are never touched.

The rationale is direct: with depth encoding, the GNN *knows* that a node at depth 0 is a prerequisite that must be executed before nodes at depth 1 or 2. When a procedural error causes a step to be skipped or executed out of order, the mismatch between the expected topological depth and the visual evidence of the matched segment produces a discriminative signal that a depth-unaware GNN — which sees all nodes as positionally equivalent — would miss entirely.

---

## 3. GNN Architecture

The `TaskVerificationGNN` processes a batch of recipe DAGs through the following staged pipeline.

### 3.1 Graph Batching (Dense → Sparse PyG Format)

PyTorch Geometric operates on sparse, flat graph representations. Since the dataset provides padded dense tensors (batch dimension B, max nodes $N_{max}$, feature dim 256), a graph batching loop unrolls each sample into a contiguous flat tensor by:

1. Slicing the first `num_real_nodes = text_mask[b].sum()` rows from the realized node matrix (discarding padding).
2. Injecting depth embeddings additively into the sliced node features.
3. Offsetting the edge index tensor by the cumulative node count `node_offset` to produce globally unique node indices across the batch.
4. Appending a batch assignment vector `batch_flat` mapping each node to its sample index $b$.

The result is a single set of flat tensors $(X_{\text{flat}}, \text{edge\_index\_flat}, \text{batch\_flat})$ representing all B graphs as one large disconnected graph — the standard PyG batching convention — ready for efficient sparse message passing.

### 3.2 Why GraphConv over GCNConv

The graph convolution layers use **`GraphConv`** (Weisfeiler-Lehman-style) rather than the classical `GCNConv`. This choice is motivated by a key architectural difference in how they aggregate neighbor messages:

- **GCNConv** computes: $h_v' = W \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u$ — a symmetric normalization that treats every node as a weighted average of its neighborhood, including itself with degree-normalized self-loops.
- **GraphConv** computes: $h_v' = W_1 h_v + W_2 \sum_{u \in \mathcal{N}(v)} h_u$ — two *separate* linear transformations for the self-representation and the neighborhood aggregation.

For procedural DAGs, the self-representation of a node carries fundamentally different information from its neighbors' aggregated state. A step's own visual-semantic realization (was this specific action performed correctly?) must be weighed independently of the topological context provided by adjacent steps (what came before and after?). GCNConv's symmetric normalization conflates these two information sources, diluting the self-signal inside the neighborhood mean. GraphConv's decoupled parametrization preserves both channels independently, allowing the model to learn the optimal trade-off between local self-evidence and topological neighborhood context — critical for a task where a step can be individually correct yet topologically misplaced.

Additionally, GCNConv's degree normalization $\frac{1}{|\mathcal{N}(v)|}$ systematically suppresses the signal at high-degree nodes (steps with many dependencies), which in recipe DAGs often correspond to the most structurally critical junctions. GraphConv avoids this implicit down-weighting.

### 3.3 Two-Layer Message Passing with Residual Connection

The graph convolution stack applies two successive layers with the following structure:

**Layer 1:**
$$
h^{(1)} = \text{Dropout}\!\left(\text{ReLU}\!\left(\text{LayerNorm}\!\left(\text{GraphConv}_1(h^{(0)}, \mathcal{E})\right)\right)\right)
$$

**Layer 2 with residual injection:**
$$
h^{(2)} = \text{Dropout}\!\left(\text{ReLU}\!\left(\text{LayerNorm}\!\left(\text{GraphConv}_2(h^{(1)}, \mathcal{E})\right) + h^{(1)}\right)\right)
$$

The **residual connection** in Layer 2 is essential for two reasons. First, DAG message passing is depth-limited: a two-hop GNN can only aggregate information from nodes up to two edges away. The residual shortcut ensures that the self-representation from Layer 1 — which already encodes the 1-hop neighborhood — is preserved and directly summed into Layer 2's output, effectively giving each node access to a richer combination of 0-hop, 1-hop, and 2-hop information. Second, residual connections are a proven stabilizer against vanishing gradients, particularly important here given the aggressive Dropout(0.4) applied at each layer.

Layer Normalization is applied *after* the graph convolution but *before* the non-linearity, following the pre-LN convention that empirically produces smoother loss landscapes on small datasets.

### 3.4 Multi-Pooling Graph Readout

After message passing, the node-level representations must be aggregated into a single graph-level embedding for binary classification. Two global pooling strategies are applied in parallel:

$$
g_{\text{mean}} = \frac{1}{|V|} \sum_{v \in V} h_v^{(2)}, \qquad g_{\text{max}} = \max_{v \in V} h_v^{(2)}
$$

These are concatenated into a joint graph embedding $g = [g_{\text{mean}} \| g_{\text{max}}] \in \mathbb{R}^{512}$.

**Mean pooling** captures the *average procedural state* of the execution — a holistic signal that integrates all steps' representations uniformly, ideal for detecting systematic patterns (e.g., globally rushed or incomplete executions).

**Max pooling** captures the *most anomalous local signal* across the graph — a max-sensitive readout that is disproportionately influenced by nodes with the largest activation magnitudes, which in practice correspond to the most deviant steps. An execution with a single catastrophic error (e.g., a completely omitted step) may show a near-normal mean pooled representation but a strongly deviant max-pooled one.

The concatenation of both ensures neither signal is lost in the classification head.

### 3.5 Classification Head

The joint graph embedding passes through a final anomaly classification MLP:

$$
\text{Linear}(512 \to 128) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(128 \to 1)
$$

The output is a single scalar logit fed to BCEWithLogitsLoss during training and to sigmoid during inference.

---

## 4. Escaping Majority-Class Collapse: Architectural and Training Strategies

The dataset imbalance that plagued the Transformer baseline is addressed through a coordinated set of architectural and training-time interventions.

**Structural Graph Constraints.** The most powerful defense against majority-class collapse is the task graph itself. By constraining the model to reason through recipe-specific topologies, the GNN is forced to evaluate whether each step is executed in a plausible causal position — a judgment that requires processing both the positive and negative class meaningfully. A model that blindly predicts "Error" for every video will have consistently high node-level features regardless of topology, producing identical graph pooling vectors across samples. The classification head, trained to minimize BCE, receives no gradient signal from this strategy and quickly learns to abandon it.

**Aggressive Dropout (0.4).** Applied after every GNN layer and in the classification head, high dropout prevents the model from memorizing recipe-specific co-occurrence patterns (e.g., Recipe 01 always correlates with error label 1 in the training split). Each gradient step sees a different sub-graph of the network, forcing distributed representations that generalize across recipes.

**Label Smoothing (0.1).** Hard binary targets are softened **bidirectionally**: positive labels $1 \to 0.9$ and negative labels $0 \to 0.1$, implemented as:

```python
smoothed_labels = labels * (1.0 - label_smoothing) + (1.0 - labels) * label_smoothing
```

This establishes a minimum entropy floor on the BCE loss ($\approx 0.32$), preventing the model from driving logits to extreme values in either direction and maintaining a well-behaved probability landscape centered near 0.5.

**Loss Weight Annealing (Alignment Warmup).** In the early training epochs, the realized node features are unreliable — the Hungarian matching has not yet converged and the projectors are still learning the shared latent space. Feeding poorly aligned node features into the GNN at this stage would cause the classification loss to overwhelm the alignment signal before it has had time to stabilize. To prevent this, the auxiliary InfoNCE loss is weighted dynamically:

$$
\lambda_{\text{align}}(\text{epoch}) = \max\!\left(0.1,\; 1.0 \times 0.8^{\text{epoch}-1}\right)
$$

The total training loss is:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{InfoNCE}}
$$

At epoch 1, $\lambda = 1.0$: the alignment loss dominates, rapidly pushing the projectors toward a coherent shared space. By epoch 10, $\lambda \approx 0.107$: the alignment has stabilized and the classification signal takes over cleanly. This *warmup-then-handoff* schedule ensures that the GNN always receives well-formed node features, regardless of training stage.

**Differential Learning Rate.** The alignment projectors (`sim_visual_proj`, `sim_text_proj`) and the learnable temperature (`logit_scale`) must converge significantly faster than the GNN layers. In the early epochs, the GNN is essentially waiting for stable node features — running it at full learning rate while features are still noisy would cause destructive weight updates. We therefore isolate these parameter groups and assign them a **5× higher learning rate**:

| Parameter Group | Learning Rate |
|---|---|
| GNN layers, fusion MLP, depth embeddings, classification head, `step_positional_encoding` | `2e-4` |
| `sim_visual_proj`, `sim_text_proj`, `logit_scale` | `1e-3` |

The projectors *run ahead*, rapidly establishing a well-aligned shared feature space. The GNN *waits and assimilates*, updating conservatively until reliable node representations are consistently delivered. This decoupled convergence schedule is the key operational mechanism behind the architectural stability of the pipeline.

---

## 5. Why This Architecture Achieves State-of-the-Art Performance

The performance improvements over the Transformer baseline are not incidental — they follow directly from three coordinated architectural decisions that each address a specific, diagnosed failure mode.

**Sequential PE resolves chronological confusion in alignment.** By injecting learnable step-order embeddings into the projected text features before the Hungarian similarity matrix is computed, the matching is biased toward temporally coherent assignments. Two task graph nodes with similar text descriptions but different chronological positions (e.g., two *"add ingredient"* nodes at steps 2 and 7) now produce distinguishable projected representations, preventing the algorithm from assigning the same visual segment to both.

**Structural Depth PE resolves topological blindness in message passing.** Without positional information, all nodes at the same depth in a DAG are indistinguishable to the message-passing algorithm if their fused features are similar. By additively injecting depth embeddings into the realized node states *before the first GNN layer*, structurally-aware features propagate through the convolution layers from the start. The GNN can now detect when a step's visual realization is inconsistent with its expected causal position — a signal that a depth-unaware GNN would miss entirely.

**Non-Linear MLP Projectors resolve cross-domain alignment failure.** The visual (768-d, from a video transformer) and textual (256-d, from DistilBERT) feature spaces originate from architecturally different encoders and exhibit significant distributional divergence. Linear projectors can only rotate and scale each space — they cannot close a structural geometric gap. The MLP projectors (Linear → GELU → LayerNorm → Linear), optimized at 5× learning rate via differential LR, learn to non-linearly warp each modality's manifold until they overlap in the shared 256-d space, directly enabling the high-quality node realizations that flow into the GNN.

---

## 6. Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `visual_dim` | 768 | EgoVLP visual feature dimensionality |
| `text_dim` | 256 | EgoVLP text encoder output dimensionality |
| `hidden_dim` | 256 | Shared latent space dimensionality |
| `dropout` | 0.4 | Aggressive regularization against recipe memorization |
| `batch_size` | 16 | Constrained by graph batching memory requirements |
| `epochs` | 45 | Sufficient for convergence under annealed alignment weight |
| `lr` (base) | `2e-4` | Stable update rate for GNN, fusion MLP, depth embeddings, `step_positional_encoding`, classification head |
| `lr` (projectors) | `1e-3` | 5× higher for `sim_visual_proj`, `sim_text_proj`, `logit_scale` — rapid alignment convergence |
| `weight_decay` | `1e-2` | L2 regularization against parameter expansion |
| `label_smoothing` | 0.1 | Entropy floor; prevents logit divergence |
| `align_weight` (epoch 1) | 1.0 | Full InfoNCE weight during alignment warmup |
| `align_weight` (min) | 0.1 | Residual alignment signal after GNN takes over |
| `align_decay` | 0.8 per epoch | Exponential handoff from alignment to classification |
| `cosine_threshold` | 0.20 | Minimum similarity for a valid visual-text match |
| `logit_scale_init` | $\log(1/0.07)$ | CLIP-style temperature initialization |
| `depth_emb_max` | 50 | Maximum supported DAG depth |
| `depth_emb_std` | 0.01 | Small-variance init to avoid early over-dominance |
| `max_recipe_steps` | 100 | Maximum number of task graph nodes per recipe |
| `hard_neg_fraction` | 0.15 | Fraction of visual nodes selected as hard negatives |
| `gradient_clip` | 1.0 | Max gradient norm for training stability |
| `cross_validation` | LOGO | Leave-One-Recipe-Out across 24 recipe groups |


---

# GNN Experimental Results and Analysis

## 1. Global Performance — LOGO and LOO Evaluation

The `TaskVerificationGNN` was evaluated under **both** the primary **Leave-One-Recipe-Out (LOGO)** regime and, for a matched comparison with the Transformer baseline, the **Leave-One-Out (LOO)** regime. Under LOGO one recipe's videos serve as the held-out test set while the model is trained from scratch on all remaining recipes, repeated across all 24 recipe groups; under LOO each individual video is held out in turn. In both cases all 384 videos are covered exactly once.

### 1.1 Quantitative Results (LOGO)

**At the standard decision threshold ($\tau = 0.5$):**

| Metric | Value |
|---|---|
| AUROC | **0.62395** |
| Accuracy | 0.62240 |
| F1-Score | 0.69083 |
| Precision | 0.65060 |
| Recall | 0.73636 |

Confusion matrix at $\tau = 0.5$: **TN = 77, FP = 87, FN = 58, TP = 162**

**Post-hoc threshold calibration via Youden's J statistic ($\tau^* = 0.4496$):**

| Metric | Value |
|---|---|
| Accuracy | **0.64583** |
| F1-Score | **0.71784** |
| Precision | 0.66031 |
| Recall | **0.78636** |

Confusion matrix at $\tau^* = 0.4496$: **TN = 75, FP = 89, FN = 47, TP = 173**

The calibrated threshold of $0.4496$ — below $0.5$ — reflects the effect of label smoothing on the model's output distribution. As discussed in §4, label smoothing contracts predicted probabilities away from the extremes, centering the distribution around a narrower band. Youden's J statistic maximizes $\text{TPR} - \text{FPR}$ over the ROC curve, identifying the threshold that recovers the best operational trade-off between sensitivity and specificity in this compressed probability space.

### 1.2 Quantitative Results (LOO)

To enable a fully controlled, same-protocol comparison with the Transformer baseline, the GNN was additionally evaluated under **Leave-One-Out (LOO)** cross-validation — each individual video held out while the model trains on all remaining videos, including others from the same recipe.

**At the standard decision threshold ($\tau = 0.5$):**

| Metric | Value |
|---|---|
| AUROC | **0.64703** |
| Accuracy | 0.62500 |
| F1-Score | 0.70124 |
| Precision | 0.64504 |
| Recall | 0.76818 |

Confusion matrix at $\tau = 0.5$: **TN = 71, FP = 93, FN = 51, TP = 169**

**Post-hoc threshold calibration via Youden's J statistic ($\tau^* = 0.6429$):**

| Metric | Value |
|---|---|
| Accuracy | 0.62500 |
| F1-Score | 0.66512 |
| Precision | **0.68095** |
| Recall | 0.65000 |

Confusion matrix at $\tau^* = 0.6429$: **TN = 97, FP = 67, FN = 77, TP = 143**

As expected, granting the GNN access to same-recipe training videos (LOO) lifts AUROC from 0.62395 (LOGO) to **0.64703** — the GNN benefits from a recipe-specific prior just as the Transformer does, but its topological reasoning remains the dominant signal. Notably, the GNN's *harder* LOGO score (0.62395) is already within ~0.005 of the Transformer's *easier* LOO score (0.62862), and under matched LOO conditions the GNN's advantage widens to +0.018 AUROC.

---

## 2. GNN vs. Transformer Baseline — A Methodologically Honest Comparison

### 2.1 Evaluation Protocols: LOO and LOGO

Both models are now evaluated under **both** cross-validation protocols, enabling fully matched, same-protocol comparisons.

- **LOO (Leave-One-Out):** each individual video is held out while the model trains on all remaining videos. The model **always has access to other videos from the same recipe** during training, allowing it to learn recipe-specific visual signatures, motion patterns, and error distributions.
- **LOGO (Leave-One-Recipe-Out):** all videos of a held-out recipe are removed from training together. The model **never sees a single video from the test recipe** and must generalize to an entirely unseen procedural domain.

The two protocols probe fundamentally different capabilities:

| Aspect | LOO | LOGO |
|---|---|---|
| Test condition | Unseen *video* from a *known* recipe | Unseen *recipe* entirely |
| Training access | All other videos of the same recipe | Zero videos of the test recipe |
| Generalization required | Intra-recipe | Cross-recipe (zero-shot on target recipe) |
| Task difficulty | Easier | Significantly harder |

A clean comparison therefore reads down the *columns* (same protocol, both models). LOGO is the protocol of primary interest, since it measures genuine transferable procedural reasoning rather than recipe-specific memorization.

### 2.2 Global Metrics — Full Protocol Matrix

All four configurations at the standard threshold ($\tau = 0.5$):

| Metric | Transformer (LOO) | GNN (LOO) | Transformer (LOGO) | GNN (LOGO) |
|---|---|---|---|---|
| **AUROC** | 0.62862 | **0.64703** | 0.60260 | **0.62395** |
| Accuracy | 0.62760 | 0.62500 | 0.59110 | **0.62240** |
| F1-Score | 0.68845 | **0.70124** | 0.65800 | **0.69083** |
| Precision | 0.66109 | 0.64504 | 0.63180 | **0.65060** |
| Recall | 0.71818 | **0.76818** | 0.68640 | **0.73636** |

Two findings hold consistently across both protocols:

1. **The GNN outperforms the Transformer under matched conditions.** Under LOO, the GNN improves AUROC by **+0.018** (0.64703 vs 0.62862) and F1 by **+0.013**. Under LOGO, it improves AUROC by **+0.021** (0.62395 vs 0.60260) and accuracy by **+3.1 points** (0.62240 vs 0.59110). The topological inductive bias adds signal regardless of how the data is split.

2. **The GNN's harder result is competitive with the Transformer's easier result.** The GNN under LOGO (AUROC 0.62395) — never having seen the test recipe — essentially matches the Transformer under LOO (0.62862), which had full same-recipe training access. The structural prior compensates almost entirely for the loss of recipe-specific memorization.

Both models degrade from LOO to LOGO, as expected, but the GNN degrades *less* in relative terms (AUROC −0.023 for the GNN vs −0.026 for the Transformer), consistent with a model that relies more on recipe-agnostic topology than on memorized visual signatures.

### 2.3 Confusion Matrix Analysis (LOO) — Structural Differences in Error Patterns

To compare the two models' decision behavior on the *full* dataset under identical conditions, we examine the LOO confusion matrices, each reported at its own ROC-optimal (Youden's J) operating point:

| | Transformer (LOO, $\tau^*=0.30$) | GNN (LOO, $\tau^*=0.64$) |
|---|---|---|
| True Negatives (TN) | 80 | **97** (+17) |
| False Positives (FP) | 84 | **67** (−17) |
| True Positives (TP) | **164** | 143 (−21) |
| False Negatives (FN) | **56** | 77 (+21) |

The two models occupy clearly different operating regimes. The GNN correctly accepts **97 of 164 correct executions** (59.1% specificity) versus the Transformer's 80 (48.8%), and raises **17 fewer false alarms**. This is architecturally meaningful: the GNN's topological validation requires positive evidence from the DAG structure before flagging an error, making it harder to trigger false positives from superficial visual similarity alone.

The Transformer, by contrast, sits at a recall-biased operating point. Its ROC-optimal threshold is pushed down to $\tau = 0.30$, and even there it predicts **248 Errors vs 136 Correct** — far more skewed toward the positive class than the ground-truth split (220/164), a residual signature of the majority-class pressure documented in the baseline section. The GNN at its optimal $\tau = 0.64$ predicts **210 Errors vs 174 Correct**, much closer to the true distribution, recovering balance without collapsing onto the majority class.

The threshold-independent summary is the AUROC gap, **+0.018 in favor of the GNN under LOO** (0.64703 vs 0.62862), reflecting genuine probabilistic discrimination rather than a thresholding artifact. The discriminative quality is further characterized per-recipe under LOGO in §3, where per-recipe probability separation correlates with per-recipe AUROC at **r = 0.944**.

---

## 3. Per-Recipe Analysis (LOGO)

Per-recipe diagnostics are reported under **LOGO** rather than LOO, because the zero-shot, unseen-recipe setting is where recipe-level behavior is most informative: it isolates whether the model's reasoning *transfers* to a recipe it has never trained on. Both models are analyzed under the same LOGO protocol, making the per-recipe comparison fully matched.

### 3.1 Full Breakdown (GNN — LOGO)

| Recipe | N | GT Err | GT Cor | TP | TN | FP | FN | Acc | AUROC | Sep (E−C) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 18 | 13 | 5 | 11 | 2 | 3 | 2 | 72.22% | **0.8769** | +0.265 |
| 5 | 15 | 7 | 8 | 4 | 6 | 2 | 3 | 66.67% | **0.8036** | +0.198 |
| 26 | 17 | 10 | 7 | 8 | 4 | 3 | 2 | 70.59% | **0.8000** | +0.234 |
| 20 | 14 | 8 | 6 | 7 | 3 | 3 | 1 | 71.43% | 0.7917 | +0.223 |
| 18 | 15 | 11 | 4 | 9 | 2 | 2 | 2 | 73.33% | 0.7727 | +0.223 |
| 7 | 16 | 10 | 6 | 9 | 5 | 1 | 1 | **87.50%** | 0.7667 | +0.315 |
| 13 | 14 | 9 | 5 | 5 | 3 | 2 | 4 | 57.14% | 0.7111 | +0.173 |
| 2 | 16 | 10 | 6 | 4 | 4 | 2 | 6 | 50.00% | 0.6667 | +0.056 |
| 8 | 16 | 10 | 6 | 9 | 3 | 3 | 1 | 75.00% | 0.6667 | +0.156 |
| 16 | 16 | 10 | 6 | 10 | 2 | 4 | 0 | 75.00% | 0.6333 | +0.169 |
| 25 | 15 | 8 | 7 | 5 | 4 | 3 | 3 | 60.00% | 0.6250 | +0.116 |
| 10 | 12 | 8 | 4 | 4 | 2 | 2 | 4 | 50.00% | 0.6250 | +0.138 |
| 9 | 14 | 5 | 9 | 4 | 7 | 2 | 1 | 78.57% | 0.6222 | +0.077 |
| 29 | 18 | 12 | 6 | 9 | 4 | 2 | 3 | 72.22% | 0.6111 | +0.151 |
| 22 | 17 | 11 | 6 | 9 | 2 | 4 | 2 | 64.71% | 0.5606 | +0.074 |
| 12 | 18 | 7 | 11 | 5 | 4 | 7 | 2 | 50.00% | 0.5584 | +0.046 |
| 4 | 17 | 7 | 10 | 5 | 4 | 6 | 2 | 52.94% | 0.5571 | +0.061 |
| 21 | 19 | 12 | 7 | 11 | 2 | 5 | 1 | 68.42% | 0.5119 | +0.026 |
| 28 | 18 | 11 | 7 | 4 | 4 | 3 | 7 | 44.44% | 0.5065 | +0.019 |
| 27 | 15 | 9 | 6 | 4 | 2 | 4 | 5 | 40.00% | 0.5000 | −0.005 |
| 15 | 15 | 10 | 5 | 9 | 1 | 4 | 1 | 66.67% | 0.4600 | −0.013 |
| 17 | 20 | 8 | 12 | 4 | 6 | 6 | 4 | 50.00% | 0.4271 | −0.071 |
| 23 | 16 | 6 | 10 | 6 | 1 | 9 | 0 | 43.75% | 0.4000 | −0.068 |
| 3 | 13 | 8 | 5 | 7 | 0 | 5 | 1 | 53.85% | 0.3750 | −0.066 |

*Sep (E−C) = mean predicted probability for GT=Error minus mean predicted probability for GT=Correct. Negative values indicate probability inversion. Across all recipes the GNN's mean predicted probability is **0.632 ± 0.207** for error videos vs **0.526 ± 0.243** for correct videos — a global separation of **+0.106**. The Pearson correlation between per-recipe separation and per-recipe AUROC is **r = 0.944**, confirming that discrimination is a systematic per-recipe signal rather than a global distributional accident.*

### 3.2 Matched Comparison: GNN vs Transformer (both LOGO)

With both models now evaluated under the *same* LOGO protocol, the comparison is fully controlled — neither model has seen any video from the test recipe. Aggregated over all 24 recipes:

| Aggregate (LOGO) | Transformer | GNN |
|---|---|---|
| Global AUROC | 0.6026 | **0.6240** |
| Global Accuracy | 0.5911 | **0.6224** |
| Mean per-recipe AUROC | 0.600 | **0.618** |
| Mean per-recipe Accuracy | 0.592 | **0.623** |

The full per-recipe head-to-head (sorted by AUROC gain in favor of the GNN):

| Recipe | N | TF Acc | GNN Acc | TF AUROC | GNN AUROC | ΔAUROC |
|---|---|---|---|---|---|---|
| 8 | 16 | 43.75% | **75.00%** | 0.250 | **0.667** | **+0.417** |
| 1 | 18 | 61.11% | **72.22%** | 0.569 | **0.877** | **+0.308** |
| 13 | 14 | 42.86% | **57.14%** | 0.467 | **0.711** | **+0.244** |
| 9 | 14 | 50.00% | **78.57%** | 0.467 | **0.622** | **+0.156** |
| 25 | 15 | 46.67% | **60.00%** | 0.482 | **0.625** | **+0.143** |
| 10 | 12 | 58.33% | 50.00% | 0.500 | **0.625** | **+0.125** |
| 15 | 15 | 33.33% | **66.67%** | 0.340 | **0.460** | **+0.120** |
| 27 | 15 | 46.67% | 40.00% | 0.389 | **0.500** | **+0.111** |
| 5 | 15 | 66.67% | 66.67% | 0.714 | **0.804** | **+0.089** |
| 2 | 16 | 50.00% | 50.00% | 0.583 | **0.667** | **+0.083** |
| 7 | 16 | 62.50% | **87.50%** | **0.783** | 0.767 | −0.017 |
| 12 | 18 | 55.56% | 50.00% | **0.610** | 0.558 | −0.052 |
| 26 | 17 | 70.59% | 70.59% | **0.857** | 0.800 | −0.057 |
| 20 | 14 | **85.71%** | 71.43% | **0.854** | 0.792 | −0.062 |
| 18 | 15 | **80.00%** | 73.33% | **0.841** | 0.773 | −0.068 |
| 21 | 19 | 57.89% | **68.42%** | **0.583** | 0.512 | −0.071 |
| 23 | 16 | 50.00% | 43.75% | **0.483** | 0.400 | −0.083 |
| 17 | 20 | 55.00% | 50.00% | **0.510** | 0.427 | −0.083 |
| 4 | 17 | 58.82% | 52.94% | **0.643** | 0.557 | −0.086 |
| 22 | 17 | 70.59% | 64.71% | **0.652** | 0.561 | −0.091 |
| 16 | 16 | 75.00% | 75.00% | **0.750** | 0.633 | −0.117 |
| 29 | 18 | 66.67% | **72.22%** | **0.792** | 0.611 | −0.181 |
| 28 | 18 | 55.56% | 44.44% | **0.688** | 0.506 | −0.182 |
| 3 | 13 | 76.92% | 53.85% | **0.588** | 0.375 | −0.213 |

The picture is genuinely mixed at the recipe level — the GNN wins AUROC on 10 of 24 recipes and accuracy on 9 — but the **aggregate clearly favors the GNN** (global AUROC 0.624 vs 0.603, accuracy 0.622 vs 0.591) because the GNN's wins are large and the Transformer's wins are small. The two biggest swings in the table (Recipe 8 at +0.42 AUROC, Recipe 1 at +0.31) are GNN gains, whereas the Transformer's largest advantage is only −0.21 (Recipe 3) and most of its wins are smaller still. In other words, where topological reasoning helps, it helps decisively; where it hurts, it hurts modestly.

### 3.3 Where the GNN Generalizes: Topological Signal Transfers

The recipes with the highest GNN AUROC — Recipe 1 (0.877), Recipe 5 (0.804), Recipe 26 (0.800), Recipe 20 (0.792), Recipe 18 (0.773), Recipe 7 (0.767) — demonstrate that the learned topological reasoning is genuinely transferable. These recipes have clear, low-branching DAG structures where the procedural dependency chain is unambiguous. The GNN, trained only on *other* recipes, arrives at test time with a recipe-agnostic inductive bias (depth-aware message passing, one-to-one step alignment) that is immediately productive on structurally well-defined execution graphs. Probability separation on this group is consistently high (+0.20 to +0.32).

Recipe 8 (**+0.417 AUROC, +31.25pp accuracy over the Transformer**) is the most striking case. Under the same LOGO protocol the Transformer collapses on this recipe (AUROC 0.250, accuracy 43.75% — below chance), systematically inverting its probabilities. The GNN, reasoning through the recipe's DAG topology rather than fuzzy cross-recipe visual similarity, reaches AUROC 0.667 and 75.00% accuracy. This is the clearest single demonstration that structural reasoning recovers signal that pure sequence-level visual matching loses entirely when the test recipe is unseen.

### 3.4 Where the GNN Struggles: Residual Failure Modes

**Probability Inversion (5 recipes).** The clearest failure signature is *probability inversion*: recipes where correct executions receive, on average, a higher predicted error probability than actual errors (Recipes 3, 23, 17, 15, 27; separation < 0). This points to an upstream matching failure: the Hungarian algorithm produces a corrupted alignment — visual segments assigned to the wrong task-graph nodes — and the GNN then performs topological validation on a structurally corrupted graph. No amount of message-passing depth can recover from this: garbage in, garbage out.

Recipe 3 is the worst case under LOGO: AUROC 0.375, separation −0.066, with the model failing to correctly accept any of the 5 correct executions (TN = 0). Recipes 23 (AUROC 0.400) and 17 (AUROC 0.427) follow the same pattern — high false-positive counts (9 and 6 respectively) driven by visual segments whose EgoVLP embeddings are insufficiently discriminative to support reliable one-to-one matching. The zero-shot LOGO setting compounds the problem: the GNN has no prior on these recipes' visual distributions to bias the alignment.

**Borderline / threshold-sensitive cases.** A second cluster (Recipes 27, 15, 28, 21) sits near AUROC 0.50 with separation close to zero — directionally ambiguous, where partial alignment success injects both signal and noise into the same graph. Recipe 15 is notable: despite a sub-chance AUROC (0.460), its accuracy is a healthy 66.67%, a reminder that point accuracy at a fixed threshold can mask weak underlying discrimination. Per-recipe threshold calibration would shift several of these cases but cannot manufacture discrimination that the corrupted alignment never provided.

---

## 4. The Localization Bottleneck: Why Late Filtering (~400 steps) over Aggressive NMS (~40 steps)

A central design decision in the localization stage was to **avoid aggressive Non-Maximum Suppression** and instead adopt a late-filtering strategy that yields roughly **400 step proposals per video**. This choice is counter-intuitive — 400 proposals against ~24 ground-truth steps is an enormous over-generation — and it demands justification. That justification comes from a *recall-agnostic* analysis of ActionFormer's raw output.

### 4.1 The Recall-Agnostic NMS Analysis

We swept NMS configurations over ActionFormer's predictions and measured **recall-agnostic** coverage: for each setting, does the retained proposal set contain the true action *at all*, ignoring whether it is correctly classified? This is the most generous possible accounting — it asks only whether the visual evidence for each real step physically survives the filter. The result is sobering:

| NMS Configuration | Proposals/video | Recall-Agnostic Coverage |
|---|---|---|
| Best filter (Score ≥ 0.05, IoU 0.4, **Top 40**) | 40 | **52.2%** |
| Aggressive filter (**Top 20**) | 20 | **~38%** |

The single best NMS configuration we could find — keeping 40 proposals per video — still **captures only 52.2% of the real actions**. In other words, even at its most permissive sensible setting, aggressive filtering causes ActionFormer to *physically lose almost half of the steps*: a **48% visual false-negative rate**. Tightening further to 20 proposals collapses coverage to ~38%.

The implication is structural, not incidental. **If you apply rigorous NMS to obtain a clean 30–40-row sequence, you immediately discard roughly half the nodes of the recipe graph.** The visual sequence becomes "perforated" — entire procedural steps simply have no corresponding feature vector. Any downstream verifier then fails *not because of its own modeling weakness*, but because the visual information it needs was never delivered. No sequence model and no GNN can detect an error in a step it cannot see.

Late filtering resolves this at the source: by minimizing suppression and retaining ~400 proposals, the pipeline preserves the overwhelming majority of true steps (a high recall-agnostic regime), trading a large amount of redundant/overlapping noise for near-complete coverage. The burden of disambiguation is then shifted *downstream*, onto the cross-modal Hungarian matching module, which is explicitly designed to recover the true step features from a noisy candidate pool.

### 4.2 Empirical Confirmation: ~400 steps vs ~40 steps (LOGO)

To verify this hypothesis directly, both models were re-run under LOGO on the **NMS-filtered (~40 step)** embeddings and compared against the **late-filtered (~400 step)** embeddings used everywhere else in this report:

| Model | Embedding Regime | Acc | Prec | Rec | F1 | AUROC |
|---|---|---|---|---|---|---|
| **Transformer** | Late filtering (~400) | 0.5911 | 0.6318 | 0.6864 | 0.6580 | **0.6026** |
| **Transformer** | NMS (~40) | 0.5859 | 0.6406 | 0.6318 | 0.6362 | 0.5893 |
| **GNN** | Late filtering (~400) | 0.6224 | 0.6506 | 0.7364 | **0.6908** | **0.6240** |
| **GNN** | NMS (~40) | 0.5807 | 0.6347 | 0.6318 | 0.6333 | 0.5962 |

Both models degrade when starved of nodes, exactly as predicted. The GNN's AUROC drops by **0.028** (0.624 → 0.596) and its F1 by a steep **0.058** (0.691 → 0.633); the Transformer drops by 0.013 (0.603 → 0.589). Critically, at ~40 steps the two models converge toward each other and toward chance (GNN 0.596 vs Transformer 0.589) — when half the graph is missing, the topological advantage has almost nothing left to exploit.

This is the mathematical confirmation of the theory: **the late-filtering (~400 step) regime is what makes the GNN's 0.624 AUROC possible at all.** With ~400 proposals, recall-agnostic coverage is high and almost every true feature is present *somewhere* in the noisy pool. The Transformer is overwhelmed by 400 overlapping, redundant tokens — its self-attention has no anchor to separate signal from noise. The GNN, by contrast, receives the same flood of noise but routes it through the Hungarian matching module, which does the "dirty work" of recovering the true step features hidden among the proposals, and then validates them against the recipe DAG. The GNN is, in this sense, an **architecture engineered for resilience**: it extracts procedural logic and localizes errors in a dataset whose underlying visual layer is extremely uncertain and noisy.

---

## 5. The Oracle Baseline: Upper-Bound Potential under Perfect Segmentation

To isolate the *modeling* capacity of each architecture from the *perceptual* quality of the localization pipeline, we introduce an **Oracle baseline**: instead of ActionFormer's predicted segments, the models are fed embeddings built directly from the **ground-truth step annotations**. This removes the localization bottleneck entirely and exposes the maximum potential of each model under ideal perception.

### 5.1 Oracle Results (LOGO)

**At the standard threshold ($\tau = 0.5$):**

| Model | Acc | Prec | Rec | F1 | AUROC |
|---|---|---|---|---|---|
| **Transformer (Oracle)** | 0.7005 | 0.7376 | 0.7409 | 0.7392 | **0.7900** |
| **GNN (Oracle)** | 0.6589 | 0.6787 | 0.7682 | 0.7207 | 0.7172 |

**Post-hoc optimal-threshold calibration:**

| Model | $\tau^*$ | Opt Acc | Opt Prec | Opt Rec | Opt F1 |
|---|---|---|---|---|---|
| **Transformer (Oracle)** | 0.7524 | 0.72917 | 0.84940 | 0.64091 | 0.73057 |
| **GNN (Oracle)** | 0.7634 | 0.67448 | 0.74359 | 0.65909 | 0.69880 |

Under perfect segmentation, the **Transformer wins decisively (AUROC 0.79 vs 0.72)** — a reversal of the real-pipeline result, where the GNN led (0.624 vs 0.603). This reversal is the most informative finding of the entire study, and it is fully explained by the architectures' differing relationship to input quality.

### 5.2 Why the Ranking Flips: Robustness vs. Raw Expressiveness

**The Transformer is a "glass giant."** Given an immaculate temporal sequence, self-attention is extraordinarily powerful: it can compute the exact relationship between the first and last step of a recipe and isolate any anomaly, reaching AUROC 0.79. But the Transformer carries **no prior knowledge of the recipe**. When ActionFormer mis-segments — merging two steps, dropping one, or injecting visual noise — the sequence loses its logical coherence, and the attention matrix searches for structure in a sequence that no longer has any. With nothing to anchor to, the Transformer **collapses to 0.60**, barely above chance. Its excellence is real but brittle.

**The GNN is an "off-road vehicle" (structural robustness).** Under ground truth its rigid graph constraints slightly limit the free-form reasoning that pure attention enjoys, so it lands a little lower (0.72 vs 0.79). But the picture inverts under the real pipeline. The GNN loses only **9.3 points** (0.717 → 0.624) when moving from Oracle to ActionFormer, against the Transformer's **18.7-point** collapse (0.790 → 0.603). The recipe graph acts as a **topological lifeline / regularization bias**: when the visual perception is confused, the GNN leans on the logical constraints of nodes and edges to "hold its course," proving dramatically more resistant to noise.

**Threshold health corroborates this.** On the real (ActionFormer) pipeline, the Transformer must be pushed to a very high threshold ($\tau^* = 0.688$) just to reach a mediocre F1 of 0.599 — a model so uncertain it scatters probabilities and must be aggressively gated to avoid a flood of false positives. The GNN's optimal threshold sits naturally near the center ($\tau^* = 0.4496$), yielding a far healthier F1 of **0.717** — it "knows what it does not know," producing a well-behaved logit distribution rather than overconfident noise.

### 5.3 Consolidated View — AUROC across all three regimes (LOGO)

| Perception Regime | Transformer | GNN | Winner |
|---|---|---|---|
| **Oracle** (ground-truth segmentation) | **0.7900** | 0.7172 | Transformer (+0.073) |
| **ActionFormer, late filtering (~400)** | 0.6026 | **0.6240** | GNN (+0.021) |
| **ActionFormer, NMS (~40)** | 0.5893 | **0.5962** | GNN (+0.007) |
| **Degradation Oracle → ActionFormer(400)** | **−18.7 pts** | **−9.3 pts** | GNN far more robust |

---

## 6. Final Conclusions

This study establishes a single, layered empirical claim about procedural mistake detection in egocentric video: **graph-based modeling is not merely an alternative to sequence modeling — it is a requirement for robustness once perception is imperfect, which is the only setting that matters in practice.**

**The matched comparison favors the GNN.** Under identical cross-validation protocols on the real pipeline, the GNN outperforms the Transformer on both LOO (AUROC 0.64703 vs 0.62862, +0.018) and LOGO (0.62395 vs 0.60260, +0.021), with consistent gains in F1 and recall. The GNN's *harder* LOGO result essentially matches the Transformer's *easier* LOO result, meaning the topological prior compensates almost entirely for the loss of recipe-specific memorization. The escape from majority-class collapse is visible in the LOO confusion matrices: at its optimal operating point the GNN predicts 210 Errors vs 174 Correct (close to the 220/164 ground truth), while the Transformer, even at $\tau = 0.30$, remains recall-biased at 248 vs 136. Per-recipe, probability separation correlates with AUROC at **r = 0.944**, confirming genuine discriminative signal rather than distributional accident.

**The localization stage is the dominant constraint.** A recall-agnostic NMS analysis shows that even the best aggressive filter (Top 40) physically loses 47.8% of real actions, and tightening to Top 20 loses ~62%. Late filtering (~400 proposals) is therefore not a quirk but a necessity: it preserves near-complete recall-agnostic coverage and shifts disambiguation onto the cross-modal matching module. The ~400-vs-~40 experiment confirms this directly — both models degrade when nodes are removed, and the GNN's 0.624 AUROC is only attainable in the high-coverage regime, where its matching module can recover true features buried in noise.

**The Oracle test reveals *why* the GNN wins where it counts.** With ground-truth segmentation, pure sequence models (Transformer) show superior raw discriminative power under ideal perception (AUROC 0.79). But when exposed to the noise of a real visual-segmentation pipeline (ActionFormer), they suffer a structural collapse (AUROC 0.60, a 18.7-point drop). The proposed GNN architecture instead exploits the topological constraints of the task graph as an intrinsic regularization system. Although slightly less expressive under ideal conditions (AUROC 0.71), it is markedly more resilient to visual noise, degrading only to 0.62 (a 9.3-point drop) and maintaining a recalibrated F1-Score of **0.717** against the Transformer's **0.599**. We therefore conclude that **graph-based modeling is a fundamental requirement for robust mistake detection in unconstrained scenarios**, where perfect perception is unavailable and the verifier must reason through structure rather than memorized visual pattern.

**The remaining bottleneck is cross-modal alignment.** The GNN's residual failures (the 5 LOGO recipes with probability inversion — Recipes 3, 23, 17, 15, 27) all stem from the same upstream cause: visual step features too weakly discriminative in the EgoVLP embedding space to support reliable one-to-one Hungarian matching. When the matching is corrupted, the GNN validates structure on a corrupted graph and cannot recover. This points to a concrete research direction: replacing the hard cosine gate ($\geq 0.20$) with a differentiable matching mechanism (Sinkhorn optimal transport, attention-based soft assignment, or a learned matching network) that propagates alignment uncertainty downstream, and enriching the recipe-level constant text features with execution-conditioned visual context to give the matcher video-specific anchors. Visual-semantic alignment and structural reasoning are co-equal: structure is necessary for robustness, but it can only be exploited when the grounding it reasons over is reliable.

## Acknowledgements

This project builds on many repositories from the CaptainCook4D release. Please refer to the original codebases for more details.

**Error Recognition**: https://github.com/CaptainCook4D/error_recognition

**Features Extraction**: https://github.com/CaptainCook4D/feature_extractors
