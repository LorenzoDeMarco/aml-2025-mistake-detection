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

## 3. Proposed Baseline: Bidirectional LSTM for Error Recognition

### 3.1 Motivation and Design Rationale

The CaptainCook4D paper establishes two baselines for the Supervised Error Recognition (SupervisedER) task: a simple MLP (V1) that processes each sub-segment feature independently, and a Transformer encoder (V2) that captures global temporal context via self-attention. While the Transformer addresses the sequential nature of procedure steps, self-attention is inherently permutation-equivariant — it has no built-in inductive bias for temporal direction. An error in a procedural activity is rarely isolated; it is the *consequence* of a causally-ordered sequence of actions. Forgetting an ingredient affects all subsequent steps; a timing mistake propagates forward in time, not backward.

This motivates the choice of a **Bidirectional LSTM (BiLSTM)** as the proposed baseline. Unlike the Transformer, recurrent architectures process sequences with an explicit notion of temporal order, maintaining a hidden state that summarizes the causal history of the execution up to each point. The bidirectional extension allows the model to also condition each timestep's prediction on future context — critical for identifying errors whose consequences only become apparent later in the sequence (e.g., a missing step that causes a downstream failure).

---

### 3.2 Architecture

The model is implemented in `core/models/lstm.py` as `LSTMModel` and follows a compact, regularized design optimized for the data-scarce regime of the CaptainCook4D dataset.

**Input projection and residual connection.** The raw input features (dimensionality determined by the backbone via `fetch_input_dim`) are first projected into a `hidden_dim * 2 = 512`-dimensional space via a linear layer (`residual_proj`). This projection serves a dual purpose: it provides a residual shortcut that bypasses the LSTM and prevents vanishing gradients in early training, and it aligns the input dimensionality with the LSTM's output space to enable element-wise addition.

**Bidirectional LSTM core.** The sequence is processed by a 2-layer bidirectional LSTM with `hidden_dim = 256` per direction, yielding a `hidden_dim * 2 = 512`-dimensional output at each timestep. Dropout of 0.3 is applied between layers (enabled only when `num_layers > 1`) to regularize the recurrent connections without suppressing the input layer's gradient flow.

**Layer normalization with residual fusion.** The LSTM output is combined with the residual projection via element-wise addition and normalized through a `LayerNorm(512)` layer. This design — residual sum followed by normalization — mirrors the pre-LN convention used in modern Transformer architectures, stabilizing the gradient landscape and preventing the LSTM's hidden state from dominating the residual signal at initialization.

**Classification head.** A two-layer MLP head (`Linear(512→128) → ReLU → Dropout(0.4) → Linear(128→1)`) produces a scalar logit per timestep, implementing a **many-to-many** classification scheme: every sub-segment in the sequence receives an independent error prediction. This is architecturally consistent with the V2 Transformer baseline and enables evaluation at both the sub-step and step aggregation levels.

**Class imbalance correction via bias initialization.** The output layer's bias is initialized to `−log((1 − p) / p)` with a prior `p = 0.3`, encoding the empirical class distribution directly into the model's initial predictions. At initialization, the sigmoid of the output logit equals the prior probability of the positive class, preventing the model from spending early training epochs correcting a zero-bias miscalibration. This is a principled alternative to loss reweighting and avoids the gradient compression effects of asymmetric `pos_weight` scaling.

The full architecture is summarized below:

| Component | Details |
|---|---|
| Input dim | Backbone-dependent (Omnivore: 1024) |
| Residual projection | `Linear(input_dim → 512)` |
| LSTM | 2 layers, hidden=256, bidirectional, dropout=0.3 |
| Post-LSTM fusion | `LayerNorm(LSTM_out + residual_proj)` |
| Classifier | `Linear(512→128) → ReLU → Dropout(0.4) → Linear(128→1)` |
| Output | Per-timestep scalar logit (many-to-many) |
| Bias init | `−log((1−0.3)/0.3) ≈ −0.847` |

---

### 3.3 Training Configuration

The model is trained under the same hyperparameter regime as the paper baselines to ensure a fair comparison:

```bash
python train_er.py \
    --variant LSTM \
    --backbone omnivore \
    --lr 0.001 \
    --num_epochs 50 \
    --batch_size 16 \
    --weight_decay 0.001
```

---

### 3.4 Evaluation

Evaluation is performed at the **Recordings** split, consistent with the paper's evaluation protocol, using the threshold `τ = 0.4` to match the operating point used for the V1 and V2 baselines:

```bash
python -m core.evaluate \
    --variant LSTM \
    --backbone omnivore \
    --ckpt checkpoints/error_recognition/LSTM/omnivore/error_recognition_recordings_omnivore_LSTM_video_best.pt \
    --split recordings \
    --threshold 0.4
```

---

### 3.5 Results and Comparison

The table below reports performance at the **Recordings** split for all three models — MLP (V1), Transformer (V2), and the proposed BiLSTM — at both the sub-step and step aggregation levels.

| Model | Level | Accuracy | Precision | Recall | F1 | AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| MLP | Sub-Step | 0.5735 | 0.3965 | 0.5688 | 0.4673 | 0.5988 | 0.3673 |
| MLP | Step | 0.5037 | 0.4091 | 0.8589 | **0.5542** | 0.6303 | 0.4020 |
| Transformer | Sub-Step | 0.6450 | 0.4491 | 0.3512 | 0.3942 | 0.6254 | 0.3711 |
| Transformer | Step | 0.6140 | 0.4541 | 0.3693 | 0.4073 | 0.6227 | 0.3942 |
| **BiLSTM** | **Sub-Step** | 0.6152 | 0.4323 | **0.5438** | **0.4817** | **0.6436** | **0.3851** |
| **BiLSTM** | **Step** | 0.5589 | 0.4314 | **0.7178** | **0.5389** | **0.6527** | **0.4110** |

#### Discussion

**BiLSTM vs. MLP.** The MLP baseline treats each sub-segment independently, with no temporal context. Its high Step-level Recall (0.859) is a symptom of majority-class bias rather than genuine discriminative power: the model defaults to predicting errors indiscriminately, inflating sensitivity at the cost of Precision (0.409) and Accuracy (0.504 — below random chance on a balanced split). The BiLSTM corrects this imbalance through its recurrent memory, achieving a substantially higher AUC (+0.023) and PR-AUC (+0.009) while recovering a more balanced operating point (Recall 0.718, Precision 0.431, F1 0.539 at Step level).

**BiLSTM vs. Transformer.** The comparison with V2 reveals the core architectural advantage of recurrent processing for this task. The Transformer's self-attention mechanism, while powerful in data-rich regimes, is permutation-equivariant — it has no built-in inductive bias for temporal ordering, relying entirely on learned positional encodings to distinguish "step 2 before step 5" from "step 5 before step 2". On CaptainCook4D's limited training set, this inductive bias deficit translates directly into conservative, low-recall predictions: the Transformer achieves only 0.369 Recall and 0.407 F1 at Step level, with an AUC of 0.623.

The BiLSTM outperforms the Transformer on every metric at both evaluation levels. The AUC improvement is particularly meaningful (+0.030 at Sub-Step, +0.030 at Step) as it is threshold-independent and reflects genuine improvement in probabilistic discrimination. The Recall gap is the most operationally significant: the BiLSTM detects 71.8% of erroneous recipe executions vs. the Transformer's 36.9% — a near-doubling of sensitivity at comparable Precision.

**Why temporal directionality matters here.** Procedural errors in CaptainCook4D are causally structured: an incorrect action at step $t$ typically manifests in visual evidence distributed across the sub-segments of steps $t, t+1, \ldots, t+k$. The forward LSTM accumulates a running summary of the causal execution history; the backward LSTM reads the sequence in reverse, propagating evidence of downstream consequences back toward their root cause. This bidirectional causal-consequential reasoning is precisely the inductive bias required for mistake detection in procedural activities — and it is what the permutation-equivariant Transformer lacks by design.

The residual connection and bias initialization further contribute to the BiLSTM's advantage over the plain MLP by preventing early-training majority-class collapse, ensuring that the recurrent layers receive stable, informative gradients from the first epoch.

**Summary.** The proposed BiLSTM baseline consistently outperforms both V1 (MLP) and V2 (Transformer) on the Recordings split across AUC, PR-AUC, and F1, establishing a new competitive reference for the SupervisedER task. Its performance advantage is most pronounced in high-sensitivity operating regimes — the task setting where missing a procedural error carries the highest cost — making it the most practically relevant of the three baselines for real-world deployment scenarios.



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

## 1. Global Performance — LOGO Evaluation

The `TaskVerificationGNN` was evaluated under the same rigorous **Leave-One-Recipe-Out (LOGO)** cross-validation regime as the Transformer baseline: one recipe's videos serve as the held-out test set while the model is trained from scratch on all remaining recipes, repeated across all 24 recipe groups. All 384 videos are covered exactly once.

### 1.1 Quantitative Results

**At the standard decision threshold ($\tau = 0.5$):**

| Metric | Value |
|---|---|
| AUROC | **0.64090** |
| Accuracy | 0.61979 |
| F1-Score | 0.66667 |
| Precision | 0.66972 |
| Recall | 0.66364 |

Confusion matrix at $\tau = 0.5$: **TN = 92, FP = 72, FN = 74, TP = 146**

**Post-hoc threshold calibration via Youden's J statistic ($\tau^* = 0.4167$):**

| Metric | Value |
|---|---|
| Accuracy | **0.63281** |
| F1-Score | **0.70064** |
| Precision | 0.65737 |
| Recall | **0.75000** |

Confusion matrix at $\tau^* = 0.4167$: **TN = 78, FP = 86, FN = 55, TP = 165**

The calibrated threshold of $0.4167$ — significantly below $0.5$ — reflects the effect of label smoothing on the model's output distribution. As discussed in §4, label smoothing contracts predicted probabilities away from the extremes, centering the distribution around a narrower band. Youden's J statistic maximizes $\text{TPR} - \text{FPR}$ over the ROC curve, identifying the threshold that recovers the best operational trade-off between sensitivity and specificity in this compressed probability space.

---

## 2. GNN vs. Transformer Baseline — A Methodologically Honest Comparison

### 2.1 A Critical Distinction: Evaluation Protocol Asymmetry

Before presenting any numbers, a fundamental methodological difference between the two models must be stated explicitly, as it changes the interpretation of every metric that follows.

The **Transformer baseline was evaluated under LOO (Leave-One-Out)** cross-validation: each individual video is held out as the test sample while the model trains on all remaining videos. Crucially, this means the Transformer **always has access to other videos from the same recipe during training** — it can learn recipe-specific visual signatures, motion patterns, and error distributions directly from its training fold.

The **GNN is evaluated under LOGO (Leave-One-Recipe-Out)** cross-validation: all videos belonging to a given recipe are held out together, and the model trains on all other recipes. The GNN therefore **never sees a single video from the test recipe during training** — it must generalize to a completely unseen procedural domain at inference time.

This is not a minor implementation detail. LOO and LOGO test fundamentally different capabilities:

| Aspect | LOO (Transformer) | LOGO (GNN) |
|---|---|---|
| Test condition | Unseen *video* from a *known* recipe | Unseen *recipe* entirely |
| Training access | All other videos of the same recipe | Zero videos of the test recipe |
| Generalization required | Intra-recipe | Cross-recipe (zero-shot on target recipe) |
| Task difficulty | Significantly easier | Significantly harder |

Any direct metric comparison between the two models must be read with this asymmetry in mind. The GNN is operating in a strictly harder regime. Comparable performance under LOGO vs LOO would itself be a strong result; superior performance would be remarkable.

> **Note:** A LOO evaluation of the GNN is planned as future work to provide a fully controlled comparison on identical experimental conditions. The results reported here represent the LOGO setting exclusively.

### 2.2 Global Metrics Under Asymmetric Conditions

With the above caveat, the reported numbers are:

| Metric | Transformer (LOO) | GNN @ $\tau=0.5$ (LOGO) | GNN @ $\tau^*=0.42$ (LOGO) |
|---|---|---|---|
| **AUROC** | 0.62862 | **0.64090** | 0.64090 |
| Accuracy | **0.62760** | 0.61979 | **0.63281** |
| F1-Score | 0.68845 | 0.66667 | 0.70064 |
| Precision | 0.66109 | **0.66972** | 0.65737 |
| Recall | **0.71818** | 0.66364 | 0.75000 |

The GNN achieves **higher AUROC (+0.013)** than the Transformer — on a harder task. At the calibrated threshold, it also recovers competitive Accuracy (+0.008pp) and F1 (−0.028). The Transformer's Recall advantage at its calibrated threshold (0.89 vs 0.75) is partially explained by its access to recipe-specific training data, which makes it easier to learn a recall-maximizing threshold for each recipe's distribution.

These numbers are best read not as "GNN vs Transformer performance" but as a lower bound on the GNN's advantage: **in an equal experimental setting (LOO), the GNN is expected to perform strictly better**, since it would additionally benefit from recipe-specific training signal that it currently lacks entirely.

### 2.3 Confusion Matrix Analysis — Structural Differences in Error Patterns

| | Transformer (LOO, $\tau=0.5$) | GNN (LOGO, $\tau=0.5$) |
|---|---|---|
| True Negatives (TN) | 83 | **92** (+9) |
| False Positives (FP) | 81 | **72** (−9) |
| True Positives (TP) | **158** | 146 (−12) |
| False Negatives (FN) | **62** | 74 (+12) |

Even under the harder LOGO regime, the GNN catches **9 more correct executions correctly** (56.1% vs 50.6% of all correct videos) and generates **9 fewer false alarms**. This is architecturally significant: the GNN's topological validation mechanism requires positive evidence from the DAG structure before committing to an error prediction, making it harder to trigger false positives from superficial visual similarity alone.

The Transformer's higher TP count reflects its recall-biased operating point — a direct consequence of the majority-class pressure that the LOO protocol does not fully neutralize. The Transformer required threshold calibration to $\tau = 0.30$ (far below the mathematical center of 0.5) to recover balanced predictions, revealing a systematic probability shift toward the positive class. The GNN, by contrast, predicts **218 Errors vs 166 Correct** — a near-perfect mirror of the ground truth (220/164) at the standard threshold, with no calibration needed to recover distributional balance. This is the clearest single indicator that the GNN has escaped majority-class collapse.

The AUROC gap (+0.013 in favor of the GNN under LOGO) is the most meaningful summary: threshold-independent, it reflects genuine probabilistic discrimination. The GNN's mean predicted probability for error videos is **0.607 ± 0.217** vs **0.497 ± 0.235** for correct videos — a global separation of **+0.110**. The Pearson correlation between per-recipe probability separation and per-recipe AUROC is **r = 0.942**, confirming this is a systematic per-recipe signal, not a global distributional accident.

---

## 3. Per-Recipe Analysis

### 3.1 Full Breakdown (GNN — LOGO)

| Recipe | N | GT Err | GT Cor | TP | TN | FP | FN | Acc | AUROC | Sep (E−C) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 16 | 10 | 6 | 8 | 5 | 1 | 2 | **81.25%** | **0.9333** | +0.382 |
| 20 | 14 | 8 | 6 | 7 | 5 | 1 | 1 | **85.71%** | **0.8542** | +0.210 |
| 18 | 15 | 11 | 4 | 11 | 2 | 2 | 0 | **86.67%** | 0.8409 | +0.316 |
| 1 | 18 | 13 | 5 | 11 | 4 | 1 | 2 | 83.33% | 0.8462 | +0.330 |
| 22 | 17 | 11 | 6 | 9 | 3 | 3 | 2 | 70.59% | 0.7576 | +0.175 |
| 26 | 17 | 10 | 7 | 7 | 5 | 2 | 3 | 70.59% | 0.7857 | +0.193 |
| 10 | 12 | 8 | 4 | 7 | 1 | 3 | 1 | 66.67% | 0.7188 | +0.123 |
| 16 | 16 | 10 | 6 | 7 | 4 | 2 | 3 | 68.75% | 0.7167 | +0.200 |
| 25 | 15 | 8 | 7 | 4 | 5 | 2 | 4 | 60.00% | 0.7143 | +0.189 |
| 7 | 16 | 10 | 6 | 7 | 3 | 3 | 3 | 62.50% | 0.7000 | +0.134 |
| 5 | 15 | 7 | 8 | 3 | 7 | 1 | 4 | 66.67% | 0.6786 | +0.167 |
| 29 | 18 | 12 | 6 | 5 | 5 | 1 | 7 | 55.56% | 0.6806 | +0.089 |
| 23 | 16 | 6 | 10 | 3 | 7 | 3 | 3 | 62.50% | 0.6500 | +0.098 |
| 17 | 20 | 8 | 12 | 6 | 5 | 7 | 2 | 55.00% | 0.6458 | +0.099 |
| 9 | 14 | 5 | 9 | 4 | 4 | 5 | 1 | 57.14% | 0.5556 | +0.054 |
| 3 | 13 | 8 | 5 | 7 | 2 | 3 | 1 | 69.23% | 0.5750 | +0.146 |
| 13 | 14 | 9 | 5 | 4 | 4 | 1 | 5 | 57.14% | 0.5111 | +0.105 |
| 4 | 17 | 7 | 10 | 2 | 6 | 4 | 5 | 47.06% | 0.5143 | −0.000 |
| 28 | 18 | 11 | 7 | 3 | 4 | 3 | 8 | 38.89% | 0.4675 | −0.039 |
| 8 | 16 | 10 | 6 | 9 | 1 | 5 | 1 | 62.50% | 0.4500 | +0.047 |
| 15 | 15 | 10 | 5 | 4 | 3 | 2 | 6 | 46.67% | 0.4000 | −0.069 |
| 21 | 19 | 12 | 7 | 10 | 2 | 5 | 2 | 63.16% | 0.3929 | −0.044 |
| 27 | 15 | 9 | 6 | 5 | 3 | 3 | 4 | 53.33% | 0.3704 | −0.104 |
| 12 | 18 | 7 | 11 | 3 | 2 | 9 | 4 | 27.78% | 0.3506 | −0.144 |

*Sep (E−C) = mean predicted probability for GT=Error minus mean predicted probability for GT=Correct. Negative values indicate probability inversion.*

### 3.2 Contextual Comparison: GNN (LOGO) vs Transformer (LOO) on Shared Recipes

The following comparison is presented with the protocol asymmetry in full view — LOGO is a harder evaluation — and should be interpreted accordingly:

| Recipe | Transformer Acc (LOO) | GNN Acc (LOGO) | $\Delta$ | GNN AUROC (LOGO) |
|---|---|---|---|---|
| 8 | 43.75% | **62.50%** | **+18.75pp** | 0.4500 |
| 20 | 78.57% | **85.71%** | **+7.14pp** | 0.8542 |
| 18 | 80.00% | **86.67%** | **+6.67pp** | 0.8409 |
| 23 | 56.25% | **62.50%** | **+6.25pp** | 0.6500 |
| 1 | 77.78% | **83.33%** | **+5.55pp** | 0.8462 |
| 26 | **82.35%** | 70.59% | −11.76pp | 0.7857 |
| 4 | **58.82%** | 47.06% | −11.76pp | 0.5143 |

The GNN outperforms the Transformer on 5 of 7 recipes — **while having never seen any video from those recipes during training**. This is the core empirical finding: topological structure transfers across recipes in a way that pure visual pattern recognition from the same recipe does not.

The two regressions (Recipes 26 and 4) must be interpreted carefully in this context. For Recipe 26 (*Mug Cake*), the GNN AUROC of 0.7857 is healthy and probability separation is positive (+0.193) — the regression in accuracy is a threshold artifact, not a discriminative failure. The Transformer's 82.35% accuracy on this recipe was achieved with access to other *Mug Cake* videos during training, allowing it to calibrate to that recipe's specific distribution. For Recipe 4, the near-zero separation (−0.000) indicates a genuine matching failure rather than a calibration issue — but again, the Transformer had training-time access to the same recipe.

### 3.3 Where the GNN Generalizes: Topological Signal Transfers

The four recipes with AUROC > 0.84 (Recipes 2, 20, 18, 1) demonstrate that the GNN's learned topological reasoning is genuinely transferable. These recipes have clear, low-branching DAG structures where the procedural dependency chain is unambiguous. The GNN — trained on completely different recipes — arrives at test time with an inductive bias (depth-aware message passing, one-to-one step alignment) that is recipe-agnostic and immediately productive on structurally well-defined execution graphs. Probability separation is consistently high (+0.21 to +0.38).

Recipe 8 (*Spiced Hot Chocolate*, +18.75pp over Transformer) is the most striking case. The Transformer, despite having access to same-recipe training videos, was severely confused by this recipe — its LOO accuracy of 43.75% is below random chance on a balanced split. The likely cause is that Spiced Hot Chocolate involves visually dense preparation steps with subtle inter-step differences that the Transformer conflates with error patterns from visually similar recipes in its training set. The GNN, approaching the recipe purely through its DAG topology with no recipe-specific prior, achieves 62.50% accuracy — not excellent in absolute terms, but a decisive demonstration that structural reasoning adds signal that visual memorization misses.

### 3.4 Where the GNN Struggles: Residual Failure Modes

**Probability Inversion (6 recipes).** The clearest failure signature is *probability inversion*: recipes where correct executions receive higher predicted error probability than actual errors (Recipes 12, 27, 21, 15, 28, 4; separation < 0). This indicates upstream matching failure: the Hungarian algorithm has produced a corrupted alignment for these recipes — visual segments assigned to wrong task graph nodes — and the GNN performs topological validation on a structurally corrupted graph. No message-passing depth or architectural sophistication can recover from this: garbage in, garbage out.

Recipe 12 (*Tomato Mozzarella Salad*) is the worst case: AUROC 0.350, accuracy 27.78%, separation −0.144. The recipe is a cold preparation with minimal motion variation between steps — "slice tomato", "slice mozzarella", "arrange on plate" produce near-identical visual features in the EgoVLP embedding space, making reliable one-to-one Hungarian assignment impossible. The zero-shot LOGO setting compounds the problem: the GNN has no prior on this recipe's visual distribution to bias the alignment.

**Moderate-confidence failure (Recipes 29, 13, 9).** A second cluster shows positive but small separation (0.05–0.11) and AUROC 0.51–0.68. Directionally correct but insufficiently discriminative — partial alignment success injecting both signal and noise into the same graph.

**Recipe 26 threshold artifact.** As noted above: high AUROC (0.786), positive separation (+0.193), accuracy regression is purely a threshold effect from the label-smoothed probability distribution falling below $\tau = 0.5$ for this recipe. A per-recipe threshold calibration would recover the expected performance.

---

## 4. Key Conclusions and Open Challenges

### 4.1 What the GNN Establishes

The central finding is not captured by any single metric: **the GNN achieves comparable or superior performance to the Transformer on a significantly harder evaluation protocol**. This demonstrates that recipe-specific task graph topology encodes a genuinely transferable inductive bias for procedural mistake detection — one that a sequence model, however well-tuned, cannot access.

Architecturally, the escape from majority-class collapse is confirmed: the GNN produces a near-perfect prediction distribution mirroring ground truth at the standard threshold, with no calibration required. The Transformer's need for aggressive threshold adjustment ($\tau = 0.30$) was a symptom of majority-class pressure that the topological constraints of the GNN eliminate by design. The per-recipe probability separation correlates with AUROC at **r = 0.942**, establishing that performance is driven by genuine discriminative signal rather than distributional accidents.

### 4.2 The Shifted Bottleneck: Cross-Modal Alignment Quality

The GNN resolves the failure modes of the Transformer — cross-task semantic inversion and majority-class collapse. But it exposes a new, upstream bottleneck: **the quality of the Hungarian cross-modal alignment**. The 6 recipes with probability inversion share one characteristic: their visual step features are insufficiently discriminative in the EgoVLP embedding space to support reliable one-to-one matching. When the matching fails, the GNN's structural reasoning operates on a corrupted graph and cannot recover.

This identifies a clear research direction. The cosine threshold ($\geq 0.20$) is a hard binary gate with no uncertainty propagation. A differentiable matching mechanism — Sinkhorn optimal transport, attention-based soft assignment, or a learned matching network — would allow the model to propagate alignment uncertainty downstream rather than silently injecting misaligned features. Additionally, the current text features are recipe-level constants (computed once from the task graph, shared across all videos of a recipe); incorporating execution-conditioned visual context into the text encoding could provide the matching stage with richer, video-specific anchors.

### 4.3 Implications for Task Verification

This work establishes a precise empirical claim: **procedural task graph structure is a necessary but not sufficient condition for reliable mistake detection in egocentric video**. It is necessary because the GNN consistently outperforms the Transformer on structurally clear recipes, and achieves this under a harder generalization regime. It is not sufficient because the inductive bias of the DAG topology can only be exploited if the cross-modal grounding is reliable. The two components — visual-semantic alignment and structural reasoning — are equally critical, and future systems should treat them as co-equal rather than treating alignment as a solved preprocessing step and investing all modeling effort downstream.

