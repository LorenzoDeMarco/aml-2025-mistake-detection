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

## Acknowledgements

This project builds on many repositories from the CaptainCook4D release. Please refer to the original codebases for more details.

**Error Recognition**: https://github.com/CaptainCook4D/error_recognition

**Features Extraction**: https://github.com/CaptainCook4D/feature_extractors
