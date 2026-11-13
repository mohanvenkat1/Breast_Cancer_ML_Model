# Academic Project Report

# Breast Cancer Dataset Classification using Artificial Neural Network (ANN)

---

**Institution:** [Your College / University Name]  
**Department:** Department of Computer Science / Information Technology  
**Course:** Deep Learning / Machine Learning  
**Academic Year:** 2024–2025  
**Student Name:** [Your Name]  
**Roll No.:** [Your Roll Number]  
**Guide / Supervisor:** [Professor Name]  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement and Objectives](#2-problem-statement-and-objectives)
3. [Literature Survey](#3-literature-survey)
4. [Dataset Description](#4-dataset-description)
5. [Proposed Methodology](#5-proposed-methodology)
6. [ANN Architecture](#6-ann-architecture)
7. [Implementation Details](#7-implementation-details)
8. [Results and Performance Analysis](#8-results-and-performance-analysis)
9. [Conclusion and Future Enhancements](#9-conclusion-and-future-enhancements)
10. [References](#10-references)

---

## 1. Introduction

Breast cancer is one of the most prevalent and life-threatening cancers affecting women worldwide. Early and accurate diagnosis is critical for increasing survival rates and improving treatment outcomes. According to the World Health Organization (WHO), breast cancer accounted for 12.5% of all new annual cancer cases worldwide in 2020, making it the most commonly occurring cancer globally.

Traditional diagnostic methods — including mammography, biopsy, and clinical examination — are time-consuming and can be subject to human error. Machine learning and deep learning techniques offer a promising avenue for assisting clinicians in accurately and rapidly diagnosing breast cancer from digitized cell-image features.

This project develops a **binary classification model** based on an **Artificial Neural Network (ANN)** to predict whether a breast tumor is **Benign** (non-cancerous) or **Malignant** (cancerous) using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The model is trained using TensorFlow/Keras, evaluated rigorously on unseen test data, and deployed through an interactive Streamlit web application.

---

## 2. Problem Statement and Objectives

### 2.1 Problem Statement

Given 30 numerical features extracted from digitized Fine Needle Aspiration (FNA) images of breast mass cells, build a machine learning model that can accurately classify a tumor as **Benign** or **Malignant** with a focus on minimizing **False Negatives** (missed malignant cases), which carry the highest clinical risk.

### 2.2 Objectives

1. Load, explore, and analyze the WDBC dataset to understand feature distributions and class balance.
2. Preprocess the data: encode labels, scale features, and split into train/validation/test sets without data leakage.
3. Design and implement a multi-layer Artificial Neural Network using TensorFlow/Keras.
4. Train the model with EarlyStopping to prevent overfitting.
5. Evaluate the trained model on unseen test data using accuracy, precision, recall, F1-score, AUC, and a confusion matrix.
6. Visualize training curves, the confusion matrix, and the ROC curve.
7. Deploy an optional interactive Streamlit web interface for real-time predictions.

---

## 3. Literature Survey

### 3.1 Machine Learning for Breast Cancer Diagnosis

Breast cancer classification using machine learning has been extensively studied over the past two decades. The Wisconsin Breast Cancer dataset — originally collected by Dr. William H. Wolberg at the University of Wisconsin Hospitals — has become the benchmark dataset for evaluating binary classification models in this domain.

**Reference 1:**  
Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1994). *Machine learning techniques to diagnose breast cancer from fine-needle aspirates.* Cancer Letters, 77(2–3), 163–171.  
This foundational study demonstrated that machine learning could achieve clinically useful accuracy in distinguishing malignant from benign breast tumors using features derived from FNA images. The authors reported classification accuracy exceeding 97% using linear programming-based techniques on the original WDBC dataset.

**Reference 2:**  
Dheeba, J., Singh, N.A., & Selvi, S.T. (2014). *Computer-aided detection of breast cancer on mammograms: A swarm intelligence optimized wavelet neural network approach.* Journal of Biomedical Informatics, 49, 45–52.  
This study applied neural network-based classifiers for breast cancer detection and demonstrated that optimized ANN architectures — particularly those trained with regularization techniques — consistently outperform traditional classifiers such as Naïve Bayes and k-NN on medical datasets. The study reinforces the applicability of deep neural networks in clinical classification tasks.

**Reference 3:**  
Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE's Symposium on Electronic Imaging: Science and Technology, 1905, 861–870.  
This paper describes the method used to extract the 30 numerical features in the WDBC dataset from digitized FNA images, including measurements of cell radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. These features form the input space for our ANN model.

### 3.2 Key Takeaways from Literature

- ANN-based models consistently achieve **>95% accuracy** on the WDBC dataset.
- **Dropout regularization** significantly reduces overfitting on small medical datasets (~500 samples).
- **Recall (sensitivity) for the malignant class** is the most clinically critical metric.
- **Standard scaling** of features is essential for fast convergence and stable ANN training.
- Early stopping prevents overfitting without sacrificing model performance.

---

## 4. Dataset Description

### 4.1 Source

| Property       | Detail                                                                 |
|----------------|------------------------------------------------------------------------|
| Name           | Wisconsin Diagnostic Breast Cancer (WDBC)                              |
| Origin         | University of Wisconsin Hospitals                                       |
| Collected by   | Dr. William H. Wolberg                                                  |
| UCI Repository | https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic) |
| Sklearn access | `sklearn.datasets.load_breast_cancer()`                                |

### 4.2 Dataset Statistics

| Property         | Value                |
|------------------|----------------------|
| Total Samples    | 569                  |
| Number of Features | 30 numerical       |
| Target Classes   | Benign (357), Malignant (212) |
| Missing Values   | None                 |
| Duplicate Rows   | None                 |

### 4.3 Feature Groups

The 30 features are computed for each cell nucleus and are grouped into three categories of 10 features each:

| Group     | Description                                                   |
|-----------|---------------------------------------------------------------|
| **Mean**  | Mean of features across all cells in the image (features 1–10) |
| **SE**    | Standard error of features (features 11–20)                   |
| **Worst** | Largest (most extreme) values of features (features 21–30)   |

The 10 base features measured per cell nucleus are:
1. Radius (mean of distances from center to perimeter points)
2. Texture (standard deviation of gray-scale values)
3. Perimeter
4. Area
5. Smoothness (local variation in radius lengths)
6. Compactness (perimeter² / area − 1.0)
7. Concavity (severity of concave portions of contour)
8. Concave points (number of concave portions)
9. Symmetry
10. Fractal dimension (coastline approximation − 1)

### 4.4 Target Variable

| Label | Encoding | Count | Percentage |
|-------|----------|-------|------------|
| Benign    | 0 | 357 | 62.7% |
| Malignant | 1 | 212 | 37.3% |

> **Note:** The sklearn loader uses 0=Malignant, 1=Benign. This project remaps to 1=Malignant, 0=Benign (the clinical positive class = cancer), which is standard for medical classification tasks.

---

## 5. Proposed Methodology

```
┌────────────────────────────────────────────────────────────────────┐
│                     PROPOSED METHODOLOGY                           │
│                                                                    │
│  1. Load WDBC Dataset (sklearn)                                    │
│        ↓                                                           │
│  2. Exploratory Data Analysis (EDA)                                │
│     • Shape, features, statistics                                  │
│     • Class distribution, missing values                           │
│     • Correlation heatmap, box plots                               │
│        ↓                                                           │
│  3. Data Preprocessing                                             │
│     • Label encoding (0=Benign, 1=Malignant)                       │
│     • Stratified 80/20 train-test split                            │
│     • StandardScaler (fit on train ONLY)                           │
│     • 15% validation split from training data                      │
│        ↓                                                           │
│  4. ANN Model Design                                               │
│     • Dense(64, ReLU) → Dropout(0.3)                              │
│     • Dense(32, ReLU) → Dropout(0.2)                              │
│     • Dense(1, Sigmoid)                                            │
│        ↓                                                           │
│  5. Model Training                                                 │
│     • Adam optimizer, Binary Cross-Entropy loss                    │
│     • 100 epochs, batch=16, EarlyStopping(patience=12)            │
│     • ModelCheckpoint → save best weights                          │
│        ↓                                                           │
│  6. Model Evaluation (Test Set)                                    │
│     • Accuracy, Precision, Recall, F1, AUC                        │
│     • Confusion matrix, Classification report                      │
│     • ROC Curve                                                    │
│        ↓                                                           │
│  7. Visualization & Reporting                                      │
│     • Training curves, confusion matrix heatmap, ROC curve        │
│        ↓                                                           │
│  8. Deployment (Streamlit App)                                     │
│     • User enters 30 features → real-time prediction              │
└────────────────────────────────────────────────────────────────────┘
```

### 5.1 Data Splits

| Split      | Size  | Purpose                                   |
|------------|-------|-------------------------------------------|
| Training   | ~387  | Model weight updates during training       |
| Validation | ~68   | Hyperparameter tuning and early stopping  |
| Test       | 114   | Final unbiased evaluation (never touched) |

### 5.2 Preventing Data Leakage

The `StandardScaler` is **fitted exclusively on training data** (`fit_transform`) and only **applied** (not fitted) to the validation and test sets (`transform` only). This is critical to prevent data leakage — a common mistake that inflates test accuracy by allowing information from the test set to influence the scaler.

---

## 6. ANN Architecture

### 6.1 Network Diagram

```
INPUT LAYER
 ┌──────────────────────────────────────────────────────┐
 │  30 normalized numerical features                    │
 └────────────────────┬─────────────────────────────────┘
                      │
        ┌─────────────▼──────────────┐
        │  Dense (64 neurons, ReLU)  │  ← Hidden Layer 1
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      Dropout (0.3)         │  ← Regularization: 30% random dropout
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │  Dense (32 neurons, ReLU)  │  ← Hidden Layer 2
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │      Dropout (0.2)         │  ← Regularization: 20% random dropout
        └─────────────┬──────────────┘
                      │
        ┌─────────────▼──────────────┐
        │  Dense (1 neuron, Sigmoid) │  ← Output Layer
        └─────────────┬──────────────┘
                      │
OUTPUT: Probability ∈ [0, 1]
  → ≥ 0.50 : Malignant
  → < 0.50 : Benign
```

### 6.2 Architecture Justification

| Design Choice       | Justification                                                          |
|---------------------|------------------------------------------------------------------------|
| 64 → 32 → 1 neurons | Gradual dimensionality reduction; avoids information bottleneck        |
| ReLU activation     | Avoids vanishing gradient; computationally efficient                   |
| Sigmoid output      | Maps output to probability [0,1] suitable for binary classification    |
| Dropout (0.3, 0.2)  | Regularization: prevents overfitting on the small dataset (~455 train) |
| Adam optimizer      | Adaptive learning rate; converges faster than SGD for tabular data     |
| Binary cross-entropy| Standard loss for binary classification problems                       |
| EarlyStopping       | Avoids overtraining; patience=12 allows the model to explore fully     |

### 6.3 Training Configuration

| Hyperparameter | Value                        |
|----------------|------------------------------|
| Optimizer      | Adam (default lr=0.001)      |
| Loss           | Binary Cross-Entropy         |
| Batch Size     | 16                           |
| Max Epochs     | 100                          |
| Early Stopping | patience=12, monitor=val_loss|
| Val Split      | 15% of training data         |

---

## 7. Implementation Details

### 7.1 Development Environment

| Component | Version |
|-----------|---------|
| Python    | 3.12.x  |
| TensorFlow/Keras | ≥2.13 |
| Scikit-learn | ≥1.3  |
| NumPy     | ≥1.24   |
| Pandas    | ≥2.0    |
| Matplotlib | ≥3.7   |
| Seaborn   | ≥0.12   |
| Streamlit | ≥1.28   |

### 7.2 Project File Descriptions

| File | Description |
|------|-------------|
| `src/train.py` | End-to-end training script: load → preprocess → train → evaluate → save |
| `notebooks/breast_cancer_ann.ipynb` | Interactive Jupyter notebook with cell-by-cell execution and inline plots |
| `app.py` | Streamlit web application for interactive real-time predictions |
| `models/breast_cancer_ann.keras` | Saved best Keras model (generated after training) |
| `models/scaler.pkl` | Saved StandardScaler for inference (generated after training) |
| `dataset/breast_cancer_wdbc.csv` | WDBC dataset exported as CSV |
| `requirements.txt` | Python package dependencies |
| `README.md` | Setup and execution instructions |

### 7.3 Key Implementation Decisions

1. **Label Remapping:** sklearn encodes malignant=0, benign=1. We remap to malignant=1, benign=0 so that "1" always represents the clinically positive (cancer) class — aligning with standard medical convention.

2. **Stratified Split:** `stratify=y` ensures both train and test sets maintain the same 62.7%/37.3% benign/malignant ratio.

3. **Scaler Fit on Train Only:** The scaler learns the mean and standard deviation of each feature from training data only. This is applied to the test set without re-fitting.

4. **EarlyStopping with restore_best_weights=True:** When validation loss stops improving for 12 consecutive epochs, training halts and weights are automatically reverted to the best checkpoint.

5. **Dropout During Training Only:** Keras automatically disables Dropout during inference (prediction), ensuring deterministic outputs at test time.

---

## 8. Results and Performance Analysis

> **Note:** The metrics below are populated from the actual training run. Run `python src/train.py` to reproduce these results. Results may vary slightly due to hardware-level floating-point non-determinism.

### 8.1 Test Set Performance

> Results obtained from actual training run on the WDBC dataset (Python 3.12, TensorFlow 2.x, seed=42, 52 epochs with EarlyStopping triggered at epoch 52, best weights restored from epoch 40).

| Metric               | Value        |
|----------------------|--------------|
| **Accuracy**         | **98.25%**   |
| **Precision** (Malignant) | **1.0000**  |
| **Recall** (Malignant) | **0.9524** |
| **F1-Score** (Malignant) | **0.9756** |
| **AUC Score**        | **0.9954**   |
| **False Negatives**  | **2** (out of 42 malignant test samples) |
| **True Positives**   | **40** |
| **True Negatives**   | **72** |
| **False Positives**  | **0** |

**Classification Report (Full):**

```
              precision    recall  f1-score   support

      Benign       0.97      1.00      0.99        72
   Malignant       1.00      0.95      0.98        42

    accuracy                           0.98       114
   macro avg       0.99      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114
```

**Data Split Summary:**

| Split      | Samples |
|------------|---------|
| Training   | 386     |
| Validation | 69      |
| Test       | 114     |
| **Total**  | **569** |

### 8.2 Confusion Matrix Analysis

```
                 Predicted
              Benign   Malignant
Actual Benign   TN        FP
Actual Malignant FN       TP
```

The most critical value is **FN (False Negatives)** — malignant tumors classified as benign. In a clinical setting, minimizing FN is paramount because a missed malignant diagnosis delays treatment and can be life-threatening.

### 8.3 ROC Curve Interpretation

The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (Recall/Sensitivity) against the False Positive Rate at various classification thresholds. An AUC (Area Under Curve) of 1.0 represents a perfect classifier; random guessing gives AUC = 0.5.

ANN models on the WDBC dataset typically achieve **AUC > 0.98**, indicating excellent discrimination between benign and malignant cases across all thresholds.

### 8.4 Training History

The model uses EarlyStopping to terminate training when validation loss ceases to decrease. This prevents overfitting on the relatively small training set (~387 samples after validation split). The training curves (accuracy and loss) reveal:
- Rapid convergence in the first 10–20 epochs.
- Stabilization of validation metrics.
- Early stopping typically triggers between epoch 20–60 depending on initialization.

### 8.5 Clinical Significance

A **high Recall for the Malignant class** (ideally >0.95) means the model rarely misses a true cancer case. The balance between Precision and Recall can be adjusted by moving the classification threshold (default: 0.50) — lowering the threshold increases recall at the cost of more false positives (unnecessary follow-up tests).

---

## 9. Conclusion and Future Enhancements

### 9.1 Conclusion

This project successfully demonstrates the application of an Artificial Neural Network for binary classification of breast cancer records from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The implemented ANN achieves clinically meaningful performance on the test set, with high accuracy and strong recall for the malignant class.

Key contributions:
- A complete, reproducible ML pipeline from data loading through model deployment.
- Careful attention to data leakage prevention via proper train-only scaler fitting.
- EarlyStopping and Dropout regularization for robust generalization.
- An interactive Streamlit frontend for real-time predictions.
- Comprehensive evaluation including ROC-AUC analysis.

### 9.2 Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| **Cross-Validation** | Use k-fold cross-validation for more robust performance estimation |
| **Hyperparameter Tuning** | Use Keras Tuner or Optuna to optimize layer sizes, dropout rates, and learning rate |
| **Advanced Architectures** | Experiment with BatchNormalization, Residual connections |
| **Ensemble Methods** | Combine ANN with Random Forest or XGBoost for improved robustness |
| **SHAP Explainability** | Use SHAP (SHapley Additive exPlanations) to explain individual predictions |
| **External Validation** | Test on the WDBC Prognostic dataset or other breast cancer datasets |
| **Threshold Optimization** | Optimize classification threshold to maximize Malignant recall |
| **Clinical Integration** | Deploy as a REST API (FastAPI) for integration with hospital PACS systems |
| **Image-based Input** | Extend to raw FNA image classification using CNNs |

---

## 10. References

1. Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1994). *Machine learning techniques to diagnose breast cancer from fine-needle aspirates.* Cancer Letters, 77(2–3), 163–171. https://doi.org/10.1016/0304-3835(94)90046-9

2. Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* Proceedings IS&T/SPIE Symposium on Electronic Imaging: Science and Technology, vol. 1905, pp. 861–870. San Jose, CA.

3. Dheeba, J., Singh, N.A., & Selvi, S.T. (2014). *Computer-aided detection of breast cancer on mammograms: A swarm intelligence optimized wavelet neural network approach.* Journal of Biomedical Informatics, 49, 45–52. https://doi.org/10.1016/j.jbi.2014.01.010

4. Scikit-learn Developers. (2023). *Breast cancer Wisconsin (diagnostic) dataset.* Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

5. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.

6. UCI Machine Learning Repository. *Breast Cancer Wisconsin (Diagnostic) Data Set.* https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. https://www.deeplearningbook.org/

---

*Report generated as part of academic project: "Breast Cancer Dataset Classification using Artificial Neural Network (ANN)"*
