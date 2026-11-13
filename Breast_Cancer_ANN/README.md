# Breast Cancer ANN Classification

> **A College Machine Learning Project** — Binary classification of breast cancer records as **Benign** or **Malignant** using an Artificial Neural Network (ANN) trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

---

## 📋 Project Structure

```
Breast_Cancer_ANN/
│
├── dataset/
│   └── breast_cancer_wdbc.csv       ← Auto-generated CSV from sklearn
│
├── notebooks/
│   └── breast_cancer_ann.ipynb      ← Full interactive Jupyter Notebook
│
├── models/
│   ├── breast_cancer_ann.keras      ← Saved trained model (after training)
│   ├── scaler.pkl                   ← Saved StandardScaler
│   ├── metrics.txt                  ← Auto-saved evaluation metrics
│   ├── 01_class_distribution.png
│   ├── 02_accuracy_curve.png
│   ├── 03_loss_curve.png
│   ├── 04_confusion_matrix.png
│   └── 05_roc_curve.png
│
├── src/
│   └── train.py                     ← Full training pipeline script
│
├── app.py                           ← Streamlit web application
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
│
└── report/
    └── project_report.md            ← Academic project report
```

---

## ⚙️ Prerequisites

- **Python 3.9 – 3.11** recommended (TensorFlow compatibility)
- pip package manager

---

## 🚀 Setup & Execution

### Step 1 — Clone or Navigate to the Project Folder

```bash
cd Breast_Cancer_ANN
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Training Script

```bash
python src/train.py
```

This will:
- Load and explore the WDBC dataset
- Preprocess and split the data
- Build, train, and evaluate the ANN
- Save the model to `models/breast_cancer_ann.keras`
- Save the scaler to `models/scaler.pkl`
- Generate all 5 visualization plots in `models/`
- Print a full evaluation report to the console

### Step 5 — Launch the Streamlit App (Optional)

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Step 6 — Run the Jupyter Notebook (Optional)

```bash
jupyter notebook notebooks/breast_cancer_ann.ipynb
```

---

## 🧠 ANN Architecture

```
Input (30 features)
        ↓
Dense(64, activation='relu')
        ↓
Dropout(0.3)
        ↓
Dense(32, activation='relu')
        ↓
Dropout(0.2)
        ↓
Dense(1, activation='sigmoid')
        ↓
Binary Output: 0 = Benign | 1 = Malignant
```

| Component  | Detail                               |
|------------|--------------------------------------|
| Optimizer  | Adam                                 |
| Loss       | Binary Cross-Entropy                 |
| Metrics    | Accuracy, Precision, Recall          |
| Epochs     | Up to 100 (with Early Stopping)      |
| Batch Size | 16                                   |
| Patience   | 12 (Early Stopping)                  |

---

## 📊 Dataset

| Property           | Value                                   |
|--------------------|-----------------------------------------|
| Name               | Wisconsin Diagnostic Breast Cancer (WDBC) |
| Source             | `sklearn.datasets.load_breast_cancer()` |
| Total Samples      | 569                                     |
| Features           | 30 numerical                            |
| Classes            | Benign (357), Malignant (212)           |
| Missing Values     | None                                    |

---

## 🎗️ Medical Context

This model classifies tumors as **Benign** (non-cancerous) or **Malignant** (cancerous).  
In a medical setting, **Recall for the Malignant class** (sensitivity) is the most critical metric — missing a malignant case (false negative) has serious clinical consequences.

---

## ⚕️ Disclaimer

> This project is for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a licensed physician for medical advice.

---

## 🧰 Technology Stack

| Library       | Purpose                         |
|---------------|---------------------------------|
| TensorFlow    | ANN model building & training   |
| Scikit-learn  | Dataset, preprocessing, metrics |
| NumPy / Pandas| Data manipulation               |
| Matplotlib    | Plotting                        |
| Seaborn       | Statistical visualization       |
| Streamlit     | Web application frontend        |
| Joblib        | Scaler persistence              |
