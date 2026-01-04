# Healthcare_Prediction

# 🏥 Healthcare Predictive Model – Patient Risk Scoring

## 📌 Overview
This project provides an **end-to-end machine learning pipeline** for patient risk scoring.  
It is designed for **educational and research purposes only** and should not be used as a medical device or clinical decision tool.  

The pipeline is built to:
- Train a predictive model on healthcare datasets (binary risk labels).
- Evaluate performance using ROC-AUC, precision/recall, and classification reports.
- Generate **risk scores** for new patient datasets.
- Provide an **interactive workflow in Google Colab** (upload datasets, train, predict).

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **Google Colab** (interactive execution)
- **Libraries:**
  - `scikit-learn` – preprocessing, metrics, pipeline
  - `xgboost` – gradient boosting classifier
  - `imbalanced-learn` – handling class imbalance (SMOTE)
  - `pandas`, `numpy` – data manipulation
  - `matplotlib`, `seaborn` – visualization
  - `joblib` – model persistence

---

## 🚀 Features
- Interactive dataset upload (CSV).
- User prompt to specify target column (binary 0/1 risk label).
- Train/test split with stratification.
- Standardization + XGBoost model pipeline.
- Model evaluation (classification report, ROC-AUC).
- Save trained model (`risk_model.joblib`).
- Upload new dataset for prediction → outputs `risk_scores_output.csv`.

---

## 📂 Project Structure
