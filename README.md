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


---

## 🖥️ How to Run (Google Colab)
1. Open the notebook in **Google Colab**.
2. Upload your **training dataset (CSV)** when prompted.
   - Ensure it contains a **binary target column** (0 = low risk, 1 = high risk).
3. Enter the **target column name** when asked.
4. The pipeline trains automatically and prints evaluation metrics.
5. Upload a **new dataset (without target column)** for prediction.
6. Risk scores will be displayed and saved as `risk_scores_output.csv`.

---

## 📊 Example Output
Classification Report:
precision    recall  f1-score   support
0       0.92      0.89      0.91       100
1       0.88      0.91      0.89        80

Test ROC-AUC: 0.9456

Predicted Risk Scores (first 5 patients):
risk_score
0    0.812345
1    0.102345
2    0.567890
3    0.934567
4    0.223456


---

## ⚠️ Disclaimer
This project is intended for **educational and research use only**.  
It is **not a medical device** and should not be used for clinical decision-making.  
Always consult qualified healthcare professionals for medical advice.

---

## 📈 Future Enhancements
- Add ROC curve and confusion matrix visualizations.
- Integrate SHAP for feature importance explanations.
- Support multi-class risk categories.
- Deploy as a simple web app (Streamlit/Flask).

---

## 👨‍💻 Contributors
- **Mohammad Razeen Iqbal** – Strategic AI  Product Manager & Systems Architect


---

