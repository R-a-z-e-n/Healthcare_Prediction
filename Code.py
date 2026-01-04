
# Healthcare Predictive Model – Interactive Dataset Input


# Install dependencies if needed
# !pip install -q scikit-learn xgboost shap imbalanced-learn pandas numpy matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from google.colab import files


print("Please upload your dataset CSV file...")
uploaded = files.upload()

for fn in uploaded.keys():
    dataset_path = fn

print(f"Dataset uploaded: {dataset_path}")


df = pd.read_csv(dataset_path)
print("First 5 rows of your dataset:")
print(df.head())

target_col = input("Enter the name of the target column (binary 0/1 risk label): ")

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    ))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")


print("\nRisk scores for first 10 patients in test set:")
risk_scores = pd.DataFrame({"risk_score": y_proba[:10]}, index=X_test.index[:10])
print(risk_scores)


joblib.dump(pipeline, "risk_model.joblib")
print("\nModel saved as risk_model.joblib")
