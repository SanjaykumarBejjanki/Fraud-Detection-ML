import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE


# =========================
# 1. Paths
# =========================

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. Load Dataset
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# =========================
# 3. Separate Features/Target
# =========================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nOriginal class distribution:")
print(y.value_counts())


# =========================
# 4. Scale Features
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================
# 5. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================
# 6. Handle Class Imbalance
# =========================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# =========================
# 7. Train Model
# =========================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)


# =========================
# 8. Predictions
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# 9. Evaluation
# =========================

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

roc_auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nROC-AUC:   {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# =========================
# 10. Save Model + Scaler
# =========================

model_path = os.path.join(MODEL_DIR, "fraud_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("\nModel saved to:", model_path)
print("Scaler saved to:", scaler_path)


# =========================
# 11. Save Metrics
# =========================

metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")

with open(metrics_path, "w") as file:
    file.write("Credit Card Fraud Detection Model\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"Dataset shape: {df.shape}\n")
    file.write(f"Normal transactions: {(y == 0).sum()}\n")
    file.write(f"Fraud transactions: {(y == 1).sum()}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

print("Metrics saved to:", metrics_path)

print("\nTraining completed successfully! ✅")