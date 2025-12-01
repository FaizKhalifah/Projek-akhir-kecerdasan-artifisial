import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score


# ================================
# 1. Load Dataset
# ================================
df = pd.read_csv("data/software_defect.csv")

print("Before cleaning:", df.shape)

# ================================
# 2. Bersihkan data ('?' → NaN → median)
# ================================
df = df.replace("?", np.nan)

# Ubah semua kolom menjadi numerik (kalau ada string akan otomatis error)
df = df.apply(pd.to_numeric)

# Isi NaN dengan median
df = df.fillna(df.median(numeric_only=True))

print("After cleaning:", df.shape)

# ================================
# 3. Pisahkan fitur dan label
# ================================
X = df.drop(columns=["defects"])
y = df["defects"].astype(int)   # pastikan biner 0/1

# ================================
# 4. Normalisasi (opsional)
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 5. Train-test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 6. Train XGBoost
# ================================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ================================
# 7. Evaluasi
# ================================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# 8. Feature importance
# ================================
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance - XGBoost")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# MCC
mcc = matthews_corrcoef(y_test, y_pred)
print("MCC:", mcc)

# Cohen Kappa
kappa = cohen_kappa_score(y_test, y_pred)
print("Cohen Kappa:", kappa)

# ROC AUC
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print("Average Precision (AP):", ap)

plt.plot(recall, precision)
plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


#inference
new_data = pd.DataFrame([{
    "loc": 50,
    "v(g)": 4,
    "ev(g)": 1,
    "iv(g)": 3,
    "n": 200,
    "v": 1200,
    "l": 0.05,
    "d": 1.21,
    "i": 1.12,
    "e": 0.23,
    "b": 4,
    "t": 2,
    "lOCode": 30,
    "lOComment": 5,
    "lOBlank": 5,
    "locCodeAndComment": 35,
    "uniq_Op": 10,
    "uniq_Opnd": 18,
    "total_Op": 70,
    "total_Opnd": 55,
    "branchCount": 6
}])


new_scaled = scaler.transform(new_data)

pred = model.predict(new_scaled)
prob = model.predict_proba(new_scaled)[0][1]

print("Prediction:", pred[0])
print("Probability of defect:", prob)

