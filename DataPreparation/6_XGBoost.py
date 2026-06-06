import pandas as pd


df = pd.read_csv(r"Dataset\cleaned_data\final_dataset.csv")


df = df[(df['snr'] >= -25) & (df['snr'] <= 20)]


df["connection"] = (df["snr"] > -15).astype(int)


features = [
    "distance_km", "dt", "kp", "sfi",
    "xray_flux", "azimuth_sin", "azimuth_cos"
]

X = df[features]
y = df["connection"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,  # 🔥 daha stabil
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


xgb.fit(X_train, y_train)


from sklearn.metrics import classification_report, roc_auc_score

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

print("=== XGBOOST RESULTS ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# =====================================================
# XGBOOST CONFUSION MATRIX
# =====================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(ax=ax)

plt.title("XGBoost Confusion Matrix")

plt.savefig(
    "xgb_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# XGBOOST FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": xgb.feature_importances_
})


importance_df = importance_df[
    importance_df["Feature"] != "dt"
]

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.title("XGBoost Feature Importance")

plt.grid(axis="x")

plt.savefig(
    "xgb_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(importance_df)