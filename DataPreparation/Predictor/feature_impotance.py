import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


df = pd.read_csv(
    r"Dataset\cleaned_data\final_dataset.csv"
)


df = df[(df['snr'] >= -25) & (df['snr'] <= 20)]


features = [
    "distance_km",
    "dt",
    "kp",
    "sfi",
    "xray_flux",
    "azimuth_sin",
    "azimuth_cos"
]


df["connection"] = (df["snr"] > -15).astype(int)

X = df[features]
y = df["connection"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


best_rf = joblib.load(r"DataPreparation\Predictor\models\rf_model.pk1")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": best_rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df = importance_df[
    importance_df["Feature"] != "dt"
]

plt.figure(figsize=(8,5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.title("Random Forest Feature Importance")

plt.grid(axis="x")

plt.savefig(
    "rf_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(importance_df)

# =====================================================
# CONFUSION MATRIX
# =====================================================

y_pred = best_rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(ax=ax)

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "rf_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()