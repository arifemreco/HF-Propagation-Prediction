import pandas as pd
import joblib

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


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(class_weight="balanced", random_state=42)

rf_search = RandomizedSearchCV(
    rf, param_grid, n_iter=15, cv=3,
    scoring="f1", n_jobs=-1, verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_


from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import numpy as np

y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

print("Optimal threshold:", best_threshold)


joblib.dump(best_rf, r"DataPreparation\Predictor\models\rf_model.pk1")
joblib.dump(best_threshold, r"DataPreparation\Predictor\models\threshold.pkl")

print("Model ve threshold kaydedildi")