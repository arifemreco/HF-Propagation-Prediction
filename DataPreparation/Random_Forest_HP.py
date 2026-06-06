import pandas as pd

df = pd.read_csv(r"Dataset\cleaned_data\final_dataset.csv")


df = df[(df['snr'] >= -25) & (df['snr'] <= 20)]

features = [
    "distance_km", "dt", "kp", "sfi",
    "xray_flux", "azimuth_sin", "azimuth_cos"
]

df["connection"] = (df["snr"] > -15).astype(int)

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
    rf,
    param_grid,
    n_iter=15,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_


from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    from sklearn.metrics import roc_curve
    import numpy as np

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    best_threshold = thresholds[best_idx]

    print("Optimal threshold:", best_threshold)
    
    return best_threshold


best_treshold = evaluate_model(best_rf, X_test, y_test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def propagation_analysis(model, df_sample, kp, sfi, xray, threshold=0.8):

    distances = np.arange(100, 8000, 200)
    probs = []

    for d in distances:

        
        subset = df_sample[
            (df_sample["distance_km"] > d - 500) &
            (df_sample["distance_km"] < d + 500)
        ]

        if len(subset) < 20:
            probs.append(np.nan)
            continue

        subset = subset.sample(min(100, len(subset)), random_state=42)

        
        samples = pd.DataFrame({
            "distance_km": d,
            "dt": subset["dt"].values,
            "kp": kp,
            "sfi": sfi,
            "xray_flux": xray,
            "azimuth_sin": subset["azimuth_sin"].values,
            "azimuth_cos": subset["azimuth_cos"].values
        })

    
        p = model.predict_proba(samples)[:, 1]

        
        prob = np.mean(p)
        probs.append(prob)

    
    max_distance = None
    for i in range(len(probs) - 3):
        if (probs[i] < threshold and
            probs[i+1] < threshold and
            probs[i+2] < threshold):
        
            max_distance = distances[i]
            break
    if max_distance is None:
        max_distance = max(distances)

    
    plt.figure(figsize=(10,5))

    plt.plot(distances, probs, marker='o')
    plt.axhline(0.5, linestyle="--", label="Reliability Threshold")

    plt.xlabel("Distance (km)")
    plt.ylabel("Connection Probability")
    plt.title(f"Propagation Analysis | Kp={kp}, SFI={sfi} | Max ≈ {max_distance} km")

    plt.legend()
    plt.grid(True)

    plt.savefig(f"propagation_kp{kp}_sfi{sfi}.png", dpi=300)
    plt.show()

    return max_distance

print("Best params:", rf_search.best_params_)

max_dist = propagation_analysis(
    best_rf,
    df,
    kp=2,
    sfi=120,
    xray=1.5,
    threshold=best_treshold
)

print("Max distance:", max_dist)
