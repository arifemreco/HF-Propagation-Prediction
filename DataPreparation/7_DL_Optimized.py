import pandas as pd
import numpy as np


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


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(16, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5
)


history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)


from sklearn.metrics import classification_report, roc_auc_score

y_prob = model.predict(X_test)


threshold = 0.5
y_pred = (y_prob > threshold).astype(int)

print("\n=== DEEP LEARNING RESULTS ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# =====================================================
# DEEP LEARNING CONFUSION MATRIX
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

plt.title("Deep Learning Confusion Matrix")

plt.savefig(
    "dl_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# DEEP LEARNING FEATURE IMPORTANCE
# =====================================================

import numpy as np
import matplotlib.pyplot as plt


weights = model.layers[0].get_weights()[0]


importance = np.mean(np.abs(weights), axis=1)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
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

plt.title("Deep Learning Feature Importance")

plt.grid(axis="x")

plt.savefig(
    "dl_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(importance_df)