import pandas as pd
import joblib
from propagation import propagation_analysis


df = pd.read_csv(r"Dataset\cleaned_data\final_dataset.csv")

df = df[(df['snr'] >= -25) & (df['snr'] <= 20)]
df["connection"] = (df["snr"] > -15).astype(int)


model = joblib.load(r"DataPreparation\Predictor\models\rf_model.pk1")
threshold = joblib.load(r"DataPreparation\Predictor\models\threshold.pkl")


propagation_analysis(model, df, kp=1, sfi=130, xray=1.2, threshold=threshold)


propagation_analysis(model, df, kp=3, sfi=115, xray=1.5, threshold=threshold)


propagation_analysis(model, df, kp=5, sfi=150, xray=2.0, threshold=threshold)