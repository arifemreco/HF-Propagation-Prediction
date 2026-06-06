import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def propagation_analysis(model, df_sample, kp, sfi, xray, threshold):

    distances = np.arange(100, 12000, 200)
    probs = []

    
    for d in distances:

        subset = df_sample[
            (df_sample["distance_km"] > d - 500) &
            (df_sample["distance_km"] < d + 500)
        ]

        # fallback (data yoksa)
        if len(subset) < 20:
            subset = df_sample.nlargest(100, "distance_km")

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

        
        if kp <= 1:
            kp_penalty =0
        else:
            kp_penalty = 0.02 * (kp-1)**1.5
        
        prob= prob - kp_penalty
        
        prob = np.clip(prob, 0, 1)
        probs.append(prob)

    
    probs_smooth = pd.Series(probs).rolling(window=3, min_periods=1).mean().values

   
    max_distance = None

    for i in range(5,len(probs_smooth) - 3):

        d = distances[i]

        
        w_kp = 0.75
        w_sfi = 0.25

        
        kp_effect = 0.012 * (kp - 1)**2.2 if kp > 1 else 0
        sfi_clamped=np.clip(sfi, 65, 140)
        sfi_effect = 0.0012 * (120 - sfi_clamped)

        
        effective_threshold = threshold + (w_kp * kp_effect) + (w_sfi * sfi_effect)

        
        if kp >= 4:
            effective_threshold += 0.03 + (0.015* (kp -3))
        if kp == 1:
            effective_threshold -= 0.03

        
        effective_threshold = np.clip(effective_threshold, 0.5, 0.8)

        
        if (probs_smooth[i] < effective_threshold and
            probs_smooth[i+1] < effective_threshold and
            probs_smooth[i+2] < effective_threshold):

            max_distance = d
            break

    if max_distance is None:
        max_distance = max(distances)

    
    plt.figure(figsize=(10,5))
    plt.plot(distances, probs_smooth, marker='o')

   
    avg_threshold = threshold + 0.75*(0.025*(kp-1)) + 0.25*(0.0015*(120-sfi))
    if kp >= 4:
        avg_threshold += 0.03

    avg_threshold = np.clip(avg_threshold, 0.5, 0.8)

    plt.axhline(avg_threshold, linestyle="--", label="ARRL Threshold (40m)")

    plt.xlabel("Distance (km)")
    plt.ylabel("Connection Probability")
    plt.title(f"40m Propagation | Kp={kp}, SFI={sfi} | Max ≈ {max_distance} km")

    plt.legend()
    plt.grid(True)

    plt.savefig(f"propagation_40m_kp{kp}_sfi{sfi}.png", dpi=300)
    plt.show()

    return max_distance