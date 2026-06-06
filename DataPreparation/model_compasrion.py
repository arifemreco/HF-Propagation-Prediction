import pandas as pd

results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "Deep Learning"],
    "Accuracy": [0.80, 0.71, 0.67],
    "ROC-AUC": [0.86, 0.79, 0.73],
    "F1 Score (Class 1)": [0.86, 0.79, 0.77]
})

print(results)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.axis('off')

table = ax.table(
    cellText=results.values,
    colLabels=results.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.savefig("model_results.png", bbox_inches='tight', dpi=300)
plt.show()