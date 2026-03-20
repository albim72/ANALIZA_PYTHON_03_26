import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
np.random.seed(42)

n = 120

group = np.random.choice(["Model_A", "Model_B", "Model_C"], size=n, p=[0.35, 0.35, 0.30])

accuracy = []
precision = []
recall = []
f1 = []

for g in group:
    if g == "Model_A":
        a = np.random.normal(0.84, 0.03)
        p = np.random.normal(0.82, 0.04)
        r = np.random.normal(0.80, 0.05)
    elif g == "Model_B":
        a = np.random.normal(0.88, 0.025)
        p = np.random.normal(0.86, 0.03)
        r = np.random.normal(0.84, 0.04)
    else:
        a = np.random.normal(0.81, 0.035)
        p = np.random.normal(0.79, 0.04)
        r = np.random.normal(0.87, 0.035)

    f = 2 * (p * r) / (p + r)

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    f1.append(f)

df = pd.DataFrame({
    "model": group,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
})

g = sns.pairplot(
    df,
    vars=["accuracy", "precision", "recall", "f1_score"],
    hue="model",
    diag_kind="kde",
    corner=False
)

g.fig.suptitle("Pairplot metryk modeli ML", y=1.02)
plt.show()
