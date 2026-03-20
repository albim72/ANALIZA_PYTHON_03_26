import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")
np.random.seed(42)

months = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze", "Lip", "Sie", "Wrz", "Paź", "Lis", "Gru"]
products = ["AI Assistant", "Forecast Engine", "Vision System", "Fraud Detector"]

rows = []
for product in products:
    for i, month in enumerate(months):
        base = {
            "AI Assistant": 120,
            "Forecast Engine": 90,
            "Vision System": 70,
            "Fraud Detector": 100
        }[product]

        season = 15 * np.sin(i / 12 * 2 * np.pi)
        trend = i * 3
        value = base + season + trend + np.random.randint(-8, 9)

        rows.append([product, month, round(value)])

df = pd.DataFrame(rows, columns=["product", "month", "sales"])

pivot = df.pivot(index="product", columns="month", values="sales")

plt.figure(figsize=(12, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
plt.title("Sprzedaż produktów w miesiącach")
plt.xlabel("Miesiąc")
plt.ylabel("Produkt")
plt.tight_layout()
plt.show()
