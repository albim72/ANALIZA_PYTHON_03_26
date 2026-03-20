import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")
np.random.seed(42)

n = 240

region = np.random.choice(["Północ", "Południe", "Wschód", "Zachód"], size=n)
ad_budget = np.random.uniform(5, 50, size=n)

sales = []
for r, b in zip(region, ad_budget):
    if r == "Północ":
        s = 20 + 3.2 * b + np.random.normal(0, 12)
    elif r == "Południe":
        s = 15 + 2.6 * b + np.random.normal(0, 10)
    elif r == "Wschód":
        s = 18 + 2.2 * b + np.random.normal(0, 11)
    else:
        s = 25 + 3.8 * b + np.random.normal(0, 13)
    sales.append(s)

df = pd.DataFrame({
    "region": region,
    "ad_budget": ad_budget,
    "sales": sales
})

g = sns.lmplot(
    data=df,
    x="ad_budget",
    y="sales",
    col="region",
    hue="region",
    height=4,
    aspect=1,
    ci=95,
    scatter_kws={"alpha": 0.7, "s": 50}
)

g.fig.suptitle("Wpływ budżetu reklamowego na sprzedaż w regionach", y=1.05)
plt.show()
