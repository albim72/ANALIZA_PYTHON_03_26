import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
np.random.seed(42)

n = 180
team = np.random.choice(["Data Science", "ML Engineering", "Analytics"], size=n, p=[0.35, 0.35, 0.30])

training_hours = []
for t in team:
    if t == "Data Science":
        val = np.random.normal(42, 8)
    elif t == "ML Engineering":
        val = np.random.normal(35, 6)
    else:
        val = np.random.normal(28, 7)
    training_hours.append(max(5, val))

df = pd.DataFrame({
    "team": team,
    "training_hours": training_hours
})

plt.figure(figsize=(10, 6))
sns.violinplot(
    data=df,
    x="team",
    y="training_hours",
    inner="box",
    cut=0
)
sns.swarmplot(
    data=df,
    x="team",
    y="training_hours",
    size=3,
    alpha=0.7
)

plt.title("Rozkład godzin szkoleniowych w zespołach")
plt.xlabel("Zespół")
plt.ylabel("Liczba godzin")
plt.tight_layout()
plt.show()
