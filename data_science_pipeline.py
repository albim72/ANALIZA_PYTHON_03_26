import random
from datetime import datetime
from faker import Faker

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# USTAWIENIA
# =========================================================
N = 100_000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
fake = Faker("pl_PL")
Faker.seed(SEED)

# =========================================================
# SŁOWNIKI I POMOCE
# =========================================================
cities = [
    "Warszawa", "Kraków", "Wrocław", "Poznań", "Gdańsk",
    "Łódź", "Katowice", "Lublin", "Szczecin", "Białystok"
]

countries = ["Poland", "Germany", "Czech Republic", "Slovakia", "Lithuania"]
devices = ["mobile", "desktop", "tablet"]
payment_methods = ["card", "blik", "bank_transfer", "paypal"]
times_of_day = ["morning", "afternoon", "evening", "night"]
product_categories = ["electronics", "fashion", "beauty", "home", "sports", "books"]

country_weights = [0.78, 0.08, 0.05, 0.04, 0.05]
device_weights = [0.58, 0.34, 0.08]
payment_weights = [0.45, 0.28, 0.15, 0.12]
time_weights = [0.22, 0.31, 0.33, 0.14]
category_weights = [0.24, 0.18, 0.12, 0.16, 0.14, 0.16]


def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# GENERATOR DANYCH
# =========================================================
rows = []

for i in range(N):
    customer_id = f"CUST-{i:06d}"
    first_name = fake.first_name()
    last_name = fake.last_name()
    city = random.choice(cities)
    country = weighted_choice(countries, country_weights)

    email = fake.email()
    age = int(np.clip(np.random.normal(37, 11), 18, 80))

    # Dochód miesięczny
    income = round(np.clip(np.random.normal(8500, 3500), 1800, 40000), 2)

    # Historia klienta
    previous_transactions = np.random.poisson(lam=18)
    days_since_last_login = int(np.clip(np.random.exponential(scale=12), 0, 180))

    device_type = weighted_choice(devices, device_weights)
    payment_method = weighted_choice(payment_methods, payment_weights)
    time_of_day = weighted_choice(times_of_day, time_weights)
    product_category = weighted_choice(product_categories, category_weights)

    billing_shipping_match = np.random.choice([0, 1], p=[0.12, 0.88])

    # Kwota transakcji zależna od kategorii
    category_base = {
        "electronics": 900,
        "fashion": 240,
        "beauty": 130,
        "home": 320,
        "sports": 280,
        "books": 85
    }[product_category]

    amount = np.random.lognormal(mean=np.log(category_base), sigma=0.55)
    amount = round(min(amount, 25000), 2)

    # IP risk score - abstrakcyjny wskaźnik ryzyka
    ip_risk_score = np.clip(np.random.beta(2, 8) * 100, 0, 100)

    # Liczba chargebacków w przeszłości
    past_chargebacks = np.random.choice([0, 1, 2, 3], p=[0.92, 0.06, 0.015, 0.005])

    # Cel: oszustwo
    # Budujemy logikę prawdopodobieństwa fraudu
    risk = -4.8

    if amount > 1500:
        risk += 0.8
    if amount > 4000:
        risk += 1.0

    if time_of_day == "night":
        risk += 0.9

    if device_type == "mobile":
        risk += 0.25

    if payment_method == "paypal":
        risk += 0.2

    if billing_shipping_match == 0:
        risk += 1.2

    if days_since_last_login > 45:
        risk += 0.55

    if previous_transactions <= 2:
        risk += 0.75

    if past_chargebacks >= 1:
        risk += 1.4 * past_chargebacks

    risk += (ip_risk_score / 100) * 2.0

    if country != "Poland":
        risk += 0.35

    # Dochód vs kwota
    if amount > 0.45 * income:
        risk += 0.9

    fraud_prob = sigmoid(risk)
    is_fraud = np.random.binomial(1, fraud_prob)

    rows.append({
        "customer_id": customer_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "city": city,
        "country": country,
        "age": age,
        "income": income,
        "previous_transactions": previous_transactions,
        "days_since_last_login": days_since_last_login,
        "device_type": device_type,
        "payment_method": payment_method,
        "time_of_day": time_of_day,
        "product_category": product_category,
        "billing_shipping_match": billing_shipping_match,
        "amount": amount,
        "ip_risk_score": round(float(ip_risk_score), 2),
        "past_chargebacks": past_chargebacks,
        "is_fraud": is_fraud
    })

df = pd.DataFrame(rows)

# =========================================================
# WPROWADZENIE "BRUDU DANYCH"
# =========================================================
# Braki danych
missing_income_idx = np.random.choice(df.index, size=2500, replace=False)
missing_age_idx = np.random.choice(df.index, size=1800, replace=False)
missing_city_idx = np.random.choice(df.index, size=1200, replace=False)

df.loc[missing_income_idx, "income"] = np.nan
df.loc[missing_age_idx, "age"] = np.nan
df.loc[missing_city_idx, "city"] = None

# Duplikaty maili / lekkie artefakty
duplicate_idx = np.random.choice(df.index, size=400, replace=False)
df.loc[duplicate_idx, "email"] = "unknown@example.com"

# Odstające kwoty
outlier_idx = np.random.choice(df.index, size=80, replace=False)
df.loc[outlier_idx, "amount"] = df.loc[outlier_idx, "amount"] * 12

# Losowe błędne formaty w payment_method
weird_idx = np.random.choice(df.index, size=200, replace=False)
df.loc[weird_idx, "payment_method"] = "unknown"

# =========================================================
# SZYBKA ANALIZA
# =========================================================
print("=== SHAPE ===")
print(df.shape)

print("\n=== HEAD ===")
print(df.head())

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== FRAUD RATE ===")
print(df["is_fraud"].mean())

print("\n=== NUMERIC SUMMARY ===")
print(df.describe())

# =========================================================
# ZAPIS SUROWYCH DANYCH
# =========================================================
df.to_csv("transactions_100k_raw.csv", index=False)
df.to_parquet("transactions_100k_raw.parquet", index=False)

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df["amount_to_income_ratio"] = df["amount"] / df["income"]
df["is_high_amount"] = (df["amount"] > 2000).astype(int)
df["is_night"] = (df["time_of_day"] == "night").astype(int)
df["has_chargeback_history"] = (df["past_chargebacks"] > 0).astype(int)

# Ograniczenie skrajności - winsoryzacja prosta
upper_cap = df["amount"].quantile(0.995)
df["amount_capped"] = df["amount"].clip(upper=upper_cap)

# =========================================================
# WYBÓR CECH
# =========================================================
target = "is_fraud"

numeric_features = [
    "age",
    "income",
    "previous_transactions",
    "days_since_last_login",
    "billing_shipping_match",
    "amount",
    "ip_risk_score",
    "past_chargebacks",
    "amount_to_income_ratio",
    "is_high_amount",
    "is_night",
    "has_chargeback_history",
    "amount_capped"
]

categorical_features = [
    "city",
    "country",
    "device_type",
    "payment_method",
    "time_of_day",
    "product_category"
]

X = df[numeric_features + categorical_features]
y = df[target]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# =========================================================
# PREPROCESSING PIPELINE
# =========================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =========================================================
# PEŁNY PIPELINE ML
# =========================================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# =========================================================
# TRENOWANIE
# =========================================================
model.fit(X_train, y_train)

# =========================================================
# EWALUACJA
# =========================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

print("\n=== ROC AUC ===")
print(roc_auc_score(y_test, y_proba))

# =========================================================
# ZAPIS DANYCH PO FEATURE ENGINEERING
# =========================================================
df.to_csv("transactions_100k_features.csv", index=False)
df.to_parquet("transactions_100k_features.parquet", index=False)

print("\nZapisano pliki:")
print("- transactions_100k_raw.csv")
print("- transactions_100k_raw.parquet")
print("- transactions_100k_features.csv")
print("- transactions_100k_features.parquet")
