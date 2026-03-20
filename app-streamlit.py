import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Tytuł aplikacji
st.title("Prosty formularz i wykres w Streamlit")

# Krótki opis
st.write("Podaj kilka wartości, a aplikacja narysuje wykres słupkowy.")

# Formularz
with st.form("moj_formularz"):
    nazwa = st.text_input("Podaj nazwę serii danych:", "Moje dane")
    a = st.number_input("Wartość A", value=10)
    b = st.number_input("Wartość B", value=20)
    c = st.number_input("Wartość C", value=15)

    submitted = st.form_submit_button("Pokaż wykres")

# Po kliknięciu przycisku
if submitted:
    # Tworzenie danych
    df = pd.DataFrame({
        "Kategoria": ["A", "B", "C"],
        "Wartość": [a, b, c]
    })

    # Wyświetlenie tekstu i tabeli
    st.success(f"Wczytano dane dla serii: {nazwa}")
    st.write("Dane wejściowe:")
    st.dataframe(df)

    # Rysowanie wykresu
    fig, ax = plt.subplots()
    ax.bar(df["Kategoria"], df["Wartość"])
    ax.set_title(f"Wykres dla: {nazwa}")
    ax.set_xlabel("Kategoria")
    ax.set_ylabel("Wartość")

    st.pyplot(fig)
