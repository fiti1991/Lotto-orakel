
# Einfaches Streamlit-Webtool für Lotto-Tipp basierend auf XGBoost-Modell
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

st.title("Lotto-Orakel: Dein KI-basierter 6er-Tipp")

st.markdown("Lade deine SwissLotto-Daten im CSV-Format hoch:")
file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

if file:
    df = pd.read_csv(file)
    df["Zahlen_Liste"] = df["Numbers"].apply(lambda x: list(map(int, x.strip("[]").split(", "))))

    def berechne_hotness(liste_von_ziehungen, fenster=10):
        hotness_liste = []
        for i in range(len(liste_von_ziehungen)):
            start = max(0, i - fenster)
            vergangene = liste_von_ziehungen[start:i]
            flat = [zahl for ziehung in vergangene for zahl in ziehung]
            counts = {n: flat.count(n) for n in range(1, 43)}
            hotness_liste.append([counts[n] for n in range(1, 43)])
        return pd.DataFrame(hotness_liste, columns=[f"Hot_{n}" for n in range(1, 43)])

    def erstelle_klassifikationsmatrix(zahlen_liste, anzahl_zahlen=42):
        matrix = []
        for ziehung in zahlen_liste:
            row = [1 if i in ziehung else 0 for i in range(1, anzahl_zahlen + 1)]
            matrix.append(row)
        return pd.DataFrame(matrix, columns=[f"Zahl_{i}" for i in range(1, anzahl_zahlen + 1)])

    X = berechne_hotness(df["Zahlen_Liste"]).iloc[10:].reset_index(drop=True)
    y = erstelle_klassifikationsmatrix(df["Zahlen_Liste"]).iloc[10:].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50)
    model = MultiOutputClassifier(xgb_base)
    model.fit(X_train, y_train)

    st.success("Modell trainiert!")

    if st.button("6 Zahlen für die nächste Ziehung generieren"):
        neueste_ziehung = X.iloc[[-1]]
        probs = model.predict_proba(neueste_ziehung)
        probs = np.array([p[0][1] for p in probs])
        tipp = list(np.argsort(probs)[-6:][::-1] + 1)
        st.subheader("Dein KI-basierter Tipp:")
        st.write(tipp)
