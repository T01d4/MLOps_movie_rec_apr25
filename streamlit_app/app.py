import streamlit as st
import pandas as pd
import os

# === Titel und Einleitung ===
st.set_page_config(page_title="🎬 Filmempfehlungen", layout="wide")
st.title("🎥 Deine Filmempfehlungen")
st.markdown("Hier siehst du die Filme, die für dich empfohlen wurden, basierend auf deinem Nutzerprofil.")

# === Lade die Predictions ===
pred_csv_path = "data/predictions/predicted_titles.csv"
pred_html_path = "data/predictions/predicted_titles.html"

# === Debug-Log anzeigen ===
st.sidebar.title("🔍 Debug-Log")
st.sidebar.write(f"📁 Erwarteter Pfad: `{pred_csv_path}`")

if os.path.exists(pred_csv_path):
    st.sidebar.success("✅ Datei gefunden")

    df = pd.read_csv(pred_csv_path)

    # Debug: Zeige erste Zeilen
    st.sidebar.write("🔢 Vorschau:")
    st.sidebar.dataframe(df.head())

    if "Unknown ID: 0" in str(df.iloc[0, 0]):
        st.sidebar.warning("⚠️ Entferne Header-Zeile mit 'Unknown ID'")
        df = df.iloc[1:].reset_index(drop=True)

    df = df.transpose()
    df.columns = [f"Empfehlung {i+1}" for i in range(df.shape[1])]
    df.index = [f"Nutzer {i+1}" for i in range(df.shape[0])]

    for idx in df.index:
        with st.expander(f"🎯 Empfehlungen für {idx}"):
            for title in df.loc[idx].dropna():
                st.write(f"- {title}")

    if os.path.exists(pred_html_path):
        with open(pred_html_path, "r", encoding="utf-8") as f:
            html = f.read()
        with st.expander("🔍 Vorschau als HTML-Tabelle"):
            st.components.v1.html(html, height=400, scrolling=True)

else:
    st.sidebar.error("❌ Datei nicht gefunden")
    st.error("Keine Vorhersagen gefunden. Bitte führe die Vorhersage-Pipeline zuerst aus.")