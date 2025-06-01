import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow

def validate_predictions(output_path):
    # Lade Vorhersagen
    df = pd.read_csv(output_path, header=0)
    
    # Prüfe auf 'predicted_title' oder alternative title-Spalten
    if "predicted_title" in df.columns:
        df_flat = df["predicted_title"].values.flatten()
    else:
        title_cols = [col for col in df.columns if col.startswith("title_")]
        if not title_cols:
            raise ValueError("Spalte 'predicted_title' oder 'title_#' nicht gefunden in der Vorhersagedatei.")
        df_flat = df[title_cols].values.flatten()

    # Erstelle Output-Ordner
    os.makedirs("reports", exist_ok=True)

    # Visualisierung: Verteilung der empfohlenen Titel
    plt.figure(figsize=(12, 6))
    sns.histplot(df_flat, bins=50, kde=False, color='skyblue')
    plt.title("Häufigkeit der empfohlenen Filme")
    plt.xlabel("Movie ID")
    plt.ylabel("Anzahl")
    plot_path = "reports/prediction_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    # 🟢 Experiment setzen
    mlflow.set_experiment("movie_recommendation")

    # Logge das Artefakt in MLflow
    with mlflow.start_run(run_name="validate_predictions", nested=False):
        mlflow.log_artifact(plot_path)

    print(f"✅ Prediction-Visualisierung gespeichert unter: {plot_path}")

# Optionaler manueller Test
if __name__ == "__main__":
    validate_predictions("data/predictions/predicted_titles.csv")