import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow


def validate_predictions():
    # Lade Vorhersagen
    df = pd.read_csv("data/predictions/predictions.csv", header=None)
    df_flat = df.values.flatten()

    # Erstelle Output-Ordner
    os.makedirs("reports", exist_ok=True)

    # Visualisierung: Verteilung der empfohlenen Movie-IDs
    plt.figure(figsize=(12, 6))
    sns.histplot(df_flat, bins=50, kde=False, color='skyblue')
    plt.title("Häufigkeit der empfohlenen Filme")
    plt.xlabel("Movie ID")
    plt.ylabel("Anzahl")
    plot_path = "reports/prediction_distribution.png"
    plt.savefig(plot_path)
    plt.close()

    # Logge das Artefakt in MLflow
    mlflow.set_experiment("movie_recommendation")
    with mlflow.start_run(run_name="validate_predictions"):
        mlflow.log_artifact(plot_path)

    print(f"Prediction-Visualisierung gespeichert unter: {plot_path}")


if __name__ == "__main__":
    validate_predictions()
