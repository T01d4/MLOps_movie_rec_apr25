import pandas as pd
import os
import mlflow
import logging

def predict_titles(pred_path, movies_path, output_dir):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starte Konvertierung von IDs zu Titeln...")

    # Lade Predictions
    df = pd.read_csv(pred_path, header=None)

    # Lade Movie-Metadaten
    movies = pd.read_csv(movies_path)
    id_to_title = dict(zip(movies["movieId"], movies["title"]))

    # IDs → Titel umwandeln
    df_titles = df.applymap(lambda x: id_to_title.get(x, f"Unknown ID: {x}"))

    # Sicherstellen, dass der Zielordner existiert
    os.makedirs(output_dir, exist_ok=True)

    # Breites Format
    csv_path = os.path.join(output_dir, "predicted_titles.csv")
    html_path = os.path.join(output_dir, "predicted_titles.html")

    df_titles.to_csv(csv_path, index=False)
    df_titles.to_html(html_path, index=False)

    # Long Format zusätzlich speichern
    df_long = df_titles.melt(var_name="userId", value_name="title").dropna()
    long_csv_path = os.path.join(output_dir, "predicted_titles_long.csv")
    df_long.to_csv(long_csv_path, index=False)

    print(f"Titel gespeichert unter:\n→ {csv_path}\n→ {html_path}\n→ {long_csv_path}")

    # MLflow Logging
    mlflow.set_experiment("movie_recommendation")
    with mlflow.start_run(run_name="map_ids_to_titles"):
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(long_csv_path)


def main():
    predict_titles(
        pred_path="data/predictions/predictions.csv",
        movies_path="data/raw/movies.csv",
        output_dir="data/predictions"
    )

if __name__ == "__main__":
    main()
