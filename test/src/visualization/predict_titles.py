import pandas as pd
import os
import mlflow
import logging

def predict_titles(pred_path, movies_path, output_dir, mapping_path=None):
    logging.basicConfig(level=logging.INFO)
    logging.info("📥 Starte Konvertierung von IDs zu Titeln...")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"❌ Prediction-Datei nicht gefunden: {pred_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"❌ Movie-Metadaten nicht gefunden: {movies_path}")

    # Robust laden – ohne oder mit Header
    try:
        df = pd.read_csv(pred_path, header=None)
        if any(col.startswith("title_") for col in df.iloc[0].astype(str)):
            df = pd.read_csv(pred_path)  # Neu laden mit Header
    except Exception:
        df = pd.read_csv(pred_path)

    movies_df = pd.read_csv(movies_path)

    # Optional: Mapping von Index zu movieId laden
    if mapping_path and os.path.exists(mapping_path):
        logging.info("🔁 Mapping-Datei erkannt – entschlüssele encodierte IDs.")
        import pickle
        with open(mapping_path, "rb") as f:
            movie_ids = pickle.load(f)
        index_to_movieId = dict(enumerate(movie_ids))
        df = df.applymap(lambda x: index_to_movieId.get(x, f"UnknownIndex:{x}"))

    # movieId → Titel
    id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    df_titles = df.applymap(lambda x: id_to_title.get(x, f"Unknown ID: {x}"))

    # Zielordner anlegen
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "predicted_titles.csv")
    html_path = os.path.join(output_dir, "predicted_titles.html")
    df_titles.to_csv(csv_path, index=False)
    df_titles.to_html(html_path, index=False)

    # Long Format erzeugen
    df_long = df_titles.melt(var_name="userId", value_name="title").dropna()
    long_csv_path = os.path.join(output_dir, "predicted_titles_long.csv")
    df_long.to_csv(long_csv_path, index=False)

    logging.info(f"✅ Titel gespeichert:\n→ {csv_path}\n→ {html_path}\n→ {long_csv_path}")

    # MLflow Logging
    mlflow.set_experiment("movie_recommendation")
    with mlflow.start_run(run_name="map_ids_to_titles"):
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(long_csv_path)


def main():
    base_path = "/opt/airflow"
    mlflow.set_tracking_uri("https://dagshub.com/sacer11/MLOps_movie_rec_apr25.mlflow")

    predict_titles(
        pred_path=os.path.join(base_path, "data/predictions/predictions.csv"),
        movies_path=os.path.join(base_path, "data/raw/movies.csv"),
        output_dir=os.path.join(base_path, "data/predictions"),
        mapping_path=os.path.join(base_path, "models/model_ids.pkl")  # ✅ Richtige Pickle-Datei
    )

if __name__ == "__main__":
    main()