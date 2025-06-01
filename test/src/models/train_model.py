# train_model.py

import pandas as pd
import numpy as np
import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# === Metriken ===

def precision_at_k(relevant_items, recommended_items, k=10):
    recommended_k = recommended_items[:k]
    return len(set(recommended_k) & set(relevant_items)) / k

def mean_average_precision_at_k(true_items_list, predicted_items_list, k=10):
    return np.mean([
        precision_at_k(true, pred, k)
        for true, pred in zip(true_items_list, predicted_items_list)
        if true
    ])

# === Modell-Training ===

def train_model(user_matrix):
    X = user_matrix.drop("userId", axis=1)

    # Verzeichnisse vorbereiten
    for path in ["/opt/airflow/model_cache", "models"]:
        os.makedirs(path, exist_ok=True)

    # Spalten speichern für spätere Inferenz
    with open("/opt/airflow/model_cache/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    # Optional: Feature-Namen loggen
    with open("models/feature_names.txt", "w") as f:
        f.writelines([col + "\n" for col in X.columns])

    return NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(X)

# === Evaluation mit Holdout ===

def get_holdout_eval_data(df, test_fraction=0.2):
    return train_test_split(df, test_size=test_fraction, random_state=42)

def evaluate_model(model, train_df, test_df):
    item_cols = train_df.drop("userId", axis=1).columns.tolist()
    test_df = test_df.sample(n=100, random_state=42)

    predictions = []
    ground_truths = []

    for _, row in test_df.iterrows():
        user_input = row[item_cols].to_frame().T
        _, indices = model.kneighbors(user_input)

        neighbors = train_df.iloc[indices[0]]
        item_scores = neighbors[item_cols].sum().sort_values(ascending=False)
        recommended_items = item_scores.index.tolist()[:10]

        actual_items = list(row[item_cols][row[item_cols] > 0].index)

        predictions.append(recommended_items)
        ground_truths.append(actual_items)

    return {
        "map_10": mean_average_precision_at_k(ground_truths, predictions, 10),
        "precision_10": np.mean([
            precision_at_k(gt, pred, 10)
            for gt, pred in zip(ground_truths, predictions)
            if gt
        ])
    }

# === Hauptfunktion ===

def main(
    input_filepath="/opt/airflow/data/processed",
    model_path="/opt/airflow/models/model.pkl"
):
    user_matrix_path = os.path.join(input_filepath, "movies_matrix.csv")

    mlflow.set_experiment("movie_recommendation")
    with mlflow.start_run(run_name="knn_model") as run:
        try:
            print(f"📥 Lade Daten aus: {user_matrix_path}")
            user_matrix = pd.read_csv(user_matrix_path)

            train_df, test_df = get_holdout_eval_data(user_matrix)
            model = train_model(train_df)

            mlflow.log_param("n_neighbors", 20)
            mlflow.log_param("algorithm", "ball_tree")

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact("models/feature_names.txt")

            metrics = evaluate_model(model, train_df, test_df)
            mlflow.log_metric("map_10", metrics["map_10"])
            mlflow.log_metric("precision_10", metrics["precision_10"])

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="movie_model",
                registered_model_name="movie_model"
            )

            print("✅ Modell erfolgreich trainiert & registriert.")
            print("📊 Metriken:", metrics)

        except Exception as e:
            print(f"❌ Fehler beim Training: {e}")
            mlflow.set_tag("status", "failed")
            raise

if __name__ == "__main__":
    main()