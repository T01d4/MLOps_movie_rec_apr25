#predict_model-py
import pandas as pd
import pickle
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def make_predictions(users_id, model_filename, user_matrix_filename, columns_path):
    users = pd.read_csv(user_matrix_filename)
    users = users[users["userId"].isin(users_id)]
    users = users.drop("userId", axis=1)

    # Erwartete Feature-Namen laden
    with open(columns_path, "rb") as f:
        expected_columns = pickle.load(f)

    users = users.reindex(columns=expected_columns, fill_value=0)

    with open(model_filename, "rb") as filehandler:
        model = pickle.load(filehandler)

    _, indices = model.kneighbors(users)
    selection = np.array([
        np.random.choice(row, size=10, replace=False) for row in indices
    ])

    return selection

def main(model_path, user_matrix_path, columns_path, output_path):
    users_id = [1, 2, 3, 4, 5]  # TODO: Übergabe als Parameter möglich

    predictions = make_predictions(users_id, model_path, user_matrix_path, columns_path)

    col_names = [f"title_{i+1}" for i in range(predictions.shape[1])]
    df = pd.DataFrame(predictions, columns=col_names)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logging.info(f"✅ Prediction gespeichert unter {output_path}")

if __name__ == "__main__":
    base_path = "."  # oder z. B. "/opt/airflow"

    main(
        model_path=os.path.join(base_path, "models/model.pkl"),
        user_matrix_path=os.path.join(base_path, "data/processed/user_matrix.csv"),
        columns_path=os.path.join(base_path, "model_cache/columns.pkl"),
        output_path=os.path.join(base_path, "data/predictions/predictions.csv")
    )