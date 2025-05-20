import pandas as pd
import pickle
import numpy as np
import os


def make_predictions(users_id, model_filename, user_matrix_filename):
    # Read user_matrix
    users = pd.read_csv(user_matrix_filename)

    # Filter with the list of users_id
    users = users[users["userId"].isin(users_id)]

    # Delete userId
    users = users.drop("userId", axis=1)

    # Open model
    with open(model_filename, "rb") as filehandler:
        model = pickle.load(filehandler)

    # Calculate nearest neighbors
    _, indices = model.kneighbors(users)

    # Select 10 random numbers from each row
    selection = np.array(
        [np.random.choice(row, size=10, replace=False) for row in indices]
    )

    return selection


def main(model_path, user_matrix_path, output_path):
    # Nimm z.B. die ersten 5 Nutzer
    users_id = [1, 2, 3, 4, 5]

    predictions = make_predictions(users_id, model_path, user_matrix_path)

    # Als DataFrame speichern
    df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Prediction gespeichert unter {output_path}")


# Lokaler Test
if __name__ == "__main__":
    main(
        model_path="models/model.pkl",
        user_matrix_path="data/processed/user_matrix.csv",
        output_path="data/predictions/predictions.csv"
    )
