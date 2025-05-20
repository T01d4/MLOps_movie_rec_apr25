import pandas as pd
import os
from src.visualization.predict_titles import predict_titles

def test_predict_titles(tmp_path):
    # Testdaten vorbereiten
    pred_path = tmp_path / "predictions.csv"
    movies_path = tmp_path / "movies.csv"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Dummy-Vorhersagen (MovieIDs)
    pd.DataFrame([[1, 2], [3, 4]]).to_csv(pred_path, index=False, header=False)

    # Dummy-Filmtitel
    pd.DataFrame({
        "movieId": [1, 2, 3, 4],
        "title": ["Movie A", "Movie B", "Movie C", "Movie D"]
    }).to_csv(movies_path, index=False)

    # Funktion aufrufen
    predict_titles(pred_path, movies_path, output_dir)

    # Assertions
    csv_output = output_dir / "predicted_titles.csv"
    html_output = output_dir / "predicted_titles.html"

    assert csv_output.exists()
    assert html_output.exists()

    df = pd.read_csv(csv_output)
    assert "Movie A" in df.values
    assert "Movie D" in df.values