import unittest
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# sys.path setzen, wenn du src-layout nutzt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.train_model import train_model  # Pfad ggf. anpassen


class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Beispiel-Matrix mit 3 Nutzern und 3 Features + movieId
        self.df = pd.DataFrame({
            "movieId": [1, 2, 3],
            "feature1": [1.0, 0.0, 1.0],
            "feature2": [0.0, 1.0, 1.0],
            "feature3": [1.0, 1.0, 0.0],
        })

    def test_model_is_trained(self):
        model = train_model(self.df)
        self.assertIsInstance(model, NearestNeighbors)

    def test_model_knn_neighbors(self):
        model = train_model(self.df)
        self.assertEqual(model.n_neighbors, 20)

    def test_error_when_movieId_missing(self):
        df_invalid = self.df.drop(columns=["movieId"])
        with self.assertRaises(KeyError):
            train_model(df_invalid)

    def test_error_with_too_few_rows(self):
        # Weniger Zeilen als n_neighbors → sollte klappen, aber Warnung kann auftreten
        small_df = self.df.iloc[:2]
        try:
            model = train_model(small_df)
            self.assertIsInstance(model, NearestNeighbors)
        except Exception as e:
            self.fail(f"train_model raised {type(e).__name__} unexpectedly!")


if __name__ == '__main__':
    unittest.main()
