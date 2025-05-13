import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# sys.path für src-layout ergänzen, falls nötig
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.predict_model import make_predictions  # ggf. Pfad anpassen


class TestPredictModel(unittest.TestCase):

    @patch("models.predict_model.pd.read_csv")
    @patch("models.predict_model.pickle.load")
    @patch("models.predict_model.open", create=True)
    def test_make_predictions_shape_and_content(self, mock_open, mock_pickle_load, mock_read_csv):
        # Mock user matrix
        mock_user_matrix = pd.DataFrame({
            "userId": [1, 2, 3],
            "f1": [0.1, 0.2, 0.3],
            "f2": [0.4, 0.5, 0.6]
        })
        mock_read_csv.return_value = mock_user_matrix

        # Mock model with kneighbors method
        mock_model = MagicMock()
        mock_model.kneighbors.return_value = (
            np.zeros((3, 20)),  # dummy distances
            np.array([np.arange(20), np.arange(20), np.arange(20)])
        )
        mock_pickle_load.return_value = mock_model

        users_id = [1, 2, 3]
        predictions = make_predictions(users_id, "dummy_model.pkl", "dummy_users.csv")

        # Assertions
        self.assertEqual(predictions.shape, (3, 10))
        self.assertTrue(np.all(predictions < 20))  # da die mock-indices 0–19 sind
        self.assertEqual(predictions.dtype, int)  # standardmäßig bei choice int

    @patch("models.predict_model.pd.read_csv")
    def test_invalid_user_ids_return_empty(self, mock_read_csv):
        mock_user_matrix = pd.DataFrame({
            "userId": [10, 11],
            "f1": [0.1, 0.2],
            "f2": [0.3, 0.4]
        })
        mock_read_csv.return_value = mock_user_matrix

        with self.assertRaises(ValueError):
            make_predictions([1, 2], "model.pkl", "user_matrix.csv")

    @patch("models.predict_model.pd.read_csv")
    @patch("models.predict_model.pickle.load")
    @patch("models.predict_model.open", create=True)
    def test_model_kneighbors_called_correctly(self, mock_open, mock_pickle_load, mock_read_csv):
        mock_user_matrix = pd.DataFrame({
            "userId": [1],
            "f1": [0.1],
            "f2": [0.2]
        })
        mock_read_csv.return_value = mock_user_matrix

        mock_model = MagicMock()
        mock_model.kneighbors.return_value = (np.zeros((1, 20)), np.array([np.arange(20)]))
        mock_pickle_load.return_value = mock_model

        make_predictions([1], "model.pkl", "user_matrix.csv")
        mock_model.kneighbors.assert_called_once()


if __name__ == "__main__":
    unittest.main()
