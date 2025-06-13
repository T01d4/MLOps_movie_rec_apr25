import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.movie.models.predict_model import make_predictions

class TestPredictModel(unittest.TestCase):
    @patch("src.movie.models.predict_model.pd.read_csv")
    @patch("src.movie.models.predict_model.pickle.load")
    def test_make_predictions(self, mock_pickle_load, mock_read_csv):
        # Mock data
        mock_users = pd.DataFrame({
            "userId": [1, 2, 3],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [0.4, 0.5, 0.6],
            "feature3": [0.7, 0.8, 0.9]
        })
        mock_model = MagicMock()
        mock_model.kneighbors.return_value = (None, np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]))

        mock_read_csv.return_value = mock_users
        mock_pickle_load.return_value = mock_model

        # Test
        users_id = [1, 2]
        predictions = make_predictions(users_id, "mock_model.pkl", "mock_user_matrix.csv")

        # Assertions
        mock_read_csv.assert_called_once_with("mock_user_matrix.csv")
        mock_pickle_load.assert_called_once_with(open("mock_model.pkl", "rb"))
        mock_model.kneighbors.assert_called_once()
        self.assertEqual(predictions.shape, (2, 10))  # Ensure predictions have the correct shape
        self.assertTrue(np.all(np.isin(predictions, [10, 20, 30, 40, 50, 60, 70, 80, 90])))

if __name__ == "__main__":
    unittest.main()
