import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.movie.models.train_model import train_model

class TestTrainModel(unittest.TestCase):
    @patch("src.movie.models.train_model.pd.read_csv")
    @patch("src.movie.models.train_model.pickle.dump")
    @patch("src.movie.models.train_model.os.makedirs")
    def test_train_model(self, mock_makedirs, mock_pickle_dump, mock_read_csv):
        # Mock data
        mock_movie_matrix = pd.DataFrame({
            "movieId": [1, 2],
            "feature1": [0.1, 0.2],
            "feature2": [0.3, 0.4],
            "feature3": [0.5, 0.6]
        })
        mock_read_csv.return_value = mock_movie_matrix

        # Test
        movie_matrix_path = "/app/data/processed/movie_matrix.csv"
        model_path = "/app/models/model.pkl"
        mock_read_csv.assert_called_once_with(movie_matrix_path)
        model = train_model(mock_movie_matrix)
        mock_makedirs.assert_called_once_with("/app/models", exist_ok=True)
        mock_pickle_dump.assert_called_once_with(model, MagicMock())

if __name__ == "__main__":
    unittest.main()
