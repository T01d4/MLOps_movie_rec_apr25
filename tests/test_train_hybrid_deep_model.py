import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.movie.models.train_hybrid_deep_model import train_hybrid_deep_model

class TestTrainHybridDeepModel(unittest.TestCase):
    @patch("src.movie.models.train_hybrid_deep_model.pd.read_csv")
    @patch("src.movie.models.train_hybrid_deep_model.os.makedirs")
    @patch("src.movie.models.train_hybrid_deep_model.pickle.dump")
    @patch("src.movie.models.train_hybrid_deep_model.mlflow.start_run")
    def test_train_hybrid_deep_model(self, mock_mlflow_start_run, mock_pickle_dump, mock_makedirs, mock_read_csv):
        # Mock data
        mock_movies = pd.DataFrame({
            "movieId": [1, 2],
            "genres": ["Action|Adventure", "Comedy|Drama"],
            "tag": ["hero epic", "funny emotional"]
        })
        mock_tags = pd.DataFrame({
            "movieId": [1, 2],
            "tag": ["hero epic", "funny emotional"]
        })
        mock_scores = pd.DataFrame({
            "movieId": [1, 2],
            "tagId": [101, 102],
            "relevance": [0.8, 0.9]
        })

        mock_read_csv.side_effect = [mock_movies, mock_tags, mock_scores]

        # Test
        train_hybrid_deep_model(n_neighbors=5, latent_dim=32, epochs=10, tfidf_features=100)

        # Assertions
        mock_read_csv.assert_any_call("/opt/airflow/data/raw/movies.csv")
        mock_read_csv.assert_any_call("/opt/airflow/data/raw/tags.csv")
        mock_read_csv.assert_any_call("/opt/airflow/data/raw/genome-scores.csv")
        mock_makedirs.assert_called_once_with("/opt/airflow/models", exist_ok=True)
        mock_pickle_dump.assert_called_once()
        mock_mlflow_start_run.assert_called_once()

if __name__ == "__main__":
    unittest.main()
