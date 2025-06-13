import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.models.validate_model import validate_deep_hybrid, update_best_model_in_mlflow, get_latest_model_version

class TestValidateModel(unittest.TestCase):
    @patch("src.models.validate_model.pd.read_csv")
    @patch("src.models.validate_model.pickle.load")
    def test_validate_deep_hybrid(self, mock_pickle_load, mock_read_csv):
        # Mock data
        mock_ratings = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [101, 102, 103],
            "rating": [4.0, 5.0, 3.0]
        })
        mock_embedding = pd.DataFrame({
            101: [0.1, 0.2],
            102: [0.3, 0.4],
            103: [0.5, 0.6]
        }, index=[1, 2])
        mock_knn_model = MagicMock()
        mock_knn_model.n_features_in_ = 3
        mock_knn_model.kneighbors.return_value = (None, [[101, 102]])

        mock_read_csv.side_effect = [mock_ratings, mock_embedding]
        mock_pickle_load.return_value = mock_knn_model

        # Test
        validate_deep_hybrid(test_user_count=2)

        # Assertions
        mock_read_csv.assert_any_call("./data/raw/ratings.csv")
        mock_read_csv.assert_any_call("./data/processed/hybrid_deep_embedding.csv")
        mock_pickle_load.assert_called_once()

    @patch("src.models.validate_model.MlflowClient")
    def test_update_best_model_in_mlflow(self, mock_mlflow_client):
        # Mock client
        mock_client = MagicMock()
        mock_mlflow_client.return_value = mock_client

        # Test
        update_best_model_in_mlflow(0.9, mock_client, "test_model", "1")

        # Assertions
        mock_client.set_registered_model_alias.assert_called_once_with("test_model", "best_model", "1")

    @patch("src.models.validate_model.MlflowClient")
    def test_get_latest_model_version(self, mock_mlflow_client):
        # Mock client
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.creation_timestamp = 123456789
        mock_version.version = "1"
        mock_version.run_id = "run_123"
        mock_client.search_model_versions.return_value = [mock_version]

        # Test
        version, run_id = get_latest_model_version(mock_client, "test_model")

        # Assertions
        self.assertEqual(version, "1")
        self.assertEqual(run_id, "run_123")

if __name__ == "__main__":
    unittest.main()
