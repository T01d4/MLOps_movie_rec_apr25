import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.movie.models.validate_model import validate_deep_hybrid, update_best_model_in_mlflow, get_latest_model_version

class TestValidateModel(unittest.TestCase):
    @patch("src.movie.models.validate_model.pd.read_csv")
    def test_validate_deep_hybrid(self, mock_read_csv):
        # Adjusted mock data to match expected structure
        mock_read_csv.side_effect = [
            pd.DataFrame({"userId": [1, 2], "movieId": [101, 102], "rating": [4.0, 5.0]}),
            pd.DataFrame({"movieId": [101, 102], "embedding": [0.1, 0.3]})
        ]
        validate_deep_hybrid(test_user_count=2)
        # Corrected assertion to match the exact file path
        mock_read_csv.assert_any_call("/opt/airflow/data/processed/hybrid_deep_embedding.csv")
        mock_read_csv.assert_any_call("/opt/airflow/data/processed/user_ratings.csv")

    @patch("src.movie.models.validate_model.MlflowClient")
    def test_update_best_model_in_mlflow(self, mock_mlflow_client):
        # Mock client
        mock_client = MagicMock()
        mock_mlflow_client.return_value = mock_client

        # Adjusted mock behavior to ensure alias setting is called
        mock_client.set_registered_model_alias = MagicMock()

        # Test
        update_best_model_in_mlflow(0.9, mock_client, "test_model", "1")

        # Assertions
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="test_model", alias="best_model", version="1"
        )

    @patch("src.movie.models.validate_model.MlflowClient")
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
