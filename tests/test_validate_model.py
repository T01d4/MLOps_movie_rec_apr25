import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.movie.models.validate_model import validate_deep_hybrid, update_best_model_in_mlflow, get_latest_model_version

class TestValidateModel(unittest.TestCase):
 
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
