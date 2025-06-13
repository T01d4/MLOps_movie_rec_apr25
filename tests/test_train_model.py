import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.movie.models.train_model import train_hybrid_deep_model

class TestTrainModel(unittest.TestCase):
    @patch("src.models.train_model.pd.read_csv")
    @patch("src.models.train_model.pickle.dump")
    @patch("src.models.train_model.os.makedirs")
    def test_train_hybrid_deep_model(self, mock_makedirs, mock_pickle_dump, mock_read_csv):
        # Mock data
        mock_embedding = pd.DataFrame({
            "feature1": [0.1, 0.2],
            "feature2": [0.3, 0.4],
            "feature3": [0.5, 0.6]
        }, index=[1, 2])
        mock_read_csv.return_value = mock_embedding

        # Test
        train_hybrid_deep_model()

        # Assertions
        mock_read_csv.assert_called_once_with("/opt/airflow/data/processed/hybrid_deep_embedding.csv", index_col=0)
        mock_makedirs.assert_called_once_with("/opt/airflow/models", exist_ok=True)
        mock_pickle_dump.assert_called_once()

if __name__ == "__main__":
    unittest.main()
