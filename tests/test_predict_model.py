import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.movie.models.predict_model import predict_hybrid_deep_model

class TestPredictModel(unittest.TestCase):
    @patch("src.movie.models.predict_model.pd.read_csv")
    @patch("src.movie.models.predict_model.pickle.load")
    def test_predict_hybrid_deep_model(self, mock_pickle_load, mock_read_csv):
        # Mock data
        mock_embedding = pd.DataFrame({
            "feature1": [0.1, 0.2],
            "feature2": [0.3, 0.4],
            "feature3": [0.5, 0.6]
        }, index=[1, 2])
        mock_knn_model = MagicMock()
        mock_knn_model.n_features_in_ = 3
        mock_knn_model.kneighbors.return_value = (None, [[101, 102]])

        mock_read_csv.return_value = mock_embedding
        mock_pickle_load.return_value = mock_knn_model

        # Test
        predictions = predict_hybrid_deep_model()

        # Assertions
        mock_read_csv.assert_called_once_with("/opt/airflow/data/processed/hybrid_deep_embedding.csv", index_col=0)
        mock_pickle_load.assert_called_once()
        self.assertEqual(len(predictions), 2)

if __name__ == "__main__":
    unittest.main()
