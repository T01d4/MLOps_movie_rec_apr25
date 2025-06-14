import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.movie.models.predict_best_model import predict_best_model

class TestPredictBestModel(unittest.TestCase):
    @patch("src.movie.models.predict_best_model.load_artifact_df_from_best_model")
    @patch("src.movie.models.predict_best_model.mlflow.pyfunc.load_model")
    def test_predict_best_model(self, mock_load_model, mock_load_artifact_df):
        # Mock data
        mock_embedding = pd.DataFrame({
            "emb_0": [0.1, 0.2],
            "emb_1": [0.3, 0.4],
            "emb_2": [0.5, 0.6]
        }, index=[1, 2])
        mock_model = MagicMock()
        mock_model.predict.return_value = [[101, 102], [103, 104]]

        mock_load_artifact_df.return_value = mock_embedding
        mock_load_model.return_value = mock_model

        # Test
        predict_best_model(n_users=2)

        # Assertions
        mock_load_artifact_df.assert_called_once_with("hybrid_deep_model", "./data/processed/best_embedding/hybrid_deep_embedding_best.csv")
        mock_load_model.assert_called_once_with("models:/hybrid_deep_model@best_model")
        mock_model.predict.assert_called_once()

if __name__ == "__main__":
    unittest.main()
