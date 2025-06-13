import os
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.movie.data.make_dataset import main

class TestMakeDataset(unittest.TestCase):
    @patch("src.movie.data.make_dataset.os.makedirs")
    @patch("src.movie.data.make_dataset.pd.read_csv")
    @patch("src.movie.data.make_dataset.pd.DataFrame.to_csv")
    def test_main(self, mock_to_csv, mock_read_csv, mock_makedirs):
        # Setup
        input_filepath = "/tmp/test_raw_data"
        output_filepath = "/tmp/test_processed_data"
        movie_matrix_path = os.path.join(output_filepath, "movies_matrix.csv")

        # Mock data
        mock_scores = pd.DataFrame({
            "movieId": [1, 2],
            "tagId": [101, 102],
            "relevance": [0.8, 0.9]
        })
        mock_ratings = pd.DataFrame({
            "userId": [1, 1],
            "movieId": [1, 2],
            "rating": [4.0, 5.0]
        })
        mock_read_csv.side_effect = [mock_scores, mock_ratings]

        # Test
        main(input_filepath=input_filepath, output_filepath=output_filepath)

        # Assertions
        mock_makedirs.assert_called_once_with(output_filepath, exist_ok=True)
        mock_read_csv.assert_any_call(os.path.join(input_filepath, "genome-scores.csv"), dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})
        mock_read_csv.assert_any_call(os.path.join(input_filepath, "ratings.csv"), usecols=["userId", "movieId", "rating"], dtype={"userId": "int32", "movieId": "int32", "rating": "float32"})
        mock_to_csv.assert_called_once_with(movie_matrix_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    unittest.main()
