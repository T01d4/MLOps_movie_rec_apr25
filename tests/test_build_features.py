import os
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.movie.features.build_features import read_ratings, read_movies, create_user_matrix, main

class TestBuildFeatures(unittest.TestCase):
    @patch("src.movie.features.build_features.pd.read_csv")
    def test_read_ratings(self, mock_read_csv):
        # Mock data
        mock_data = pd.DataFrame({"movieId": [1, 2], "userId": [101, 102], "rating": [4.5, 3.0]})
        mock_read_csv.return_value = mock_data

        # Test
        result = read_ratings("ratings.csv")

        # Assertions
        mock_read_csv.assert_called_once_with(os.path.join("/opt/airflow/data/raw", "ratings.csv"))
        self.assertIn("movieId", result.columns)

    @patch("src.movie.features.build_features.pd.read_csv")
    def test_read_movies(self, mock_read_csv):
        # Mock data
        mock_data = pd.DataFrame({"movieId": [1, 2], "title": ["Movie1", "Movie2"], "genres": ["Action|Comedy", "Drama"]})
        mock_read_csv.return_value = mock_data

        # Test
        result = read_movies("movies.csv")

        # Assertions
        mock_read_csv.assert_called_once_with(os.path.join("/opt/airflow/data/raw", "movies.csv"))
        self.assertIn("Action", result.columns)
        self.assertIn("Comedy", result.columns)

    def test_create_user_matrix(self):
        # Mock data
        ratings = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [1, 2, 1],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [123456, 123457, 123458]
        })
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Movie1", "Movie2"],
            "Action": [1, 0],
            "Comedy": [0, 1]
        })

        # Test
        result = create_user_matrix(ratings, movies)

        # Assertions
        self.assertEqual(result.shape[0], 2)  # Two users
        self.assertIn("Action", result.columns)
        self.assertIn("Comedy", result.columns)

    @patch("src.movie.features.build_features.os.makedirs")
    @patch("src.movie.features.build_features.pd.DataFrame.to_csv")
    def test_main(self, mock_to_csv, mock_makedirs):
        # Test
        main(force_rebuild=True)

        # Assertions
        mock_makedirs.assert_called_once_with("/opt/airflow/data/processed", exist_ok=True)
        mock_to_csv.assert_any_call(os.path.join("/opt/airflow/data/processed", "movie_matrix.csv"), index=False)
        mock_to_csv.assert_any_call(os.path.join("/opt/airflow/data/processed", "user_matrix.csv"))

if __name__ == "__main__":
    unittest.main()
