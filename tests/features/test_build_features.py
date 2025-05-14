import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.build_features import read_ratings, read_movies, create_user_matrix  # ggf. Pfad anpassen


class TestBuildFeatures(unittest.TestCase):

    @patch("features.build_features.pd.read_csv")
    def test_read_ratings_encodes_movie_ids(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "userId": [1, 2],
            "movieId": [101, 202],
            "rating": [4.0, 5.0],
            "timestamp": [111111, 222222]
        })
        mock_read_csv.return_value = mock_df

        result = read_ratings("ratings.csv", data_dir="dummy/")
        self.assertListEqual(list(result.columns), ["userId", "movieId", "rating", "timestamp"])
        self.assertTrue(result["movieId"].isin([0, 1]).all())

    @patch("features.build_features.pd.read_csv")
    def test_read_movies_extracts_genres(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "movieId": [1, 2],
            "title": ["Movie A", "Movie B"],
            "genres": ["Action|Comedy", "Drama"]
        })
        mock_read_csv.return_value = mock_df

        result = read_movies("movies.csv", data_dir="dummy/")
        expected_columns = ["movieId", "title", "Action", "Comedy", "Drama"]
        for col in expected_columns:
            self.assertIn(col, result.columns)

    def test_create_user_matrix_mean_aggregation(self):
        ratings = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [10, 20, 30],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [123, 456, 789]
        })
        movies = pd.DataFrame({
            "movieId": [10, 20, 30],
            "title": ["M1", "M2", "M3"],
            "Action": [1, 0, 0],
            "Comedy": [0, 1, 0]
        })

        user_matrix = create_user_matrix(ratings, movies)
        self.assertListEqual(list(user_matrix.columns), ["Action", "Comedy"])
        self.assertEqual(user_matrix.shape[0], 2)
        self.assertAlmostEqual(user_matrix.loc[1, "Action"], 0.5)  # Mittelwert von [1,0]
        self.assertAlmostEqual(user_matrix.loc[1, "Comedy"], 0.5)

    def test_create_user_matrix_handles_missing_merge(self):
        ratings = pd.DataFrame({
            "userId": [1],
            "movieId": [999],  # kommt nicht in movies vor
            "rating": [5.0],
            "timestamp": [123]
        })
        movies = pd.DataFrame({
            "movieId": [1],
            "title": ["X"],
            "Action": [1],
            "Comedy": [0]
        })

        result = create_user_matrix(ratings, movies)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
