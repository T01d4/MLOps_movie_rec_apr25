import unittest
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
from data.make_dataset import process_data


class TestProcessData(unittest.TestCase):
    def setUp(self):
        self.test_dir = TemporaryDirectory()
        self.input_dir = Path(self.test_dir.name) / "raw"
        self.output_dir = Path(self.test_dir.name) / "processed"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sample input data
        (self.input_dir / "genome-scores.csv").write_text(
            "movieId,tagId,relevance\n1,10,0.9\n2,20,0.8\n"
        )
        (self.input_dir / "genome-tags.csv").write_text(
            "tagId,tag\n10,action\n20,comedy\n"
        )
        (self.input_dir / "movies.csv").write_text(
            "movieId,title,genres\n1,Movie A,Action\n2,Movie B,Comedy\n"
        )
        (self.input_dir / "ratings.csv").write_text(
            "userId,movieId,rating\n100,1,4.5\n101,2,3.5\n"
        )
        (self.input_dir / "tags.csv").write_text(
            "userId,movieId,tag\n100,1,fun\n101,2,exciting\n"
        )

        self.logger = logging.getLogger("TestLogger")

    def test_process_data_output(self):
        process_data(
            input_filepath_scores=self.input_dir / "genome-scores.csv",
            input_filepath_gtags=self.input_dir / "genome-tags.csv",
            input_filepath_movies=self.input_dir / "movies.csv",
            input_filepath_ratings=self.input_dir / "ratings.csv",
            input_filepath_tags=self.input_dir / "tags.csv",
            output_filepath=self.output_dir,
            logger=self.logger
        )

        output_file = self.output_dir / "movies_matrix.csv"
        self.assertTrue(output_file.exists(), "Output file was not created.")

        df = pd.read_csv(output_file)
        self.assertIn("userId", df.columns)
        self.assertGreater(len(df), 0)

    def tearDown(self):
        self.test_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
