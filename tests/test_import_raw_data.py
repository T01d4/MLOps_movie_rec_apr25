import os
import unittest
from unittest.mock import patch, MagicMock
from src.movie.data.import_raw_data import import_raw_data

class TestImportRawData(unittest.TestCase):
    @patch("src.movie.data.import_raw_data.requests.get")
    @patch("os.makedirs")
    def test_import_raw_data(self, mock_makedirs, mock_requests_get):
        # Setup
        raw_data_path = "/tmp/test_raw_data"
        filenames = ["test_file.csv"]
        bucket_folder_url = "https://example.com/"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test content"
        mock_requests_get.return_value = mock_response

        # Test
        import_raw_data(raw_data_path, filenames, bucket_folder_url)

        # Assertions
        mock_makedirs.assert_called_once_with(raw_data_path, exist_ok=True)
        mock_requests_get.assert_called_once_with(
            os.path.join(bucket_folder_url, filenames[0]), timeout=10
        )
        output_file = os.path.join(raw_data_path, filenames[0])
        self.assertTrue(os.path.exists(output_file))

        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    unittest.main()
