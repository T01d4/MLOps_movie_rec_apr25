import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
from src.movie.data.import_raw_data import import_raw_data

class TestImportRawData(unittest.TestCase):
    @patch("src.movie.data.import_raw_data.requests.get")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_import_raw_data(self, mock_open, mock_makedirs, mock_requests_get):
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
        mock_requests_get.assert_called_once_with("https://example.com/test_file.csv", timeout=10)
        mock_open.assert_called_once_with(os.path.join(raw_data_path, "test_file.csv"), "wb")
        mock_open().write.assert_called_once_with(b"test content")

if __name__ == "__main__":
    unittest.main()