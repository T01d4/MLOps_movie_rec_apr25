import unittest
from unittest.mock import patch, mock_open, MagicMock
import os

# sys.path setzen, falls du src-layout nutzt
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.import_raw_data import import_raw_data


class TestImportRawData(unittest.TestCase):

    @patch("data.import_raw_data.os.makedirs")
    @patch("data.import_raw_data.os.path.exists")
    def test_directory_created_if_not_exists(self, mock_exists, mock_makedirs):
        mock_exists.side_effect = [False] + [True]*10  # First: dir doesn't exist, then all files do
        import_raw_data("dummy/path", ["file1.csv"], "https://example.com/")
        mock_makedirs.assert_called_once_with("dummy/path")

    @patch("data.import_raw_data.requests.get")
    @patch("data.import_raw_data.open", new_callable=mock_open)
    @patch("data.import_raw_data.os.path.exists")
    def test_file_downloaded_when_not_exists(self, mock_exists, mock_open_func, mock_requests_get):
        mock_exists.side_effect = [True, False]  # dir exists, file doesn't
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "test content"
        mock_requests_get.return_value = mock_response

        import_raw_data("some/path", ["file1.csv"], "https://example.com/")

        mock_requests_get.assert_called_once_with("https://example.com/file1.csv")
        mock_open_func.assert_called_once_with(os.path.join("some/path", "file1.csv"), "wb")
        handle = mock_open_func()
        handle.write.assert_called_once_with(b"test content")

    @patch("data.import_raw_data.requests.get")
    @patch("data.import_raw_data.os.path.exists")
    def test_file_skipped_if_already_exists(self, mock_exists, mock_requests_get):
        mock_exists.side_effect = [True, True]  # dir exists, file exists
        import_raw_data("existing/path", ["file1.csv"], "https://example.com/")
        mock_requests_get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
