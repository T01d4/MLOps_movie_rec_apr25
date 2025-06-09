import os
import unittest
from src.data.check_structure import check_existing_file, check_existing_folder

class TestCheckStructure(unittest.TestCase):
    # ...existing code...

    def test_check_existing_file(self):
        # Create a temporary file for testing
        temp_file = "temp_test_file.txt"
        with open(temp_file, "w") as f:
            f.write("test")
        
        # Test if the file exists
        self.assertTrue(check_existing_file(temp_file))
        
        # Remove the temporary file
        os.remove(temp_file)
        
        # Test if the file does not exist
        self.assertFalse(check_existing_file(temp_file))

    def test_check_existing_folder(self):
        # Create a temporary folder path
        temp_folder = "temp_test_folder"
        
        # Ensure the folder does not exist initially
        if os.path.exists(temp_folder):
            os.rmdir(temp_folder)
        
        # Test folder creation
        self.assertTrue(check_existing_folder(temp_folder))
        self.assertTrue(os.path.exists(temp_folder))
        
        # Test if the folder already exists
        self.assertFalse(check_existing_folder(temp_folder))
        
        # Remove the temporary folder
        os.rmdir(temp_folder)

# ...existing code...
if __name__ == "__main__":
    unittest.main()
