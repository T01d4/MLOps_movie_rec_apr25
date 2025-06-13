import os

def check_existing_file(file_path):
    '''Check if a file already exists. Log the result.'''
    if os.path.isfile(file_path):
        print(f"⚠️ File already exists: {file_path}")
        return True
    else:
        print(f"✅ File does not exist (expected): {file_path}")
        return False

def check_existing_folder(folder_path):
    '''Check if a folder exists. If not, create it.'''
    if not os.path.exists(folder_path):
        print(f"📁 Folder not found, creating: {folder_path}")
        os.makedirs(folder_path)
        return True
    else:
        print(f"✅ Folder exists: {folder_path}")
        return False