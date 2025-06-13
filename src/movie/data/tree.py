import os

def load_ignore_patterns(ignore_file_paths):
    """Load ignore patterns from .dockerignore and .gitignore files."""
    ignore_patterns = set()
    for ignore_file_path in ignore_file_paths:
        if os.path.exists(ignore_file_path):
            with open(ignore_file_path, "r") as f:
                for line in f:
                    pattern = line.strip()
                    if pattern and not pattern.startswith("#"):
                        ignore_patterns.add(pattern)
    ignore_patterns.update(["*.log", "*run_id=", "*task_id="])  # Explicitly exclude .log files, run_id, and task_id information
    return ignore_patterns

def should_ignore(name, ignore_patterns):
    """Check if a file or directory should be excluded based on patterns."""
    for pattern in ignore_patterns:
        if name.startswith(pattern) or name.endswith(pattern) or pattern in name:
            return True
    return False

def print_directory_structure(root_dir, ignore_file_paths):
    ignore_patterns = load_ignore_patterns(ignore_file_paths)
    for root, dirs, files in os.walk(root_dir):
        # Exclude directories and files based on ignore patterns
        dirs[:] = [d for d in dirs if not should_ignore(d, ignore_patterns)]
        files = [f for f in files if not should_ignore(f, ignore_patterns)]
        # Print the directory structure excluding ignored items
        level = root.replace(root_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for d in dirs:
            print(f"{sub_indent}{d}/")
        for f in files:
            print(f"{sub_indent}{f}")

print_directory_structure(
    "/Users/moustafaal-halabi/Documents/GitHub/MLOps_movie_rec_apr25",
    [
        "/Users/moustafaal-halabi/Documents/GitHub/MLOps_movie_rec_apr25/src/streamlit_app/.dockerignore",
        "/Users/moustafaal-halabi/Documents/GitHub/MLOps_movie_rec_apr25/.gitignore",
        "/Users/moustafaal-halabi/Documents/GitHub/MLOps_movie_rec_apr25/src/airflow/.dockerignore",
        "/Users/moustafaal-halabi/Documents/GitHub/MLOps_movie_rec_apr25/src/airflow/.gitignore"
    ]
)