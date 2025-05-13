import os
python_exe = "venv312\\Scripts\\python.exe"

steps = [
    f"{python_exe} src/data/import_raw_data.py",
    f"{python_exe} src/data/make_dataset.py data/raw data/processed",
    f"{python_exe} src/features/build_features.py",
    f"{python_exe} src/models/train_model.py"
]

for step in steps:
    print(f"\n Starte: {step}")
    if os.system(step) != 0:
        print(f" Fehler bei: {step}")
        break
    else:
        print(f" Fertig: {step}")