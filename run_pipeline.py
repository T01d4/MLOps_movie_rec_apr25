import os

os.environ["PYTHONPATH"] = "src"

steps = [
    f"python src/data/import_raw_data.py",
    f"python src/data/make_dataset.py",
    f"python src/features/build_features.py",
    f"python src/models/train_model.py"
]

for step in steps:
    print(f"\n Starte: {step}")
    if os.system(step) != 0:
        print(f" Fehler bei: {step}")
        break
    else:
        print(f" Fertig: {step}")
