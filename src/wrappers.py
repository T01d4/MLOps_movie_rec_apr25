def import_raw_data():
    from src.data.import_raw_data import main
    main()

def make_dataset():
    from src.data.make_dataset import main
    main(input_filepath="data/raw", output_filepath="data/processed")

def build_features():
    from src.features.build_features import main
    main()  # 
    
def train_model():
    from src.models.train_model import main
    main(input_filepath="data/processed", model_path="models/model.pkl")

def predict_model():
    from src.models.predict_model import main
    main(
        model_path="models/model.pkl",
        user_matrix_path="data/processed/user_matrix.csv",
        output_path="data/predictions/predictions.csv"
    )
