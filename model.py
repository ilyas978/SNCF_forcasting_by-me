import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "model_lgb.pkl"

class SNCFModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        try:
            self.model = joblib.load(model_path)
            print(f"Modèle chargé depuis : {model_path}")
        except Exception as e:
            print(f"ERREUR lors du chargement du modèle : {e}")
            self.model = None

    def predict(self, input_df: pd.DataFrame):
        if self.model is None:
            return None
        return self.model.predict(input_df)