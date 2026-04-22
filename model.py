import lightgbm as lgb
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "illyyyaaaasssss/sncf-model"
FILENAME = "model_lgb.txt"

class SNCFModel:
    def __init__(self):
        try:
            print("Téléchargement du modèle depuis Hugging Face...")
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            self.model = lgb.Booster(model_file=model_path)
            print("Modèle chargé avec succès !")
        except Exception as e:
            print(f"ERREUR : {e}")
            self.model = None

    def predict(self, input_df: pd.DataFrame):
        if self.model is None:
            return None
        return self.model.predict(input_df)