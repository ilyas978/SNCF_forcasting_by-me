from fastapi import FastAPI, HTTPException
import pandas as pd
import traceback
from features import apply_feature_engineering
from model import SNCFModel

app = FastAPI(title="SNCF Delay Predictor API")
predictor = SNCFModel()


@app.get("/")
def read_root():
    return {"message": "SNCF Forecasting API is running"}


@app.post("/predict")
def predict(data: dict):
    """
    Receives a dictionary (JSON) with raw data,
    applies feature engineering, and returns the prediction.
    """
    print("--- Début de la requête ---")

    try:
        # 1. Load data into DataFrame
        df_raw = pd.DataFrame([data])
        print("Data chargée")

        # 2. Apply feature engineering
        df_features = apply_feature_engineering(df_raw)
        print("Feature engineering terminé")

        # 3. Drop columns not used by the model (e.g., date)
        X = df_features.drop(columns=["date"], errors="ignore")
        print("Colonnes envoyées au modèle :", X.columns.tolist())
        print("Nombre de colonnes :", len(X.columns))
        # 4. Predict
        if predictor.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prediction = predictor.predict(X)
        print(f"Prédiction réussie : {prediction[0]}")

        return {"prediction": float(prediction[0])}

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is

    except Exception as e:
        error_details = traceback.format_exc()
        print("!!! ERREUR !!!")
        print(error_details)
        return {"error": str(e), "traceback": error_details}  # ← return au lieu de raise