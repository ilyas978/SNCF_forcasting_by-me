import pandas as pd
import numpy as np

# Ordre exact des colonnes utilisées à l'entraînement


def apply_feature_engineering(df):
    FEATURE_COLS = [
    'train', 'gare', 'arret', 'p2q0', 'p3q0', 'p4q0',
    'p0q2', 'p0q3', 'p0q4', 'dow', 'month', 'woy',
    'train_lag_mean', 'train_lag_std', 'train_lag_max', 'train_lag_min',
    'train_lag_pondere', 'S_pondere', 'S_moy', 'S_std',
    'trend_station_1', 'trend_station_2', 'trend_train_1', 'trend_train_2',
    'train_pos_lag', 'S_pos_moy', 'moy_per_station'
]
    df = df.copy()

    # 1. Temps
    if 'date' in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["dow"]   = df["date"].dt.day_of_week
        df["month"] = df["date"].dt.month
        df["woy"]   = df["date"].dt.isocalendar().week.astype(int)

    # 2. Lag features Train
    df["train_lag_mean"] = df[["p0q2", "p0q3", "p0q4"]].mean(axis=1)
    df["train_lag_std"]  = df[["p0q2", "p0q3", "p0q4"]].std(axis=1).fillna(0.0)
    df["train_lag_max"]  = df[["p0q2", "p0q3", "p0q4"]].max(axis=1)
    df["train_lag_min"]  = df[["p0q2", "p0q3", "p0q4"]].min(axis=1)

    # 3. Lag features Station
    df["S_moy"] = df[["p2q0", "p3q0", "p4q0"]].mean(axis=1)
    df["S_std"] = df[["p2q0", "p3q0", "p4q0"]].std(axis=1).fillna(0.0)

    # 4. Pondérations
    df["train_lag_pondere"] = 0.5*df["p0q2"] + 0.3*df["p0q3"] + 0.2*df["p0q4"]
    df["S_pondere"]         = 0.5*df["p2q0"] + 0.3*df["p3q0"] + 0.2*df["p4q0"]

    # 5. Trends
    df["trend_station_1"] = df["p2q0"] - df["p3q0"]
    df["trend_station_2"] = df["p3q0"] - df["p4q0"]
    df["trend_train_1"]   = df["p0q2"] - df["p0q3"]
    df["trend_train_2"]   = df["p0q3"] - df["p0q4"]

    # 6. Features binaires
    df["train_pos_lag"] = (df["train_lag_mean"] > 0).astype(int)
    df["S_pos_moy"]     = (df["S_moy"] > 0).astype(int)

    # 7. moy_per_station — fallback sur p2q0 en production (1 seule ligne)
    df["moy_per_station"] = df["p2q0"]

    # 8. Types catégoriels
    for col in ["train", "gare"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 9. Retourner exactement les colonnes dans le bon ordre
    return df[FEATURE_COLS]