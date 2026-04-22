# SNCF Delay Forecasting — End-to-End ML Pipeline

Prediction of train delays using a LightGBM model served via a FastAPI REST API, containerized with Docker and deployed on Render.

## Architecture

```
Raw Data → Feature Engineering → LightGBM Model → FastAPI → Docker → Render
```

## Project Structure

```
├── main.py          # FastAPI app
├── model.py         # Model loader
├── features.py      # Feature engineering
├── models/
│   └── model_lgb.pkl
├── Dockerfile
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run with Docker

```bash
docker build -t sncf-api .
docker run -p 8000:8000 sncf-api
```

## API Usage

**POST** `/predict`

```json
{
  "date": "2023-11-13",
  "train": "ZPQEKP",
  "gare": "VXY",
  "arret": 0,
  "p0q2": -4.0,
  "p0q3": -2.0,
  "p0q4": -4.0,
  "p2q0": 0.0,
  "p3q0": 0.0,
  "p4q0": -2.0
}
```

**Response**

```json
{
  "prediction": -1.52
}
```

## Model

- Algorithm: LightGBM
- Target: `p0q0` (delay in minutes)
- Features: 27 engineered features (lag, trends, time, station stats)
- Evaluation metric: MAE
