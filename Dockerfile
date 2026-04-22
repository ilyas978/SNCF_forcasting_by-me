FROM python:3.11-slim

WORKDIR /app

# Copier les dépendances en premier (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Port exposé par Render
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
