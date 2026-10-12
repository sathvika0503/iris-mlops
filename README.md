# Basic MLOps Iris Starter (Docker + FastAPI + MLflow)

This is a starter template. Add or replace `data/iris.csv` with your dataset, then adapt `app/train.py` if your dataset has different columns or labels.

## Included
- FastAPI prediction API
- scikit-learn baseline classifier (kept lightweight so the starter builds more easily)
- MLflow experiment tracking
- Docker Compose
- CSV validation and manual retraining endpoint
- Persistent local folders for dataset, model, and MLflow tracking data

> This is a basic starter, not a guarantee that arbitrary dataset changes will produce correct predictions. Keep the CSV schema consistent and test after changing data.

## Expected CSV schema
The included Iris CSV should have these columns:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (setosa, versicolor, virginica)

## Run on Windows
1. Install Docker Desktop and start it.
2. Extract this ZIP to a simple path, for example `C:\mlops-iris-starter`.
3. Open Command Prompt in the extracted folder.
4. Run:
   ```bat
   docker compose up --build
   ```
5. Open:
   - API docs: http://localhost:8000/docs
   - Health: http://localhost:8000/health
   - MLflow UI: http://localhost:5000

Keep the terminal open while it runs. First build can take a few minutes.

## Try a prediction
In `http://localhost:8000/docs`, expand `POST /predict`, click **Try it out**, and use:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Update the dataset
1. Stop the app with `Ctrl+C`.
2. Edit `data/iris.csv` in Excel or VS Code. Keep the same column names and valid labels.
3. Start again with `docker compose up --build`.
4. Or call `POST /retrain` to train again after saving the CSV.

This starter uses a manual retraining endpoint. Automatic file-watching can be added later.

## Add your own dataset
If your CSV has different columns:
1. Replace `data/iris.csv`.
2. Edit `FEATURE_COLUMNS` and `TARGET_COLUMN` in `app/train.py`.
3. Update the input model in `app/main.py` so `/predict` accepts the feature fields you need.
4. Update validation rules for your target labels.
5. Rebuild/restart the service.

## API keys
No API keys are required for this local starter. If you later connect an external service, store secrets in a `.env` file (never commit it to GitHub) and add the variable names to `docker-compose.yml`. Do not paste real API keys into public chats or source code.

## Project structure
```text
basic_mlops_iris_starter/
├── app/
│   ├── main.py
│   └── train.py
├── data/
│   └── iris.csv
├── models/             # saved model artefacts
├── mlruns/             # persistent MLflow backend/artifacts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Notes
- MLflow is exposed only for local development. Do not expose it publicly without authentication.
- This template uses a simple scikit-learn RandomForest so you can verify the end-to-end Docker/API workflow first. Once it works, you can replace the model with TensorFlow/Keras if required by your project rubric.
