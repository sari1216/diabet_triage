# Changelog

## v0.1 – Baseline

- Model: StandardScaler + LinearRegression
- Held-out split: 80/20, random_state=42
- Test RMSE: 53.85
- FastAPI service with GET /health and POST /predict
- Invalid input returns HTTP 422 with a JSON error
- Docker image with the model baked in, published to GHCR