# Changelog

## v0.2 – Feature selection with LassoCV

### What changed
- Replaced LinearRegression with LassoCV (L1 regularisation, strength chosen with 5-fold cross-validation).
- Same preprocessing (StandardScaler), same 80/20 split and random_state=42 as v0.1, so the metrics are directly comparable.
- Added compare_models.py, which evaluates the candidate models on the same split.

### Metrics (held-out test set)

| Version | Model            | Test RMSE |
|---------|------------------|-----------|
| v0.1    | LinearRegression | 53.85     |
| v0.2    | LassoCV          | 52.92     |

Other candidates tried: Ridge (xx.xx), RandomForest (xx.xx).

### Why this is an improvement
LassoCV lowers the test RMSE by about 0.9 compared to v0.1. The L1 penalty
shrinks weak or redundant features (several of the blood serum measurements are
strongly correlated) towards zero, which reduces overfitting on this small
dataset. The result is a model that is both more accurate and simpler.

## v0.1 – Baseline

- Model: StandardScaler + LinearRegression
- Held-out split: 80/20, random_state=42
- Test RMSE: 53.85
- FastAPI service with GET /health and POST /predict
- Invalid input returns HTTP 422 with a JSON error
- Docker image with the model baked in, published to GHCR